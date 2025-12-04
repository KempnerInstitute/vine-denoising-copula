#!/usr/bin/env python
"""
Conditional Diffusion Training V2 - Fixed Conditioning Collapse

Key fixes over V1:
1. Classifier-Free Guidance (CFG): Randomly drop histogram conditioning during training
2. Anti-copying loss: Penalize when output is too similar to input histogram
3. Stronger smoothness (TV) loss: Force smooth outputs
4. Log-space density matching: Better magnitude learning

Usage:
    python scripts/train_conditional_diffusion_v2.py --config configs/train_conditional_diffusion_v2.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project
from vdc.data.onthefly import OnTheFlyCopulaDataset
from vdc.data.hist import scatter_to_hist
from vdc.utils.smoothing import log_total_variation_loss


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def normalize_histogram(hist: torch.Tensor, m: int) -> torch.Tensor:
    """Normalize histogram to have unit mass."""
    du = dv = 1.0 / m
    mass = (hist * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return hist / mass


def create_histogram_from_samples(
    samples: torch.Tensor,
    m: int,
    device: torch.device,
) -> torch.Tensor:
    """Create normalized histogram from sample points."""
    B = samples.shape[0]
    histograms = []
    
    for b in range(B):
        pts = samples[b].cpu().numpy()
        hist = scatter_to_hist(pts, m, reflect=True)
        histograms.append(hist)
    
    hist_tensor = torch.from_numpy(np.stack(histograms)).float().unsqueeze(1).to(device)
    return normalize_histogram(hist_tensor, m)


def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute correlation between pred and target.
    Returns value in [0, 1] where 0 = uncorrelated, 1 = perfectly correlated.
    """
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    
    pred_std = pred_centered.std(dim=1, keepdim=True).clamp_min(1e-8)
    target_std = target_centered.std(dim=1, keepdim=True).clamp_min(1e-8)
    
    corr = (pred_centered * target_centered).mean(dim=1) / (pred_std.squeeze() * target_std.squeeze())
    return corr.mean()


def training_step(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: Dict,
    step: int,
) -> Dict[str, float]:
    """
    Training step with fixes for conditioning collapse:
    1. Classifier-free guidance (random histogram dropout)
    2. Anti-copying loss
    3. Strong smoothness regularization
    """
    m = config['data']['m']
    training_cfg = config['training']
    loss_weights = training_cfg.get('loss_weights', {})
    use_amp = training_cfg.get('use_amp', True)
    
    # CFG parameters
    cfg_dropout_prob = training_cfg.get('cfg_dropout_prob', 0.15)  # Drop histogram 15% of time
    
    # Get data
    density = batch['density'].to(device)  # (B, 1, m, m) ground truth density
    samples = batch['samples'].to(device)  # (B, N, 2) sample points
    B = density.shape[0]
    
    # Create histogram from samples
    histogram = create_histogram_from_samples(samples, m, device)
    
    # Normalize histogram for model input
    log_histogram = torch.log(histogram.clamp(min=1e-12))
    
    # TARGET: log-density (what we want to predict)
    target_log = torch.log(density.clamp(min=1e-12))
    
    # Sample random timesteps
    t = torch.randint(0, diffusion.timesteps, (B,), device=device)
    t_normalized = t.float() / diffusion.timesteps
    
    # Add noise to target log-density
    real_noise = torch.randn_like(target_log)
    noisy_log = diffusion.q_sample(target_log, t, real_noise)
    
    # ============================================================
    # FIX 1: Classifier-Free Guidance - randomly drop histogram
    # ============================================================
    cfg_mask = torch.rand(B, 1, 1, 1, device=device) > cfg_dropout_prob
    log_histogram_masked = log_histogram * cfg_mask.float()  # Zero out histogram for some samples
    
    # Concatenate: [noisy_log_density, histogram (possibly zeroed)]
    model_input = torch.cat([noisy_log, log_histogram_masked], dim=1)
    
    optimizer.zero_grad()
    
    with autocast(device_type='cuda', enabled=use_amp):
        # Model predicts noise
        pred_noise = model(model_input, t_normalized)
        
        # Primary loss: noise prediction
        loss_noise = F.mse_loss(pred_noise, real_noise)
        
        # Reconstruct density for auxiliary losses
        alpha_t = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        recon_log = (noisy_log - sqrt_one_minus_alpha_t * pred_noise.clamp(-10, 10)) / sqrt_alpha_t
        recon_log = recon_log.clamp(-20, 20)
        recon_density = torch.exp(recon_log).clamp(1e-12, 1e6)
        
        # Normalize reconstructed density
        du = dv = 1.0 / m
        mass = (recon_density * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        recon_density = recon_density / mass
        
        # Auxiliary weight (more weight at low noise)
        aux_weight = (1.0 - t_normalized).clamp(min=0.0).view(-1, 1, 1, 1) ** 2
        
        # ISE loss (density space)
        loss_ise = (aux_weight * (recon_density - density) ** 2).mean()
        
        # Marginal uniformity
        row_marg = (recon_density * dv).sum(dim=-1)
        col_marg = (recon_density * du).sum(dim=-2)
        loss_marginal = (
            (aux_weight.squeeze(-1) * (row_marg - 1.0) ** 2).mean() +
            (aux_weight.squeeze(-2) * (col_marg - 1.0) ** 2).mean()
        )
        
        # Log-space loss for magnitude matching
        target_log_clipped = torch.log(density.clamp(min=1e-12))
        recon_log_clipped = torch.log(recon_density.clamp(min=1e-12))
        loss_log = (aux_weight * (target_log_clipped - recon_log_clipped) ** 2).mean()
        
        # Tail loss (corners)
        tail_size = max(5, m // 12)
        corner_mask = torch.zeros_like(density)
        corner_mask[:, :, :tail_size, :tail_size] = 1.0
        corner_mask[:, :, :tail_size, -tail_size:] = 1.0
        corner_mask[:, :, -tail_size:, :tail_size] = 1.0
        corner_mask[:, :, -tail_size:, -tail_size:] = 1.0
        tail_weight = corner_mask * 4.0 + (1 - corner_mask) * 1.0
        loss_tail = (aux_weight * tail_weight * (target_log_clipped - recon_log_clipped) ** 2).mean()
        
        # ============================================================
        # FIX 2: Strong TV smoothness loss
        # ============================================================
        loss_tv = log_total_variation_loss(recon_log, weight=1.0)
        
        # ============================================================
        # FIX 3: Anti-copying loss - penalize correlation with histogram
        # ============================================================
        # We want the output to NOT be a copy of the histogram
        corr_with_hist = correlation_loss(recon_density, histogram)
        # Correlation with true density (we want this to be high)
        corr_with_true = correlation_loss(recon_density, density)
        # Anti-copying: penalize if corr_with_hist > corr_with_true
        loss_anticopy = F.relu(corr_with_hist - corr_with_true + 0.1)  # Margin of 0.1
        
        # Total loss
        total_loss = (
            loss_weights.get('noise', 1.0) * loss_noise +
            loss_weights.get('ise', 0.5) * loss_ise +
            loss_weights.get('marginal', 0.1) * loss_marginal +
            loss_weights.get('log', 0.3) * loss_log +
            loss_weights.get('tail', 0.3) * loss_tail +
            loss_weights.get('tv', 0.1) * loss_tv +
            loss_weights.get('anticopy', 0.5) * loss_anticopy  # NEW: anti-copying
        )
    
    # Backward pass
    if use_amp:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.get('gradient_clip', 1.0))
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.get('gradient_clip', 1.0))
        optimizer.step()
    
    return {
        'loss': float(total_loss.item()),
        'noise': float(loss_noise.item()),
        'ise': float(loss_ise.item()),
        'marginal': float(loss_marginal.item()),
        'log': float(loss_log.item()),
        'tail': float(loss_tail.item()),
        'tv': float(loss_tv.item()),
        'anticopy': float(loss_anticopy.item()),
        'corr_hist': float(corr_with_hist.item()),
        'corr_true': float(corr_with_true.item()),
    }


@torch.no_grad()
def sample_conditional(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    histogram: torch.Tensor,
    device: torch.device,
    num_steps: int = 50,
    cfg_scale: float = 2.0,  # Classifier-free guidance scale
) -> torch.Tensor:
    """
    Sample with classifier-free guidance.
    
    cfg_scale > 1 emphasizes the conditioning (histogram).
    """
    B = histogram.shape[0]
    m = histogram.shape[-1]
    T = diffusion.timesteps
    
    # Normalize histogram
    du = dv = 1.0 / m
    mass = (histogram * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    histogram = histogram / mass
    log_histogram = torch.log(histogram.clamp(min=1e-12))
    
    # Start from noise
    x_t = torch.randn(B, 1, m, m, device=device)
    
    # Timestep schedule
    if num_steps >= T:
        timesteps = list(range(T - 1, -1, -1))
    else:
        step_size = T // num_steps
        timesteps = list(range(T - 1, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)
    
    alphas_cumprod = diffusion.alphas_cumprod.to(device)
    
    for i, t in enumerate(timesteps):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        t_normalized = t_tensor.float() / T
        
        # Conditional prediction (with histogram)
        model_input_cond = torch.cat([x_t, log_histogram], dim=1)
        pred_noise_cond = model(model_input_cond, t_normalized)
        
        # Unconditional prediction (without histogram) for CFG
        log_histogram_uncond = torch.zeros_like(log_histogram)
        model_input_uncond = torch.cat([x_t, log_histogram_uncond], dim=1)
        pred_noise_uncond = model(model_input_uncond, t_normalized)
        
        # CFG: pred = uncond + scale * (cond - uncond)
        pred_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
        
        # DDIM update
        alpha_t = alphas_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        pred_x0 = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        pred_x0 = pred_x0.clamp(-20, 20)
        
        if t == 0:
            x_t = pred_x0
        else:
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
            else:
                t_prev = 0
            
            alpha_t_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            dir_xt = torch.sqrt(1 - alpha_t_prev) * pred_noise
            x_t = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
    
    # Convert to density
    density = torch.exp(x_t).clamp(1e-12, 1e6)
    mass = (density * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return density / mass


def visualize_samples(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    batch: Dict,
    device: torch.device,
    config: Dict,
    step: int,
    output_dir: Path,
):
    """Generate visualization with correlation metrics."""
    import matplotlib.pyplot as plt
    
    model.eval()
    m = config['data']['m']
    
    density_true = batch['density'][:1].to(device)
    samples = batch['samples'][:1].to(device)
    histogram = create_histogram_from_samples(samples, m, device)
    
    # Sample with CFG
    density_pred = sample_conditional(model, diffusion, histogram, device, num_steps=50, cfg_scale=2.0)
    density_pred = copula_project(density_pred, iters=10)
    
    # Compute correlations
    corr_hist = correlation_loss(density_pred, histogram).item()
    corr_true = correlation_loss(density_pred, density_true).item()
    
    # Convert to numpy
    density_true_np = density_true[0, 0].cpu().numpy()
    density_pred_np = density_pred[0, 0].cpu().numpy()
    histogram_np = histogram[0, 0].cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    vmax = min(50, max(density_true_np.max(), density_pred_np.max()))
    
    axes[0].imshow(histogram_np, origin='lower', cmap='hot', vmin=0, vmax=vmax)
    axes[0].set_title('Input Histogram')
    
    axes[1].imshow(density_pred_np, origin='lower', cmap='hot', vmin=0, vmax=vmax)
    axes[1].set_title(f'Predicted (CFG=2.0)\nCorr w/ hist: {corr_hist:.3f}')
    
    axes[2].imshow(np.clip(density_true_np, 0, vmax), origin='lower', cmap='hot', vmin=0, vmax=vmax)
    axes[2].set_title(f'True Density\nCorr w/ pred: {corr_true:.3f}')
    
    # Difference
    diff = density_pred_np - np.clip(density_true_np, 0, vmax)
    axes[3].imshow(diff, origin='lower', cmap='RdBu', vmin=-vmax/2, vmax=vmax/2)
    axes[3].set_title('Difference')
    
    plt.suptitle(f'Step {step} | Goal: corr_true > corr_hist', fontsize=12)
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'step_{step}.png', dpi=150)
    plt.close()
    
    print(f'[Step {step}] Corr w/ histogram: {corr_hist:.4f}, Corr w/ true: {corr_true:.4f}')
    
    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print('=' * 60)
        print('Conditional Diffusion Training V2 (Fixed Conditioning Collapse)')
        print('=' * 60)
        print(f'World size: {world_size}')
        print(f'Config: {args.config}')
        print(f'Device: {device}')
        print('Key fixes:')
        print(f'  - CFG dropout prob: {config["training"].get("cfg_dropout_prob", 0.15)}')
        print(f'  - Anti-copy loss weight: {config["training"]["loss_weights"].get("anticopy", 0.5)}')
        print('=' * 60)
    
    # Build model
    model_cfg = config['model']
    model = GridUNet(
        m=config['data']['m'],
        in_channels=model_cfg.get('in_channels', 2),
        base_channels=model_cfg.get('base_channels', 64),
        channel_mults=tuple(model_cfg.get('channel_mults', (1, 2, 3, 4))),
        num_res_blocks=model_cfg.get('num_res_blocks', 2),
        attention_resolutions=tuple(model_cfg.get('attention_resolutions', (16, 8))),
        dropout=model_cfg.get('dropout', 0.1),
    ).to(device)
    
    if rank == 0:
        print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Build diffusion
    diff_cfg = config.get('diffusion', {})
    diffusion = CopulaAwareDiffusion(
        timesteps=diff_cfg.get('timesteps', 1000),
        beta_schedule=diff_cfg.get('noise_schedule', 'cosine'),
        beta_start=diff_cfg.get('beta_start', 1e-4),
        beta_end=diff_cfg.get('beta_end', 0.02),
    ).to(device)
    
    # Wrap for DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Dataset and dataloader
    dataset = OnTheFlyCopulaDataset(
        m=config['data']['m'],
        n_samples_per_batch=config['data'].get('n_samples_per_copula', 2000),
        families=config['data'].get('copula_families', {}).get('single', None),
        param_ranges=config['data'].get('param_ranges', {}),
        rotation_prob=config['data'].get('rotation_prob', 0.25),
        mixture_prob=config['data'].get('copula_families', {}).get('mixture', {}).get('prob', 0.0),
    )
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )
    
    # Optimizer
    training_cfg = config['training']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get('learning_rate', 1e-4),
        weight_decay=training_cfg.get('weight_decay', 1e-4),
    )
    
    scaler = GradScaler(enabled=training_cfg.get('use_amp', True))
    
    # Training loop
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/conditional_diffusion_v2'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = checkpoint_dir / 'visualizations'
    
    max_steps = training_cfg.get('max_steps', 150000)
    log_every = training_cfg.get('log_every', 100)
    save_every = training_cfg.get('save_every', 10000)
    vis_every = training_cfg.get('visualize_every', 5000)
    
    step = 0
    epoch = 0
    
    model.train()
    pbar = tqdm(total=max_steps, disable=(rank != 0))
    
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        for batch in dataloader:
            if step >= max_steps:
                break
            
            metrics = training_step(
                model, diffusion, batch, optimizer, scaler, device, config, step
            )
            
            step += 1
            pbar.update(1)
            
            if rank == 0 and step % log_every == 0:
                pbar.set_postfix({
                    'loss': f'{metrics["loss"]:.4f}',
                    'c_hist': f'{metrics["corr_hist"]:.3f}',
                    'c_true': f'{metrics["corr_true"]:.3f}',
                })
            
            if rank == 0 and step % vis_every == 0:
                visualize_samples(
                    model.module if world_size > 1 else model,
                    diffusion, batch, device, config, step, vis_dir
                )
            
            if rank == 0 and step % save_every == 0:
                torch.save({
                    'step': step,
                    'model_state_dict': (model.module if world_size > 1 else model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                }, checkpoint_dir / f'model_step_{step}.pt')
                print(f'[Step {step}] Checkpoint saved')
        
        epoch += 1
    
    pbar.close()
    
    # Save final
    if rank == 0:
        torch.save({
            'step': step,
            'model_state_dict': (model.module if world_size > 1 else model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
        }, checkpoint_dir / 'model_final.pt')
        print(f'Training complete. Final checkpoint saved to {checkpoint_dir}')
    
    cleanup_distributed()


if __name__ == '__main__':
    main()

