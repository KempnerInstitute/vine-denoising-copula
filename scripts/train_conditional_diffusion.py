#!/usr/bin/env python
"""
Conditional Diffusion Training for Copula Density Estimation.

This script trains a diffusion model that is CONDITIONED on histogram input.
Unlike unconditional diffusion which learns the marginal distribution of densities,
this model learns p(density | histogram).

Key differences from unconditional training:
1. Model input: [noisy_log_density, histogram] concatenated along channel dimension
2. The histogram provides context that guides the denoising process
3. At inference: given a histogram, we can sample the corresponding density

This is the approach needed for vine copula integration where we need to
predict density from observed samples.

Usage:
    python scripts/train_conditional_diffusion.py --config configs/train_conditional_diffusion.yaml
    
    # With multiple GPUs
    torchrun --nproc_per_node=4 scripts/train_conditional_diffusion.py --config configs/train_conditional_diffusion.yaml
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

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project
from vdc.data.onthefly import OnTheFlyCopulaDataset
from vdc.data.hist import scatter_to_hist


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
    """Clean up distributed training."""
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
    """
    Create normalized histogram from sample points.
    
    Args:
        samples: (B, N, 2) sample points in [0, 1]^2
        m: Grid resolution
        device: Target device
        
    Returns:
        (B, 1, m, m) normalized histogram
    """
    B = samples.shape[0]
    histograms = []
    
    for b in range(B):
        pts = samples[b].cpu().numpy()
        hist = scatter_to_hist(pts, m, reflect=True)
        histograms.append(hist)
    
    hist_tensor = torch.from_numpy(np.stack(histograms)).float().unsqueeze(1).to(device)
    return normalize_histogram(hist_tensor, m)


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
    Single training step for conditional diffusion.
    
    The key difference from unconditional diffusion:
    - Input to model: [noisy_log_density, histogram] (2 channels)
    - Model learns to predict noise given both noisy density and histogram context
    """
    m = config['data']['m']
    training_cfg = config['training']
    loss_weights = training_cfg.get('loss_weights', {})
    use_amp = training_cfg.get('use_amp', True)
    
    # Get data from batch
    density = batch['density'].to(device)  # (B, 1, m, m) ground truth density
    samples = batch['samples'].to(device)  # (B, N, 2) sample points
    B = density.shape[0]
    
    # Create histogram from samples (this is our conditioning input)
    histogram = create_histogram_from_samples(samples, m, device)
    
    # Normalize histogram for model input (convert to log-space for consistency)
    log_histogram = torch.log(histogram.clamp(min=1e-12))
    
    # Target log-density
    target_log = torch.log(density.clamp(min=1e-12))
    
    # Sample random timesteps
    t = torch.randint(0, diffusion.timesteps, (B,), device=device)
    t_normalized = t.float() / diffusion.timesteps
    
    # Add noise to target log-density
    real_noise = torch.randn_like(target_log)
    noisy_log = diffusion.q_sample(target_log, t, real_noise)
    
    # Concatenate noisy log-density with histogram (conditional input)
    # Shape: (B, 2, m, m)
    model_input = torch.cat([noisy_log, log_histogram], dim=1)
    
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
        
        # Auxiliary losses (weighted by timestep - more weight at low noise)
        aux_weight = (1.0 - t_normalized).clamp(min=0.0).view(-1, 1, 1, 1) ** 2
        
        # ISE loss
        loss_ise = (aux_weight * (recon_density - density) ** 2).mean()
        
        # Marginal uniformity penalty
        row_marg = (recon_density * dv).sum(dim=-1)  # (B, 1, m)
        col_marg = (recon_density * du).sum(dim=-2)  # (B, 1, m)
        loss_marginal = (
            (aux_weight.squeeze(-1) * (row_marg - 1.0) ** 2).mean() +
            (aux_weight.squeeze(-2) * (col_marg - 1.0) ** 2).mean()
        )
        
        # Total loss
        total_loss = (
            loss_weights.get('noise', 1.0) * loss_noise +
            loss_weights.get('ise', 0.5) * loss_ise +
            loss_weights.get('marginal', 0.1) * loss_marginal
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
    }


@torch.no_grad()
def sample_conditional(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    histogram: torch.Tensor,
    device: torch.device,
    num_steps: int = 50,
) -> torch.Tensor:
    """
    Sample density conditioned on histogram using reverse diffusion.
    
    Args:
        model: Trained conditional diffusion model
        diffusion: Diffusion scheduler
        histogram: (B, 1, m, m) conditioning histogram
        device: Target device
        num_steps: Number of sampling steps
        
    Returns:
        (B, 1, m, m) sampled density
    """
    B = histogram.shape[0]
    m = histogram.shape[-1]
    T = diffusion.timesteps
    
    # Normalize histogram and convert to log-space
    du = dv = 1.0 / m
    mass = (histogram * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    histogram = histogram / mass
    log_histogram = torch.log(histogram.clamp(min=1e-12))
    
    # Start from pure noise
    x_t = torch.randn(B, 1, m, m, device=device)
    
    # Create timestep schedule
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
        
        # Concatenate current state with histogram
        model_input = torch.cat([x_t, log_histogram], dim=1)
        
        # Predict noise
        pred_noise = model(model_input, t_normalized)
        
        # Get alpha values
        alpha_t = alphas_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        # Predict x_0
        pred_x0 = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        pred_x0 = pred_x0.clamp(-20, 20)
        
        if t == 0:
            x_t = pred_x0
        else:
            # Get next timestep
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
            else:
                t_prev = 0
            
            alpha_t_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            
            # DDIM update (deterministic)
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
    """Generate and save visualization of model predictions."""
    import matplotlib.pyplot as plt
    
    model.eval()
    m = config['data']['m']
    
    # Get first sample from batch
    density_true = batch['density'][:1].to(device)
    samples = batch['samples'][:1].to(device)
    
    # Create histogram
    histogram = create_histogram_from_samples(samples, m, device)
    
    # Sample from model
    density_pred = sample_conditional(model, diffusion, histogram, device, num_steps=50)
    
    # Apply copula projection
    density_pred = copula_project(density_pred, iters=10)
    
    # Convert to numpy
    density_true_np = density_true[0, 0].cpu().numpy()
    density_pred_np = density_pred[0, 0].cpu().numpy()
    histogram_np = histogram[0, 0].cpu().numpy()
    samples_np = samples[0].cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Samples
    axes[0].scatter(samples_np[:, 0], samples_np[:, 1], s=1, alpha=0.5)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Samples')
    axes[0].set_aspect('equal')
    
    # Histogram
    im1 = axes[1].imshow(histogram_np.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes[1].set_title('Histogram (Input)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Predicted density
    vmax = max(density_true_np.max(), density_pred_np.max())
    im2 = axes[2].imshow(density_pred_np.T, origin='lower', extent=[0, 1, 0, 1], cmap='hot', vmax=vmax)
    axes[2].set_title('Predicted Density')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # True density
    im3 = axes[3].imshow(density_true_np.T, origin='lower', extent=[0, 1, 0, 1], cmap='hot', vmax=vmax)
    axes[3].set_title('True Density')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    
    # Compute correlation
    corr = np.corrcoef(density_true_np.flatten(), density_pred_np.flatten())[0, 1]
    fig.suptitle(f'Step {step} | Correlation: {corr:.4f}')
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'step_{step}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    model.train()
    return corr


def main():
    parser = argparse.ArgumentParser(description='Train conditional diffusion model')
    parser.add_argument('--config', type=Path, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=Path, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if rank == 0:
        print("=" * 60)
        print("Conditional Diffusion Training for Copula Density")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Config: {args.config}")
        print(f"Device: {device}")
        print("=" * 60)
    
    # Create dataset
    data_cfg = config['data']
    dataset = OnTheFlyCopulaDataset(
        n_samples_per_batch=data_cfg.get('n_samples_per_copula', 2000),
        m=data_cfg['m'],
        families=data_cfg['copula_families']['single'],
        param_ranges=data_cfg['param_ranges'],
        rotation_prob=data_cfg.get('rotation_prob', 0.25),
        transform_to_probit_space=data_cfg.get('transform_to_probit_space', False),
        seed=42 + rank,
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )
    
    # Create model
    model_cfg = config['model']
    model = GridUNet(
        m=model_cfg['m'],
        in_channels=model_cfg.get('in_channels', 2),  # [noisy_log_density, histogram]
        base_channels=model_cfg.get('base_channels', 64),
        channel_mults=tuple(model_cfg.get('channel_mults', [1, 2, 3, 4])),
        num_res_blocks=model_cfg.get('num_res_blocks', 2),
        attention_resolutions=tuple(model_cfg.get('attention_resolutions', [16, 8])),
        dropout=model_cfg.get('dropout', 0.1),
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
    
    # Create diffusion
    diff_cfg = config['diffusion']
    diffusion = CopulaAwareDiffusion(
        timesteps=diff_cfg.get('timesteps', 1000),
        beta_schedule=diff_cfg.get('noise_schedule', 'cosine'),
    ).to(device)
    
    # Optimizer
    training_cfg = config['training']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg['learning_rate'],
        weight_decay=training_cfg.get('weight_decay', 1e-4),
    )
    
    scaler = GradScaler(enabled=training_cfg.get('use_amp', True))
    
    # Resume from checkpoint
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model_state = checkpoint['model_state_dict']
        if any(k.startswith('module.') for k in model_state.keys()):
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
        (model.module if hasattr(model, 'module') else model).load_state_dict(model_state)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        if rank == 0:
            print(f"Resumed from step {start_step}")
    
    # Training loop
    max_steps = training_cfg['max_steps']
    log_every = training_cfg.get('log_every', 100)
    save_every = training_cfg.get('save_every', 5000)
    visualize_every = training_cfg.get('visualize_every', 2000)
    
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    step = start_step
    epoch = 0
    
    if rank == 0:
        pbar = tqdm(total=max_steps - start_step, desc='Training', initial=0)
    
    model.train()
    
    try:
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
                
                if rank == 0:
                    pbar.update(1)
                    
                    if step % log_every == 0:
                        pbar.set_postfix({
                            'loss': f'{metrics["loss"]:.4f}',
                            'noise': f'{metrics["noise"]:.4f}',
                            'ise': f'{metrics["ise"]:.4f}',
                        })
                    
                    if step % visualize_every == 0:
                        corr = visualize_samples(
                            model.module if hasattr(model, 'module') else model,
                            diffusion,
                            batch,
                            device,
                            config,
                            step,
                            checkpoint_dir / 'visualizations',
                        )
                        pbar.write(f'[Step {step}] Visualization saved, correlation: {corr:.4f}')
                    
                    if step % save_every == 0:
                        save_model = model.module if hasattr(model, 'module') else model
                        torch.save({
                            'step': step,
                            'epoch': epoch,
                            'model_state_dict': save_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'config': config,
                        }, checkpoint_dir / f'model_step_{step}.pt')
                        pbar.write(f'[Step {step}] Checkpoint saved')
            
            epoch += 1
    
    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted")
    
    finally:
        if rank == 0:
            pbar.close()
            # Save final checkpoint
            save_model = model.module if hasattr(model, 'module') else model
            torch.save({
                'step': step,
                'epoch': epoch,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, checkpoint_dir / 'model_final.pt')
            print(f"Training complete. Final checkpoint saved to {checkpoint_dir}")
        
        cleanup_distributed()


if __name__ == '__main__':
    main()

