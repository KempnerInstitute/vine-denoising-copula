#!/usr/bin/env python
"""
Direct Histogram-to-Density Prediction Training.

This is NOT a diffusion model. Instead, it's a direct regression model:
- Input: 2D histogram from samples
- Output: Copula log-density

This approach is simpler and faster than diffusion, and directly optimizes
for density accuracy. Good for vine copula integration where speed matters.

The model learns: density = f(histogram)

Usage:
    python scripts/train_direct_hist2density.py --config configs/train_direct_hist2density.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

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
    
    # Normalize to unit mass
    du = dv = 1.0 / m
    mass = (hist_tensor * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return hist_tensor / mass


def add_coordinate_channels(x: torch.Tensor, m: int) -> torch.Tensor:
    """Add coordinate channels to input tensor."""
    B = x.shape[0]
    device = x.device
    
    # Create coordinate grids
    u = torch.linspace(0, 1, m, device=device)
    v = torch.linspace(0, 1, m, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    
    # Expand to batch size
    uu = uu.unsqueeze(0).unsqueeze(0).expand(B, 1, m, m)
    vv = vv.unsqueeze(0).unsqueeze(0).expand(B, 1, m, m)
    
    return torch.cat([x, uu, vv], dim=1)


def compute_losses(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    target_density: torch.Tensor,
    m: int,
    loss_weights: Dict,
) -> Dict[str, torch.Tensor]:
    """Compute all loss components."""
    du = dv = 1.0 / m
    
    # Predicted density
    pred_density = torch.exp(pred_log.clamp(-20, 20)).clamp(1e-12, 1e6)
    mass = (pred_density * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    pred_density = pred_density / mass
    
    losses = {}
    
    # ISE: Integrated squared error on density
    if loss_weights.get('ise', 0) > 0:
        losses['ise'] = ((pred_density - target_density) ** 2).mean()
    
    # Log-ISE: ISE in log-space for better shape matching
    if loss_weights.get('log_ise', 0) > 0:
        pred_log_safe = torch.log(pred_density.clamp(min=1e-12))
        target_log_safe = torch.log(target_density.clamp(min=1e-12))
        losses['log_ise'] = ((pred_log_safe - target_log_safe) ** 2).mean()
    
    # Marginal uniformity
    if loss_weights.get('marginal', 0) > 0:
        row_marg = (pred_density * dv).sum(dim=-1)  # (B, 1, m)
        col_marg = (pred_density * du).sum(dim=-2)  # (B, 1, m)
        losses['marginal'] = (
            ((row_marg - 1.0) ** 2).mean() +
            ((col_marg - 1.0) ** 2).mean()
        )
    
    # Tail loss: emphasize accuracy in tail regions (corners)
    if loss_weights.get('tail', 0) > 0:
        # Create tail mask (corners and edges)
        u = torch.linspace(0, 1, m, device=pred_density.device)
        v = torch.linspace(0, 1, m, device=pred_density.device)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        
        # Distance to boundary
        dist_to_boundary = torch.min(
            torch.min(uu, 1 - uu),
            torch.min(vv, 1 - vv)
        )
        tail_weight = torch.exp(-dist_to_boundary * 10)  # Higher weight near boundaries
        tail_weight = tail_weight.unsqueeze(0).unsqueeze(0)
        
        losses['tail'] = (tail_weight * (pred_density - target_density) ** 2).mean()
    
    # NLL: Negative log-likelihood
    if loss_weights.get('nll', 0) > 0:
        # Weight by target density (important regions)
        nll = -target_density * pred_log_safe
        losses['nll'] = (nll * du * dv).sum(dim=(-2, -1)).mean()
    
    return losses, pred_density


def training_step(
    model: torch.nn.Module,
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: Dict,
    step: int,
) -> Dict[str, float]:
    """Single training step for direct prediction."""
    m = config['data']['m']
    training_cfg = config['training']
    loss_weights = training_cfg.get('loss_weights', {})
    use_amp = training_cfg.get('use_amp', True)
    use_coords = config['model'].get('use_coordinates', True)
    
    # Get data from batch
    density = batch['density'].to(device)  # (B, 1, m, m) ground truth density
    samples = batch['samples'].to(device)  # (B, N, 2) sample points
    
    # Create histogram from samples (this is our input)
    histogram = create_histogram_from_samples(samples, m, device)
    
    # Convert to log-space for model input
    log_histogram = torch.log(histogram.clamp(min=1e-12))
    
    # Add coordinate channels if requested
    if use_coords:
        model_input = add_coordinate_channels(log_histogram, m)
    else:
        model_input = log_histogram
    
    # Target log-density
    target_log = torch.log(density.clamp(min=1e-12))
    
    optimizer.zero_grad()
    
    # Dummy time embedding (not used for direct prediction, but required by UNet)
    t_dummy = torch.zeros(density.shape[0], device=device)
    
    with autocast(device_type='cuda', enabled=use_amp):
        # Direct prediction: histogram -> log-density
        pred_log = model(model_input, t_dummy)
        
        # Compute losses
        losses, pred_density = compute_losses(
            pred_log, target_log, density, m, loss_weights
        )
        
        # Total loss
        total_loss = sum(
            loss_weights.get(name, 0) * loss
            for name, loss in losses.items()
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
        **{k: float(v.item()) for k, v in losses.items()},
    }


@torch.no_grad()
def predict_density(
    model: torch.nn.Module,
    histogram: torch.Tensor,
    m: int,
    use_coords: bool = True,
    projection_iters: int = 15,
) -> torch.Tensor:
    """
    Predict density from histogram using direct model.
    
    Args:
        model: Trained direct prediction model
        histogram: (B, 1, m, m) input histogram
        m: Grid resolution
        use_coords: Whether to add coordinate channels
        projection_iters: Number of copula projection iterations
        
    Returns:
        (B, 1, m, m) predicted density
    """
    device = histogram.device
    du = dv = 1.0 / m
    
    # Normalize histogram
    mass = (histogram * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    histogram = histogram / mass
    
    # Convert to log-space
    log_histogram = torch.log(histogram.clamp(min=1e-12))
    
    # Add coordinate channels if requested
    if use_coords:
        model_input = add_coordinate_channels(log_histogram, m)
    else:
        model_input = log_histogram
    
    # Dummy time embedding
    t_dummy = torch.zeros(histogram.shape[0], device=device)
    
    # Predict
    pred_log = model(model_input, t_dummy)
    
    # Convert to density
    pred_density = torch.exp(pred_log.clamp(-20, 20)).clamp(1e-12, 1e6)
    mass = (pred_density * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    pred_density = pred_density / mass
    
    # Apply copula projection
    if projection_iters > 0:
        pred_density = copula_project(pred_density, iters=projection_iters)
        mass = (pred_density * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        pred_density = pred_density / mass
    
    return pred_density


def visualize_samples(
    model: torch.nn.Module,
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
    use_coords = config['model'].get('use_coordinates', True)
    
    # Get first sample from batch
    density_true = batch['density'][:1].to(device)
    samples = batch['samples'][:1].to(device)
    
    # Create histogram
    histogram = create_histogram_from_samples(samples, m, device)
    
    # Predict density
    density_pred = predict_density(model, histogram, m, use_coords, projection_iters=15)
    
    # Convert to numpy
    density_true_np = density_true[0, 0].cpu().numpy()
    density_pred_np = density_pred[0, 0].cpu().numpy()
    histogram_np = histogram[0, 0].cpu().numpy()
    samples_np = samples[0].cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Linear scale
    axes[0, 0].scatter(samples_np[:, 0], samples_np[:, 1], s=1, alpha=0.5)
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title('Samples')
    axes[0, 0].set_aspect('equal')
    
    im1 = axes[0, 1].imshow(histogram_np.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes[0, 1].set_title('Histogram (Input)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    vmax_pred = np.percentile(density_pred_np, 99)
    im2 = axes[0, 2].imshow(density_pred_np.T, origin='lower', extent=[0, 1, 0, 1], cmap='hot', vmin=0, vmax=vmax_pred)
    axes[0, 2].set_title(f'Predicted (max={density_pred_np.max():.1f})')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    vmax_true = np.percentile(density_true_np, 99)
    im3 = axes[0, 3].imshow(density_true_np.T, origin='lower', extent=[0, 1, 0, 1], cmap='hot', vmin=0, vmax=vmax_true)
    axes[0, 3].set_title(f'True (max={density_true_np.max():.1f})')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)
    
    # Row 2: Log scale
    log_hist = np.log(histogram_np.clip(min=1e-10))
    im4 = axes[1, 0].imshow(log_hist.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes[1, 0].set_title('Log Histogram')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    log_pred = np.log(density_pred_np.clip(min=1e-10))
    log_true = np.log(density_true_np.clip(min=1e-10))
    vmin_log = min(np.percentile(log_pred, 1), np.percentile(log_true, 1))
    vmax_log = max(np.percentile(log_pred, 99), np.percentile(log_true, 99))
    
    im5 = axes[1, 1].imshow(log_pred.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r', vmin=vmin_log, vmax=vmax_log)
    axes[1, 1].set_title('Log Predicted')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    im6 = axes[1, 2].imshow(log_true.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r', vmin=vmin_log, vmax=vmax_log)
    axes[1, 2].set_title('Log True')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    # Error map
    error = density_pred_np - density_true_np
    vmax_err = np.percentile(np.abs(error), 95)
    im7 = axes[1, 3].imshow(error.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r', vmin=-vmax_err, vmax=vmax_err)
    axes[1, 3].set_title('Error (Pred - True)')
    plt.colorbar(im7, ax=axes[1, 3], fraction=0.046)
    
    # Compute metrics
    corr = np.corrcoef(density_true_np.flatten(), density_pred_np.flatten())[0, 1]
    log_corr = np.corrcoef(log_true.flatten(), log_pred.flatten())[0, 1]
    mse = np.mean((density_true_np - density_pred_np) ** 2)
    
    fig.suptitle(f'Step {step} | Corr: {corr:.4f} | Log-Corr: {log_corr:.4f} | MSE: {mse:.2f}')
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'step_{step}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    model.train()
    return corr, log_corr


def main():
    parser = argparse.ArgumentParser(description='Train direct histogram-to-density model')
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
        print("Direct Histogram-to-Density Training")
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
    use_coords = model_cfg.get('use_coordinates', True)
    in_channels = 3 if use_coords else 1  # histogram + optional coordinates
    
    model = GridUNet(
        m=model_cfg['m'],
        in_channels=in_channels,
        out_channels=model_cfg.get('out_channels', 1),
        base_channels=model_cfg.get('base_channels', 64),
        channel_mults=tuple(model_cfg.get('channel_mults', [1, 2, 4, 8])),
        num_res_blocks=model_cfg.get('num_res_blocks', 3),
        attention_resolutions=tuple(model_cfg.get('attention_resolutions', [16, 8, 4])),
        dropout=model_cfg.get('dropout', 0.1),
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        print(f"Using coordinates: {use_coords}")
    
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
                    model, batch, optimizer, scaler, device, config, step
                )
                step += 1
                
                if rank == 0:
                    pbar.update(1)
                    
                    if step % log_every == 0:
                        loss_str = f'loss={metrics["loss"]:.4f}'
                        if 'ise' in metrics:
                            loss_str += f' ise={metrics["ise"]:.4f}'
                        if 'log_ise' in metrics:
                            loss_str += f' log_ise={metrics["log_ise"]:.4f}'
                        pbar.set_postfix_str(loss_str)
                    
                    if step % visualize_every == 0:
                        corr, log_corr = visualize_samples(
                            model.module if hasattr(model, 'module') else model,
                            batch,
                            device,
                            config,
                            step,
                            checkpoint_dir / 'visualizations',
                        )
                        pbar.write(f'[Step {step}] Visualization saved, corr={corr:.4f}, log_corr={log_corr:.4f}')
                    
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

