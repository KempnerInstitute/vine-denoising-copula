#!/usr/bin/env python
"""
Inference Script for Vine Diffusion Copula.

Supports:
- Density estimation from samples
- Sample generation from fitted vine
- Visualization of results

Usage:
    # Estimate density for bivariate data
    python scripts/infer.py density --checkpoint path/to/model.pt --data samples.npy
    
    # Generate samples from fitted vine
    python scripts/infer.py sample --checkpoint path/to/model.pt --n-samples 1000
    
    # Visualize model outputs
    python scripts/infer.py visualize --checkpoint path/to/model.pt
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.config import get_run_dir
from vdc.inference.density import sample_density_grid, scatter_to_hist


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, CopulaAwareDiffusion, Dict]:
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model_cfg = config.get('model', {})
    m = config.get('data', {}).get('m', 64)
    
    # Determine input channels based on config.
    # For conditional diffusion, diffusion_unet may use in_channels=2: [x_t, log_histogram].
    in_channels = int(model_cfg.get('in_channels', 1))
    
    model = GridUNet(
        m=m,
        in_channels=in_channels,
        base_channels=model_cfg.get('base_channels', 64),
        channel_mults=tuple(model_cfg.get('channel_mults', (1, 2, 3, 4))),
        num_res_blocks=model_cfg.get('num_res_blocks', 2),
        attention_resolutions=tuple(model_cfg.get('attention_resolutions', (16, 8))),
        dropout=model_cfg.get('dropout', 0.1),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    diffusion = CopulaAwareDiffusion(
        timesteps=config.get('diffusion', {}).get('timesteps', 1000),
        beta_schedule=config.get('diffusion', {}).get('noise_schedule', 'cosine'),
    ).to(device)
    
    step = checkpoint.get('step', 'unknown')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded (step {step}, {n_params:,} parameters)")
    
    return model, diffusion, config


def visualize_density(
    density: np.ndarray,
    samples: Optional[np.ndarray],
    output_path: Path,
    title: str = "Estimated Copula Density",
):
    """Create visualization of estimated density."""
    fig, axes = plt.subplots(1, 3 if samples is not None else 2, figsize=(15, 5))
    
    # Density heatmap
    im = axes[0].imshow(density, origin='lower', cmap='hot', extent=[0, 1, 0, 1])
    axes[0].set_xlabel('u')
    axes[0].set_ylabel('v')
    axes[0].set_title('Estimated Density')
    plt.colorbar(im, ax=axes[0])
    
    # Log density
    log_density = np.log(np.clip(density, 1e-12, None))
    im2 = axes[1].imshow(log_density, origin='lower', cmap='viridis', extent=[0, 1, 0, 1])
    axes[1].set_xlabel('u')
    axes[1].set_ylabel('v')
    axes[1].set_title('Log Density')
    plt.colorbar(im2, ax=axes[1])
    
    # Scatter plot of samples
    if samples is not None:
        axes[2].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].set_xlabel('u')
        axes[2].set_ylabel('v')
        axes[2].set_title(f'Samples (n={len(samples)})')
        axes[2].set_aspect('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization: {output_path}")


def cmd_density(args):
    """Estimate density from samples."""
    device = torch.device(args.device)
    
    # Load model
    model, diffusion, config = load_model(args.checkpoint, device)
    m = config.get('data', {}).get('m', 64)
    
    # Check if model uses histogram conditioning
    use_histogram = config.get('model', {}).get('in_channels', 1) > 1
    
    # Load samples
    if args.data.suffix == '.npy':
        samples = np.load(args.data)
    elif args.data.suffix == '.csv':
        samples = np.loadtxt(args.data, delimiter=',')
    else:
        raise ValueError(f"Unsupported data format: {args.data.suffix}")
    
    if samples.shape[1] != 2:
        raise ValueError(f"Expected bivariate data (n, 2), got shape {samples.shape}")
    
    print(f"Loaded {len(samples)} samples from {args.data}")
    
    # Estimate density
    print("Estimating density...")
    density = estimate_density(
        model=model,
        diffusion=diffusion,
        samples=samples,
        m=m,
        device=device,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        use_histogram_conditioning=use_histogram,
    )
    
    # Create output directory
    output_dir = get_run_dir(
        base_dir=args.output,
        experiment_name="density_estimation",
        include_timestamp=True,
        include_job_id=True,
    )
    
    # Save density
    np.save(output_dir / 'density.npy', density)
    print(f"✓ Saved density: {output_dir / 'density.npy'}")
    
    # Visualize
    visualize_density(density, samples, output_dir / 'figures' / 'density.png')
    
    # Save metadata
    metadata = {
        'checkpoint': str(args.checkpoint),
        'data': str(args.data),
        'n_samples': len(samples),
        'grid_size': m,
        'num_steps': args.num_steps,
        'cfg_scale': args.cfg_scale,
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}")


def cmd_sample(args):
    """Generate samples from model."""
    device = torch.device(args.device)
    
    # Load model
    model, diffusion, config = load_model(args.checkpoint, device)
    m = config.get('data', {}).get('m', 64)
    
    print(f"Generating {args.n_samples} samples...")
    
    # For now, generate from uniform and use inverse Rosenblatt
    # This is a placeholder - full vine sampling would be more sophisticated
    np.random.seed(args.seed)
    samples = np.random.uniform(0, 1, (args.n_samples, args.dim))
    
    # Create output directory
    output_dir = get_run_dir(
        base_dir=args.output,
        experiment_name="sampling",
        include_timestamp=True,
        include_job_id=True,
    )
    
    # Save samples
    np.save(output_dir / 'samples.npy', samples)
    print(f"✓ Saved samples: {output_dir / 'samples.npy'}")
    
    # Visualize first two dimensions
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('$u_1$')
    ax.set_ylabel('$u_2$')
    ax.set_title(f'Generated Samples (n={args.n_samples})')
    ax.set_aspect('equal')
    plt.savefig(output_dir / 'figures' / 'samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Results saved to: {output_dir}")


def cmd_visualize(args):
    """Visualize model on test copulas."""
    device = torch.device(args.device)
    
    # Load model
    model, diffusion, config = load_model(args.checkpoint, device)
    m = config.get('data', {}).get('m', 64)
    use_histogram = config.get('model', {}).get('in_channels', 1) > 1
    
    # Test copulas
    from vdc.data.generators import (
        sample_gaussian_copula, sample_clayton_copula, 
        sample_frank_copula, sample_gumbel_copula
    )
    
    test_copulas = [
        ('Gaussian(ρ=0.7)', lambda: sample_gaussian_copula(1000, 0.7)),
        ('Gaussian(ρ=-0.7)', lambda: sample_gaussian_copula(1000, -0.7)),
        ('Clayton(θ=3)', lambda: sample_clayton_copula(1000, 3.0)),
        ('Frank(θ=5)', lambda: sample_frank_copula(1000, 5.0)),
    ]
    
    # Create output directory
    output_dir = get_run_dir(
        base_dir=args.output,
        experiment_name="visualization",
        include_timestamp=True,
        include_job_id=True,
    )
    
    print("Generating visualizations...")
    
    for name, sampler in test_copulas:
        print(f"  Processing: {name}")
        samples = sampler()
        
        density = estimate_density(
            model, diffusion, samples, m, device,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            use_histogram_conditioning=use_histogram,
        )
        
        safe_name = name.replace('(', '_').replace(')', '').replace('=', '').replace(' ', '_')
        visualize_density(
            density, samples,
            output_dir / 'figures' / f'{safe_name}.png',
            title=name
        )
    
    print(f"\n✓ Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference for Vine Diffusion Copula",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Inference task')
    
    # Density estimation
    density_parser = subparsers.add_parser('density', help='Estimate copula density from samples')
    density_parser.add_argument('--checkpoint', type=Path, required=True, help='Model checkpoint')
    density_parser.add_argument('--data', type=Path, required=True, help='Input samples (npy/csv)')
    density_parser.add_argument('--output', type=Path, default=Path('results'), help='Output directory')
    density_parser.add_argument('--num-steps', type=int, default=50, help='Diffusion sampling steps')
    density_parser.add_argument('--cfg-scale', type=float, default=2.0, help='CFG scale')
    density_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sample generation
    sample_parser = subparsers.add_parser('sample', help='Generate samples from model')
    sample_parser.add_argument('--checkpoint', type=Path, required=True, help='Model checkpoint')
    sample_parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples')
    sample_parser.add_argument('--dim', type=int, default=2, help='Dimension')
    sample_parser.add_argument('--output', type=Path, default=Path('results'), help='Output directory')
    sample_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    sample_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Visualization
    viz_parser = subparsers.add_parser('visualize', help='Visualize model on test copulas')
    viz_parser.add_argument('--checkpoint', type=Path, required=True, help='Model checkpoint')
    viz_parser.add_argument('--output', type=Path, default=Path('results'), help='Output directory')
    viz_parser.add_argument('--num-steps', type=int, default=50, help='Diffusion sampling steps')
    viz_parser.add_argument('--cfg-scale', type=float, default=2.0, help='CFG scale')
    viz_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    print("=" * 60)
    print("Vine Diffusion Copula - Inference")
    print("=" * 60)
    
    if args.command == 'density':
        cmd_density(args)
    elif args.command == 'sample':
        cmd_sample(args)
    elif args.command == 'visualize':
        cmd_visualize(args)


if __name__ == '__main__':
    main()
