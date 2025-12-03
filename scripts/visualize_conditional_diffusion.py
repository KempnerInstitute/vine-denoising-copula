#!/usr/bin/env python
"""
Visualization script for conditional diffusion copula models.

This script generates visualizations comparing predicted vs true copula densities
for models trained with conditional diffusion (histogram as input).

Usage:
    python scripts/visualize_conditional_diffusion.py \
        --checkpoint checkpoints/conditional_diffusion_m64/model_step_5000.pt \
        --output-dir results/conditional_viz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project
from vdc.data.generators import analytic_logpdf_grid, sample_bicop
from vdc.data.hist import scatter_to_hist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize conditional diffusion model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to checkpoint_dir/visualizations/eval)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of samples per copula",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=50,
        help="Number of diffusion sampling steps",
    )
    parser.add_argument(
        "--sdedit",
        action="store_true",
        help="Use SDEdit-style sampling (start from noisy histogram instead of pure noise)",
    )
    parser.add_argument(
        "--start-t",
        type=int,
        default=300,
        help="Starting timestep for SDEdit (lower = less denoising, preserve histogram more)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[GridUNet, CopulaAwareDiffusion, Dict]:
    """Load model and diffusion from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model_cfg = config.get('model', {})
    m = model_cfg.get('m', 64)
    
    model = GridUNet(
        m=m,
        in_channels=model_cfg.get('in_channels', 2),
        base_channels=model_cfg.get('base_channels', 64),
        channel_mults=tuple(model_cfg.get('channel_mults', [1, 2, 3, 4])),
        num_res_blocks=model_cfg.get('num_res_blocks', 2),
        attention_resolutions=tuple(model_cfg.get('attention_resolutions', [16, 8])),
        dropout=model_cfg.get('dropout', 0.1),
    ).to(device)
    
    # Load weights
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create diffusion
    diff_cfg = config.get('diffusion', {})
    diffusion = CopulaAwareDiffusion(
        timesteps=diff_cfg.get('timesteps', 1000),
        beta_schedule=diff_cfg.get('noise_schedule', 'cosine'),
    ).to(device)
    
    return model, diffusion, config


@torch.no_grad()
def sample_conditional(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    histogram: torch.Tensor,
    device: torch.device,
    num_steps: int = 50,
    start_from_histogram: bool = True,
    start_t: int = 500,
) -> torch.Tensor:
    """
    Sample density conditioned on histogram using reverse diffusion.
    
    Args:
        model: Conditional diffusion model
        diffusion: Diffusion scheduler
        histogram: Input histogram (B, 1, m, m)
        device: Target device
        num_steps: Number of sampling steps
        start_from_histogram: If True, use SDEdit-style sampling starting from noisy histogram
        start_t: Starting timestep for SDEdit (only used if start_from_histogram=True)
    """
    B = histogram.shape[0]
    m = histogram.shape[-1]
    T = diffusion.timesteps
    
    # Normalize histogram and convert to log-space
    du = dv = 1.0 / m
    mass = (histogram * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    histogram = histogram / mass
    log_histogram = torch.log(histogram.clamp(min=1e-12))
    
    alphas_cumprod = diffusion.alphas_cumprod.to(device)
    
    if start_from_histogram:
        # SDEdit-style: start from noisy histogram instead of pure noise
        start_t = min(start_t, T - 1)
        alpha_start = alphas_cumprod[start_t]
        noise = torch.randn_like(log_histogram)
        x_t = torch.sqrt(alpha_start) * log_histogram + torch.sqrt(1 - alpha_start) * noise
        
        # Create timestep schedule from start_t down to 0
        if num_steps >= start_t:
            timesteps = list(range(start_t, -1, -1))
        else:
            step_size = max(1, start_t // num_steps)
            timesteps = list(range(start_t, -1, -step_size))
            if timesteps[-1] != 0:
                timesteps.append(0)
    else:
        # Standard: start from pure noise
        x_t = torch.randn(B, 1, m, m, device=device)
        
        if num_steps >= T:
            timesteps = list(range(T - 1, -1, -1))
        else:
            step_size = T // num_steps
            timesteps = list(range(T - 1, -1, -step_size))
            if timesteps[-1] != 0:
                timesteps.append(0)
    
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


def get_test_cases() -> List[Tuple[str, Dict, str]]:
    """Return test copula configurations."""
    return [
        ("gaussian", {"rho": 0.75}, "gaussian_rho0.75"),
        ("gaussian", {"rho": -0.65}, "gaussian_rho-0.65"),
        ("clayton", {"theta": 2.5}, "clayton_theta2.5"),
        ("gumbel", {"theta": 3.0}, "gumbel_theta3.0"),
        ("frank", {"theta": 5.0}, "frank_theta5.0"),
        ("joe", {"theta": 2.0}, "joe_theta2.0"),
        ("student", {"rho": 0.7, "nu": 6.0}, "student_rho0.7_nu6"),
    ]


def visualize_comparison(
    samples: np.ndarray,
    histogram: np.ndarray,
    density_pred: np.ndarray,
    density_true: np.ndarray,
    name: str,
    save_path: Path,
):
    """Create comparison visualization with independent color scales."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Linear scale
    # Samples
    axes[0, 0].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, c='blue')
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title('Samples')
    axes[0, 0].set_aspect('equal')
    
    # Histogram
    im1 = axes[0, 1].imshow(histogram.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes[0, 1].set_title('Histogram (Input)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Predicted density - independent scale
    vmax_pred = np.percentile(density_pred, 99)
    im2 = axes[0, 2].imshow(density_pred.T, origin='lower', extent=[0, 1, 0, 1], cmap='hot', vmin=0, vmax=vmax_pred)
    axes[0, 2].set_title(f'Predicted (max={density_pred.max():.1f})')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # True density - independent scale
    vmax_true = np.percentile(density_true, 99)
    im3 = axes[0, 3].imshow(density_true.T, origin='lower', extent=[0, 1, 0, 1], cmap='hot', vmin=0, vmax=vmax_true)
    axes[0, 3].set_title(f'True (max={density_true.max():.1f})')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)
    
    # Row 2: Log scale (better for comparing shapes)
    # Log histogram
    log_hist = np.log(histogram.clip(min=1e-10))
    im4 = axes[1, 0].imshow(log_hist.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes[1, 0].set_title('Log Histogram')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    # Log predicted - use same scale as log true for comparison
    log_pred = np.log(density_pred.clip(min=1e-10))
    log_true = np.log(density_true.clip(min=1e-10))
    vmin_log = min(np.percentile(log_pred, 1), np.percentile(log_true, 1))
    vmax_log = max(np.percentile(log_pred, 99), np.percentile(log_true, 99))
    
    im5 = axes[1, 1].imshow(log_pred.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r', vmin=vmin_log, vmax=vmax_log)
    axes[1, 1].set_title('Log Predicted (shared scale)')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    im6 = axes[1, 2].imshow(log_true.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r', vmin=vmin_log, vmax=vmax_log)
    axes[1, 2].set_title('Log True (shared scale)')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    # Error map
    error = density_pred - density_true
    vmax_err = np.percentile(np.abs(error), 95)
    im7 = axes[1, 3].imshow(error.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r', vmin=-vmax_err, vmax=vmax_err)
    axes[1, 3].set_title('Error (Pred - True)')
    plt.colorbar(im7, ax=axes[1, 3], fraction=0.046)
    
    # Compute metrics
    corr = np.corrcoef(density_true.flatten(), density_pred.flatten())[0, 1]
    log_corr = np.corrcoef(log_true.flatten(), log_pred.flatten())[0, 1]
    mse = np.mean((density_true - density_pred) ** 2)
    
    fig.suptitle(f'{name} | Corr: {corr:.4f} | Log-Corr: {log_corr:.4f} | MSE: {mse:.2f}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {'correlation': corr, 'log_correlation': log_corr, 'mse': mse}


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, diffusion, config = load_model(args.checkpoint, device)
    m = config.get('model', {}).get('m', 64)
    print(f"Model loaded (m={m})")
    
    # Output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.checkpoint.parent / "visualizations" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Test cases
    test_cases = get_test_cases()
    results = []
    
    print(f"\nEvaluating {len(test_cases)} copula configurations...")
    print(f"Using {args.sampling_steps} diffusion steps\n")
    
    for family, params, name in tqdm(test_cases, desc="Evaluating"):
        # Generate samples
        samples = sample_bicop(family, params, n=args.num_samples, rotation=0)
        
        # Create histogram
        histogram = scatter_to_hist(samples, m, reflect=True)
        hist_tensor = torch.from_numpy(histogram).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Get true density
        log_density_true = analytic_logpdf_grid(family, params, m, rotation=0)
        density_true = np.exp(np.clip(log_density_true, -20, 20))
        du = dv = 1.0 / m
        density_true = density_true / (density_true.sum() * du * dv)
        
        # Sample from model
        density_pred = sample_conditional(
            model, diffusion, hist_tensor, device, 
            num_steps=args.sampling_steps,
            start_from_histogram=args.sdedit,
            start_t=args.start_t,
        )
        
        # Apply copula projection
        density_pred = copula_project(density_pred, iters=10)
        density_pred = density_pred[0, 0].cpu().numpy()
        
        # Normalize
        density_pred = density_pred / (density_pred.sum() * du * dv)
        
        # Visualize
        save_path = output_dir / f"{name}.png"
        metrics = visualize_comparison(
            samples, histogram, density_pred, density_true, name, save_path
        )
        results.append((name, metrics))
        
        print(f"  {name}: corr={metrics['correlation']:.4f}, mse={metrics['mse']:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    correlations = [r[1]['correlation'] for r in results]
    mses = [r[1]['mse'] for r in results]
    print(f"Mean correlation: {np.mean(correlations):.4f}")
    print(f"Mean MSE: {np.mean(mses):.4f}")
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

