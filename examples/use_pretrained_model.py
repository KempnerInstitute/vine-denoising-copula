#!/usr/bin/env python3
"""
Example: Using a pretrained diffusion copula model for inference.

This shows how to:
1. Load a trained diffusion model from checkpoint
2. Estimate copula density from bivariate samples
3. Compute h-functions for vine construction
4. Fit a vine copula to high-dimensional data

Usage:
    python examples/use_pretrained_model.py --checkpoint path/to/model.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project
from vdc.models.hfunc import HFuncLookup
from vdc.data.hist import scatter_to_hist


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained diffusion model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    # Extract configuration
    model_cfg = config.get('model', {})
    m = config.get('data', {}).get('m', 64)
    
    # Build model
    model = GridUNet(
        m=m,
        in_channels=model_cfg.get('in_channels', 1),
        base_channels=model_cfg.get('base_channels', 64),
        channel_mults=tuple(model_cfg.get('channel_mults', (1, 2, 3, 4))),
        num_res_blocks=model_cfg.get('num_res_blocks', 2),
        attention_resolutions=tuple(model_cfg.get('attention_resolutions', (16, 8))),
        dropout=model_cfg.get('dropout', 0.1),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Build diffusion
    diff_cfg = config.get('diffusion', {})
    diffusion = CopulaAwareDiffusion(
        timesteps=diff_cfg.get('timesteps', 1000),
        beta_schedule=diff_cfg.get('noise_schedule', 'cosine'),
    ).to(device)
    
    step = checkpoint.get('step', 'unknown')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded (step {step}, {n_params:,} parameters)")
    
    return model, diffusion, config


@torch.no_grad()
def estimate_density(model, diffusion, samples, m, device, num_steps=50):
    """
    Estimate copula density from bivariate samples using DDIM sampling.
    
    Args:
        model: Trained GridUNet
        diffusion: CopulaAwareDiffusion
        samples: (n, 2) array of pseudo-observations in [0,1]²
        m: Grid resolution
        device: Compute device
        num_steps: Number of DDIM sampling steps
        
    Returns:
        density: (m, m) numpy array of estimated copula density
    """
    # Create histogram from samples
    hist = scatter_to_hist(samples, m, reflect=True)
    du = 1.0 / m
    hist = hist / (hist.sum() * du * du + 1e-12)
    
    # DDIM sampling from noise to density
    T = diffusion.timesteps
    x_t = torch.randn(1, 1, m, m, device=device)
    
    step_size = max(1, T // num_steps)
    timesteps = list(range(T - 1, -1, -step_size))
    if timesteps[-1] != 0:
        timesteps.append(0)
    
    alphas_cumprod = diffusion.alphas_cumprod.to(device)
    
    for i, t in enumerate(timesteps):
        t_normalized = torch.full((1,), t / T, device=device)
        
        # Predict noise
        pred_noise = model(x_t, t_normalized)
        
        # Compute predicted x_0 (clean log-density)
        alpha_t = alphas_cumprod[t]
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        pred_x0 = pred_x0.clamp(-20, 20)
        
        if t == 0:
            x_t = pred_x0
        else:
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            alpha_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            x_t = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * pred_noise
    
    # Convert log-density to density and project
    density = torch.exp(x_t).clamp(1e-12, 1e6)
    density = density / (density.sum() * du * du).clamp_min(1e-12)
    density = copula_project(density, iters=50)
    
    return density[0, 0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Use pretrained diffusion copula model")
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Vine Diffusion Copula - Pretrained Model Example")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Generate sample data (Gaussian copula with rho=0.7)
    # =========================================================================
    print("\n[Step 1] Generating sample data...")
    
    np.random.seed(42)
    rho = 0.7
    n_samples = 2000
    
    # Generate correlated normals and transform to uniform
    Z = np.random.randn(n_samples, 2)
    Z[:, 1] = rho * Z[:, 0] + np.sqrt(1 - rho**2) * Z[:, 1]
    samples = norm.cdf(Z)
    
    print(f"  Generated {n_samples} samples from Gaussian copula (ρ={rho})")
    print(f"    Shape: {samples.shape}")
    print(f"    Range: [{samples.min():.4f}, {samples.max():.4f}]")
    
    # =========================================================================
    # Step 2: Load model (if checkpoint provided)
    # =========================================================================
    if args.checkpoint:
        print(f"\n[Step 2] Loading trained model...")
        model, diffusion, config = load_model(args.checkpoint, args.device)
        m = config.get('data', {}).get('m', 64)
        
        # =====================================================================
        # Step 3: Estimate copula density
        # =====================================================================
        print(f"\n[Step 3] Estimating copula density...")
        density = estimate_density(model, diffusion, samples, m, args.device)
        
        print(f"  Estimated density on {m}×{m} grid")
        print(f"    Density range: [{density.min():.4f}, {density.max():.4f}]")
        print(f"    Total mass: {density.sum() * (1/m)**2:.4f}")
        
        # =====================================================================
        # Step 4: Compute h-functions
        # =====================================================================
        print(f"\n[Step 4] Computing h-functions...")
        hfunc = HFuncLookup(density)
        
        # Test h-function at some points
        test_u = np.array([0.3, 0.5, 0.7])
        test_v = np.array([0.4, 0.5, 0.6])
        
        h1 = hfunc.h_u_given_v(test_u, test_v)
        h2 = hfunc.h_v_given_u(test_u, test_v)
        
        print(f"  H-functions computed")
        print(f"    Test points (u, v): {list(zip(test_u, test_v))}")
        print(f"    H(u|v): {h1}")
        print(f"    H(v|u): {h2}")
        
        # =====================================================================
        # Step 5: Visualize (optional)
        # =====================================================================
        try:
            import matplotlib.pyplot as plt
            
            print(f"\n[Step 5] Creating visualization...")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Scatter plot
            axes[0].scatter(samples[:500, 0], samples[:500, 1], s=1, alpha=0.5)
            axes[0].set_xlim(0, 1)
            axes[0].set_ylim(0, 1)
            axes[0].set_xlabel('u')
            axes[0].set_ylabel('v')
            axes[0].set_title(f'Samples (n={n_samples})')
            axes[0].set_aspect('equal')
            
            # Estimated density
            im = axes[1].imshow(density, origin='lower', extent=[0,1,0,1], cmap='hot')
            axes[1].set_xlabel('u')
            axes[1].set_ylabel('v')
            axes[1].set_title('Estimated Density')
            plt.colorbar(im, ax=axes[1])
            
            # Log density
            log_density = np.log(np.clip(density, 1e-6, None))
            im2 = axes[2].imshow(log_density, origin='lower', extent=[0,1,0,1], cmap='viridis')
            axes[2].set_xlabel('u')
            axes[2].set_ylabel('v')
            axes[2].set_title('Log Density')
            plt.colorbar(im2, ax=axes[2])
            
            plt.tight_layout()
            output_path = 'example_output.png'
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            print(f"  Saved visualization: {output_path}")
            
        except ImportError:
            print("  (matplotlib not available, skipping visualization)")
    
    else:
        print("\n[Step 2] No checkpoint provided - showing data preparation only")
        print("    To run with a trained model, use:")
        print("    python examples/use_pretrained_model.py --checkpoint path/to/model.pt")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print("""
Key steps demonstrated:
  1. Generate/load bivariate pseudo-observations in [0,1]²
  2. Load trained diffusion model from checkpoint
  3. Estimate copula density using DDIM sampling
  4. Compute h-functions for vine copula construction
  5. Visualize results

For high-dimensional vine copulas, see:
  - examples/fit_vine_copula.py
  - vdc/vine/api.py (VineCopulaModel class)
""")


if __name__ == '__main__':
    main()

