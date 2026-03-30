#!/usr/bin/env python
"""
Vine Copula Demo: Density Visualization and Sampling

This script demonstrates:
1. Fitting a D-vine copula using the diffusion model
2. Visualizing estimated pair copula densities
3. Sampling from the fitted vine copula
4. Comparing samples to original data
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from scipy.stats import norm, kstest, gaussian_kde
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project
from vdc.models.hfunc import HFuncLookup


def scatter_to_hist(pts, m, reflect=True):
    """Create 2D histogram from scatter points."""
    if reflect:
        pts_reflected = []
        for dx, dy in [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            shifted = pts.copy()
            shifted[:, 0] += dx
            shifted[:, 1] += dy
            mask = (shifted[:, 0] >= 0) & (shifted[:, 0] <= 1) & \
                   (shifted[:, 1] >= 0) & (shifted[:, 1] <= 1)
            pts_reflected.append(shifted[mask])
        pts_all = np.vstack(pts_reflected)
    else:
        pts_all = pts
    hist, _, _ = np.histogram2d(pts_all[:, 0], pts_all[:, 1], bins=m, range=[[0, 1], [0, 1]])
    return hist.astype(np.float64)


def generate_gaussian_copula(n, d, rho=0.5, seed=None):
    """Generate samples from Gaussian copula with AR(1) correlation."""
    if seed is not None:
        np.random.seed(seed)
    Sigma = np.array([[rho ** abs(i - j) for j in range(d)] for i in range(d)])
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    return norm.cdf(Z)


def generate_clayton_vine(n, d, theta=2.0, seed=None):
    """Generate samples from Clayton D-vine."""
    if seed is not None:
        np.random.seed(seed)
    
    def clayton_hinv(u, v, theta):
        if theta < 1e-10:
            return u
        t = v ** (-theta) * (u ** (-theta / (1 + theta)) - 1) + 1
        return np.clip(t ** (-1 / theta), 1e-10, 1 - 1e-10)
    
    U = np.zeros((n, d))
    W = np.random.uniform(0, 1, (n, d))
    U[:, 0] = W[:, 0]
    for j in range(1, d):
        U[:, j] = clayton_hinv(W[:, j], U[:, j-1], theta)
    return U


def load_model(checkpoint_path, device):
    """Load trained diffusion model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model_cfg = config.get('model', {})
    m = config.get('data', {}).get('m', 64)
    
    model = GridUNet(
        m=m,
        in_channels=model_cfg.get('in_channels', 2),
        base_channels=model_cfg.get('base_channels', 64),
        channel_mults=tuple(model_cfg.get('channel_mults', (1, 2, 3, 4))),
        num_res_blocks=model_cfg.get('num_res_blocks', 2),
        attention_resolutions=tuple(model_cfg.get('attention_resolutions', (16, 8))),
        dropout=model_cfg.get('dropout', 0.1),
    ).to(device)
    
    model.load_state_dict(checkpoint.get('model_state_dict', {}))
    model.eval()
    
    diffusion = CopulaAwareDiffusion(
        timesteps=config.get('diffusion', {}).get('timesteps', 1000),
        beta_schedule=config.get('diffusion', {}).get('noise_schedule', 'cosine'),
    ).to(device)
    
    return model, diffusion, m


@torch.no_grad()
def estimate_pair_copula(model, diffusion, pair_data, m, device, 
                         num_steps=50, cfg_scale=2.0):
    """Estimate pair copula density."""
    hist = scatter_to_hist(pair_data, m, reflect=True)
    du = 1.0 / m
    hist = hist / (hist.sum() * du * du + 1e-12)
    
    hist_tensor = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)
    log_histogram = torch.log(hist_tensor.clamp(min=1e-12))
    
    T = diffusion.timesteps
    alphas_cumprod = diffusion.alphas_cumprod.to(device)
    
    step_size = max(1, T // num_steps)
    timesteps = list(range(T - 1, -1, -step_size))
    if timesteps[-1] != 0:
        timesteps.append(0)
    
    x_t = torch.randn(1, 1, m, m, device=device)
    
    for i, t in enumerate(timesteps):
        t_normalized = torch.full((1,), t, device=device).float() / T
        
        model_input_cond = torch.cat([x_t, log_histogram], dim=1)
        pred_noise_cond = model(model_input_cond, t_normalized)
        
        model_input_uncond = torch.cat([x_t, torch.zeros_like(log_histogram)], dim=1)
        pred_noise_uncond = model(model_input_uncond, t_normalized)
        
        pred_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
        
        alpha_t = alphas_cumprod[t]
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        pred_x0 = pred_x0.clamp(-20, 20)
        
        if t == 0:
            x_t = pred_x0
        else:
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            alpha_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            x_t = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * pred_noise
    
    density = torch.exp(x_t).clamp(1e-12, 1e6)
    density = density / (density.sum() * du * du).clamp_min(1e-12)
    density = copula_project(density, iters=50)
    
    density_np = density[0, 0].cpu().numpy()
    hfunc = HFuncLookup(density_np)
    
    return density_np, hfunc


class DVineCopula:
    """D-Vine copula fitted with diffusion model."""
    
    def __init__(self, pair_copulas, d):
        """
        Args:
            pair_copulas: List of lists of (density, hfunc) tuples
            d: Dimension
        """
        self.pair_copulas = pair_copulas
        self.d = d
    
    @classmethod
    def fit(cls, U, model, diffusion, device, m=64, num_steps=50, cfg_scale=2.0):
        """Fit D-vine to data."""
        n, d = U.shape
        pair_copulas = []
        V = [U.copy()]
        
        print(f"Fitting D-vine with d={d}...")
        for tree in range(d - 1):
            print(f"  Tree {tree + 1}/{d - 1}...")
            tree_copulas = []
            V_next = np.zeros((n, d - tree - 1))
            
            for edge in range(d - tree - 1):
                u_data = V[tree][:, edge]
                v_data = V[tree][:, edge + 1]
                
                pair_data = np.column_stack([u_data, v_data])
                pair_data = np.clip(pair_data, 1e-6, 1-1e-6)
                
                density, hfunc = estimate_pair_copula(
                    model, diffusion, pair_data, m, device,
                    num_steps=num_steps, cfg_scale=cfg_scale
                )
                tree_copulas.append((density, hfunc))
                
                h_val = hfunc.h_u_given_v(u_data, v_data)
                V_next[:, edge] = np.clip(h_val, 1e-6, 1-1e-6)
            
            pair_copulas.append(tree_copulas)
            V.append(V_next)
        
        return cls(pair_copulas, d)
    
    def sample(self, n, seed=None):
        """
        Sample from the fitted D-vine using inverse Rosenblatt transform.
        
        For a D-vine with d variables, the algorithm is:
        1. Generate W_1, ..., W_d ~ iid U(0,1)
        2. U_1 = W_1
        3. For j = 2, ..., d:
           - Compute pseudo-observations V needed for conditioning
           - Apply inverse h-functions in reverse order
        """
        if seed is not None:
            np.random.seed(seed)
        
        d = self.d
        W = np.random.uniform(0, 1, (n, d))
        U = np.zeros((n, d))
        
        # Store pseudo-observations at each tree level
        # V[tree][edge] = pseudo-observation for that position
        V = [[None for _ in range(d)] for _ in range(d)]
        
        # Initialize: V[0][j] = U_j (the actual samples)
        # First variable is just uniform
        U[:, 0] = W[:, 0]
        V[0][0] = U[:, 0]
        
        # For j = 2, ..., d
        for j in range(1, d):
            # We need to invert the Rosenblatt transform
            # Starting from W_j, apply inverse h-functions
            
            # For D-vine, use simplified algorithm:
            # U_j = h^{-1}_{j|j-1}(W_j | U_{j-1})
            # This is correct for the first tree
            
            _, hfunc = self.pair_copulas[0][j-1]
            U[:, j] = hfunc.hinv_u_given_v(W[:, j], U[:, j-1])
            U[:, j] = np.clip(U[:, j], 1e-6, 1-1e-6)
            V[0][j] = U[:, j]
        
        return U
    
    def logpdf(self, U):
        """Compute log-pdf at given points."""
        n, d = U.shape
        logpdf = np.zeros(n)
        V = [U.copy()]
        
        for tree in range(d - 1):
            V_next = np.zeros((n, d - tree - 1))
            
            for edge in range(d - tree - 1):
                density, hfunc = self.pair_copulas[tree][edge]
                
                u_data = V[tree][:, edge]
                v_data = V[tree][:, edge + 1]
                
                pair_pdf = hfunc.pdf(u_data, v_data)
                logpdf += np.log(np.clip(pair_pdf, 1e-10, None))
                
                h_val = hfunc.h_u_given_v(u_data, v_data)
                V_next[:, edge] = np.clip(h_val, 1e-6, 1-1e-6)
            
            V.append(V_next)
        
        return logpdf


def visualize_densities(vine, output_dir):
    """Visualize all pair copula densities."""
    d = vine.d
    n_trees = len(vine.pair_copulas)
    
    # Count total copulas
    total_copulas = sum(len(tree) for tree in vine.pair_copulas)
    
    # Create grid
    n_cols = min(4, d - 1)
    n_rows = (total_copulas + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if total_copulas == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    idx = 0
    for tree_idx, tree_copulas in enumerate(vine.pair_copulas):
        for edge_idx, (density, _) in enumerate(tree_copulas):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            im = ax.imshow(density.T, origin='lower', extent=[0, 1, 0, 1],
                          aspect='equal', cmap='hot', vmin=0)
            ax.set_xlabel('u')
            ax.set_ylabel('v')
            ax.set_title(f'Tree {tree_idx + 1}, Edge {edge_idx + 1}')
            plt.colorbar(im, ax=ax, shrink=0.8)
            
            idx += 1
    
    # Hide unused axes
    for i in range(idx, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(f'Estimated Pair Copula Densities (D-Vine, d={d})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'pair_copula_densities.png', dpi=150)
    plt.close()
    print(f"Saved: pair_copula_densities.png")


def visualize_samples(U_original, U_sampled, output_dir, scenario_name):
    """Compare original and sampled data."""
    d = U_original.shape[1]
    
    # Pairwise scatter plots for first few dimensions
    n_pairs = min(6, d * (d-1) // 2)
    fig, axes = plt.subplots(2, n_pairs, figsize=(4*n_pairs, 8))
    
    pair_idx = 0
    for i in range(d):
        for j in range(i+1, d):
            if pair_idx >= n_pairs:
                break
            
            # Original data
            ax = axes[0, pair_idx]
            ax.scatter(U_original[:, i], U_original[:, j], alpha=0.3, s=10)
            ax.set_xlabel(f'U_{i+1}')
            ax.set_ylabel(f'U_{j+1}')
            ax.set_title(f'Original')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Sampled data
            ax = axes[1, pair_idx]
            ax.scatter(U_sampled[:, i], U_sampled[:, j], alpha=0.3, s=10, c='C1')
            ax.set_xlabel(f'U_{i+1}')
            ax.set_ylabel(f'U_{j+1}')
            ax.set_title(f'Sampled')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            pair_idx += 1
        if pair_idx >= n_pairs:
            break
    
    plt.suptitle(f'{scenario_name}: Original vs Sampled (n={len(U_sampled)})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'sample_comparison_{scenario_name}.png', dpi=150)
    plt.close()
    print(f"Saved: sample_comparison_{scenario_name}.png")
    
    # Marginal distributions
    fig, axes = plt.subplots(2, d, figsize=(3*d, 6))
    
    for i in range(d):
        # Original marginal
        ax = axes[0, i]
        ax.hist(U_original[:, i], bins=30, density=True, alpha=0.7)
        ax.axhline(y=1, color='r', linestyle='--', label='Uniform')
        ax.set_xlabel(f'U_{i+1}')
        ax.set_title('Original')
        ax.set_xlim(0, 1)
        
        # Sampled marginal
        ax = axes[1, i]
        ax.hist(U_sampled[:, i], bins=30, density=True, alpha=0.7, color='C1')
        ax.axhline(y=1, color='r', linestyle='--', label='Uniform')
        ax.set_xlabel(f'U_{i+1}')
        ax.set_title('Sampled')
        ax.set_xlim(0, 1)
    
    plt.suptitle(f'{scenario_name}: Marginal Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'marginals_{scenario_name}.png', dpi=150)
    plt.close()
    print(f"Saved: marginals_{scenario_name}.png")


def compute_dependence_metrics(U):
    """Compute pairwise Kendall's tau."""
    from scipy.stats import kendalltau
    d = U.shape[1]
    tau_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                tau_matrix[i, j] = 1.0
            else:
                tau_matrix[i, j], _ = kendalltau(U[:, i], U[:, j])
    return tau_matrix


def main():
    parser = argparse.ArgumentParser(description='Vine Copula Demo')
    parser.add_argument('--dimension', type=int, default=5, help='Dimension')
    parser.add_argument('--n-train', type=int, default=2000, help='Training samples')
    parser.add_argument('--n-sample', type=int, default=1000, help='Samples to generate')
    parser.add_argument('--scenario', type=str, default='gaussian', 
                       choices=['gaussian', 'clayton'], help='Data scenario')
    parser.add_argument('--checkpoint', type=Path,
                       default=Path('checkpoints/conditional_diffusion_v2/model_step_120000.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/vine_demo'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Dimension: {args.dimension}")
    print(f"Scenario: {args.scenario}")
    
    # Load model
    model, diffusion, m = load_model(REPO_ROOT / args.checkpoint, device)
    print("Model loaded")
    
    # Generate training data
    print(f"\nGenerating {args.n_train} training samples...")
    if args.scenario == 'gaussian':
        U_train = generate_gaussian_copula(args.n_train, args.dimension, rho=0.6, seed=42)
    else:
        U_train = generate_clayton_vine(args.n_train, args.dimension, theta=2.0, seed=42)
    
    # Create output directory
    output_dir = REPO_ROOT / args.output / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fit vine copula
    print("\nFitting D-vine copula...")
    vine = DVineCopula.fit(U_train, model, diffusion, device, m=m)
    
    # Visualize densities
    print("\nVisualizing pair copula densities...")
    visualize_densities(vine, output_dir)
    
    # Sample from vine
    print(f"\nSampling {args.n_sample} points from fitted vine...")
    U_sampled = vine.sample(args.n_sample, seed=123)
    
    # Visualize samples
    print("\nVisualizing samples...")
    visualize_samples(U_train, U_sampled, output_dir, args.scenario)
    
    # Compare dependence structure
    print("\nComputing Kendall's tau matrices...")
    tau_original = compute_dependence_metrics(U_train)
    tau_sampled = compute_dependence_metrics(U_sampled)
    
    # Visualize tau comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im = axes[0].imshow(tau_original, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title("Original Kendall's τ")
    plt.colorbar(im, ax=axes[0])
    
    im = axes[1].imshow(tau_sampled, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title("Sampled Kendall's τ")
    plt.colorbar(im, ax=axes[1])
    
    im = axes[2].imshow(tau_original - tau_sampled, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[2].set_title("Difference (Original - Sampled)")
    plt.colorbar(im, ax=axes[2])
    
    plt.suptitle(f"Kendall's Tau Comparison ({args.scenario}, d={args.dimension})")
    plt.tight_layout()
    plt.savefig(output_dir / 'kendall_tau_comparison.png', dpi=150)
    plt.close()
    print("Saved: kendall_tau_comparison.png")
    
    # Summary stats
    tau_error = np.abs(tau_original - tau_sampled).mean()
    print(f"\nMean absolute Kendall's tau error: {tau_error:.4f}")
    
    # Marginal uniformity test
    print("\nMarginal uniformity tests (KS p-values):")
    for i in range(args.dimension):
        _, p_orig = kstest(U_train[:, i], 'uniform')
        _, p_samp = kstest(U_sampled[:, i], 'uniform')
        print(f"  U_{i+1}: Original={p_orig:.4f}, Sampled={p_samp:.4f}")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
