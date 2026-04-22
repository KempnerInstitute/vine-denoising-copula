#!/usr/bin/env python3
"""
Example: Fitting a Vine Copula to High-Dimensional Data.

This shows how to:
1. Load a trained diffusion model
2. Generate high-dimensional copula data
3. Fit a D-vine copula using diffusion-estimated pair copulas
4. Evaluate log-likelihood and uniformity
5. Generate samples from the fitted vine

Usage:
    python examples/fit_vine_copula.py --checkpoint path/to/model.pt
"""
import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from scipy.stats import norm, kstest

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained diffusion model."""
    from vdc.models.unet_grid import GridUNet
    from vdc.models.copula_diffusion import CopulaAwareDiffusion
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model_cfg = config.get('model', {})
    m = config.get('data', {}).get('m', 64)
    
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
    
    diffusion = CopulaAwareDiffusion(
        timesteps=config.get('diffusion', {}).get('timesteps', 1000),
        beta_schedule=config.get('diffusion', {}).get('noise_schedule', 'cosine'),
    ).to(device)
    
    return model, diffusion, config


def generate_gaussian_vine(n: int, d: int, rho: float = 0.5, seed: int = None) -> np.ndarray:
    """
    Generate samples from Gaussian copula with AR(1) correlation structure.
    
    The correlation matrix is: Sigma[i,j] = rho^|i-j|
    
    Args:
        n: Number of samples
        d: Dimension
        rho: Correlation parameter
        seed: Random seed
        
    Returns:
        U: (n, d) copula samples in [0,1]^d
    """
    if seed is not None:
        np.random.seed(seed)
    
    # AR(1) correlation structure
    Sigma = np.array([[rho ** abs(i - j) for j in range(d)] for i in range(d)])
    
    # Generate correlated normals and transform to uniform
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    U = norm.cdf(Z)
    
    return U


def generate_clayton_vine(n: int, d: int, theta: float = 2.0, seed: int = None) -> np.ndarray:
    """
    Generate samples from D-vine with Clayton pair copulas.
    
    Args:
        n: Number of samples
        d: Dimension
        theta: Clayton parameter (> 0)
        seed: Random seed
        
    Returns:
        U: (n, d) copula samples in [0,1]^d
    """
    if seed is not None:
        np.random.seed(seed)
    
    def clayton_hinv(u, v, theta):
        """Inverse h-function for Clayton copula."""
        t = v ** (-theta) * (u ** (-theta / (1 + theta)) - 1) + 1
        return np.clip(t ** (-1 / theta), 1e-10, 1 - 1e-10)
    
    U = np.zeros((n, d))
    W = np.random.uniform(0, 1, (n, d))
    U[:, 0] = W[:, 0]
    
    for j in range(1, d):
        U[:, j] = clayton_hinv(W[:, j], U[:, j-1], theta)
    
    return U


class SimpleDVine:
    """
    Simple D-vine implementation for demonstration.
    
    Uses the diffusion model to estimate pair copulas.
    """
    
    def __init__(self, d: int, m: int = 64, device: str = 'cuda'):
        self.d = d
        self.m = m
        self.device = device
        self.pair_copulas = []  # List of lists: pair_copulas[tree][edge]
    
    def fit(self, U: np.ndarray, model, diffusion, verbose: bool = True):
        """
        Fit D-vine to data.
        
        Args:
            U: (n, d) pseudo-observations
            model: Trained GridUNet
            diffusion: CopulaAwareDiffusion
            verbose: Print progress
        """
        from vdc.models.projection import copula_project
        from vdc.models.hfunc import HFuncLookup
        from vdc.data.hist import scatter_to_hist
        
        n, d = U.shape
        assert d == self.d
        
        # Store transformed data for each tree
        V = [U.copy()]
        
        total_pairs = d * (d - 1) // 2
        pair_count = 0
        
        for tree in range(d - 1):
            tree_copulas = []
            V_next = np.zeros((n, d - tree - 1))
            
            for edge in range(d - tree - 1):
                pair_count += 1
                if verbose:
                    print(f"  Fitting pair {pair_count}/{total_pairs} (tree {tree+1}, edge {edge+1})...")
                
                # Get data for this pair
                u_data = V[tree][:, edge]
                v_data = V[tree][:, edge + 1]
                pair_data = np.column_stack([u_data, v_data])
                pair_data = np.clip(pair_data, 1e-6, 1-1e-6)
                
                # Estimate pair copula with diffusion model
                density, hfunc = self._estimate_pair_copula(
                    model, diffusion, pair_data
                )
                tree_copulas.append((density, hfunc))
                
                # Compute h-transform for next tree
                h_val = hfunc.h_u_given_v(u_data, v_data)
                V_next[:, edge] = np.clip(h_val, 1e-6, 1-1e-6)
            
            self.pair_copulas.append(tree_copulas)
            V.append(V_next)
        
        if verbose:
            print(f"  Fitted {total_pairs} pair copulas")
    
    @torch.no_grad()
    def _estimate_pair_copula(self, model, diffusion, pair_data) -> Tuple:
        """Estimate bivariate copula from pair samples."""
        from vdc.models.projection import copula_project
        from vdc.models.hfunc import HFuncLookup
        from vdc.data.hist import scatter_to_hist
        
        m = self.m
        device = self.device
        
        # Create histogram
        hist = scatter_to_hist(pair_data, m, reflect=True)
        du = 1.0 / m
        hist = hist / (hist.sum() * du * du + 1e-12)
        
        # DDIM sampling
        T = diffusion.timesteps
        x_t = torch.randn(1, 1, m, m, device=device)
        
        num_steps = 50
        step_size = max(1, T // num_steps)
        timesteps = list(range(T - 1, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)
        
        alphas_cumprod = diffusion.alphas_cumprod.to(device)
        
        for i, t in enumerate(timesteps):
            t_norm = torch.full((1,), t / T, device=device)
            pred_noise = model(x_t, t_norm)
            
            alpha_t = alphas_cumprod[t]
            pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            pred_x0 = pred_x0.clamp(-20, 20)
            
            if t == 0:
                x_t = pred_x0
            else:
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                alpha_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
                x_t = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * pred_noise
        
        # Convert to density and project
        density = torch.exp(x_t).clamp(1e-12, 1e6)
        density = density / (density.sum() * du * du).clamp_min(1e-12)
        density = copula_project(density, iters=50)
        
        density_np = density[0, 0].cpu().numpy()
        hfunc = HFuncLookup(density_np)
        
        return density_np, hfunc
    
    def logpdf(self, U: np.ndarray) -> np.ndarray:
        """Compute log-density at points."""
        n, d = U.shape
        assert d == self.d
        
        logpdf = np.zeros(n)
        V = [U.copy()]
        
        for tree in range(d - 1):
            V_next = np.zeros((n, d - tree - 1))
            
            for edge in range(d - tree - 1):
                density, hfunc = self.pair_copulas[tree][edge]
                
                u_data = V[tree][:, edge]
                v_data = V[tree][:, edge + 1]
                
                # Add log-density contribution
                pair_pdf = hfunc.pdf(u_data, v_data)
                logpdf += np.log(np.clip(pair_pdf, 1e-10, None))
                
                # Compute h-transform
                h_val = hfunc.h_u_given_v(u_data, v_data)
                V_next[:, edge] = np.clip(h_val, 1e-6, 1-1e-6)
            
            V.append(V_next)
        
        return logpdf
    
    def rosenblatt(self, U: np.ndarray) -> np.ndarray:
        """Rosenblatt transform to independent uniforms."""
        n, d = U.shape
        W = np.zeros_like(U)
        W[:, 0] = U[:, 0]
        
        V = [U.copy()]
        
        for tree in range(d - 1):
            V_next = np.zeros((n, d - tree - 1))
            
            for edge in range(d - tree - 1):
                _, hfunc = self.pair_copulas[tree][edge]
                
                u_data = V[tree][:, edge]
                v_data = V[tree][:, edge + 1]
                
                h_val = hfunc.h_u_given_v(u_data, v_data)
                V_next[:, edge] = np.clip(h_val, 1e-6, 1-1e-6)
                
                if edge == 0:
                    W[:, tree + 1] = h_val
            
            V.append(V_next)
        
        return W


def main():
    parser = argparse.ArgumentParser(description="Fit vine copula example")
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--dimension', type=int, default=5,
                       help='Dimension of vine copula')
    parser.add_argument('--n-train', type=int, default=2000,
                       help='Number of training samples')
    parser.add_argument('--n-test', type=int, default=500,
                       help='Number of test samples')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Vine Denoising Copula - Vine Copula Fitting Example")
    print("=" * 70)
    
    d = args.dimension
    n_train = args.n_train
    n_test = args.n_test
    
    # =========================================================================
    # Step 1: Generate training and test data
    # =========================================================================
    print(f"\n[Step 1] Generating {d}-dimensional Gaussian copula data...")
    
    rho = 0.6
    U_train = generate_gaussian_vine(n_train, d, rho=rho, seed=42)
    U_test = generate_gaussian_vine(n_test, d, rho=rho, seed=123)
    
    print(f"  Training data: {U_train.shape}")
    print(f"  Test data: {U_test.shape}")
    print(f"  True correlation: ρ = {rho} (AR(1) structure)")
    
    if args.checkpoint is None:
        print("\n" + "-" * 50)
        print("No checkpoint provided. Showing data generation only.")
        print("To run full example with vine fitting, use:")
        print(f"  python examples/fit_vine_copula.py --checkpoint path/to/model.pt")
        print("-" * 50)
        
        # Show true pairwise correlations
        print("\n  True pairwise correlations:")
        for i in range(min(d-1, 3)):
            corr = np.corrcoef(U_train[:, i], U_train[:, i+1])[0, 1]
            print(f"    Corr(U{i+1}, U{i+2}) = {corr:.3f} (expected: {rho:.3f})")
        
        return
    
    # =========================================================================
    # Step 2: Load diffusion model
    # =========================================================================
    print(f"\n[Step 2] Loading diffusion model...")
    
    model, diffusion, config = load_model(args.checkpoint, args.device)
    m = config.get('data', {}).get('m', 64)
    
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"  Grid resolution: {m}×{m}")
    
    # =========================================================================
    # Step 3: Fit D-vine
    # =========================================================================
    print(f"\n[Step 3] Fitting {d}-dimensional D-vine copula...")
    
    import time
    start_time = time.time()
    
    vine = SimpleDVine(d, m=m, device=args.device)
    vine.fit(U_train, model, diffusion, verbose=True)
    
    fit_time = time.time() - start_time
    print(f"  Fitting completed in {fit_time:.1f} seconds")
    
    # =========================================================================
    # Step 4: Evaluate on test data
    # =========================================================================
    print(f"\n[Step 4] Evaluating on test data...")
    
    # Log-likelihood
    logpdf = vine.logpdf(U_test)
    mean_logpdf = logpdf.mean()
    
    print(f"  Log-likelihood:")
    print(f"    Mean: {mean_logpdf:.4f}")
    print(f"    Std:  {logpdf.std():.4f}")
    
    # Rosenblatt uniformity test
    W = vine.rosenblatt(U_test)
    
    print(f"\n  Rosenblatt uniformity test (KS p-values):")
    ks_pvalues = []
    for j in range(d):
        _, pval = kstest(W[:, j], 'uniform')
        ks_pvalues.append(pval)
        status = "pass" if pval > 0.05 else "fail"
        print(f"    W{j+1}: p = {pval:.4f} {status}")
    
    print(f"\n  Mean KS p-value: {np.mean(ks_pvalues):.4f}")
    print(f"  Min KS p-value:  {np.min(ks_pvalues):.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
  Dimension: {d}
  Training samples: {n_train}
  Test samples: {n_test}
  Fit time: {fit_time:.1f}s
  
  Mean log-likelihood: {mean_logpdf:.4f}
  Mean KS p-value: {np.mean(ks_pvalues):.4f}
  
  The KS p-values > 0.05 indicate that the Rosenblatt-transformed
  data are approximately uniform, suggesting a good vine fit.
""")
    
    print("=" * 70)
    print("Example Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
