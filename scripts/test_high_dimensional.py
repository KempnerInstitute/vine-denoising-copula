#!/usr/bin/env python
"""
High-Dimensional Vine Copula Test

Tests the best V2 diffusion model on high-dimensional copula problems using
vine copula decomposition. Evaluates on:
- Various dimensions: 3, 5, 10, 15, 20
- Different dependence structures: Gaussian, Student-t, Clayton cascade
- Multiple metrics: log-likelihood, Rosenblatt uniformity, sampling quality

Usage:
    python scripts/test_high_dimensional.py --checkpoint checkpoints/conditional_diffusion_v2/model_step_120000.pt

Author: Generated for vine_diffusion_copula project
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.stats import norm, kstest, anderson
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project
from vdc.models.hfunc import HFuncLookup

# Import scatter_to_hist directly to avoid pyvinecopulib dependency
def scatter_to_hist(pts: np.ndarray, m: int, reflect: bool = True) -> np.ndarray:
    """
    Create 2D histogram from scatter points.
    
    Args:
        pts: (n, 2) array of points in [0, 1]^2
        m: Grid resolution
        reflect: Whether to use reflection at boundaries
        
    Returns:
        (m, m) histogram array
    """
    if reflect:
        # Reflection at boundaries helps with edge effects
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
    
    # Create histogram
    hist, _, _ = np.histogram2d(
        pts_all[:, 0], pts_all[:, 1],
        bins=m,
        range=[[0, 1], [0, 1]]
    )
    
    return hist.astype(np.float64)


# =============================================================================
# Data Generation for High-Dimensional Copulas
# =============================================================================

def generate_gaussian_copula(n: int, d: int, rho: float = 0.5, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate samples from a d-dimensional Gaussian copula with AR(1) correlation.
    
    Returns:
        U: (n, d) copula samples
        Sigma: (d, d) correlation matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    # AR(1) correlation structure: Sigma[i,j] = rho^|i-j|
    Sigma = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            Sigma[i, j] = rho ** abs(i - j)
    
    # Generate correlated Gaussians
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    
    # Transform to uniform
    U = norm.cdf(Z)
    
    return U, Sigma


def generate_student_copula(n: int, d: int, rho: float = 0.5, nu: float = 5.0, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate samples from a d-dimensional Student-t copula with AR(1) correlation.
    
    Returns:
        U: (n, d) copula samples
        Sigma: (d, d) correlation matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    # AR(1) correlation structure
    Sigma = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            Sigma[i, j] = rho ** abs(i - j)
    
    # Generate from multivariate t via: X = sqrt(nu/W) * Z where W ~ chi2(nu), Z ~ MVN
    W = np.random.chisquare(nu, n)
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    T = Z / np.sqrt(W[:, np.newaxis] / nu)
    
    # Transform to uniform via t-CDF
    U = stats.t.cdf(T, df=nu)
    
    return U, Sigma


def generate_clayton_vine(n: int, d: int, theta: float = 2.0, seed: int = None) -> np.ndarray:
    """
    Generate samples from a D-vine with Clayton pair-copulas.
    Uses sequential conditional sampling.
    
    Returns:
        U: (n, d) copula samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    U = np.zeros((n, d))
    U[:, 0] = np.random.uniform(0, 1, n)
    
    for j in range(1, d):
        # Generate using Clayton inverse h-function
        # For Clayton: h^{-1}(w|v) = v * (w^(-theta/(1+theta)) * (v^(-theta) - 1) + 1)^(-1/theta)
        v = U[:, j-1]
        w = np.random.uniform(0, 1, n)
        
        # Avoid numerical issues
        v = np.clip(v, 1e-10, 1-1e-10)
        w = np.clip(w, 1e-10, 1-1e-10)
        
        # Clayton inverse h-function
        t1 = w ** (-theta / (1 + theta))
        t2 = v ** (-theta) - 1
        U[:, j] = np.clip(v * (t1 * t2 + 1) ** (-1/theta), 1e-10, 1-1e-10)
    
    return U


def generate_mixed_vine(n: int, d: int, seed: int = None) -> np.ndarray:
    """
    Generate from a vine with mixed copula families.
    Uses alternating Gaussian and Clayton in the first tree.
    """
    if seed is not None:
        np.random.seed(seed)
    
    U = np.zeros((n, d))
    U[:, 0] = np.random.uniform(0, 1, n)
    
    for j in range(1, d):
        v = U[:, j-1]
        w = np.random.uniform(0, 1, n)
        
        v = np.clip(v, 1e-6, 1-1e-6)
        w = np.clip(w, 1e-6, 1-1e-6)
        
        if j % 2 == 1:  # Odd: use Gaussian
            rho = 0.6
            z_v = norm.ppf(v)
            z_w = norm.ppf(w)
            z_u = z_w * np.sqrt(1 - rho**2) + rho * z_v
            U[:, j] = norm.cdf(z_u)
        else:  # Even: use Clayton
            theta = 2.0
            t1 = w ** (-theta / (1 + theta))
            t2 = v ** (-theta) - 1
            U[:, j] = np.clip(v * (t1 * t2 + 1) ** (-1/theta), 1e-10, 1-1e-10)
    
    return U


# =============================================================================
# Model Loading and Inference
# =============================================================================

def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, CopulaAwareDiffusion, Dict]:
    """Load trained V2 model from checkpoint."""
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
    
    # Load state dict
    state_dict = checkpoint.get('model_state_dict', {})
    model.load_state_dict(state_dict)
    model.eval()
    
    diffusion = CopulaAwareDiffusion(
        timesteps=config.get('diffusion', {}).get('timesteps', 1000),
        beta_schedule=config.get('diffusion', {}).get('noise_schedule', 'cosine'),
    ).to(device)
    
    return model, diffusion, config


@torch.no_grad()
def estimate_pair_copula(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    pair_data: np.ndarray,
    m: int,
    device: torch.device,
    num_steps: int = 50,
    cfg_scale: float = 2.0,
) -> Tuple[np.ndarray, HFuncLookup]:
    """
    Estimate bivariate copula density from pair samples using V2 model.
    
    Args:
        model: Trained diffusion model
        diffusion: Diffusion process
        pair_data: (n, 2) samples
        m: Grid resolution
        device: Compute device
        
    Returns:
        density: (m, m) estimated copula density
        hfunc: HFuncLookup object for h-function computations
    """
    # Create histogram from samples
    hist = scatter_to_hist(pair_data, m, reflect=True)
    du = 1.0 / m
    hist = hist / (hist.sum() * du * du + 1e-12)
    
    hist_tensor = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)
    log_histogram = torch.log(hist_tensor.clamp(min=1e-12))
    
    # Sample with CFG
    T = diffusion.timesteps
    x_t = torch.randn(1, 1, m, m, device=device)
    
    step_size = max(1, T // num_steps)
    timesteps = list(range(T - 1, -1, -step_size))
    if timesteps[-1] != 0:
        timesteps.append(0)
    
    alphas_cumprod = diffusion.alphas_cumprod.to(device)
    
    for i, t in enumerate(timesteps):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
        t_normalized = t_tensor.float() / T
        
        # Conditional
        model_input_cond = torch.cat([x_t, log_histogram], dim=1)
        pred_noise_cond = model(model_input_cond, t_normalized)
        
        # Unconditional
        model_input_uncond = torch.cat([x_t, torch.zeros_like(log_histogram)], dim=1)
        pred_noise_uncond = model(model_input_uncond, t_normalized)
        
        # CFG
        pred_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
        
        alpha_t = alphas_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        pred_x0 = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        pred_x0 = pred_x0.clamp(-20, 20)
        
        if t == 0:
            x_t = pred_x0
        else:
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            alpha_t_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            dir_xt = torch.sqrt(1 - alpha_t_prev) * pred_noise
            x_t = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
    
    # Convert to density and project
    density = torch.exp(x_t).clamp(1e-12, 1e6)
    mass = (density * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    density = density / mass
    density = copula_project(density, iters=50)
    
    density_np = density[0, 0].cpu().numpy()
    
    # Create h-function lookup (expects numpy array)
    hfunc = HFuncLookup(density_np)
    
    return density_np, hfunc


# =============================================================================
# D-Vine Fitting
# =============================================================================

class DVineFit:
    """D-vine copula fitted with diffusion-estimated pair copulas."""
    
    def __init__(self, d: int, m: int = 64):
        self.d = d
        self.m = m
        self.pair_copulas: List[List[Tuple[np.ndarray, HFuncLookup]]] = []
        self.order: List[int] = list(range(d))
    
    def fit(
        self,
        U: np.ndarray,
        model: torch.nn.Module,
        diffusion: CopulaAwareDiffusion,
        device: torch.device,
        verbose: bool = True,
    ):
        """Fit D-vine to data using diffusion model for pair copulas."""
        n, d = U.shape
        assert d == self.d
        
        # Tree 1: fit pair copulas for consecutive variables
        # Tree k: fit conditional pair copulas
        
        # Store h-transformed data for each tree
        V = [U.copy()]  # V[0] = original data
        
        total_pairs = d * (d - 1) // 2
        pbar = tqdm(total=total_pairs, disable=not verbose, desc="Fitting D-vine")
        
        for tree in range(d - 1):
            tree_copulas = []
            V_next = np.zeros((n, d - tree - 1))
            
            for edge in range(d - tree - 1):
                # Variables for this edge
                if tree == 0:
                    # Tree 1: use original data
                    u_data = V[tree][:, edge]
                    v_data = V[tree][:, edge + 1]
                else:
                    # Tree k: use h-transformed data from previous tree
                    u_data = V[tree][:, edge]
                    v_data = V[tree][:, edge + 1]
                
                # Combine into pair data
                pair_data = np.column_stack([u_data, v_data])
                pair_data = np.clip(pair_data, 1e-6, 1-1e-6)
                
                # Estimate pair copula
                density, hfunc = estimate_pair_copula(
                    model, diffusion, pair_data, self.m, device
                )
                tree_copulas.append((density, hfunc))
                
                # Compute h-transform for next tree
                # HFuncLookup expects numpy arrays
                h_val = hfunc.h_u_given_v(u_data, v_data)
                V_next[:, edge] = np.clip(h_val, 1e-6, 1-1e-6)
                
                pbar.update(1)
            
            self.pair_copulas.append(tree_copulas)
            V.append(V_next)
        
        pbar.close()
    
    def logpdf(self, U: np.ndarray) -> np.ndarray:
        """Compute log-density at points U."""
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
                
                # Add log-density (HFuncLookup expects numpy arrays)
                pair_pdf = hfunc.pdf(u_data, v_data)
                logpdf += np.log(np.clip(pair_pdf, 1e-10, None))
                
                # Compute h-transform for next tree
                h_val = hfunc.h_u_given_v(u_data, v_data)
                V_next[:, edge] = np.clip(h_val, 1e-6, 1-1e-6)
            
            V.append(V_next)
        
        return logpdf
    
    def rosenblatt(self, U: np.ndarray) -> np.ndarray:
        """Forward Rosenblatt transform."""
        n, d = U.shape
        W = np.zeros_like(U)
        W[:, 0] = U[:, 0]
        
        # Apply sequential h-transforms
        V = [U.copy()]
        
        for tree in range(d - 1):
            V_next = np.zeros((n, d - tree - 1))
            
            for edge in range(d - tree - 1):
                _, hfunc = self.pair_copulas[tree][edge]
                
                u_data = V[tree][:, edge]
                v_data = V[tree][:, edge + 1]
                
                # HFuncLookup expects numpy arrays
                h_val = hfunc.h_u_given_v(u_data, v_data)
                V_next[:, edge] = np.clip(h_val, 1e-6, 1-1e-6)
                
                if edge == 0:
                    W[:, tree + 1] = h_val
            
            V.append(V_next)
        
        return W


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_rosenblatt_uniformity(W: np.ndarray) -> Dict[str, float]:
    """
    Test uniformity of Rosenblatt-transformed data.
    
    Returns various uniformity metrics per dimension.
    """
    n, d = W.shape
    metrics = {
        'mean_ks_pvalue': 0.0,
        'min_ks_pvalue': 1.0,
        'mean_anderson_stat': 0.0,
    }
    
    ks_pvalues = []
    anderson_stats = []
    
    for j in range(d):
        # Kolmogorov-Smirnov test for uniformity
        ks_stat, ks_pvalue = kstest(W[:, j], 'uniform')
        ks_pvalues.append(ks_pvalue)
        
        # Anderson-Darling test
        ad_result = anderson(W[:, j], 'norm')  # Use transformed
        anderson_stats.append(ad_result.statistic)
    
    metrics['mean_ks_pvalue'] = float(np.mean(ks_pvalues))
    metrics['min_ks_pvalue'] = float(np.min(ks_pvalues))
    metrics['mean_anderson_stat'] = float(np.mean(anderson_stats))
    metrics['ks_pvalues'] = [float(p) for p in ks_pvalues]
    
    return metrics


def compute_correlation_metrics(U_true: np.ndarray, U_fitted: np.ndarray) -> Dict[str, float]:
    """Compare pairwise correlations between true and fitted samples."""
    d = U_true.shape[1]
    
    corr_true = np.corrcoef(U_true.T)
    corr_fitted = np.corrcoef(U_fitted.T)
    
    # Extract upper triangle
    idx = np.triu_indices(d, k=1)
    corr_true_vec = corr_true[idx]
    corr_fitted_vec = corr_fitted[idx]
    
    mae = np.mean(np.abs(corr_true_vec - corr_fitted_vec))
    rmse = np.sqrt(np.mean((corr_true_vec - corr_fitted_vec) ** 2))
    
    return {
        'correlation_mae': float(mae),
        'correlation_rmse': float(rmse),
    }


# =============================================================================
# Main Test Function
# =============================================================================

def run_test(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    config: Dict,
    device: torch.device,
    output_dir: Path,
    dimensions: List[int] = [3, 5, 10, 15],
    n_samples: int = 2000,
    n_test: int = 500,
):
    """
    Run comprehensive high-dimensional test.
    """
    m = config.get('data', {}).get('m', 64)
    
    # Test scenarios
    scenarios = [
        ('gaussian_ar1', lambda n, d, seed: generate_gaussian_copula(n, d, rho=0.6, seed=seed)[0]),
        ('student_ar1', lambda n, d, seed: generate_student_copula(n, d, rho=0.6, nu=5, seed=seed)[0]),
        ('clayton_vine', lambda n, d, seed: generate_clayton_vine(n, d, theta=2.0, seed=seed)),
        ('mixed_vine', lambda n, d, seed: generate_mixed_vine(n, d, seed=seed)),
    ]
    
    results = []
    
    for scenario_name, generator in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*60}")
        
        for d in dimensions:
            print(f"\n  Dimension d={d}")
            print(f"  {'-'*50}")
            
            # Generate training data
            seed = 42
            U_train = generator(n_samples, d, seed)
            U_test = generator(n_test, d, seed + 1000)
            
            # Fit D-vine
            print(f"    Fitting D-vine...")
            start_time = time.time()
            
            dvine = DVineFit(d, m=m)
            dvine.fit(U_train, model, diffusion, device, verbose=False)
            
            fit_time = time.time() - start_time
            print(f"    Fit time: {fit_time:.2f}s")
            
            # Evaluate
            print(f"    Evaluating...")
            
            # 1. Log-likelihood on test data
            logpdf = dvine.logpdf(U_test)
            mean_logpdf = float(np.mean(logpdf))
            
            # 2. Rosenblatt transform uniformity
            W = dvine.rosenblatt(U_test)
            uniformity = compute_rosenblatt_uniformity(W)
            
            # Store results
            result = {
                'scenario': scenario_name,
                'dimension': d,
                'n_train': n_samples,
                'n_test': n_test,
                'fit_time_sec': fit_time,
                'mean_logpdf': mean_logpdf,
                **uniformity,
            }
            results.append(result)
            
            print(f"    Mean log-pdf: {mean_logpdf:.4f}")
            print(f"    Rosenblatt KS p-value (mean): {uniformity['mean_ks_pvalue']:.4f}")
            print(f"    Rosenblatt KS p-value (min): {uniformity['min_ks_pvalue']:.4f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary plots
    create_summary_plots(results, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    return results


def create_summary_plots(results: List[Dict], output_dir: Path):
    """Create summary visualization of results."""
    
    # Group by scenario
    scenarios = list(set(r['scenario'] for r in results))
    dimensions = sorted(set(r['dimension'] for r in results))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean log-pdf by dimension
    ax = axes[0, 0]
    for scenario in scenarios:
        dims = [r['dimension'] for r in results if r['scenario'] == scenario]
        logpdfs = [r['mean_logpdf'] for r in results if r['scenario'] == scenario]
        ax.plot(dims, logpdfs, 'o-', label=scenario, markersize=8)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Mean Log-PDF')
    ax.set_title('Log-Likelihood vs Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: KS p-value by dimension
    ax = axes[0, 1]
    for scenario in scenarios:
        dims = [r['dimension'] for r in results if r['scenario'] == scenario]
        ks_pvals = [r['mean_ks_pvalue'] for r in results if r['scenario'] == scenario]
        ax.plot(dims, ks_pvals, 'o-', label=scenario, markersize=8)
    ax.axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Mean KS p-value')
    ax.set_title('Rosenblatt Uniformity Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Plot 3: Fit time by dimension
    ax = axes[1, 0]
    for scenario in scenarios:
        dims = [r['dimension'] for r in results if r['scenario'] == scenario]
        times = [r['fit_time_sec'] for r in results if r['scenario'] == scenario]
        ax.plot(dims, times, 'o-', label=scenario, markersize=8)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Fit Time (seconds)')
    ax.set_title('Computational Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary bar chart
    ax = axes[1, 1]
    x = np.arange(len(scenarios))
    width = 0.2
    
    for i, d in enumerate(dimensions[:4]):  # Show first 4 dimensions
        logpdfs = [r['mean_logpdf'] for r in results if r['dimension'] == d]
        if len(logpdfs) == len(scenarios):
            ax.bar(x + i*width, logpdfs, width, label=f'd={d}')
    
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Mean Log-PDF')
    ax.set_title('Performance by Scenario and Dimension')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=150)
    plt.close()
    
    # Per-scenario detail plots
    for scenario in scenarios:
        scenario_results = [r for r in results if r['scenario'] == scenario]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        dims = [r['dimension'] for r in scenario_results]
        
        # Log-pdf
        axes[0].bar(dims, [r['mean_logpdf'] for r in scenario_results])
        axes[0].set_xlabel('Dimension')
        axes[0].set_ylabel('Mean Log-PDF')
        axes[0].set_title(f'{scenario}: Log-Likelihood')
        
        # KS p-values per dimension (stacked)
        for r in scenario_results:
            d = r['dimension']
            ks_pvals = r.get('ks_pvalues', [])
            if ks_pvals:
                axes[1].scatter([d] * len(ks_pvals), ks_pvals, alpha=0.5)
        axes[1].axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
        axes[1].set_xlabel('Dimension')
        axes[1].set_ylabel('KS p-value')
        axes[1].set_title(f'{scenario}: Rosenblatt Uniformity')
        axes[1].set_ylim(0, 1)
        
        # Fit time
        axes[2].bar(dims, [r['fit_time_sec'] for r in scenario_results])
        axes[2].set_xlabel('Dimension')
        axes[2].set_ylabel('Fit Time (s)')
        axes[2].set_title(f'{scenario}: Computational Cost')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{scenario}_details.png', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='High-dimensional vine copula test')
    parser.add_argument('--checkpoint', type=Path, 
                       default=Path('checkpoints/conditional_diffusion_v2/model_step_120000.pt'),
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=Path,
                       default=Path('results/high_dimensional_test'),
                       help='Output directory')
    parser.add_argument('--dimensions', type=int, nargs='+', default=[3, 5, 10],
                       help='Dimensions to test')
    parser.add_argument('--n-train', type=int, default=2000,
                       help='Training samples')
    parser.add_argument('--n-test', type=int, default=500,
                       help='Test samples')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Available checkpoints:")
        ckpt_dir = REPO_ROOT / 'checkpoints' / 'conditional_diffusion_v2'
        if ckpt_dir.exists():
            for f in sorted(ckpt_dir.glob('*.pt')):
                print(f"  {f}")
        sys.exit(1)
    
    model, diffusion, config = load_model(args.checkpoint, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Run test
    results = run_test(
        model, diffusion, config, device,
        args.output,
        dimensions=args.dimensions,
        n_samples=args.n_train,
        n_test=args.n_test,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<20} {'Dim':>5} {'LogPDF':>10} {'KS-pval':>10} {'Time':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['scenario']:<20} {r['dimension']:>5} {r['mean_logpdf']:>10.4f} {r['mean_ks_pvalue']:>10.4f} {r['fit_time_sec']:>8.1f}s")


if __name__ == '__main__':
    main()
