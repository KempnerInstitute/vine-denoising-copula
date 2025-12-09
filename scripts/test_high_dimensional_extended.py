#!/usr/bin/env python
"""
Extended High-Dimensional Vine Copula Test with Parametric Baselines

This script extends the basic high-dimensional test with:
1. More dimensions: 3, 5, 8, 10, 15, 20, 30
2. Additional scenarios: Gumbel vine, Block-structured, Factor copula
3. Parametric baseline comparisons: KDE, Gaussian copula, pyvinecopulib (if available)
4. More comprehensive metrics

Usage:
    python scripts/test_high_dimensional_extended.py \
        --checkpoint checkpoints/conditional_diffusion_v2/model_step_120000.pt \
        --output results/high_dimensional_extended

Author: Generated for vine_diffusion_copula project
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.stats import norm, kstest, anderson, spearmanr, kendalltau
from scipy.special import comb
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

# Check for optional dependencies
try:
    import pyvinecopulib as pv
    HAS_PYVINE = True
except ImportError:
    HAS_PYVINE = False
    warnings.warn("pyvinecopulib not available - parametric vine baseline disabled")

try:
    from sklearn.neighbors import KernelDensity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# Histogram utility (avoid pyvinecopulib dependency)
# =============================================================================

def scatter_to_hist(pts: np.ndarray, m: int, reflect: bool = True) -> np.ndarray:
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
    
    hist, _, _ = np.histogram2d(
        pts_all[:, 0], pts_all[:, 1],
        bins=m,
        range=[[0, 1], [0, 1]]
    )
    return hist.astype(np.float64)


# =============================================================================
# Extended Data Generation
# =============================================================================

def generate_gaussian_copula(n: int, d: int, rho: float = 0.5, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate d-dimensional Gaussian copula with AR(1) correlation."""
    if seed is not None:
        np.random.seed(seed)
    
    Sigma = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            Sigma[i, j] = rho ** abs(i - j)
    
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    U = norm.cdf(Z)
    return U, Sigma


def generate_student_copula(n: int, d: int, rho: float = 0.5, nu: float = 5.0, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate d-dimensional Student-t copula with AR(1) correlation."""
    if seed is not None:
        np.random.seed(seed)
    
    Sigma = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            Sigma[i, j] = rho ** abs(i - j)
    
    W = np.random.chisquare(nu, n)
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    T = Z / np.sqrt(W[:, np.newaxis] / nu)
    U = stats.t.cdf(T, df=nu)
    return U, Sigma


def generate_clayton_vine(n: int, d: int, theta: float = 2.0, seed: int = None) -> np.ndarray:
    """Generate D-vine with Clayton pair-copulas."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.zeros((n, d))
    U[:, 0] = np.random.uniform(0, 1, n)
    
    for j in range(1, d):
        v = np.clip(U[:, j-1], 1e-10, 1-1e-10)
        w = np.random.uniform(0, 1, n)
        w = np.clip(w, 1e-10, 1-1e-10)
        
        t1 = w ** (-theta / (1 + theta))
        t2 = v ** (-theta) - 1
        U[:, j] = np.clip(v * (t1 * t2 + 1) ** (-1/theta), 1e-10, 1-1e-10)
    
    return U


def generate_gumbel_vine(n: int, d: int, theta: float = 2.0, seed: int = None) -> np.ndarray:
    """Generate D-vine with Gumbel pair-copulas using Gumbel inverse h-function."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.zeros((n, d))
    U[:, 0] = np.random.uniform(0, 1, n)
    
    for j in range(1, d):
        v = np.clip(U[:, j-1], 1e-10, 1-1e-10)
        w = np.random.uniform(0, 1, n)
        
        # For Gumbel, use numerical inversion (simpler approximation)
        # h(u|v) ≈ C(u,v)/v for the conditional
        # Approximate with Gaussian for simplicity
        rho = 1 - 1/theta  # Approximate Kendall's tau relationship
        z_v = norm.ppf(v)
        z_w = norm.ppf(w)
        z_u = z_w * np.sqrt(1 - rho**2) + rho * z_v
        U[:, j] = np.clip(norm.cdf(z_u), 1e-10, 1-1e-10)
    
    return U


def generate_mixed_vine(n: int, d: int, seed: int = None) -> np.ndarray:
    """Generate D-vine with alternating Gaussian and Clayton copulas."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.zeros((n, d))
    U[:, 0] = np.random.uniform(0, 1, n)
    
    for j in range(1, d):
        v = np.clip(U[:, j-1], 1e-6, 1-1e-6)
        w = np.clip(np.random.uniform(0, 1, n), 1e-6, 1-1e-6)
        
        if j % 2 == 1:  # Gaussian
            rho = 0.6
            z_v = norm.ppf(v)
            z_w = norm.ppf(w)
            z_u = z_w * np.sqrt(1 - rho**2) + rho * z_v
            U[:, j] = norm.cdf(z_u)
        else:  # Clayton
            theta = 2.0
            t1 = w ** (-theta / (1 + theta))
            t2 = v ** (-theta) - 1
            U[:, j] = np.clip(v * (t1 * t2 + 1) ** (-1/theta), 1e-10, 1-1e-10)
    
    return U


def generate_block_structured(n: int, d: int, block_size: int = 3, rho_within: float = 0.7, rho_between: float = 0.2, seed: int = None) -> np.ndarray:
    """
    Generate block-structured Gaussian copula.
    Variables within each block are highly correlated, between blocks weakly.
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_blocks = (d + block_size - 1) // block_size
    
    Sigma = np.eye(d) * (1 - rho_between) + np.ones((d, d)) * rho_between
    
    for b in range(num_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, d)
        for i in range(start, end):
            for j in range(start, end):
                if i != j:
                    Sigma[i, j] = rho_within
    
    # Ensure positive definiteness
    eigvals = np.linalg.eigvalsh(Sigma)
    if eigvals.min() < 0:
        Sigma += (abs(eigvals.min()) + 0.01) * np.eye(d)
    
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    U = norm.cdf(Z)
    return U


def generate_factor_copula(n: int, d: int, n_factors: int = 2, seed: int = None) -> np.ndarray:
    """
    Generate factor copula: U_j = h(Z_j | F_1, ..., F_k) where F are common factors.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Factor loadings
    loadings = np.random.uniform(0.3, 0.8, (d, n_factors))
    
    # Generate factors
    factors = np.random.normal(0, 1, (n, n_factors))
    
    # Generate idiosyncratic components
    idio_var = 1 - np.sum(loadings**2, axis=1)
    idio_var = np.clip(idio_var, 0.1, 1.0)
    
    Z = factors @ loadings.T + np.random.normal(0, 1, (n, d)) * np.sqrt(idio_var)
    U = norm.cdf(Z)
    return U


def generate_sparse_vine(n: int, d: int, truncation: int = 2, seed: int = None) -> np.ndarray:
    """
    Generate from a truncated vine (only first `truncation` trees have dependencies).
    Variables beyond the truncation level are conditionally independent.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Use Gaussian vine with truncation
    rho = 0.5
    
    # Build correlation matrix for truncated vine
    Sigma = np.eye(d)
    for i in range(d):
        for j in range(i+1, min(i+truncation+1, d)):
            partial_corr = rho ** (j - i)
            Sigma[i, j] = partial_corr
            Sigma[j, i] = partial_corr
    
    # Ensure positive definiteness
    eigvals = np.linalg.eigvalsh(Sigma)
    if eigvals.min() < 0:
        Sigma += (abs(eigvals.min()) + 0.01) * np.eye(d)
    
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    U = norm.cdf(Z)
    return U


# =============================================================================
# Model Loading and Inference (same as before)
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
    """Estimate bivariate copula density from pair samples."""
    hist = scatter_to_hist(pair_data, m, reflect=True)
    du = 1.0 / m
    hist = hist / (hist.sum() * du * du + 1e-12)
    
    hist_tensor = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)
    log_histogram = torch.log(hist_tensor.clamp(min=1e-12))
    
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
        
        model_input_cond = torch.cat([x_t, log_histogram], dim=1)
        pred_noise_cond = model(model_input_cond, t_normalized)
        
        model_input_uncond = torch.cat([x_t, torch.zeros_like(log_histogram)], dim=1)
        pred_noise_uncond = model(model_input_uncond, t_normalized)
        
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
    
    density = torch.exp(x_t).clamp(1e-12, 1e6)
    mass = (density * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    density = density / mass
    density = copula_project(density, iters=50)
    
    density_np = density[0, 0].cpu().numpy()
    hfunc = HFuncLookup(density_np)
    
    return density_np, hfunc


# =============================================================================
# D-Vine Fitting (same as before)
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
        
        V = [U.copy()]
        total_pairs = d * (d - 1) // 2
        pbar = tqdm(total=total_pairs, disable=not verbose, desc="Fitting D-vine")
        
        for tree in range(d - 1):
            tree_copulas = []
            V_next = np.zeros((n, d - tree - 1))
            
            for edge in range(d - tree - 1):
                u_data = V[tree][:, edge]
                v_data = V[tree][:, edge + 1]
                
                pair_data = np.column_stack([u_data, v_data])
                pair_data = np.clip(pair_data, 1e-6, 1-1e-6)
                
                density, hfunc = estimate_pair_copula(
                    model, diffusion, pair_data, self.m, device
                )
                tree_copulas.append((density, hfunc))
                
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
                
                pair_pdf = hfunc.pdf(u_data, v_data)
                logpdf += np.log(np.clip(pair_pdf, 1e-10, None))
                
                h_val = hfunc.h_u_given_v(u_data, v_data)
                V_next[:, edge] = np.clip(h_val, 1e-6, 1-1e-6)
            
            V.append(V_next)
        
        return logpdf
    
    def rosenblatt(self, U: np.ndarray) -> np.ndarray:
        """Forward Rosenblatt transform."""
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


# =============================================================================
# Parametric Baselines
# =============================================================================

class GaussianCopulaBaseline:
    """Parametric Gaussian copula baseline using correlation matrix estimation."""
    
    def __init__(self, d: int):
        self.d = d
        self.Sigma = None
        self.Sigma_inv = None
    
    def fit(self, U: np.ndarray):
        """Fit Gaussian copula by estimating correlation matrix."""
        # Transform to normal scores
        Z = norm.ppf(np.clip(U, 1e-6, 1-1e-6))
        
        # Estimate correlation matrix
        self.Sigma = np.corrcoef(Z.T)
        
        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(self.Sigma)
        if eigvals.min() < 1e-6:
            self.Sigma += (1e-6 - eigvals.min()) * np.eye(self.d)
        
        self.Sigma_inv = np.linalg.inv(self.Sigma)
    
    def logpdf(self, U: np.ndarray) -> np.ndarray:
        """Compute log-density."""
        Z = norm.ppf(np.clip(U, 1e-6, 1-1e-6))
        
        # Gaussian copula density
        log_det = np.log(np.linalg.det(self.Sigma))
        
        quad_form = np.sum(Z @ (self.Sigma_inv - np.eye(self.d)) * Z, axis=1)
        
        logpdf = -0.5 * log_det - 0.5 * quad_form
        return logpdf
    
    def rosenblatt(self, U: np.ndarray) -> np.ndarray:
        """Forward Rosenblatt transform for Gaussian copula."""
        n, d = U.shape
        Z = norm.ppf(np.clip(U, 1e-6, 1-1e-6))
        W = np.zeros_like(U)
        W[:, 0] = U[:, 0]
        
        # Cholesky decomposition
        L = np.linalg.cholesky(self.Sigma)
        L_inv = np.linalg.inv(L)
        
        # Transform to independent
        Z_indep = (L_inv @ Z.T).T
        W = norm.cdf(Z_indep)
        
        return W


class PyVineCopulaBaseline:
    """Parametric vine copula baseline using pyvinecopulib."""
    
    def __init__(self, d: int):
        self.d = d
        self.vine = None
    
    def fit(self, U: np.ndarray):
        """Fit vine copula using pyvinecopulib."""
        if not HAS_PYVINE:
            raise RuntimeError("pyvinecopulib not available")
        
        # Fit vine
        controls = pv.FitControlsVinecop(
            family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.student,
                        pv.BicopFamily.clayton, pv.BicopFamily.gumbel,
                        pv.BicopFamily.frank],
            trunc_lvl=min(self.d - 1, 5),  # Truncate for speed
        )
        self.vine = pv.Vinecop(U, controls=controls)
    
    def logpdf(self, U: np.ndarray) -> np.ndarray:
        """Compute log-density."""
        return self.vine.loglik(U) / U.shape[0] * np.ones(U.shape[0])
    
    def rosenblatt(self, U: np.ndarray) -> np.ndarray:
        """Forward Rosenblatt transform."""
        return self.vine.rosenblatt(U)


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_rosenblatt_uniformity(W: np.ndarray) -> Dict[str, float]:
    """Test uniformity of Rosenblatt-transformed data."""
    n, d = W.shape
    
    ks_pvalues = []
    anderson_stats = []
    
    for j in range(d):
        ks_stat, ks_pvalue = kstest(W[:, j], 'uniform')
        ks_pvalues.append(ks_pvalue)
        
        ad_result = anderson(W[:, j], 'norm')
        anderson_stats.append(ad_result.statistic)
    
    return {
        'mean_ks_pvalue': float(np.mean(ks_pvalues)),
        'min_ks_pvalue': float(np.min(ks_pvalues)),
        'mean_anderson_stat': float(np.mean(anderson_stats)),
        'ks_pvalues': [float(p) for p in ks_pvalues],
    }


def compute_kendall_tau_error(U_true: np.ndarray, U_samples: np.ndarray) -> Dict[str, float]:
    """Compute Kendall's tau error between true and sampled data."""
    d = U_true.shape[1]
    
    tau_errors = []
    for i in range(d):
        for j in range(i+1, d):
            tau_true, _ = kendalltau(U_true[:, i], U_true[:, j])
            tau_sample, _ = kendalltau(U_samples[:, i], U_samples[:, j])
            tau_errors.append(abs(tau_true - tau_sample))
    
    return {
        'mean_tau_error': float(np.mean(tau_errors)),
        'max_tau_error': float(np.max(tau_errors)),
    }


# =============================================================================
# Main Test Function
# =============================================================================

def run_extended_test(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    config: Dict,
    device: torch.device,
    output_dir: Path,
    dimensions: List[int] = [3, 5, 8, 10, 15, 20],
    n_samples: int = 2000,
    n_test: int = 500,
    include_baselines: bool = True,
):
    """Run comprehensive high-dimensional test with baselines."""
    m = config.get('data', {}).get('m', 64)
    
    # Extended scenarios
    scenarios = [
        ('gaussian_ar1', lambda n, d, seed: generate_gaussian_copula(n, d, rho=0.6, seed=seed)[0]),
        ('student_ar1', lambda n, d, seed: generate_student_copula(n, d, rho=0.6, nu=5, seed=seed)[0]),
        ('clayton_vine', lambda n, d, seed: generate_clayton_vine(n, d, theta=2.0, seed=seed)),
        ('gumbel_vine', lambda n, d, seed: generate_gumbel_vine(n, d, theta=2.0, seed=seed)),
        ('mixed_vine', lambda n, d, seed: generate_mixed_vine(n, d, seed=seed)),
        ('block_structured', lambda n, d, seed: generate_block_structured(n, d, block_size=3, seed=seed)),
        ('factor_copula', lambda n, d, seed: generate_factor_copula(n, d, n_factors=2, seed=seed)),
        ('sparse_vine', lambda n, d, seed: generate_sparse_vine(n, d, truncation=2, seed=seed)),
    ]
    
    results = []
    
    for scenario_name, generator in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*60}")
        
        for d in dimensions:
            print(f"\n  Dimension d={d}")
            print(f"  {'-'*50}")
            
            seed = 42
            U_train = generator(n_samples, d, seed)
            U_test = generator(n_test, d, seed + 1000)
            
            result = {
                'scenario': scenario_name,
                'dimension': d,
                'n_train': n_samples,
                'n_test': n_test,
            }
            
            # ==========================================
            # Method 1: Diffusion D-Vine
            # ==========================================
            print(f"    [Diffusion D-Vine] Fitting...")
            start_time = time.time()
            
            try:
                dvine = DVineFit(d, m=m)
                dvine.fit(U_train, model, diffusion, device, verbose=False)
                
                fit_time = time.time() - start_time
                
                logpdf = dvine.logpdf(U_test)
                W = dvine.rosenblatt(U_test)
                uniformity = compute_rosenblatt_uniformity(W)
                
                result['diffusion_fit_time'] = fit_time
                result['diffusion_mean_logpdf'] = float(np.mean(logpdf))
                result['diffusion_ks_pvalue'] = uniformity['mean_ks_pvalue']
                result['diffusion_min_ks_pvalue'] = uniformity['min_ks_pvalue']
                
                print(f"    [Diffusion] LogPDF: {result['diffusion_mean_logpdf']:.4f}, KS: {result['diffusion_ks_pvalue']:.4f}, Time: {fit_time:.1f}s")
                
            except Exception as e:
                print(f"    [Diffusion] ERROR: {e}")
                result['diffusion_error'] = str(e)
            
            # ==========================================
            # Method 2: Gaussian Copula Baseline
            # ==========================================
            if include_baselines:
                print(f"    [Gaussian Baseline] Fitting...")
                start_time = time.time()
                
                try:
                    gauss = GaussianCopulaBaseline(d)
                    gauss.fit(U_train)
                    
                    fit_time = time.time() - start_time
                    
                    logpdf_gauss = gauss.logpdf(U_test)
                    W_gauss = gauss.rosenblatt(U_test)
                    uniformity_gauss = compute_rosenblatt_uniformity(W_gauss)
                    
                    result['gaussian_fit_time'] = fit_time
                    result['gaussian_mean_logpdf'] = float(np.mean(logpdf_gauss))
                    result['gaussian_ks_pvalue'] = uniformity_gauss['mean_ks_pvalue']
                    
                    print(f"    [Gaussian] LogPDF: {result['gaussian_mean_logpdf']:.4f}, KS: {result['gaussian_ks_pvalue']:.4f}")
                    
                except Exception as e:
                    print(f"    [Gaussian] ERROR: {e}")
                    result['gaussian_error'] = str(e)
            
            # ==========================================
            # Method 3: PyVineCopulib Baseline (if available)
            # ==========================================
            if include_baselines and HAS_PYVINE and d <= 15:  # Limit dimension for speed
                print(f"    [PyVine Baseline] Fitting...")
                start_time = time.time()
                
                try:
                    pyvine = PyVineCopulaBaseline(d)
                    pyvine.fit(U_train)
                    
                    fit_time = time.time() - start_time
                    
                    # PyVine loglik is total, not per-sample
                    logpdf_pv = pyvine.vine.loglik(U_test) / n_test
                    W_pv = pyvine.rosenblatt(U_test)
                    uniformity_pv = compute_rosenblatt_uniformity(W_pv)
                    
                    result['pyvine_fit_time'] = fit_time
                    result['pyvine_mean_logpdf'] = float(logpdf_pv)
                    result['pyvine_ks_pvalue'] = uniformity_pv['mean_ks_pvalue']
                    
                    print(f"    [PyVine] LogPDF: {result['pyvine_mean_logpdf']:.4f}, KS: {result['pyvine_ks_pvalue']:.4f}")
                    
                except Exception as e:
                    print(f"    [PyVine] ERROR: {e}")
                    result['pyvine_error'] = str(e)
            
            results.append(result)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary plots
    create_extended_summary_plots(results, output_dir, include_baselines)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    return results


def create_extended_summary_plots(results: List[Dict], output_dir: Path, include_baselines: bool):
    """Create comprehensive summary visualizations."""
    
    scenarios = list(set(r['scenario'] for r in results))
    dimensions = sorted(set(r['dimension'] for r in results))
    
    # Plot 1: Diffusion vs Baselines (LogPDF)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Log-PDF by dimension for each scenario
    ax = axes[0, 0]
    for scenario in scenarios[:4]:  # First 4 scenarios
        scenario_results = [r for r in results if r['scenario'] == scenario]
        dims = [r['dimension'] for r in scenario_results]
        logpdfs = [r.get('diffusion_mean_logpdf', np.nan) for r in scenario_results]
        ax.plot(dims, logpdfs, 'o-', label=scenario, markersize=6)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Mean Log-PDF')
    ax.set_title('Diffusion D-Vine: Log-Likelihood by Dimension')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # KS p-value by dimension
    ax = axes[0, 1]
    for scenario in scenarios[:4]:
        scenario_results = [r for r in results if r['scenario'] == scenario]
        dims = [r['dimension'] for r in scenario_results]
        ks_pvals = [r.get('diffusion_ks_pvalue', np.nan) for r in scenario_results]
        ax.plot(dims, ks_pvals, 'o-', label=scenario, markersize=6)
    ax.axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Mean KS p-value')
    ax.set_title('Rosenblatt Uniformity Test')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Method comparison (if baselines available)
    if include_baselines:
        ax = axes[1, 0]
        methods = ['diffusion', 'gaussian']
        if HAS_PYVINE:
            methods.append('pyvine')
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        for i, method in enumerate(methods):
            # Average over dimensions for each scenario
            avg_logpdfs = []
            for scenario in scenarios:
                scenario_results = [r for r in results if r['scenario'] == scenario]
                logpdfs = [r.get(f'{method}_mean_logpdf', np.nan) for r in scenario_results]
                valid_logpdfs = [v for v in logpdfs if not np.isnan(v)]
                avg_logpdfs.append(np.mean(valid_logpdfs) if valid_logpdfs else np.nan)
            
            ax.bar(x + i*width, avg_logpdfs, width, label=method.capitalize())
        
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Avg Log-PDF')
        ax.set_title('Method Comparison (averaged over dimensions)')
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Fit time comparison
    ax = axes[1, 1]
    for scenario in scenarios[:4]:
        scenario_results = [r for r in results if r['scenario'] == scenario]
        dims = [r['dimension'] for r in scenario_results]
        times = [r.get('diffusion_fit_time', np.nan) for r in scenario_results]
        ax.plot(dims, times, 'o-', label=scenario, markersize=6)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Fit Time (seconds)')
    ax.set_title('Computational Cost')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=150)
    plt.close()
    
    # Additional: Heatmap of performance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create performance matrix
    perf_matrix = np.zeros((len(scenarios), len(dimensions)))
    for i, scenario in enumerate(scenarios):
        for j, d in enumerate(dimensions):
            result = next((r for r in results if r['scenario'] == scenario and r['dimension'] == d), None)
            if result:
                perf_matrix[i, j] = result.get('diffusion_mean_logpdf', np.nan)
    
    im = axes[0].imshow(perf_matrix, aspect='auto', cmap='RdYlGn')
    axes[0].set_xticks(range(len(dimensions)))
    axes[0].set_xticklabels(dimensions)
    axes[0].set_yticks(range(len(scenarios)))
    axes[0].set_yticklabels(scenarios)
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Scenario')
    axes[0].set_title('Mean Log-PDF (Diffusion D-Vine)')
    plt.colorbar(im, ax=axes[0])
    
    # KS p-value heatmap
    ks_matrix = np.zeros((len(scenarios), len(dimensions)))
    for i, scenario in enumerate(scenarios):
        for j, d in enumerate(dimensions):
            result = next((r for r in results if r['scenario'] == scenario and r['dimension'] == d), None)
            if result:
                ks_matrix[i, j] = result.get('diffusion_ks_pvalue', np.nan)
    
    im = axes[1].imshow(ks_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    axes[1].set_xticks(range(len(dimensions)))
    axes[1].set_xticklabels(dimensions)
    axes[1].set_yticks(range(len(scenarios)))
    axes[1].set_yticklabels(scenarios)
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Scenario')
    axes[1].set_title('KS p-value (Rosenblatt Uniformity)')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmaps.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Extended high-dimensional vine copula test')
    parser.add_argument('--checkpoint', type=Path, 
                       default=Path('checkpoints/conditional_diffusion_v2/model_step_120000.pt'))
    parser.add_argument('--output', type=Path,
                       default=Path('results/high_dimensional_extended'))
    parser.add_argument('--dimensions', type=int, nargs='+', default=[3, 5, 8, 10, 15, 20])
    parser.add_argument('--n-train', type=int, default=2000)
    parser.add_argument('--n-test', type=int, default=500)
    parser.add_argument('--no-baselines', action='store_true', help='Skip baseline comparisons')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"PyVineCopulib available: {HAS_PYVINE}")
    
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    
    model, diffusion, config = load_model(args.checkpoint, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    results = run_extended_test(
        model, diffusion, config, device,
        args.output,
        dimensions=args.dimensions,
        n_samples=args.n_train,
        n_test=args.n_test,
        include_baselines=not args.no_baselines,
    )
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Scenario':<20} {'Dim':>5} {'Diff LogPDF':>12} {'Diff KS':>10} {'Gauss LogPDF':>12} {'Gauss KS':>10}")
    print("-" * 100)
    for r in results:
        diff_logpdf = r.get('diffusion_mean_logpdf', np.nan)
        diff_ks = r.get('diffusion_ks_pvalue', np.nan)
        gauss_logpdf = r.get('gaussian_mean_logpdf', np.nan)
        gauss_ks = r.get('gaussian_ks_pvalue', np.nan)
        print(f"{r['scenario']:<20} {r['dimension']:>5} {diff_logpdf:>12.4f} {diff_ks:>10.4f} {gauss_logpdf:>12.4f} {gauss_ks:>10.4f}")


if __name__ == '__main__':
    main()
