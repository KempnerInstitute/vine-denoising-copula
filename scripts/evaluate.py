#!/usr/bin/env python
"""
Comprehensive Evaluation Script for Vine Diffusion Copula.

Evaluates the trained model on:
1. Bivariate copula density estimation (various families)
2. High-dimensional vine copula fitting (D-vine)
3. Sampling quality metrics (Rosenblatt uniformity)

Usage:
    # Full evaluation
    python scripts/evaluate.py --checkpoint path/to/model.pt
    
    # Quick evaluation (fewer copulas/dimensions)
    python scripts/evaluate.py --checkpoint path/to/model.pt --quick
    
    # Bivariate only
    python scripts/evaluate.py --checkpoint path/to/model.pt --mode bivariate
    
    # High-dimensional only
    python scripts/evaluate.py --checkpoint path/to/model.pt --mode vine --dimensions 3 5 10
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import time

import numpy as np
import torch
from scipy import stats
from scipy.stats import norm, kstest
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.hfunc import HFuncLookup
from vdc.config import get_run_dir
from vdc.inference.density import sample_density_grid, scatter_to_hist
from vdc.data.generators import (
    sample_gaussian_copula, sample_student_copula,
    sample_clayton_copula, sample_gumbel_copula,
    sample_frank_copula, sample_joe_copula,
    gaussian_copula_density, clayton_copula_density,
    generate_gaussian_vine, generate_student_vine,
    generate_clayton_vine, generate_mixed_vine,
)


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, CopulaAwareDiffusion, Dict]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model_cfg = config.get('model', {})
    m = config.get('data', {}).get('m', 64)
    model_type = model_cfg.get('type', 'diffusion_unet')
    in_channels_cfg = model_cfg.get('in_channels', 1)
    if model_type == 'diffusion_unet':
        if in_channels_cfg != 1:
            print(f"Warning: overriding in_channels={in_channels_cfg} -> 1 for diffusion_unet")
        in_channels = 1
    else:
        in_channels = in_channels_cfg
    
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
    
    return model, diffusion, config


def estimate_pair_copula(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    pair_data: np.ndarray,
    m: int,
    device: torch.device,
    num_steps: int = 50,
) -> Tuple[np.ndarray, HFuncLookup]:
    """Estimate bivariate copula density from pair samples (shared sampler)."""
    use_histogram_conditioning = bool(getattr(model, "conv_in").in_channels > 1)
    density_np = sample_density_grid(
        model=model,
        diffusion=diffusion,
        samples=pair_data,
        m=m,
        device=device,
        num_steps=num_steps,
        cfg_scale=1.0,
        use_histogram_conditioning=use_histogram_conditioning,
        projection_iters=50,
    )
    hfunc = HFuncLookup(density_np)
    return density_np, hfunc


# =============================================================================
# Bivariate Evaluation
# =============================================================================

BIVARIATE_TEST_COPULAS = [
    {'family': 'gaussian', 'params': {'rho': 0.7}, 'name': 'Gaussian(ρ=0.7)'},
    {'family': 'gaussian', 'params': {'rho': -0.7}, 'name': 'Gaussian(ρ=-0.7)'},
    {'family': 'clayton', 'params': {'theta': 3.0}, 'name': 'Clayton(θ=3)'},
    {'family': 'gumbel', 'params': {'theta': 2.5}, 'name': 'Gumbel(θ=2.5)'},
    {'family': 'frank', 'params': {'theta': 5.0}, 'name': 'Frank(θ=5)'},
    {'family': 'joe', 'params': {'theta': 3.0}, 'name': 'Joe(θ=3)'},
]

QUICK_BIVARIATE = BIVARIATE_TEST_COPULAS[:3]


def get_true_density(family: str, params: dict, m: int) -> np.ndarray:
    """Compute true copula density on grid."""
    u_grid = np.linspace(0.5/m, 1 - 0.5/m, m)
    v_grid = np.linspace(0.5/m, 1 - 0.5/m, m)
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
    
    if family == 'gaussian':
        density = gaussian_copula_density(U, V, params['rho'])
    elif family == 'clayton':
        density = clayton_copula_density(U, V, params['theta'])
    else:
        # Estimate from large sample
        sampler = {
            'student': lambda n: sample_student_copula(n, params.get('rho', 0.5), params.get('df', 5)),
            'gumbel': lambda n: sample_gumbel_copula(n, params['theta']),
            'frank': lambda n: sample_frank_copula(n, params['theta']),
            'joe': lambda n: sample_joe_copula(n, params['theta']),
        }[family]
        samples = sampler(100000)
        density = scatter_to_hist(samples, m, reflect=True)
        du = 1.0 / m
        density = density / (density.sum() * du * du + 1e-12)
    
    return np.clip(density, 1e-10, None)


def evaluate_bivariate(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    config: Dict,
    device: torch.device,
    test_copulas: List[Dict],
    n_samples: int = 2000,
) -> List[Dict]:
    """Evaluate on bivariate copulas."""
    m = config.get('data', {}).get('m', 64)
    results = []
    
    samplers = {
        'gaussian': lambda n, p: sample_gaussian_copula(n, p['rho']),
        'student': lambda n, p: sample_student_copula(n, p.get('rho', 0.5), p.get('df', 5)),
        'clayton': lambda n, p: sample_clayton_copula(n, p['theta']),
        'gumbel': lambda n, p: sample_gumbel_copula(n, p['theta']),
        'frank': lambda n, p: sample_frank_copula(n, p['theta']),
        'joe': lambda n, p: sample_joe_copula(n, p['theta']),
    }
    
    for spec in tqdm(test_copulas, desc="Bivariate evaluation"):
        family = spec['family']
        params = spec['params']
        name = spec['name']
        
        # Generate samples
        samples = samplers[family](n_samples, params)
        
        # Estimate density
        density_pred, _ = estimate_pair_copula(model, diffusion, samples, m, device)
        
        # True density
        density_true = get_true_density(family, params, m)
        
        # Metrics
        du = 1.0 / m
        ise = np.mean((density_pred - density_true) ** 2) * du * du
        corr = np.corrcoef(density_pred.flatten(), density_true.flatten())[0, 1]
        
        # Histogram correlation (to detect copying)
        hist = scatter_to_hist(samples, m, reflect=True)
        hist = hist / (hist.sum() * du * du + 1e-12)
        corr_hist = np.corrcoef(density_pred.flatten(), hist.flatten())[0, 1]
        
        results.append({
            'name': name,
            'family': family,
            'params': params,
            'ise': float(ise),
            'corr_true': float(corr),
            'corr_hist': float(corr_hist),
            'density_pred': density_pred,
            'density_true': density_true,
        })
    
    return results


# =============================================================================
# Vine Copula Evaluation
# =============================================================================

class DVineFit:
    """D-vine copula fitted with diffusion-estimated pair copulas."""
    
    def __init__(self, d: int, m: int = 64):
        self.d = d
        self.m = m
        self.pair_copulas: List[List[Tuple[np.ndarray, HFuncLookup]]] = []
    
    def fit(self, U: np.ndarray, model, diffusion, device, verbose: bool = False):
        """Fit D-vine to data."""
        n, d = U.shape
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
                
                density, hfunc = estimate_pair_copula(model, diffusion, pair_data, self.m, device)
                tree_copulas.append((density, hfunc))
                
                h_val = hfunc.h_u_given_v(u_data, v_data)
                V_next[:, edge] = np.clip(h_val, 1e-6, 1-1e-6)
                pbar.update(1)
            
            self.pair_copulas.append(tree_copulas)
            V.append(V_next)
        
        pbar.close()
    
    def logpdf(self, U: np.ndarray) -> np.ndarray:
        """Compute log-density at points."""
        n, d = U.shape
        logpdf = np.zeros(n)
        V = [U.copy()]
        
        for tree in range(d - 1):
            V_next = np.zeros((n, d - tree - 1))
            
            for edge in range(d - tree - 1):
                _, hfunc = self.pair_copulas[tree][edge]
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


def evaluate_vine(
    model: torch.nn.Module,
    diffusion: CopulaAwareDiffusion,
    config: Dict,
    device: torch.device,
    dimensions: List[int],
    n_train: int = 2000,
    n_test: int = 500,
) -> List[Dict]:
    """Evaluate on high-dimensional vine scenarios."""
    m = config.get('data', {}).get('m', 64)
    
    scenarios = [
        ('gaussian_ar1', lambda n, d, s: generate_gaussian_vine(n, d, rho=0.6, seed=s)),
        ('student_ar1', lambda n, d, s: generate_student_vine(n, d, rho=0.6, df=5, seed=s)),
        ('clayton_vine', lambda n, d, s: generate_clayton_vine(n, d, theta=2.0, seed=s)),
        ('mixed_vine', lambda n, d, s: generate_mixed_vine(n, d, seed=s)),
    ]
    
    results = []
    
    for scenario_name, generator in scenarios:
        for d in dimensions:
            print(f"  {scenario_name}, d={d}...")
            
            U_train = generator(n_train, d, 42)
            U_test = generator(n_test, d, 1042)
            
            start_time = time.time()
            dvine = DVineFit(d, m=m)
            dvine.fit(U_train, model, diffusion, device)
            fit_time = time.time() - start_time
            
            logpdf = dvine.logpdf(U_test)
            mean_logpdf = float(np.mean(logpdf))
            
            W = dvine.rosenblatt(U_test)
            ks_pvalues = [kstest(W[:, j], 'uniform')[1] for j in range(d)]
            
            results.append({
                'scenario': scenario_name,
                'dimension': d,
                'mean_logpdf': mean_logpdf,
                'mean_ks_pvalue': float(np.mean(ks_pvalues)),
                'min_ks_pvalue': float(np.min(ks_pvalues)),
                'fit_time_sec': fit_time,
            })
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def create_bivariate_plots(results: List[Dict], output_dir: Path):
    """Create bivariate evaluation plots."""
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i, r in enumerate(results):
        vmax = min(50, max(r['density_pred'].max(), r['density_true'].max()))
        
        axes[i, 0].imshow(r['density_pred'], origin='lower', cmap='hot', vmin=0, vmax=vmax)
        axes[i, 0].set_title(f'Predicted')
        
        axes[i, 1].imshow(r['density_true'], origin='lower', cmap='hot', vmin=0, vmax=vmax)
        axes[i, 1].set_title(f'True')
        
        diff = r['density_pred'] - r['density_true']
        axes[i, 2].imshow(diff, origin='lower', cmap='RdBu', vmin=-vmax/2, vmax=vmax/2)
        axes[i, 2].set_title(f'Difference')
        
        axes[i, 3].text(0.5, 0.5, f"{r['name']}\n\nISE: {r['ise']:.2e}\nCorr: {r['corr_true']:.3f}",
                       ha='center', va='center', fontsize=12, transform=axes[i, 3].transAxes)
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bivariate_results.png', dpi=150)
    plt.close()


def create_vine_plots(results: List[Dict], output_dir: Path):
    """Create vine evaluation plots."""
    scenarios = list(set(r['scenario'] for r in results))
    dimensions = sorted(set(r['dimension'] for r in results))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for scenario in scenarios:
        dims = [r['dimension'] for r in results if r['scenario'] == scenario]
        logpdfs = [r['mean_logpdf'] for r in results if r['scenario'] == scenario]
        axes[0].plot(dims, logpdfs, 'o-', label=scenario, markersize=8)
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Mean Log-PDF')
    axes[0].set_title('Log-Likelihood vs Dimension')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for scenario in scenarios:
        dims = [r['dimension'] for r in results if r['scenario'] == scenario]
        ks = [r['mean_ks_pvalue'] for r in results if r['scenario'] == scenario]
        axes[1].plot(dims, ks, 'o-', label=scenario, markersize=8)
    axes[1].axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Mean KS p-value')
    axes[1].set_title('Rosenblatt Uniformity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    for scenario in scenarios:
        dims = [r['dimension'] for r in results if r['scenario'] == scenario]
        times = [r['fit_time_sec'] for r in results if r['scenario'] == scenario]
        axes[2].plot(dims, times, 'o-', label=scenario, markersize=8)
    axes[2].set_xlabel('Dimension')
    axes[2].set_ylabel('Fit Time (s)')
    axes[2].set_title('Computational Cost')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'vine_results.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vine Diffusion Copula")
    parser.add_argument('--checkpoint', type=Path, required=True, help='Model checkpoint')
    parser.add_argument('--output', type=Path, default=Path('results'), help='Output directory')
    parser.add_argument('--mode', choices=['all', 'bivariate', 'vine'], default='all')
    parser.add_argument('--quick', action='store_true', help='Quick evaluation')
    parser.add_argument('--dimensions', type=int, nargs='+', default=[3, 5, 10])
    parser.add_argument('--n-train', type=int, default=2000)
    parser.add_argument('--n-test', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("=" * 60)
    print("Vine Diffusion Copula - Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    
    model, diffusion, config = load_model(args.checkpoint, device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    
    output_dir = get_run_dir(
        base_dir=str(args.output),
        experiment_name="evaluation",
        include_timestamp=True,
        include_job_id=True,
    )
    
    all_results = {'checkpoint': str(args.checkpoint), 'config': config}
    
    if args.mode in ['all', 'bivariate']:
        print("\n" + "-" * 40)
        print("Bivariate Evaluation")
        print("-" * 40)
        
        test_copulas = QUICK_BIVARIATE if args.quick else BIVARIATE_TEST_COPULAS
        bivariate_results = evaluate_bivariate(
            model, diffusion, config, device, test_copulas
        )
        
        print("\nBivariate Results:")
        print(f"{'Name':<25} {'ISE':>12} {'Corr(True)':>12} {'Corr(Hist)':>12}")
        print("-" * 65)
        for r in bivariate_results:
            print(f"{r['name']:<25} {r['ise']:>12.2e} {r['corr_true']:>12.3f} {r['corr_hist']:>12.3f}")
        
        create_bivariate_plots(bivariate_results, output_dir / 'figures')
        
        all_results['bivariate'] = [
            {k: v for k, v in r.items() if k not in ['density_pred', 'density_true']}
            for r in bivariate_results
        ]
    
    if args.mode in ['all', 'vine']:
        print("\n" + "-" * 40)
        print("Vine Copula Evaluation")
        print("-" * 40)
        
        dimensions = [3] if args.quick else args.dimensions
        vine_results = evaluate_vine(
            model, diffusion, config, device, dimensions,
            n_train=args.n_train, n_test=args.n_test
        )
        
        print("\nVine Results:")
        print(f"{'Scenario':<20} {'Dim':>5} {'LogPDF':>10} {'KS-pval':>10} {'Time':>8}")
        print("-" * 60)
        for r in vine_results:
            print(f"{r['scenario']:<20} {r['dimension']:>5} {r['mean_logpdf']:>10.3f} {r['mean_ks_pvalue']:>10.3f} {r['fit_time_sec']:>7.1f}s")
        
        create_vine_plots(vine_results, output_dir / 'figures')
        all_results['vine'] = vine_results
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
