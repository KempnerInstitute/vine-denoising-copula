#!/usr/bin/env python
"""
Hyperparameter Sweep for Vine Copula Diffusion Model

Tests:
1. CFG scale: 1.0, 1.5, 2.0, 2.5, 3.0
2. Diffusion steps: 25, 50, 75, 100
3. Ensemble sizes: 1, 3, 5
4. Training samples: 1000, 2000, 5000

Runs on dimension 10 with gaussian_ar1 and clayton_vine scenarios.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from scipy.stats import norm, kstest

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
        """Inverse h-function for Clayton copula."""
        if theta < 1e-10:
            return u
        t = v ** (-theta) * (u ** (-theta / (1 + theta)) - 1) + 1
        return np.clip(t ** (-1 / theta), 1e-10, 1 - 1e-10)
    
    U = np.zeros((n, d))
    W = np.random.uniform(0, 1, (n, d))
    U[:, 0] = W[:, 0]
    
    for j in range(1, d):
        v = U[:, j-1]
        U[:, j] = clayton_hinv(W[:, j], v, theta)
    
    return U


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
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
    
    return model, diffusion, config


@torch.no_grad()
def estimate_pair_copula(model, diffusion, pair_data, m, device, 
                         num_steps=50, cfg_scale=2.0, n_ensemble=1):
    """Estimate pair copula density with configurable hyperparameters."""
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
    
    densities = []
    for _ in range(n_ensemble):
        x_t = torch.randn(1, 1, m, m, device=device)
        
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
        densities.append(density)
    
    density = torch.mean(torch.stack(densities), dim=0)
    density = copula_project(density, iters=50)
    
    density_np = density[0, 0].cpu().numpy()
    hfunc = HFuncLookup(density_np)
    
    return density_np, hfunc


def fit_dvine(U, model, diffusion, device, m=64, **kwargs):
    """Fit D-vine copula using diffusion model."""
    n, d = U.shape
    pair_copulas = []
    V = [U.copy()]
    
    for tree in range(d - 1):
        tree_copulas = []
        V_next = np.zeros((n, d - tree - 1))
        
        for edge in range(d - tree - 1):
            u_data = V[tree][:, edge]
            v_data = V[tree][:, edge + 1]
            
            pair_data = np.column_stack([u_data, v_data])
            pair_data = np.clip(pair_data, 1e-6, 1-1e-6)
            
            density, hfunc = estimate_pair_copula(
                model, diffusion, pair_data, m, device, **kwargs
            )
            tree_copulas.append((density, hfunc))
            
            h_val = hfunc.h_u_given_v(u_data, v_data)
            V_next[:, edge] = np.clip(h_val, 1e-6, 1-1e-6)
        
        pair_copulas.append(tree_copulas)
        V.append(V_next)
    
    return pair_copulas


def evaluate_dvine(pair_copulas, U_test):
    """Evaluate D-vine on test data."""
    n, d = U_test.shape
    logpdf = np.zeros(n)
    W = np.zeros_like(U_test)
    W[:, 0] = U_test[:, 0]
    
    V = [U_test.copy()]
    
    for tree in range(d - 1):
        V_next = np.zeros((n, d - tree - 1))
        
        for edge in range(d - tree - 1):
            density, hfunc = pair_copulas[tree][edge]
            
            u_data = V[tree][:, edge]
            v_data = V[tree][:, edge + 1]
            
            pair_pdf = hfunc.pdf(u_data, v_data)
            logpdf += np.log(np.clip(pair_pdf, 1e-10, None))
            
            h_val = hfunc.h_u_given_v(u_data, v_data)
            V_next[:, edge] = np.clip(h_val, 1e-6, 1-1e-6)
            
            if edge == 0:
                W[:, tree + 1] = h_val
        
        V.append(V_next)
    
    ks_pvals = [kstest(W[:, j], 'uniform')[1] for j in range(d)]
    
    return {
        'mean_logpdf': float(np.mean(logpdf)),
        'mean_ks_pvalue': float(np.mean(ks_pvals)),
        'min_ks_pvalue': float(np.min(ks_pvals)),
    }


def run_sweep(model, diffusion, device, output_dir):
    """Run hyperparameter sweep."""
    results = []
    dimension = 10
    n_test = 500
    
    # Generate test data once
    U_test_gaussian = generate_gaussian_copula(n_test, dimension, rho=0.6, seed=9999)
    U_test_clayton = generate_clayton_vine(n_test, dimension, theta=2.0, seed=9999)
    
    # Sweep configurations
    configs = []
    
    # CFG sweep
    for cfg in [1.0, 1.5, 2.0, 2.5, 3.0]:
        configs.append({'name': f'cfg_{cfg}', 'cfg_scale': cfg, 'num_steps': 50, 'n_ensemble': 1, 'n_train': 2000})
    
    # Steps sweep
    for steps in [25, 50, 75, 100]:
        configs.append({'name': f'steps_{steps}', 'cfg_scale': 2.0, 'num_steps': steps, 'n_ensemble': 1, 'n_train': 2000})
    
    # Ensemble sweep
    for ens in [1, 3, 5]:
        configs.append({'name': f'ensemble_{ens}', 'cfg_scale': 2.0, 'num_steps': 50, 'n_ensemble': ens, 'n_train': 2000})
    
    # Samples sweep
    for n_train in [1000, 2000, 5000]:
        configs.append({'name': f'samples_{n_train}', 'cfg_scale': 2.0, 'num_steps': 50, 'n_ensemble': 1, 'n_train': n_train})
    
    print(f"Running {len(configs)} configurations...")
    print("=" * 80)
    
    for cfg in configs:
        print(f"\n{cfg['name']}")
        print("-" * 40)
        
        result = {'config': cfg}
        
        # Gaussian scenario
        U_train = generate_gaussian_copula(cfg['n_train'], dimension, rho=0.6, seed=42)
        
        t0 = time.time()
        copulas = fit_dvine(U_train, model, diffusion, device,
                           num_steps=cfg['num_steps'], 
                           cfg_scale=cfg['cfg_scale'],
                           n_ensemble=cfg['n_ensemble'])
        fit_time = time.time() - t0
        
        metrics = evaluate_dvine(copulas, U_test_gaussian)
        result['gaussian'] = {**metrics, 'fit_time': fit_time}
        print(f"  Gaussian: LogPDF={metrics['mean_logpdf']:.2f}, KS={metrics['mean_ks_pvalue']:.4f}, Time={fit_time:.1f}s")
        
        # Clayton scenario
        U_train = generate_clayton_vine(cfg['n_train'], dimension, theta=2.0, seed=42)
        
        t0 = time.time()
        copulas = fit_dvine(U_train, model, diffusion, device,
                           num_steps=cfg['num_steps'], 
                           cfg_scale=cfg['cfg_scale'],
                           n_ensemble=cfg['n_ensemble'])
        fit_time = time.time() - t0
        
        metrics = evaluate_dvine(copulas, U_test_clayton)
        result['clayton'] = {**metrics, 'fit_time': fit_time}
        print(f"  Clayton:  LogPDF={metrics['mean_logpdf']:.2f}, KS={metrics['mean_ks_pvalue']:.4f}, Time={fit_time:.1f}s")
        
        results.append(result)
    
    # Save results
    with open(output_dir / 'hyperparameter_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n--- CFG Scale ---")
    cfg_results = [r for r in results if r['config']['name'].startswith('cfg_')]
    for r in cfg_results:
        cfg_val = r['config']['cfg_scale']
        g_ks = r['gaussian']['mean_ks_pvalue']
        c_ks = r['clayton']['mean_ks_pvalue']
        print(f"  CFG={cfg_val}: Gaussian KS={g_ks:.4f}, Clayton KS={c_ks:.4f}")
    
    print("\n--- Diffusion Steps ---")
    step_results = [r for r in results if r['config']['name'].startswith('steps_')]
    for r in step_results:
        steps = r['config']['num_steps']
        g_ks = r['gaussian']['mean_ks_pvalue']
        c_ks = r['clayton']['mean_ks_pvalue']
        t = r['gaussian']['fit_time']
        print(f"  Steps={steps}: Gaussian KS={g_ks:.4f}, Clayton KS={c_ks:.4f}, Time={t:.1f}s")
    
    print("\n--- Ensemble Size ---")
    ens_results = [r for r in results if r['config']['name'].startswith('ensemble_')]
    for r in ens_results:
        ens = r['config']['n_ensemble']
        g_ks = r['gaussian']['mean_ks_pvalue']
        c_ks = r['clayton']['mean_ks_pvalue']
        t = r['gaussian']['fit_time']
        print(f"  Ensemble={ens}: Gaussian KS={g_ks:.4f}, Clayton KS={c_ks:.4f}, Time={t:.1f}s")
    
    print("\n--- Training Samples ---")
    samp_results = [r for r in results if r['config']['name'].startswith('samples_')]
    for r in samp_results:
        n = r['config']['n_train']
        g_ks = r['gaussian']['mean_ks_pvalue']
        c_ks = r['clayton']['mean_ks_pvalue']
        print(f"  n_train={n}: Gaussian KS={g_ks:.4f}, Clayton KS={c_ks:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, 
                       default=Path('checkpoints/conditional_diffusion_v2/model_step_120000.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/hyperparameter_sweep'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Device: {device}")
    
    checkpoint_path = REPO_ROOT / args.checkpoint
    print(f"Loading model from: {checkpoint_path}")
    model, diffusion, config = load_model(checkpoint_path, device)
    
    output_dir = REPO_ROOT / args.output / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    
    run_sweep(model, diffusion, device, output_dir)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
