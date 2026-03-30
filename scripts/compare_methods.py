#!/usr/bin/env python
"""
Quick comparison script for different training methods (V2, V4, V6).
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import json
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project
from vdc.data.generators import sample_bicop, analytic_logpdf_grid
from vdc.data.hist import scatter_to_hist

# Test copulas
TEST_COPULAS = [
    {'family': 'gaussian', 'params': {'rho': 0.7}, 'name': 'Gaussian(ρ=0.7)', 'rotation': 0},
    {'family': 'gaussian', 'params': {'rho': -0.7}, 'name': 'Gaussian(ρ=-0.7)', 'rotation': 0},
    {'family': 'clayton', 'params': {'theta': 3.0}, 'name': 'Clayton(θ=3.0)', 'rotation': 0},
    {'family': 'clayton', 'params': {'theta': 5.0}, 'name': 'Clayton(θ=5.0, 90°)', 'rotation': 90},
    {'family': 'frank', 'params': {'theta': 5.0}, 'name': 'Frank(θ=5.0)', 'rotation': 0},
]


def load_v2_model(checkpoint_path, device):
    """Load V2 model (standard UNet with 2 input channels)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = GridUNet(
        m=config['data']['m'],
        in_channels=2,
        base_channels=config['model'].get('base_channels', 64),
        channel_mults=tuple(config['model'].get('channel_mults', (1, 2, 3, 4))),
        num_res_blocks=config['model'].get('num_res_blocks', 2),
        attention_resolutions=tuple(config['model'].get('attention_resolutions', (16, 8))),
        dropout=config['model'].get('dropout', 0.1),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    diffusion = CopulaAwareDiffusion(
        timesteps=config.get('diffusion', {}).get('timesteps', 1000),
        beta_schedule=config.get('diffusion', {}).get('noise_schedule', 'cosine'),
    ).to(device)
    
    return model, diffusion, config


def load_v4_model(checkpoint_path, device):
    """Load V4 model (multi-scale UNet)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    resolutions = config.get('multiscale', {}).get('resolutions', [8, 16, 32, 64])
    num_scales = len(resolutions)
    
    model = GridUNet(
        m=config['data']['m'],
        in_channels=1 + num_scales,  # noisy + multi-scale histograms
        base_channels=config['model'].get('base_channels', 64),
        channel_mults=tuple(config['model'].get('channel_mults', (1, 2, 3, 4))),
        num_res_blocks=config['model'].get('num_res_blocks', 2),
        attention_resolutions=tuple(config['model'].get('attention_resolutions', (16, 8))),
        dropout=config['model'].get('dropout', 0.1),
    ).to(device)
    
    # Handle wrapper module prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('unet.'):
            new_state_dict[k[5:]] = v  # Remove 'unet.' prefix
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    diffusion = CopulaAwareDiffusion(
        timesteps=config.get('diffusion', {}).get('timesteps', 1000),
        beta_schedule=config.get('diffusion', {}).get('noise_schedule', 'cosine'),
    ).to(device)
    
    return model, diffusion, config


def load_v6_model(checkpoint_path, device):
    """Load V6 model (progressive resolution, same as V2 architecture)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = GridUNet(
        m=config['data']['m'],
        in_channels=2,
        base_channels=config['model'].get('base_channels', 64),
        channel_mults=tuple(config['model'].get('channel_mults', (1, 2, 3, 4))),
        num_res_blocks=config['model'].get('num_res_blocks', 2),
        attention_resolutions=tuple(config['model'].get('attention_resolutions', (16, 8))),
        dropout=config['model'].get('dropout', 0.1),
    ).to(device)
    
    # Handle wrapper module prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('unet.'):
            new_state_dict[k[5:]] = v  # Remove 'unet.' prefix
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    diffusion = CopulaAwareDiffusion(
        timesteps=config.get('diffusion', {}).get('timesteps', 1000),
        beta_schedule=config.get('diffusion', {}).get('noise_schedule', 'cosine'),
    ).to(device)
    
    return model, diffusion, config


def load_v7_model(checkpoint_path, device):
    """Load V7 model (probit space, same architecture as V2)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = GridUNet(
        m=config['data']['m'],
        in_channels=2,
        base_channels=config['model'].get('base_channels', 64),
        channel_mults=tuple(config['model'].get('channel_mults', (1, 2, 3, 4))),
        num_res_blocks=config['model'].get('num_res_blocks', 2),
        attention_resolutions=tuple(config['model'].get('attention_resolutions', (16, 8))),
        dropout=config['model'].get('dropout', 0.1),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    diffusion = CopulaAwareDiffusion(
        timesteps=config.get('diffusion', {}).get('timesteps', 1000),
        beta_schedule=config.get('diffusion', {}).get('noise_schedule', 'cosine'),
    ).to(device)
    
    return model, diffusion, config


def copula_samples_to_probit(samples, eps=1e-6):
    """Transform copula samples [0,1]² to probit space ℝ²."""
    from scipy.stats import norm
    samples_clipped = np.clip(samples, eps, 1 - eps)
    return norm.ppf(samples_clipped)


def create_probit_histogram(points, m, z_range, device):
    """Create histogram in probit space."""
    z_pts = copula_samples_to_probit(points)
    hist, _, _ = np.histogram2d(
        z_pts[:, 0], z_pts[:, 1],
        bins=m,
        range=[[-z_range, z_range], [-z_range, z_range]],
    )
    dz = 2 * z_range / m
    hist = hist / (hist.sum() * dz * dz + 1e-12)
    return torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)


def probit_density_to_copula_density(c_gaussian, m, eps=1e-7):
    """Transform probit density back to copula density."""
    import math
    device = c_gaussian.device
    NORM_CONST = 1.0 / math.sqrt(2 * math.pi)
    
    u_grid = torch.linspace(0.5/m, 1 - 0.5/m, m, device=device)
    U, V = torch.meshgrid(u_grid, u_grid, indexing='ij')
    
    u_clamped = U.clamp(eps, 1 - eps)
    v_clamped = V.clamp(eps, 1 - eps)
    Z_u = torch.erfinv(2 * u_clamped - 1) * math.sqrt(2)
    Z_v = torch.erfinv(2 * v_clamped - 1) * math.sqrt(2)
    
    phi_u = NORM_CONST * torch.exp(-0.5 * Z_u ** 2)
    phi_v = NORM_CONST * torch.exp(-0.5 * Z_v ** 2)
    jacobian = phi_u * phi_v
    
    return c_gaussian * jacobian


@torch.no_grad()
def sample_v7(model, diffusion, histogram_probit, device, z_range=4.0, num_steps=50, cfg_scale=2.0):
    """Sample from V7 model in probit space."""
    m = histogram_probit.shape[-1]
    T = diffusion.timesteps
    
    log_histogram = torch.log(histogram_probit.clamp(min=1e-12))
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
    
    # Convert to density in probit space
    dz = 2 * z_range / m
    density_probit = torch.exp(x_t).clamp(1e-12, 1e6)
    mass = (density_probit * dz * dz).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    density_probit = density_probit / mass
    
    # Transform back to copula space
    B = density_probit.shape[0]
    density_copula = probit_density_to_copula_density(density_probit.view(B, m, m), m).view(B, 1, m, m)
    
    # Normalize in copula space
    du = 1.0 / m
    mass_copula = (density_copula * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return density_copula / mass_copula


def create_histogram(samples, m, device):
    """Create normalized histogram."""
    hist = scatter_to_hist(samples, m, reflect=True)
    du = dv = 1.0 / m
    hist = hist / (hist.sum() * du * dv + 1e-12)
    return torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)


def create_multiscale_histogram(samples, resolutions, device, smooth_sigma=1.5):
    """Create multi-scale histogram pyramid for V4."""
    max_m = max(resolutions)
    num_scales = len(resolutions)
    
    histograms = torch.zeros(1, num_scales, max_m, max_m, device=device)
    
    for s, m in enumerate(resolutions):
        hist = scatter_to_hist(samples, m, reflect=True)
        du = dv = 1.0 / m
        hist = hist / (hist.sum() * du * dv + 1e-12)
        
        scale_sigma = smooth_sigma * (m / max_m)
        if scale_sigma > 0.5:
            hist = gaussian_filter(hist, sigma=scale_sigma)
            hist = hist / (hist.sum() * du * dv + 1e-12)
        
        if m < max_m:
            hist_t = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0)
            hist_t = F.interpolate(hist_t, size=(max_m, max_m), mode='bilinear', align_corners=False)
            hist = hist_t[0, 0].numpy()
            du_max = dv_max = 1.0 / max_m
            hist = hist / (hist.sum() * du_max * dv_max + 1e-12)
        
        histograms[0, s] = torch.from_numpy(hist).float()
    
    return histograms


@torch.no_grad()
def sample_v2(model, diffusion, histogram, device, num_steps=50, cfg_scale=2.0):
    """Sample from V2/V6 model."""
    m = histogram.shape[-1]
    T = diffusion.timesteps
    
    log_histogram = torch.log(histogram.clamp(min=1e-12))
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
    
    du = dv = 1.0 / m
    density = torch.exp(x_t).clamp(1e-12, 1e6)
    mass = (density * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return density / mass


@torch.no_grad()
def sample_v4(model, diffusion, histogram_pyramid, device, num_steps=50, cfg_scale=2.0):
    """Sample from V4 model."""
    m = histogram_pyramid.shape[-1]
    T = diffusion.timesteps
    
    log_histogram_pyramid = torch.log(histogram_pyramid.clamp(min=1e-12))
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
        model_input_cond = torch.cat([x_t, log_histogram_pyramid], dim=1)
        pred_noise_cond = model(model_input_cond, t_normalized)
        
        # Unconditional
        model_input_uncond = torch.cat([x_t, torch.zeros_like(log_histogram_pyramid)], dim=1)
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
    
    du = dv = 1.0 / m
    density = torch.exp(x_t).clamp(1e-12, 1e6)
    mass = (density * du * dv).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return density / mass


def evaluate_model(model, diffusion, config, version, device):
    """Evaluate model on test copulas."""
    m = config['data']['m']
    results = []
    
    for copula in TEST_COPULAS:
        family = copula['family']
        params = copula['params']
        name = copula['name']
        rotation = copula.get('rotation', 0)
        
        # Generate samples
        points = sample_bicop(family, params, 2000, rotation=rotation)
        
        # Create conditioning and sample based on version
        if version == 'v4':
            resolutions = config.get('multiscale', {}).get('resolutions', [8, 16, 32, 64])
            smooth_sigma = config.get('multiscale', {}).get('smooth_sigma', 1.5)
            histogram = create_multiscale_histogram(points, resolutions, device, smooth_sigma)
            density_pred = sample_v4(model, diffusion, histogram, device)
        elif version == 'v7':
            z_range = config.get('probit', {}).get('z_range', 4.0)
            histogram_probit = create_probit_histogram(points, m, z_range, device)
            density_pred = sample_v7(model, diffusion, histogram_probit, device, z_range=z_range)
        else:
            histogram = create_histogram(points, m, device)
            density_pred = sample_v2(model, diffusion, histogram, device)
        
        density_pred = copula_project(density_pred, iters=50)
        density_pred_np = density_pred[0, 0].cpu().numpy()
        
        # True density
        density_true_raw = analytic_logpdf_grid(family, params, m=m, rotation=rotation)
        density_true_raw = np.exp(np.clip(density_true_raw, -20, 20))
        density_true_t = torch.from_numpy(density_true_raw).float().unsqueeze(0).unsqueeze(0).to(device)
        density_true_t = copula_project(density_true_t, iters=50)
        density_true_np = density_true_t[0, 0].cpu().numpy()
        
        # Metrics
        du = dv = 1.0 / m
        ise = np.mean((density_pred_np - density_true_np) ** 2) * du * du
        
        results.append({
            'name': name,
            'ise': float(ise),
        })
    
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Models to compare
    models = {
        'V2 (200k)': ('checkpoints/conditional_diffusion_v2/model_step_200000.pt', 'v2'),
        'V4 (120k)': ('checkpoints/conditional_diffusion_v4/model_step_120000.pt', 'v4'),
        'V6 (200k)': ('checkpoints/conditional_diffusion_v6/model_final.pt', 'v6'),
        'V7 (probit)': ('checkpoints/conditional_diffusion_v7/model_final.pt', 'v7'),
    }
    
    all_results = {}
    
    for model_name, (checkpoint_path, version) in models.items():
        checkpoint_path = REPO_ROOT / checkpoint_path
        if not checkpoint_path.exists():
            print(f"Skipping {model_name}: checkpoint not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        try:
            if version == 'v4':
                model, diffusion, config = load_v4_model(checkpoint_path, device)
            elif version == 'v6':
                model, diffusion, config = load_v6_model(checkpoint_path, device)
            elif version == 'v7':
                model, diffusion, config = load_v7_model(checkpoint_path, device)
            else:
                model, diffusion, config = load_v2_model(checkpoint_path, device)
            
            results = evaluate_model(model, diffusion, config, version, device)
            all_results[model_name] = results
            
            for r in results:
                print(f"  {r['name']}: ISE = {r['ise']:.6f}")
            
            mean_ise = np.mean([r['ise'] for r in results])
            print(f"\n  Mean ISE: {mean_ise:.6f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    if len(all_results) > 1:
        copula_names = [r['name'] for r in list(all_results.values())[0]]
        
        print(f"\n{'Copula':<25}", end="")
        for model_name in all_results.keys():
            print(f"{model_name:<15}", end="")
        print()
        print("-" * (25 + 15 * len(all_results)))
        
        for i, copula_name in enumerate(copula_names):
            print(f"{copula_name:<25}", end="")
            for model_name, results in all_results.items():
                ise = results[i]['ise']
                print(f"{ise:<15.6f}", end="")
            print()
        
        print("-" * (25 + 15 * len(all_results)))
        print(f"{'Mean ISE':<25}", end="")
        for model_name, results in all_results.items():
            mean_ise = np.mean([r['ise'] for r in results])
            print(f"{mean_ise:<15.6f}", end="")
        print()
        
        # Save results
        output_path = REPO_ROOT / 'results' / 'evaluation' / 'method_comparison.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

