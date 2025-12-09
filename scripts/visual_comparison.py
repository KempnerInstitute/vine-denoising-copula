#!/usr/bin/env python
"""
Generate visual comparison of different methods.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
    {'family': 'clayton', 'params': {'theta': 5.0}, 'name': 'Clayton(θ=5.0, 90°)', 'rotation': 90},
]


def load_model(checkpoint_path, version, device):
    """Load model based on version."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    if version == 'v4':
        resolutions = config.get('multiscale', {}).get('resolutions', [8, 16, 32, 64])
        in_channels = 1 + len(resolutions)
    else:
        in_channels = 2
    
    model = GridUNet(
        m=config['data']['m'],
        in_channels=in_channels,
        base_channels=config['model'].get('base_channels', 64),
        channel_mults=tuple(config['model'].get('channel_mults', (1, 2, 3, 4))),
        num_res_blocks=config['model'].get('num_res_blocks', 2),
        attention_resolutions=tuple(config['model'].get('attention_resolutions', (16, 8))),
        dropout=config['model'].get('dropout', 0.1),
    ).to(device)
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('unet.'):
            new_state_dict[k[5:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    diffusion = CopulaAwareDiffusion(
        timesteps=config.get('diffusion', {}).get('timesteps', 1000),
        beta_schedule=config.get('diffusion', {}).get('noise_schedule', 'cosine'),
    ).to(device)
    
    return model, diffusion, config


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
def sample_density(model, diffusion, histogram, device, num_steps=50, cfg_scale=2.0):
    """Sample density from model."""
    m = histogram.shape[-1]
    T = diffusion.timesteps
    
    if histogram.shape[1] > 1:
        log_histogram = torch.log(histogram.clamp(min=1e-12))
    else:
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Models to compare
    models_info = {
        'V2': ('checkpoints/conditional_diffusion_v2/model_step_200000.pt', 'v2'),
        'V4': ('checkpoints/conditional_diffusion_v4/model_step_120000.pt', 'v4'),
        'V6': ('checkpoints/conditional_diffusion_v6/model_final.pt', 'v6'),
    }
    
    # Load models
    models = {}
    for name, (path, version) in models_info.items():
        full_path = REPO_ROOT / path
        if full_path.exists():
            try:
                model, diffusion, config = load_model(full_path, version, device)
                models[name] = (model, diffusion, config, version)
                print(f"Loaded {name}")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
    
    # Generate comparison
    output_dir = REPO_ROOT / 'results' / 'evaluation' / 'visual_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for copula in TEST_COPULAS:
        family = copula['family']
        params = copula['params']
        name = copula['name']
        rotation = copula.get('rotation', 0)
        
        print(f"\nProcessing: {name}")
        
        # Generate samples
        points = sample_bicop(family, params, 2000, rotation=rotation)
        
        # True density
        m = 64
        density_true_raw = analytic_logpdf_grid(family, params, m=m, rotation=rotation)
        density_true_raw = np.exp(np.clip(density_true_raw, -20, 20))
        density_true_t = torch.from_numpy(density_true_raw).float().unsqueeze(0).unsqueeze(0).to(device)
        density_true_t = copula_project(density_true_t, iters=50)
        density_true_np = density_true_t[0, 0].cpu().numpy()
        
        # Histogram
        histogram = create_histogram(points, m, device)
        histogram_np = histogram[0, 0].cpu().numpy()
        
        # Get predictions from each model
        predictions = {}
        for model_name, (model, diffusion, config, version) in models.items():
            if version == 'v4':
                resolutions = config.get('multiscale', {}).get('resolutions', [8, 16, 32, 64])
                smooth_sigma = config.get('multiscale', {}).get('smooth_sigma', 1.5)
                hist_input = create_multiscale_histogram(points, resolutions, device, smooth_sigma)
            else:
                hist_input = histogram
            
            pred = sample_density(model, diffusion, hist_input, device)
            pred = copula_project(pred, iters=50)
            predictions[model_name] = pred[0, 0].cpu().numpy()
        
        # Create figure
        num_models = len(predictions)
        fig, axes = plt.subplots(1, num_models + 2, figsize=(4 * (num_models + 2), 4))
        
        vmax = min(50, density_true_np.max())
        
        # Histogram
        axes[0].imshow(histogram_np, origin='lower', cmap='hot', vmin=0, vmax=vmax)
        axes[0].set_title('Input Histogram')
        axes[0].axis('off')
        
        # True density
        axes[1].imshow(np.clip(density_true_np, 0, vmax), origin='lower', cmap='hot', vmin=0, vmax=vmax)
        axes[1].set_title('True Density')
        axes[1].axis('off')
        
        # Predictions
        for i, (model_name, pred_np) in enumerate(predictions.items()):
            du = 1.0 / m
            ise = np.mean((pred_np - density_true_np) ** 2) * du * du
            axes[i + 2].imshow(pred_np, origin='lower', cmap='hot', vmin=0, vmax=vmax)
            axes[i + 2].set_title(f'{model_name}\nISE={ise:.2e}')
            axes[i + 2].axis('off')
        
        plt.suptitle(name, fontsize=14)
        plt.tight_layout()
        
        safe_name = name.replace('(', '_').replace(')', '').replace('=', '').replace('°', 'deg').replace('ρ', 'rho').replace('θ', 'theta')
        plt.savefig(output_dir / f'{safe_name}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {safe_name}_comparison.png")
    
    print(f"\nAll comparisons saved to: {output_dir}")


if __name__ == '__main__':
    main()

