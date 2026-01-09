"""
Shared utilities for turning a trained diffusion model into a copula density.

This reuses the same DDIM-style reverse diffusion used by the CLI so that
any caller (e.g., vine fitting, notebooks) produces identical results.
"""
from typing import Optional, Tuple

import numpy as np
import torch

from vdc.models.projection import copula_project


def scatter_to_hist(pts: np.ndarray, m: int, reflect: bool = True) -> np.ndarray:
    """Create a normalized 2D histogram from points in [0,1]^2."""
    if reflect:
        pts_reflected = []
        for dx, dy in [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            shifted = pts.copy()
            shifted[:, 0] += dx
            shifted[:, 1] += dy
            mask = (
                (shifted[:, 0] >= 0)
                & (shifted[:, 0] <= 1)
                & (shifted[:, 1] >= 0)
                & (shifted[:, 1] <= 1)
            )
            pts_reflected.append(shifted[mask])
        pts_all = np.vstack(pts_reflected)
    else:
        pts_all = pts

    hist, _, _ = np.histogram2d(
        pts_all[:, 0], pts_all[:, 1], bins=m, range=[[0, 1], [0, 1]]
    )
    du = 1.0 / m
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() * du * du + 1e-12)
    return hist


@torch.no_grad()
def sample_density_grid(
    model: torch.nn.Module,
    diffusion,
    samples: np.ndarray,
    m: int,
    device: torch.device,
    num_steps: int = 50,
    cfg_scale: float = 1.0,
    use_histogram_conditioning: bool = False,
    projection_iters: int = 50,
    log_n: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Run reverse diffusion to estimate a copula density grid.

    Args:
        model: Trained UNet (predicts noise)
        diffusion: CopulaAwareDiffusion instance
        samples: (n, 2) pseudo-observations in [0,1]^2
        m: Grid resolution
        device: Torch device
        num_steps: Number of DDIM steps
        cfg_scale: Guidance scale (if histogram conditioning was trained)
        use_histogram_conditioning: If model expects histogram channel
        projection_iters: IPFP iterations to enforce copula constraints

    Returns:
        (m, m) numpy array of copula density
    """
    hist = scatter_to_hist(samples, m, reflect=True)
    du = 1.0 / m

    hist_tensor = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)
    log_histogram = torch.log(hist_tensor.clamp(min=1e-12))
    dtype = hist_tensor.dtype
    
    # Compute log of sample size if not provided
    if log_n is None:
        log_n = torch.tensor([np.log(len(samples))], device=device, dtype=dtype)
    elif not isinstance(log_n, torch.Tensor):
        log_n = torch.tensor([log_n], device=device, dtype=dtype)
    else:
        log_n = log_n.to(device=device, dtype=dtype)

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

        if use_histogram_conditioning:
            # Classifier-free guidance
            model_input_cond = torch.cat([x_t, log_histogram], dim=1)
            pred_noise_cond = model(model_input_cond, t_normalized, log_n)

            model_input_uncond = torch.cat([x_t, torch.zeros_like(log_histogram)], dim=1)
            pred_noise_uncond = model(model_input_uncond, t_normalized, log_n)

            pred_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
        else:
            pred_noise = model(x_t, t_normalized, log_n)

        alpha_t = alphas_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        pred_x0 = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        pred_x0 = pred_x0.clamp(-20, 20)

        if t == 0:
            x_t = pred_x0
        else:
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            alpha_t_prev = (
                alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            )
            dir_xt = torch.sqrt(1 - alpha_t_prev) * pred_noise
            x_t = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

    density = torch.exp(x_t).clamp(1e-12, 1e6)
    mass = (density * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    density = density / mass
    density = copula_project(density, iters=projection_iters)

    return density[0, 0].cpu().numpy()
