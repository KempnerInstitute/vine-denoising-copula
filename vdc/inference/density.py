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
    pred_noise_clip: Optional[float] = 10.0,
    # Optional smoothing knobs (all in "grid-cell units")
    hist_smooth_sigma: float = 0.0,
    x0_smooth_sigma: float = 0.0,
    x0_smooth_every: int = 0,
    final_smooth_sigma: float = 0.0,
    transform_to_probit_space: bool = False,
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
        pred_noise_clip: Optional clip value for predicted noise (|eps| <= clip).
            This mirrors the training-time clamp and helps prevent spiky/unstable
            reverse diffusion trajectories. Set to None or <=0 to disable.
        hist_smooth_sigma: Optional Gaussian smoothing sigma applied to the *conditioning histogram*
            (before taking log). This can reduce "speckle" artifacts when the input histogram is
            very noisy (small n).
        x0_smooth_sigma: Optional Gaussian smoothing sigma applied to the predicted x0 (density-space,
            then converted back to log-space) during the reverse diffusion trajectory.
        x0_smooth_every: Apply x0 smoothing every k DDIM steps (0 disables in-loop smoothing).
        final_smooth_sigma: Optional Gaussian smoothing sigma applied to the final density grid
            (after exp+mass-normalize, before copula projection).

    Returns:
        (m, m) numpy array of copula density
    """
    hist = scatter_to_hist(samples, m, reflect=True)
    du = 1.0 / m

    hist_tensor = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)
    # Optional smoothing of the conditioning histogram (can reduce speckle on high-res grids).
    if hist_smooth_sigma is not None and float(hist_smooth_sigma) > 0:
        from vdc.utils.smoothing import smooth_density_gaussian

        hist_tensor = smooth_density_gaussian(hist_tensor.clamp_min(1e-12), sigma=float(hist_smooth_sigma), preserve_mass=True)
    log_histogram = torch.log(hist_tensor.clamp(min=1e-12))
    if bool(transform_to_probit_space):
        # Convert copula log-density to probit-space *joint* log-density f_Z(z),
        # where z = Φ^{-1}(u). This matches probit/KDE baselines and reduces edge bias.
        from vdc.utils.probit_transform import copula_logdensity_to_probit_logdensity

        log_histogram = copula_logdensity_to_probit_logdensity(log_histogram, m)
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

        # Training clamps predicted noise to avoid rare extreme values that can
        # produce overly peaky densities at inference time. Keep inference aligned.
        if pred_noise_clip is not None and float(pred_noise_clip) > 0:
            c = float(pred_noise_clip)
            pred_noise = pred_noise.clamp(min=-c, max=c)

        alpha_t = alphas_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        pred_x0 = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        pred_x0 = pred_x0.clamp(-20, 20)
        # Optional in-loop smoothing of x0 (convert to density -> smooth -> log).
        if x0_smooth_sigma is not None and float(x0_smooth_sigma) > 0 and int(x0_smooth_every) > 0:
            if (i % int(x0_smooth_every) == 0) or (t == 0):
                from vdc.utils.smoothing import smooth_density_gaussian

                d0 = torch.exp(pred_x0).clamp(1e-12, 1e6)
                d0 = smooth_density_gaussian(d0, sigma=float(x0_smooth_sigma), preserve_mass=True)
                pred_x0 = torch.log(d0.clamp_min(1e-12)).clamp(-20, 20)

        if t == 0:
            x_t = pred_x0
        else:
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            alpha_t_prev = (
                alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            )
            dir_xt = torch.sqrt(1 - alpha_t_prev) * pred_noise
            x_t = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

    if bool(transform_to_probit_space):
        # Model output is log f_Z(z_u,z_v); convert back to copula density c(u,v):
        # log c = log f_Z - log φ(z_u) - log φ(z_v)
        from vdc.utils.probit_transform import probit_logdensity_to_copula_logdensity

        log_c = probit_logdensity_to_copula_logdensity(x_t, m)
        density = torch.exp(log_c).clamp(1e-12, 1e6)
    else:
        density = torch.exp(x_t).clamp(1e-12, 1e6)
    density = torch.nan_to_num(density, nan=0.0, posinf=1e6, neginf=0.0)
    mass = (density * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    density = density / mass
    # Optional final smoothing before projection (smoothing after projection can reintroduce small artifacts
    # when you re-project again; smoothing-before-projection is usually cleaner).
    if final_smooth_sigma is not None and float(final_smooth_sigma) > 0:
        from vdc.utils.smoothing import smooth_density_gaussian

        density = smooth_density_gaussian(density, sigma=float(final_smooth_sigma), preserve_mass=True)
        density = torch.nan_to_num(density, nan=0.0, posinf=1e6, neginf=0.0)
        mass = (density * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        density = density / mass

    if int(projection_iters) > 0:
        density = copula_project(density, iters=int(projection_iters))
    density = torch.nan_to_num(density, nan=0.0, posinf=1e6, neginf=0.0)

    return density[0, 0].cpu().numpy()
