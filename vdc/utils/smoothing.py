"""
Smoothing utilities for copula density grids.

Provides Gaussian smoothing and TV-based denoising for post-processing
predicted density grids to remove spottiness while preserving structure.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def gaussian_kernel_2d(sigma: float, kernel_size: int = None) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for smoothing.
    
    Args:
        sigma: Standard deviation of Gaussian
        kernel_size: Size of kernel (auto-computed if None)
        
    Returns:
        (1, 1, k, k) Gaussian kernel tensor
    """
    if kernel_size is None:
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_2d = gauss_1d.outer(gauss_1d)
    gauss_2d = gauss_2d / gauss_2d.sum()
    
    return gauss_2d.view(1, 1, kernel_size, kernel_size)


def smooth_density_gaussian(
    density: torch.Tensor,
    sigma: float = 1.0,
    preserve_mass: bool = True,
) -> torch.Tensor:
    """
    Apply Gaussian smoothing to a density grid.
    
    Args:
        density: (B, 1, m, m) density tensor
        sigma: Gaussian sigma in grid units (higher = more smoothing)
        preserve_mass: If True, renormalize to preserve total mass
        
    Returns:
        Smoothed density tensor
    """
    if sigma <= 0:
        return density
    
    device = density.device
    dtype = density.dtype
    m = density.shape[-1]
    
    # Create kernel
    kernel = gaussian_kernel_2d(sigma).to(device=device, dtype=dtype)
    k = kernel.shape[-1]
    pad = k // 2
    
    # Apply convolution with reflection padding to avoid boundary shrinkage artifacts.
    # (Using zero-padding biases the density near u/v≈0,1 and looks like "edge effects".)
    if pad > 0:
        # reflect requires pad < input size; if extremely small grids are used,
        # fall back to replicate padding.
        pad_mode = "reflect"
        if pad >= density.shape[-1] or pad >= density.shape[-2]:
            pad_mode = "replicate"
        density_pad = F.pad(density, (pad, pad, pad, pad), mode=pad_mode)
        smoothed = F.conv2d(density_pad, kernel)
    else:
        smoothed = F.conv2d(density, kernel)
    
    # Ensure non-negativity
    smoothed = smoothed.clamp(min=1e-12)
    
    if preserve_mass:
        # Renormalize to preserve mass
        du = dv = 1.0 / m
        original_mass = (density * du * dv).sum(dim=(-2, -1), keepdim=True)
        smoothed_mass = (smoothed * du * dv).sum(dim=(-2, -1), keepdim=True)
        smoothed = smoothed * (original_mass / smoothed_mass.clamp_min(1e-12))
    
    return smoothed


def smooth_density_gaussian_np(
    density: np.ndarray,
    sigma: float = 1.0,
    preserve_mass: bool = True,
) -> np.ndarray:
    """
    Apply Gaussian smoothing to a numpy density grid.
    
    Args:
        density: (m, m) density array
        sigma: Gaussian sigma in grid units
        preserve_mass: If True, renormalize to preserve total mass
        
    Returns:
        Smoothed density array
    """
    density_t = torch.from_numpy(density).float().unsqueeze(0).unsqueeze(0)
    smoothed_t = smooth_density_gaussian(density_t, sigma, preserve_mass)
    return smoothed_t[0, 0].numpy()


def total_variation_loss(density: torch.Tensor, weight: float = 0.1) -> torch.Tensor:
    """
    Compute total variation loss for smoothness regularization.
    
    TV loss penalizes sharp changes between adjacent cells, encouraging
    smoother density predictions.
    
    Args:
        density: (B, 1, m, m) density tensor
        weight: Weight for TV loss
        
    Returns:
        Scalar TV loss
    """
    # Horizontal differences
    diff_h = torch.abs(density[:, :, :, 1:] - density[:, :, :, :-1])
    # Vertical differences
    diff_v = torch.abs(density[:, :, 1:, :] - density[:, :, :-1, :])
    
    tv_loss = weight * (diff_h.mean() + diff_v.mean())
    return tv_loss


def log_total_variation_loss(log_density: torch.Tensor, weight: float = 0.1) -> torch.Tensor:
    """
    Compute total variation loss on log-density for smoothness.
    
    Operating in log-space is better for peaked distributions where
    we want smooth relative changes rather than absolute changes.
    
    Args:
        log_density: (B, 1, m, m) log-density tensor
        weight: Weight for TV loss
        
    Returns:
        Scalar log-TV loss
    """
    # Horizontal differences in log-space
    diff_h = torch.abs(log_density[:, :, :, 1:] - log_density[:, :, :, :-1])
    # Vertical differences in log-space  
    diff_v = torch.abs(log_density[:, :, 1:, :] - log_density[:, :, :-1, :])
    
    tv_loss = weight * (diff_h.mean() + diff_v.mean())
    return tv_loss


def adaptive_smooth_density(
    density: torch.Tensor,
    samples: Optional[torch.Tensor] = None,
    base_sigma: float = 0.5,
    sample_adaptive: bool = True,
) -> torch.Tensor:
    """
    Apply adaptive Gaussian smoothing based on sample density.
    
    Applies more smoothing in low-sample regions and less in high-sample regions
    to preserve detail where data is abundant.
    
    Args:
        density: (B, 1, m, m) density tensor
        samples: (B, N, 2) sample points (optional, for adaptive smoothing)
        base_sigma: Base smoothing sigma
        sample_adaptive: If True, adapt sigma based on local sample count
        
    Returns:
        Smoothed density tensor
    """
    if not sample_adaptive or samples is None:
        return smooth_density_gaussian(density, base_sigma)
    
    # For now, use uniform smoothing
    # TODO: Implement sample-adaptive smoothing based on local sample density
    return smooth_density_gaussian(density, base_sigma)


def smooth_and_project(
    density: torch.Tensor,
    sigma: float = 0.5,
    projection_iters: int = 15,
    row_target: Optional[torch.Tensor] = None,
    col_target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Smooth density and then apply copula projection.
    
    The order matters: smooth first (to reduce spottiness), then project
    (to enforce copula constraints).
    
    Args:
        density: (B, 1, m, m) density tensor
        sigma: Gaussian smoothing sigma
        projection_iters: Number of IPFP iterations
        row_target: Target row marginals (default: uniform)
        col_target: Target column marginals (default: uniform)
        
    Returns:
        Smoothed and projected density tensor
    """
    from vdc.models.projection import copula_project
    
    # Step 1: Gaussian smoothing
    smoothed = smooth_density_gaussian(density, sigma, preserve_mass=True)
    
    # Step 2: Copula projection
    projected = copula_project(
        smoothed,
        iters=projection_iters,
        row_target=row_target,
        col_target=col_target,
    )
    
    return projected

