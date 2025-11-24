"""Anti-aliased histogram / bilinear splatting for bivariate samples.

Produces a smooth density estimate on an m x m grid whose cell centers are at (k+0.5)/m.
Vectorized bilinear splat accumulates fractional contributions to the four neighboring cells.
Optionally applies Gaussian blur (separable) for additional anti-aliasing.

API
----
anti_aliased_hist(samples, m, sigma=0.0, normalize=True)
  samples: (B,N,2) or (N,2) in [0,1]
  m: grid size
  sigma: Gaussian blur std in cells (0 disables)
  normalize: if True, returns density (integral 1); else returns raw counts per cell.

Returns: density tensor of shape (B,1,m,m) if batched, else (1,m,m).

Notes:
  - Density values approximate continuous density: counts/(N*du*dv) with du=dv=1/m.
  - For very large N, this matches standard histogram up to bilinear smoothing.
  - Samples falling outside [0,1] are clipped.
"""
from __future__ import annotations
from typing import Tuple
import math
import torch
import torch.nn.functional as F

def _gaussian_kernel1d(sigma: float, device, dtype, truncate: float = 3.0) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    radius = int(truncate * sigma + 0.5)
    xs = torch.arange(-radius, radius+1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (xs / sigma)**2)
    kernel = kernel / kernel.sum()
    return kernel

def _apply_gaussian_blur(grid: torch.Tensor, sigma: float) -> torch.Tensor:
    # grid shape (B,1,m,m)
    if sigma <= 0:
        return grid
    B, C, H, W = grid.shape
    device = grid.device; dtype = grid.dtype
    kx = _gaussian_kernel1d(sigma, device, dtype)
    ky = kx  # isotropic
    kx = kx.view(1,1,-1)
    ky = ky.view(1,1,-1)
    pad_x = (kx.shape[2]-1)//2
    pad_y = (ky.shape[2]-1)//2
    # Separable: first horizontal then vertical
    grid = F.conv2d(F.pad(grid, (pad_x,pad_x,0,0), mode='reflect'), kx.unsqueeze(2))
    grid = F.conv2d(F.pad(grid, (0,0,pad_y,pad_y), mode='reflect'), ky.unsqueeze(3))
    return grid

def anti_aliased_hist(samples: torch.Tensor, m: int, sigma: float = 0.0, normalize: bool = True) -> torch.Tensor:
    """Compute anti-aliased histogram via bilinear splatting.

    Args:
      samples: (B,N,2) or (N,2) tensor in [0,1]
      m: grid resolution
      sigma: optional Gaussian blur std (in cell units)
      normalize: if True, return density integrating to 1 over [0,1]^2
    Returns:
      (B,1,m,m) density tensor.
    """
    if samples.dim() == 2:
        samples = samples.unsqueeze(0)
    assert samples.dim() == 3 and samples.size(-1) == 2, "samples must have shape (B,N,2)"
    B, N, _ = samples.shape
    device = samples.device
    # Clip to [0,1]
    S = samples.clamp(0.0, 1.0)
    # Convert to continuous grid coordinates: cell centers at (i+0.5)/m => i = u*m - 0.5
    coords = S * m - 0.5  # (B,N,2)
    x = coords[...,0]
    y = coords[...,1]
    i0 = torch.floor(x).long()
    j0 = torch.floor(y).long()
    di = x - i0.float()
    dj = y - j0.float()
    i1 = i0 + 1
    j1 = j0 + 1
    # Clamp indices
    i0c = i0.clamp(0, m-1); i1c = i1.clamp(0, m-1)
    j0c = j0.clamp(0, m-1); j1c = j1.clamp(0, m-1)
    # Weights
    w00 = (1 - di) * (1 - dj)
    w10 = di * (1 - dj)
    w01 = (1 - di) * dj
    w11 = di * dj
    hist = torch.zeros(B, m, m, device=device)
    for b in range(B):
        hist[b].index_put_((i0c[b], j0c[b]), w00[b], accumulate=True)
        hist[b].index_put_((i1c[b], j0c[b]), w10[b], accumulate=True)
        hist[b].index_put_((i0c[b], j1c[b]), w01[b], accumulate=True)
        hist[b].index_put_((i1c[b], j1c[b]), w11[b], accumulate=True)
    if normalize:
        # Convert counts to density: counts/N approximates mass per cell; density = mass/(du*dv) with du=dv=1/m
        hist = hist * (m * m) / float(N)
    hist = hist.unsqueeze(1)  # (B,1,m,m)
    if sigma > 0:
        hist = _apply_gaussian_blur(hist, sigma)
    if normalize:
        # Renormalize after blur to preserve integral
        du = dv = 1.0 / m
        mass = (hist * du * dv).sum(dim=(2,3), keepdim=True).clamp_min(1e-12)
        hist = hist / mass
    return hist

__all__ = ["anti_aliased_hist"]
