"""Log-domain IPFP (Sinkhorn) projection for copula densities.

Differentiable enforcement of uniform marginals:
  - Works on cell masses in log space for stability.
  - Returns projected density grid with exact (within tolerance) uniform marginals.

Usage:
    D_proj = ipfp_project_log(D_raw, iters=20)

Supports batching: input (B,1,m,m).
"""
from typing import Optional
import torch
import math


def ipfp_project_log(density: torch.Tensor, iters: int = 20, eps: float = 1e-12, fast_path: Optional[int] = None, 
                     stabilize: bool = False, max_log_value: float = 15.0) -> torch.Tensor:
    """Project positive density to copula via log-domain IPFP.

    Args:
        density: (B,1,m,m) positive tensor
        iters: full number of iterations (if fast_path not used)
        eps: numerical stabilizer
        fast_path: if provided, run that many iterations then return early (used for adaptive scheme)
        stabilize: if True, apply gradient stabilization (clamp log values, detach extremes)
        max_log_value: maximum absolute value for log-space operations (prevents overflow)
    Returns:
        (B,1,m,m) projected density
    """
    assert density.dim() == 4 and density.size(1) == 1, "Expected (B,1,m,m)"
    B, _, m, _ = density.shape
    du = dv = 1.0 / m
    # Convert to masses and log domain, squeeze channel dim
    M0 = (density.squeeze(1).clamp_min(eps)) * (du * dv)  # (B,m,m)
    logM0 = M0.log()  # (B,m,m)
    
    if stabilize:
        # Clamp log values to prevent overflow in exp
        logM0 = logM0.clamp(-max_log_value, max_log_value)
    
    # Initialize dual potentials a (rows), b (cols)
    a = torch.zeros(B, m, device=density.device, dtype=density.dtype)
    b = torch.zeros(B, m, device=density.device, dtype=density.dtype)
    target = math.log(du + eps)  # log target row/col sum (scalar)
    total_iters = fast_path if fast_path is not None else iters
    for _ in range(total_iters):
        # Row update: log row sums of current matrix exp(logM0 + a_i + b_j)
        log_rows = torch.logsumexp(logM0 + a.unsqueeze(2) + b.unsqueeze(1), dim=2)  # (B,m)
        a = a + (target - log_rows)
        
        if stabilize:
            a = a.clamp(-max_log_value, max_log_value)
        
        # Col update
        log_cols = torch.logsumexp(logM0 + a.unsqueeze(2) + b.unsqueeze(1), dim=1)  # (B,m)
        b = b + (target - log_cols)
        
        if stabilize:
            b = b.clamp(-max_log_value, max_log_value)
    
    logM = logM0 + a.unsqueeze(2) + b.unsqueeze(1)
    
    if stabilize:
        logM = logM.clamp(-max_log_value, max_log_value)
    
    M = logM.exp()
    D = (M / (du * dv)).unsqueeze(1)
    return D


def marginal_deviation(
    density: torch.Tensor,
    row_widths: Optional[torch.Tensor] = None,
    col_widths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute mean squared marginal deviation from 1 after simple normalization (no full IPFP)."""
    B, _, m, _ = density.shape
    device = density.device
    dtype = density.dtype
    if row_widths is None:
        row_widths = torch.full((m,), 1.0 / m, device=device, dtype=dtype)
    else:
        row_widths = row_widths.to(device=device, dtype=dtype)
    if col_widths is None:
        col_widths = torch.full((m,), 1.0 / m, device=device, dtype=dtype)
    else:
        col_widths = col_widths.to(device=device, dtype=dtype)
    row_widths_bc = row_widths.view(1, 1, m, 1)
    col_widths_bc = col_widths.view(1, 1, 1, m)
    area = row_widths_bc * col_widths_bc
    D = density.clamp_min(1e-12)
    mass = (D * area).sum(dim=(2,3), keepdim=True)
    Dn = D / mass
    rows = (Dn * col_widths_bc).sum(dim=3)
    cols = (Dn * row_widths_bc).sum(dim=2)
    dev = ((rows - 1)**2).mean() + ((cols - 1)**2).mean()
    return 0.5 * dev
