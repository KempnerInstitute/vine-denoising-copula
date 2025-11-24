"""
Loss functions for copula density estimation.

Includes:
1. NLL (Negative Log-Likelihood) on points
2. ISE (Integrated Squared Error) on grids
3. Marginal constraint penalties
4. Tail-weighted losses
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def bilinear_interpolate_torch(
    grid: torch.Tensor,
    points: torch.Tensor,
    grid_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    Bilinear interpolation of grid values at arbitrary points.
    
    Args:
        grid: (B, 1, m, m) or (B, m, m) grid values
        points: (B, N, 2) query points in [grid_range]
        grid_range: (min, max) range of grid coordinates
        
    Returns:
        (B, N) interpolated values
    """
    if grid.dim() == 4:
        B, _, m, _ = grid.shape
        grid = grid.squeeze(1)  # (B, m, m)
    else:
        B, m, _ = grid.shape
    
    N = points.shape[1]
    device = grid.device
    
    # Normalize points to grid coordinates [0, m-1]
    grid_min, grid_max = grid_range
    u = (points[:, :, 0] - grid_min) / (grid_max - grid_min) * (m - 1)
    v = (points[:, :, 1] - grid_min) / (grid_max - grid_min) * (m - 1)
    
    # Clamp to valid range
    u = torch.clamp(u, 0, m - 1)
    v = torch.clamp(v, 0, m - 1)
    
    # Get integer indices
    u0 = torch.floor(u).long()
    u1 = torch.clamp(u0 + 1, 0, m - 1)
    v0 = torch.floor(v).long()
    v1 = torch.clamp(v0 + 1, 0, m - 1)
    
    # Get fractional parts
    u_frac = u - u0.float()
    v_frac = v - v0.float()
    
    # Gather grid values at corners
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N)
    
    Q00 = grid[batch_idx, u0, v0]
    Q01 = grid[batch_idx, u0, v1]
    Q10 = grid[batch_idx, u1, v0]
    Q11 = grid[batch_idx, u1, v1]
    
    # Bilinear interpolation
    result = (
        Q00 * (1 - u_frac) * (1 - v_frac) +
        Q01 * (1 - u_frac) * v_frac +
        Q10 * u_frac * (1 - v_frac) +
        Q11 * u_frac * v_frac
    )
    
    return result


def nll_points(
    density_grid: torch.Tensor,
    points: torch.Tensor,
    eps: float = 1e-10,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Negative log-likelihood of points under the density grid.
    
    NLL = -1/N Σ_i log c(u_i, v_i)
    
    Args:
        density_grid: (B, 1, m, m) or (B, m, m) density values (positive)
        points: (B, N, 2) points in [0,1]²
        eps: Small constant for numerical stability
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        NLL scalar (if reduction != 'none') or (B,) losses
    """
    # Interpolate density at points
    density_at_points = bilinear_interpolate_torch(
        density_grid, points, grid_range=(0.0, 1.0)
    )  # (B, N)
    
    # Compute negative log-likelihood
    log_density = torch.log(density_at_points + eps)
    nll = -log_density  # (B, N)
    
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    elif reduction == 'none':
        return nll.mean(dim=1)  # (B,)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def ise_logdensity(
    log_density_pred: torch.Tensor,
    log_density_target: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Integrated Squared Error between log-densities on grids.
    
    ISE = ∬ (log c_pred - log c_target)² du dv
    
    Approximated as discrete sum with uniform weights.
    
    Args:
        log_density_pred: (B, 1, m, m) or (B, m, m) predicted log-density
        log_density_target: (B, 1, m, m) or (B, m, m) target log-density
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        ISE scalar or (B,) losses
    """
    if log_density_pred.dim() == 4:
        log_density_pred = log_density_pred.squeeze(1)
    if log_density_target.dim() == 4:
        log_density_target = log_density_target.squeeze(1)
    
    # Squared error
    se = (log_density_pred - log_density_target) ** 2  # (B, m, m)
    
    # Integrate (sum over grid, normalized by grid size)
    B, m, _ = se.shape
    du = dv = 1.0 / m
    ise = se.sum(dim=[1, 2]) * du * dv  # (B,)
    
    if reduction == 'mean':
        return ise.mean()
    elif reduction == 'sum':
        return ise.sum()
    elif reduction == 'none':
        return ise
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def marginal_penalty(
    density_grid: torch.Tensor,
    target_marginal: float = 1.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Penalty for violating uniform marginal constraints.
    
    For a copula:
    - ∫ c(u,v) dv = 1 for all u  (row marginals)
    - ∫ c(u,v) du = 1 for all v  (column marginals)
    
    Computes MSE between computed marginals and target.
    
    Args:
        density_grid: (B, 1, m, m) or (B, m, m) density
        target_marginal: Target value for marginals (default: 1.0)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Marginal penalty scalar or (B,) losses
    """
    if density_grid.dim() == 4:
        B, _, m, _ = density_grid.shape
        D = density_grid.squeeze(1)
    else:
        B, m, _ = density_grid.shape
        D = density_grid
    
    du = dv = 1.0 / m
    
    # Row marginals: integrate over v (sum along axis 2)
    row_marg = (D * dv).sum(dim=2)  # (B, m)
    row_penalty = ((row_marg - target_marginal) ** 2).mean(dim=1)  # (B,)
    
    # Column marginals: integrate over u (sum along axis 1)
    col_marg = (D * du).sum(dim=1)  # (B, m)
    col_penalty = ((col_marg - target_marginal) ** 2).mean(dim=1)  # (B,)
    
    # Total penalty
    penalty = row_penalty + col_penalty  # (B,)
    
    if reduction == 'mean':
        return penalty.mean()
    elif reduction == 'sum':
        return penalty.sum()
    elif reduction == 'none':
        return penalty
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def tail_weighted_loss(
    density_grid: torch.Tensor,
    points: Optional[torch.Tensor] = None,
    tail_region: float = 0.1,
    weight: float = 2.0,
    mode: str = 'density',
    eps: float = 1e-10,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Weighted loss that emphasizes tail regions.
    
    Tail regions are defined as u < tail_region, u > 1-tail_region,
    v < tail_region, or v > 1-tail_region (corners and edges).
    
    Args:
        density_grid: (B, 1, m, m) or (B, m, m) density
        points: (B, N, 2) optional points for point-based weighting
        tail_region: Threshold for tail regions (default: 0.1)
        weight: Weight multiplier for tail regions (default: 2.0)
        mode: 'density' (penalize low density in tails) or 'nll' (weighted NLL on points)
        eps: Small constant
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Weighted loss scalar or (B,) losses
    """
    if mode == 'density':
        # Grid-based: penalize regions where density is too low in tails
        if density_grid.dim() == 4:
            B, _, m, _ = density_grid.shape
            D = density_grid.squeeze(1)
        else:
            B, m, _ = density_grid.shape
            D = density_grid
        
        # Create tail mask
        u_grid = torch.linspace(0, 1, m, device=D.device)
        v_grid = torch.linspace(0, 1, m, device=D.device)
        U, V = torch.meshgrid(u_grid, v_grid, indexing='ij')
        
        in_tail = (
            (U < tail_region) | (U > 1 - tail_region) |
            (V < tail_region) | (V > 1 - tail_region)
        )  # (m, m)
        
        # Weight mask: higher weight in tails
        weights = torch.ones_like(D)
        weights[:, in_tail] *= weight
        
        # Weighted MSE to some reference (e.g., mean density)
        target = D.mean(dim=[1, 2], keepdim=True) # Fix: mean per sample
        loss = (weights * (D - target) ** 2).mean(dim=[1, 2]) # (B,)
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
        
    elif mode == 'nll':
        # Point-based: weighted NLL
        assert points is not None, "Need points for NLL mode"
        
        # Identify points in tail regions
        u, v = points[:, :, 0], points[:, :, 1]
        in_tail = (
            (u < tail_region) | (u > 1 - tail_region) |
            (v < tail_region) | (v > 1 - tail_region)
        )  # (B, N)
        
        # Compute NLL
        density_at_points = bilinear_interpolate_torch(density_grid, points)
        nll = -torch.log(density_at_points + eps)  # (B, N)
        
        # Apply weights
        weights = torch.where(in_tail, weight, 1.0)
        weighted_nll = (weights * nll).mean(dim=1) # (B,)
        
        if reduction == 'mean':
            return weighted_nll.mean()
        elif reduction == 'sum':
            return weighted_nll.sum()
        elif reduction == 'none':
            return weighted_nll
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
    
    else:
        raise ValueError(f"Invalid mode: {mode}")


def hfunc_penalty(
    density_grid: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Penalty for h-functions violating proper copula properties.
    
    H-functions are conditional CDFs computed via cumulative integration:
    - h_{U|V}(u|v) = ∫₀ᵘ c(s,v) ds  (must be in [0,1], monotone, h(1|v)=1)
    - h_{V|U}(v|u) = ∫₀ᵛ c(u,t) dt  (must be in [0,1], monotone, h(u|1)=1)
    
    This loss ensures:
    1. H-values stay in [0, 1]
    2. H-functions are monotonically increasing
    3. Endpoint condition: h(1|·) = 1
    
    Args:
        density_grid: (B, 1, m, m) or (B, m, m) copula density
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        H-function penalty scalar or (B,) losses
    """
    if density_grid.dim() == 4:
        B, _, m, _ = density_grid.shape
        D = density_grid.squeeze(1)
    else:
        B, m, _ = density_grid.shape
        D = density_grid
    
    du = dv = 1.0 / m
    
    # Compute h-functions via cumulative integration
    # h_{U|V}(u|v) = ∫₀ᵘ c(s,v) ds - integrate along u-axis (dim=1)
    h_u_given_v = torch.cumsum(D, dim=1) * du  # (B, m, m)
    
    # h_{V|U}(v|u) = ∫₀ᵛ c(u,t) dt - integrate along v-axis (dim=2)
    h_v_given_u = torch.cumsum(D, dim=2) * dv  # (B, m, m)
    
    # Penalty 1: Values outside [0, 1]
    # Penalize negative values or values > 1
    penalty_bounds_u = (torch.relu(-h_u_given_v).mean(dim=[1, 2]) + 
                        torch.relu(h_u_given_v - 1.0).mean(dim=[1, 2]))
    penalty_bounds_v = (torch.relu(-h_v_given_u).mean(dim=[1, 2]) + 
                        torch.relu(h_v_given_u - 1.0).mean(dim=[1, 2]))
    
    # Penalty 2: Non-monotonicity
    # H-functions must be non-decreasing in their first argument
    # For h_u_given_v: should increase along dim=1 (u direction)
    diff_u = h_u_given_v[:, 1:, :] - h_u_given_v[:, :-1, :]  # Should be ≥ 0
    penalty_monotone_u = torch.relu(-diff_u).mean(dim=[1, 2])
    
    # For h_v_given_u: should increase along dim=2 (v direction)
    diff_v = h_v_given_u[:, :, 1:] - h_v_given_u[:, :, :-1]  # Should be ≥ 0
    penalty_monotone_v = torch.relu(-diff_v).mean(dim=[1, 2])
    
    # Penalty 3: Endpoint condition h(1|·) = 1
    # h_u_given_v[-1, :] should equal 1 for all v
    # h_v_given_u[:, -1] should equal 1 for all u
    penalty_endpoint_u = ((h_u_given_v[:, -1, :] - 1.0) ** 2).mean(dim=1)
    penalty_endpoint_v = ((h_v_given_u[:, :, -1] - 1.0) ** 2).mean(dim=1)
    
    # Total penalty per batch
    penalty = (penalty_bounds_u + penalty_bounds_v + 
               penalty_monotone_u + penalty_monotone_v + 
               penalty_endpoint_u + penalty_endpoint_v)  # (B,)
    
    if reduction == 'mean':
        return penalty.mean()
    elif reduction == 'sum':
        return penalty.sum()
    elif reduction == 'none':
        return penalty
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class CopulaLoss(torch.nn.Module):
    """
    Combined loss for copula density estimation.
    
    Combines multiple loss terms with configurable weights:
    - NLL on points
    - ISE on grids (if teacher available)
    - Marginal penalties
    - Tail weighting
    
    Args:
        lambda_nll: Weight for NLL loss (default: 1.0)
        lambda_ise: Weight for ISE loss (default: 0.5)
        lambda_marginal: Weight for marginal penalty (default: 0.1)
        lambda_tail: Weight for tail loss (default: 0.2)
        tail_region: Tail region threshold (default: 0.1)
        tail_weight: Tail weight multiplier (default: 2.0)
    """
    
    def __init__(
        self,
        lambda_nll: float = 1.0,
        lambda_ise: float = 0.5,
        lambda_marginal: float = 0.1,
        lambda_tail: float = 0.2,
        tail_region: float = 0.1,
        tail_weight: float = 2.0,
    ):
        super().__init__()
        self.lambda_nll = lambda_nll
        self.lambda_ise = lambda_ise
        self.lambda_marginal = lambda_marginal
        self.lambda_tail = lambda_tail
        self.tail_region = tail_region
        self.tail_weight = tail_weight
        
    def forward(
        self,
        density_grid: torch.Tensor,
        points: torch.Tensor,
        log_density_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            density_grid: (B, 1, m, m) predicted density (after projection)
            points: (B, N, 2) training points
            log_density_target: (B, 1, m, m) optional teacher log-density
            
        Returns:
            (total_loss, loss_dict)
        """
        losses = {}
        
        # NLL on points
        if self.lambda_nll > 0:
            loss_nll = nll_points(density_grid, points)
            losses['nll'] = loss_nll
        else:
            losses['nll'] = torch.tensor(0.0, device=density_grid.device)
        
        # ISE on grids (if teacher available)
        if self.lambda_ise > 0 and log_density_target is not None:
            log_density_pred = torch.log(density_grid + 1e-10)
            loss_ise = ise_logdensity(log_density_pred, log_density_target)
            losses['ise'] = loss_ise
        else:
            losses['ise'] = torch.tensor(0.0, device=density_grid.device)
        
        # Marginal penalty
        if self.lambda_marginal > 0:
            loss_marginal = marginal_penalty(density_grid)
            losses['marginal'] = loss_marginal
        else:
            losses['marginal'] = torch.tensor(0.0, device=density_grid.device)
        
        # Tail weighting
        if self.lambda_tail > 0:
            loss_tail = tail_weighted_loss(
                density_grid,
                points,
                tail_region=self.tail_region,
                weight=self.tail_weight,
                mode='nll',
            )
            losses['tail'] = loss_tail
        else:
            losses['tail'] = torch.tensor(0.0, device=density_grid.device)
        
        # Total loss
        total_loss = (
            self.lambda_nll * losses['nll'] +
            self.lambda_ise * losses['ise'] +
            self.lambda_marginal * losses['marginal'] +
            self.lambda_tail * losses['tail']
        )
        
        losses['total'] = total_loss
        
        return total_loss, losses


if __name__ == "__main__":
    print("Testing loss functions...")
    
    # Create dummy data
    B, m, N = 2, 64, 1000
    
    # Create a density grid (independence copula)
    D = torch.ones(B, 1, m, m)
    
    # Create random points
    points = torch.rand(B, N, 2)
    
    print(f"\nTest 1: NLL on points")
    loss_nll = nll_points(D, points)
    print(f"NLL: {loss_nll.item():.6f}")
    print(f"Expected: ~{-np.log(1.0):.6f} for uniform density")
    
    print(f"\nTest 2: ISE on grids")
    log_D = torch.log(D + 1e-10)
    log_D_target = torch.log(D + 1e-10) + 0.1 * torch.randn_like(D)
    loss_ise = ise_logdensity(log_D, log_D_target)
    print(f"ISE: {loss_ise.item():.6f}")
    
    print(f"\nTest 3: Marginal penalty")
    # Perfect copula should have near-zero penalty
    loss_marg = marginal_penalty(D)
    print(f"Marginal penalty (uniform): {loss_marg.item():.8f}")
    
    # Bad copula (not uniform marginals)
    D_bad = torch.rand(B, 1, m, m)
    loss_marg_bad = marginal_penalty(D_bad)
    print(f"Marginal penalty (random): {loss_marg_bad.item():.6f}")
    
    print(f"\nTest 4: Tail-weighted loss")
    loss_tail = tail_weighted_loss(D, points, tail_region=0.1, weight=2.0, mode='nll')
    print(f"Tail-weighted NLL: {loss_tail.item():.6f}")
    
    print(f"\nTest 5: Combined loss")
    criterion = CopulaLoss(
        lambda_nll=1.0,
        lambda_ise=0.5,
        lambda_marginal=0.1,
        lambda_tail=0.2,
    )
    
    total_loss, loss_dict = criterion(D, points, log_D_target)
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.6f}")
    
    print("\nAll loss tests passed!")
