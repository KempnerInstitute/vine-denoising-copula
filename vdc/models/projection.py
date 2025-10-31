"""
Copula projection using Sinkhorn/IPFP algorithm.

This module enforces copula constraints (uniform marginals, unit integral)
on a positive density grid using iterative proportional fitting (IPFP)
or Sinkhorn matrix balancing.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def sinkhorn_project_density(
    density_grid: torch.Tensor,
    iters: int = 30,
    eps: float = 1e-8,
    check_convergence: bool = False,
    tol: float = 1e-4,
) -> torch.Tensor:
    """
    Project a positive density grid to a valid copula density using Sinkhorn/IPFP.
    
    A copula density c(u,v) must satisfy:
    1. c(u,v) ≥ 0 for all u,v
    2. ∫₀¹ c(u,v) dv = 1 for all u (uniform marginal)
    3. ∫₀¹ c(u,v) du = 1 for all v (uniform marginal)
    
    We work with cell masses M_ij = D_ij * Δu * Δv and enforce:
    - Row sums = Δu = 1/m
    - Col sums = Δv = 1/m
    
    Then convert back to density: D̂_ij = M_ij / (Δu * Δv)
    
    Args:
        density_grid: (B, 1, m, m) positive density grid
        iters: Number of Sinkhorn iterations
        eps: Small value to prevent division by zero
        check_convergence: If True, check for convergence and stop early
        tol: Tolerance for convergence check
        
    Returns:
        (B, 1, m, m) projected copula density
    """
    B, C, m, _ = density_grid.shape
    assert C == 1, "Expected single-channel input"
    
    # Grid spacing
    du = dv = 1.0 / m
    
    # Convert density to cell masses
    D = density_grid.squeeze(1)  # (B, m, m)
    M = D * (du * dv)  # Cell masses
    
    # Target row and column sums
    target_row = torch.full((B, m, 1), du, device=D.device, dtype=D.dtype)
    target_col = torch.full((B, 1, m), dv, device=D.device, dtype=D.dtype)
    
    # Sinkhorn iterations
    X = M.clamp_min(eps)
    
    for iter_idx in range(iters):
        X_old = X.clone() if check_convergence else None
        
        # Normalize rows
        row_sum = X.sum(dim=2, keepdim=True)  # (B, m, 1)
        X = X * (target_row / (row_sum + eps))
        
        # Normalize columns
        col_sum = X.sum(dim=1, keepdim=True)  # (B, 1, m)
        X = X * (target_col / (col_sum + eps))
        
        # Check convergence
        if check_convergence and iter_idx > 5:
            max_change = (X - X_old).abs().max()
            if max_change < tol:
                break
    
    # Convert back to density
    D_hat = X / (du * dv)
    D_hat = D_hat.unsqueeze(1)  # (B, 1, m, m)
    
    return D_hat


def copula_project(
    density_grid: torch.Tensor,
    iters: int = 30,
    eps: float = 1e-8,
    method: str = "sinkhorn",
) -> torch.Tensor:
    """
    Main interface for copula projection.
    
    Args:
        density_grid: (B, 1, m, m) positive density grid
        iters: Number of iterations
        eps: Small value for numerical stability
        method: Projection method ('sinkhorn' or 'ipfp', currently only sinkhorn)
        
    Returns:
        (B, 1, m, m) projected copula density
    """
    if method == "sinkhorn":
        return sinkhorn_project_density(density_grid, iters, eps)
    else:
        raise ValueError(f"Unknown projection method: {method}")


def check_copula_constraints(
    density_grid: torch.Tensor,
    eps: float = 1e-8,
    verbose: bool = True,
) -> dict:
    """
    Check how well a density grid satisfies copula constraints.
    
    Args:
        density_grid: (B, 1, m, m) density grid
        eps: Epsilon for log computation
        verbose: Whether to print diagnostics
        
    Returns:
        Dictionary with constraint violation metrics
    """
    B, C, m, _ = density_grid.shape
    D = density_grid.squeeze(1)  # (B, m, m)
    
    du = dv = 1.0 / m
    
    # Check positivity
    min_density = D.min().item()
    
    # Check marginal integrals
    # ∫ c(u,v) dv should equal 1 for all u
    marginal_u = (D * dv).sum(dim=2)  # (B, m), should be all 1s
    # ∫ c(u,v) du should equal 1 for all v
    marginal_v = (D * du).sum(dim=1)  # (B, m), should be all 1s
    
    # Compute errors
    marginal_u_error = (marginal_u - 1.0).abs().max().item()
    marginal_v_error = (marginal_v - 1.0).abs().max().item()
    marginal_u_mean_error = (marginal_u - 1.0).abs().mean().item()
    marginal_v_mean_error = (marginal_v - 1.0).abs().mean().item()
    
    # Check total mass
    total_mass = (D * du * dv).sum(dim=(1, 2))  # (B,), should be all 1s
    total_mass_error = (total_mass - 1.0).abs().max().item()
    
    results = {
        'min_density': min_density,
        'marginal_u_max_error': marginal_u_error,
        'marginal_v_max_error': marginal_v_error,
        'marginal_u_mean_error': marginal_u_mean_error,
        'marginal_v_mean_error': marginal_v_mean_error,
        'total_mass_error': total_mass_error,
        'satisfies_positivity': min_density >= 0,
        'satisfies_marginals': max(marginal_u_error, marginal_v_error) < 1e-3,
        'satisfies_total_mass': total_mass_error < 1e-3,
    }
    
    if verbose:
        print("Copula constraint check:")
        print(f"  Min density: {min_density:.6f}")
        print(f"  Marginal U max error: {marginal_u_error:.6e}")
        print(f"  Marginal V max error: {marginal_v_error:.6e}")
        print(f"  Marginal U mean error: {marginal_u_mean_error:.6e}")
        print(f"  Marginal V mean error: {marginal_v_mean_error:.6e}")
        print(f"  Total mass error: {total_mass_error:.6e}")
        print(f"  Satisfies positivity: {results['satisfies_positivity']}")
        print(f"  Satisfies marginals: {results['satisfies_marginals']}")
        print(f"  Satisfies total mass: {results['satisfies_total_mass']}")
    
    return results


def project_and_check(
    density_grid: torch.Tensor,
    iters: int = 30,
    eps: float = 1e-8,
    verbose: bool = True,
) -> tuple:
    """
    Project density and check constraints.
    
    Returns:
        (projected_density, constraint_checks_before, constraint_checks_after)
    """
    if verbose:
        print("\nBefore projection:")
    checks_before = check_copula_constraints(density_grid, eps, verbose)
    
    projected = copula_project(density_grid, iters, eps)
    
    if verbose:
        print("\nAfter projection:")
    checks_after = check_copula_constraints(projected, eps, verbose)
    
    return projected, checks_before, checks_after


if __name__ == "__main__":
    # Test copula projection
    import numpy as np
    
    print("Testing copula projection...")
    
    # Create a test density (not necessarily a valid copula)
    m = 64
    torch.manual_seed(42)
    
    # Generate a non-uniform density
    x = torch.randn(1, 1, m, m)
    density = F.softplus(x)  # Make positive
    
    print(f"\nTest 1: Random positive density")
    print(f"Input density shape: {density.shape}")
    print(f"Input density range: [{density.min():.3f}, {density.max():.3f}]")
    
    projected, before, after = project_and_check(density, iters=50, verbose=True)
    
    print(f"\nProjected density range: [{projected.min():.3f}, {projected.max():.3f}]")
    
    # Test 2: Near-independence copula
    print("\n" + "="*60)
    print("Test 2: Near-independence (flat) density")
    flat_density = torch.ones(1, 1, m, m) * (m * m)  # Should integrate to 1
    
    projected_flat, _, _ = project_and_check(flat_density, iters=50, verbose=True)
    
    # Test 3: Strong correlation
    print("\n" + "="*60)
    print("Test 3: Strong correlation (diagonal bias)")
    
    u_grid = torch.linspace(0, 1, m)
    v_grid = torch.linspace(0, 1, m)
    uu, vv = torch.meshgrid(u_grid, v_grid, indexing='ij')
    
    # Create a diagonal-biased density
    diagonal_density = torch.exp(-50 * (uu - vv)**2)
    diagonal_density = diagonal_density.unsqueeze(0).unsqueeze(0)
    
    projected_diag, _, _ = project_and_check(diagonal_density, iters=50, verbose=True)
    
    print("\nAll tests completed!")
