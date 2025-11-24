"""Probit (Gaussian copula) space transformations with Jacobian corrections.

For proper Gaussian copula representation, we transform:
- Copula space [0,1]² with uniform marginals
- ↔ Probit/Gaussian space [-∞,∞]² with standard normal marginals

Key transformations:
1. Coordinate transform: u → z = Φ⁻¹(u) where Φ is standard normal CDF
2. Density transform includes Jacobian correction:
   
   c_copula(u,v) = c_gaussian(Φ⁻¹(u), Φ⁻¹(v)) · φ(Φ⁻¹(u)) · φ(Φ⁻¹(v))
   
   where φ is standard normal PDF.

This ensures that when training in probit space with normal marginals,
we can correctly evaluate copula properties (uniform marginals) after transformation.
"""
import torch
import math

# Constants for numerical stability
EPS = 1e-7
NORM_CONST = 1.0 / math.sqrt(2 * math.pi)


def copula_to_probit_coordinates(u: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Transform copula coordinates [0,1] to probit space [-∞,∞].
    
    Args:
        u: Tensor of copula coordinates in [0,1]
        eps: Small value for clamping to avoid infinities
        
    Returns:
        z = Φ⁻¹(u) in probit space
    """
    u_clamped = u.clamp(eps, 1 - eps)
    # erfinv(2u-1) * sqrt(2) = Φ⁻¹(u)
    z = torch.erfinv(2 * u_clamped - 1) * math.sqrt(2)
    return z


def probit_to_copula_coordinates(z: torch.Tensor) -> torch.Tensor:
    """Transform probit coordinates [-∞,∞] back to copula space [0,1].
    
    Args:
        z: Tensor in probit/Gaussian space
        
    Returns:
        u = Φ(z) in copula space [0,1]
    """
    # Φ(z) = 0.5 * (1 + erf(z/sqrt(2)))
    u = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return u


def standard_normal_pdf(z: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF: φ(z) = (1/√(2π)) exp(-z²/2).
    
    Args:
        z: Input values
        
    Returns:
        φ(z)
    """
    return NORM_CONST * torch.exp(-0.5 * z ** 2)


def standard_normal_logpdf(z: torch.Tensor) -> torch.Tensor:
    """Log of standard normal PDF: log φ(z) = -0.5*log(2π) - z²/2.
    
    Args:
        z: Input values
        
    Returns:
        log φ(z)
    """
    return -0.5 * math.log(2 * math.pi) - 0.5 * z ** 2


def copula_density_to_probit_density(
    c_copula: torch.Tensor,
    m: int,
    eps: float = EPS
) -> torch.Tensor:
    """Transform copula density to probit/Gaussian space density.
    
    Given copula density c(u,v) on grid [0,1]², compute the corresponding
    Gaussian copula density c_G(z_u, z_v) where z = Φ⁻¹(u).
    
    Transformation:
        c_G(z_u, z_v) = c_copula(Φ(z_u), Φ(z_v)) / (φ(z_u) · φ(z_v))
        
    Args:
        c_copula: Copula density tensor, shape (..., m, m)
        m: Grid resolution
        eps: Clamping epsilon
        
    Returns:
        c_gaussian: Density in probit space, shape (..., m, m)
    """
    device = c_copula.device
    
    # Create copula grid coordinates
    u_grid = torch.linspace(0.5/m, 1 - 0.5/m, m, device=device)
    v_grid = torch.linspace(0.5/m, 1 - 0.5/m, m, device=device)
    U, V = torch.meshgrid(u_grid, v_grid, indexing='ij')
    
    # Transform to probit coordinates
    Z_u = copula_to_probit_coordinates(U, eps)
    Z_v = copula_to_probit_coordinates(V, eps)
    
    # Compute Jacobian: product of marginal PDFs
    phi_u = standard_normal_pdf(Z_u)
    phi_v = standard_normal_pdf(Z_v)
    jacobian = phi_u * phi_v
    
    # Transform density: divide by Jacobian (copula → probit)
    # Add small epsilon to avoid division by zero
    c_gaussian = c_copula / (jacobian.clamp(min=eps))
    
    return c_gaussian


def probit_density_to_copula_density(
    c_gaussian: torch.Tensor,
    m: int,
    eps: float = EPS
) -> torch.Tensor:
    """Transform probit/Gaussian space density back to copula density.
    
    Given Gaussian copula density c_G(z_u, z_v), compute the standard
    copula density c(u,v) on [0,1]².
    
    Transformation:
        c_copula(u,v) = c_G(Φ⁻¹(u), Φ⁻¹(v)) · φ(Φ⁻¹(u)) · φ(Φ⁻¹(v))
        
    Args:
        c_gaussian: Density in probit space, shape (..., m, m)
        m: Grid resolution
        eps: Clamping epsilon
        
    Returns:
        c_copula: Copula density, shape (..., m, m)
    """
    device = c_gaussian.device
    
    # Create copula grid coordinates
    u_grid = torch.linspace(0.5/m, 1 - 0.5/m, m, device=device)
    v_grid = torch.linspace(0.5/m, 1 - 0.5/m, m, device=device)
    U, V = torch.meshgrid(u_grid, v_grid, indexing='ij')
    
    # Transform to probit coordinates
    Z_u = copula_to_probit_coordinates(U, eps)
    Z_v = copula_to_probit_coordinates(V, eps)
    
    # Compute Jacobian: product of marginal PDFs
    phi_u = standard_normal_pdf(Z_u)
    phi_v = standard_normal_pdf(Z_v)
    jacobian = phi_u * phi_v
    
    # Transform density: multiply by Jacobian (probit → copula)
    c_copula = c_gaussian * jacobian
    
    return c_copula


def get_probit_grid_spacing(m: int, quantile_range: float = 0.9999) -> tuple:
    """Compute equivalent grid spacing in probit space.
    
    When discretizing [-∞,∞] for a Gaussian copula, we typically cover
    a finite range like [Φ⁻¹(ε), Φ⁻¹(1-ε)]. This function computes the
    spacing for numerical integration.
    
    Args:
        m: Number of grid points (same as copula grid)
        quantile_range: Fraction of probability mass to cover (e.g., 0.9999)
        
    Returns:
        (z_min, z_max, dz): Min, max probit values and spacing
    """
    eps = (1 - quantile_range) / 2
    u_min = eps
    u_max = 1 - eps
    
    z_min = copula_to_probit_coordinates(torch.tensor(u_min)).item()
    z_max = copula_to_probit_coordinates(torch.tensor(u_max)).item()
    dz = (z_max - z_min) / m
    
    return z_min, z_max, dz


def copula_logdensity_to_probit_logdensity(
    log_c_copula: torch.Tensor,
    m: int,
    eps: float = EPS
) -> torch.Tensor:
    """Transform copula log-density to probit/Gaussian space log-density.
    
    This is the numerically stable version using log-space arithmetic.
    
    Transformation:
        log c_G(z_u, z_v) = log c_copula(Φ(z_u), Φ(z_v)) - log φ(z_u) - log φ(z_v)
        
    Args:
        log_c_copula: Log copula density tensor, shape (..., m, m)
        m: Grid resolution
        eps: Clamping epsilon (unused in log space, kept for API consistency)
        
    Returns:
        log_c_gaussian: Log-density in probit space, shape (..., m, m)
    """
    device = log_c_copula.device
    
    # Create copula grid coordinates
    u_grid = torch.linspace(0.5/m, 1 - 0.5/m, m, device=device)
    v_grid = torch.linspace(0.5/m, 1 - 0.5/m, m, device=device)
    U, V = torch.meshgrid(u_grid, v_grid, indexing='ij')
    
    # Transform to probit coordinates
    Z_u = copula_to_probit_coordinates(U, eps)
    Z_v = copula_to_probit_coordinates(V, eps)
    
    # Compute log-Jacobian: sum of log marginal PDFs
    log_phi_u = standard_normal_logpdf(Z_u)
    log_phi_v = standard_normal_logpdf(Z_v)
    log_jacobian = log_phi_u + log_phi_v
    
    # Transform log-density: subtract log-Jacobian (copula → probit)
    log_c_gaussian = log_c_copula - log_jacobian
    
    return log_c_gaussian


def probit_logdensity_to_copula_logdensity(
    log_c_gaussian: torch.Tensor,
    m: int,
    eps: float = EPS
) -> torch.Tensor:
    """Transform probit/Gaussian space log-density back to copula log-density.
    
    Numerically stable log-space version.
    
    Transformation:
        log c_copula(u,v) = log c_G(Φ⁻¹(u), Φ⁻¹(v)) + log φ(Φ⁻¹(u)) + log φ(Φ⁻¹(v))
        
    Args:
        log_c_gaussian: Log-density in probit space, shape (..., m, m)
        m: Grid resolution
        eps: Clamping epsilon (unused in log space)
        
    Returns:
        log_c_copula: Log copula density, shape (..., m, m)
    """
    device = log_c_gaussian.device
    
    # Create copula grid coordinates
    u_grid = torch.linspace(0.5/m, 1 - 0.5/m, m, device=device)
    v_grid = torch.linspace(0.5/m, 1 - 0.5/m, m, device=device)
    U, V = torch.meshgrid(u_grid, v_grid, indexing='ij')
    
    # Transform to probit coordinates
    Z_u = copula_to_probit_coordinates(U, eps)
    Z_v = copula_to_probit_coordinates(V, eps)
    
    # Compute log-Jacobian: sum of log marginal PDFs
    log_phi_u = standard_normal_logpdf(Z_u)
    log_phi_v = standard_normal_logpdf(Z_v)
    log_jacobian = log_phi_u + log_phi_v
    
    # Transform log-density: add log-Jacobian (probit → copula)
    log_c_copula = log_c_gaussian + log_jacobian
    
    return log_c_copula


def verify_transformation(c_copula: torch.Tensor, m: int, tol: float = 1e-3) -> dict:
    """Verify that forward and inverse transformations are consistent.
    
    Args:
        c_copula: Original copula density
        m: Grid resolution
        tol: Tolerance for relative error
        
    Returns:
        Dictionary with verification metrics
    """
    # Forward transform
    c_gaussian = copula_density_to_probit_density(c_copula, m)
    
    # Inverse transform
    c_copula_reconstructed = probit_density_to_copula_density(c_gaussian, m)
    
    # Compute error
    rel_error = torch.abs(c_copula - c_copula_reconstructed) / (c_copula.abs() + 1e-8)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()
    
    return {
        'max_relative_error': max_rel_error,
        'mean_relative_error': mean_rel_error,
        'passed': max_rel_error < tol,
    }


if __name__ == '__main__':
    # Quick test
    print("Testing probit transformations...")
    
    m = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a simple test copula (independence copula = uniform density)
    c_copula = torch.ones(1, 1, m, m, device=device)
    
    # Transform to probit space
    c_gaussian = copula_density_to_probit_density(c_copula, m)
    print(f"Copula density range: [{c_copula.min():.4f}, {c_copula.max():.4f}]")
    print(f"Gaussian density range: [{c_gaussian.min():.4f}, {c_gaussian.max():.4f}]")
    
    # Transform back
    c_reconstructed = probit_density_to_copula_density(c_gaussian, m)
    print(f"Reconstructed density range: [{c_reconstructed.min():.4f}, {c_reconstructed.max():.4f}]")
    
    # Verify
    results = verify_transformation(c_copula, m)
    print(f"Verification: {results}")
    print(f"Max relative error: {results['max_relative_error']:.6f}")
    print(f"Test {'PASSED' if results['passed'] else 'FAILED'}")
