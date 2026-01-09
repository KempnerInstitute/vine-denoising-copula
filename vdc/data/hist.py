"""
Histogram utilities for converting scatter data to grids.
"""

import numpy as np
from typing import Optional, Tuple
from scipy import stats


def scatter_to_hist(
    points: np.ndarray,
    m: int = 64,
    reflect: bool = True,
    smooth_sigma: Optional[float] = None,
    probit_smooth: bool = True,
) -> np.ndarray:
    """
    Convert 2D scatter of pseudo-observations to a normalized histogram.
    
    Args:
        points: Array of shape (n, 2) with values in (0, 1)
        m: Grid resolution (m x m)
        reflect: If True, use reflection padding at boundaries (ignored if probit_smooth=True)
        smooth_sigma: If provided, apply Gaussian smoothing with this sigma
        probit_smooth: If True, use probit transformation for boundary-aware smoothing
        
    Returns:
        Array of shape (m, m) with sum = 1.0 (probability mass)
    """
    n = points.shape[0]
    
    # Use probit transformation for boundary-aware smoothing (preferred method)
    if smooth_sigma is not None and probit_smooth:
        # Method from paper: Transform to probit space, smooth, transform back
        return _probit_smooth_hist(points, m, smooth_sigma)
    
    # Fallback: Traditional reflection method
    if reflect:
        # Reflect points at all four boundaries
        reflected_points = [points]
        
        # Reflect across left/right (u boundaries)
        reflected_points.append(np.stack([-points[:, 0], points[:, 1]], axis=1))
        reflected_points.append(np.stack([2 - points[:, 0], points[:, 1]], axis=1))
        
        # Reflect across top/bottom (v boundaries)
        reflected_points.append(np.stack([points[:, 0], -points[:, 1]], axis=1))
        reflected_points.append(np.stack([points[:, 0], 2 - points[:, 1]], axis=1))
        
        # Corners
        reflected_points.append(np.stack([-points[:, 0], -points[:, 1]], axis=1))
        reflected_points.append(np.stack([2 - points[:, 0], -points[:, 1]], axis=1))
        reflected_points.append(np.stack([-points[:, 0], 2 - points[:, 1]], axis=1))
        reflected_points.append(np.stack([2 - points[:, 0], 2 - points[:, 1]], axis=1))
        
        all_points = np.concatenate(reflected_points, axis=0)
    else:
        all_points = points
    
    # Create histogram
    hist, _, _ = np.histogram2d(
        all_points[:, 0],
        all_points[:, 1],
        bins=m,
        range=[[0, 1], [0, 1]]
    )
    
    # Normalize to probability mass (sum = 1)
    hist = hist / hist.sum()
    
    # Optional smoothing (non-probit)
    if smooth_sigma is not None and not probit_smooth:
        from scipy.ndimage import gaussian_filter
        hist = gaussian_filter(hist, sigma=smooth_sigma)
        hist = hist / hist.sum()  # Re-normalize after smoothing
    
    return hist


def _probit_smooth_hist(
    points: np.ndarray,
    m: int,
    smooth_sigma: float,
) -> np.ndarray:
    """
    Probit-based smoothing for copula histograms.
    
    This method handles boundaries correctly by:
    1. Transforming copula data (U in [0,1]) to probit space (Z ~ N(0,1))
    2. Creating histogram and smoothing in probit space (no boundaries)
    3. Transforming back to copula space
    
    This avoids boundary artifacts that occur with direct smoothing.
    
    Args:
        points: Array of shape (n, 2) with values in (0, 1)
        m: Grid resolution (m x m)
        smooth_sigma: Gaussian smoothing sigma in probit space
        
    Returns:
        Array of shape (m, m) with sum = 1.0 (probability mass)
    """
    from scipy.ndimage import gaussian_filter
    
    # Clip points to avoid infinite probit values at boundaries
    eps = 1e-6
    points_clipped = np.clip(points, eps, 1 - eps)
    
    # Transform to probit space: u -> Φ^(-1)(u) where Φ is standard normal CDF
    points_probit = stats.norm.ppf(points_clipped)  # (n, 2)
    
    # Determine probit space range (use robust range based on data)
    z_min = np.percentile(points_probit, 0.1, axis=0)
    z_max = np.percentile(points_probit, 99.9, axis=0)
    
    # Expand range slightly for safety
    z_range = z_max - z_min
    z_min -= 0.1 * z_range
    z_max += 0.1 * z_range
    
    # Create histogram in probit space
    hist_probit, z_edges_u, z_edges_v = np.histogram2d(
        points_probit[:, 0],
        points_probit[:, 1],
        bins=m,
        range=[[z_min[0], z_max[0]], [z_min[1], z_max[1]]]
    )
    
    # Normalize
    hist_probit = hist_probit / hist_probit.sum()
    
    # Smooth in probit space (no boundary issues!)
    hist_probit_smooth = gaussian_filter(hist_probit, sigma=smooth_sigma)
    hist_probit_smooth = hist_probit_smooth / hist_probit_smooth.sum()
    
    # Create grid centers in probit space
    z_centers_u = (z_edges_u[:-1] + z_edges_u[1:]) / 2
    z_centers_v = (z_edges_v[:-1] + z_edges_v[1:]) / 2
    
    # Transform grid back to copula space
    # Grid in probit space
    Z_U, Z_V = np.meshgrid(z_centers_u, z_centers_v, indexing='ij')
    
    # Transform to copula space: z -> Φ(z)
    U_grid = stats.norm.cdf(Z_U)  # (m, m)
    V_grid = stats.norm.cdf(Z_V)  # (m, m)
    
    # Now we need to map the smoothed probit histogram back to uniform copula grid
    # Use interpolation to map from irregular (U_grid, V_grid) to uniform [0,1] grid
    
    # Create uniform copula grid
    u_uniform = np.linspace(0, 1, m)
    v_uniform = np.linspace(0, 1, m)
    U_uniform, V_uniform = np.meshgrid(u_uniform, v_uniform, indexing='ij')
    
    # Interpolate: from (U_grid, V_grid) -> uniform grid
    from scipy.interpolate import griddata
    
    # Flatten for interpolation
    points_transformed = np.column_stack([U_grid.ravel(), V_grid.ravel()])
    values_smooth = hist_probit_smooth.ravel()
    points_query = np.column_stack([U_uniform.ravel(), V_uniform.ravel()])
    
    # Interpolate to uniform grid
    hist_copula = griddata(
        points_transformed,
        values_smooth,
        points_query,
        method='cubic',
        fill_value=0.0
    ).reshape(m, m)
    
    # Ensure non-negative and normalize
    hist_copula = np.maximum(hist_copula, 0)
    hist_copula = hist_copula / hist_copula.sum()
    
    return hist_copula


def create_tail_biased_grid(m: int, tail_density: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a non-uniform grid with more density near 0 and 1 (tail regions).
    
    Args:
        m: Number of grid points
        tail_density: How much to bias towards tails (>1 = more dense at tails)
        
    Returns:
        Tuple of (u_grid, v_grid), each of shape (m,)
    """
    # Use beta distribution to create biased spacing
    from scipy.stats import beta
    
    # Create symmetric biasing: dense at 0 and 1
    alpha = 1.0 / tail_density
    t = np.linspace(0, 1, m)
    
    # Apply beta CDF to create non-uniform spacing
    grid = beta.cdf(t, alpha, alpha)
    
    return grid, grid


def hist_to_density_grid(hist: np.ndarray, du: float = None, dv: float = None) -> np.ndarray:
    """
    Convert histogram (cell masses) to density grid.
    
    Args:
        hist: Histogram array of shape (m, m) with sum = 1
        du, dv: Grid spacings. If None, assumes uniform grid with du = dv = 1/m
        
    Returns:
        Density grid of shape (m, m)
    """
    m = hist.shape[0]
    if du is None:
        du = 1.0 / m
    if dv is None:
        dv = 1.0 / m
    
    density = hist / (du * dv)
    return density


def density_grid_to_hist(density: np.ndarray, du: float = None, dv: float = None) -> np.ndarray:
    """
    Convert density grid to histogram (cell masses).
    
    Args:
        density: Density grid of shape (m, m)
        du, dv: Grid spacings. If None, assumes uniform grid with du = dv = 1/m
        
    Returns:
        Histogram array of shape (m, m) with sum ≈ 1
    """
    m = density.shape[0]
    if du is None:
        du = 1.0 / m
    if dv is None:
        dv = 1.0 / m
    
    hist = density * du * dv
    return hist


def visualize_histogram(hist: np.ndarray, points: Optional[np.ndarray] = None, save_path: Optional[str] = None):
    """
    Visualize a histogram with optional scatter overlay.
    
    Args:
        hist: Histogram array of shape (m, m)
        points: Optional scatter points of shape (n, 2)
        save_path: If provided, save figure to this path
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2 if points is not None else 1, figsize=(12 if points is not None else 6, 5))
    
    if points is not None:
        ax1, ax2 = axes
    else:
        ax1 = axes
        ax2 = None
    
    # Plot histogram
    im = ax1.imshow(hist.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis', aspect='auto')
    ax1.set_xlabel('u')
    ax1.set_ylabel('v')
    ax1.set_title('Histogram')
    plt.colorbar(im, ax=ax1)
    
    # Plot scatter if provided
    if points is not None and ax2 is not None:
        ax2.scatter(points[:, 0], points[:, 1], alpha=0.3, s=1)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('u')
        ax2.set_ylabel('v')
        ax2.set_title('Scatter')
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Test histogram creation
    np.random.seed(42)
    
    # Generate some correlated uniform data
    from scipy.stats import multivariate_normal
    mean = [0, 0]
    cov = [[1, 0.7], [0.7, 1]]
    samples = multivariate_normal.rvs(mean, cov, size=1000)
    
    # Convert to pseudo-observations
    from scipy.stats import norm
    u = norm.cdf(samples[:, 0])
    v = norm.cdf(samples[:, 1])
    points = np.stack([u, v], axis=1)
    
    # Create histogram
    hist = scatter_to_hist(points, m=64, reflect=True)
    
    print(f"Histogram shape: {hist.shape}")
    print(f"Histogram sum: {hist.sum():.6f}")
    print(f"Min/max: {hist.min():.6e}, {hist.max():.6e}")
    
    # Visualize
    visualize_histogram(hist, points, save_path="test_histogram.png")
    print("Saved visualization to test_histogram.png")


# Alias for backward compatibility
points_to_histogram = scatter_to_hist
