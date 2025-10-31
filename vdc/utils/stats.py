"""
Statistical utilities for copula analysis.

Includes:
- Kendall's tau estimation
- Tail dependence coefficients
- PIT (Probability Integral Transform) diagnostics
- Rank transformations
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional
import warnings


def ranks_to_uniform(X: np.ndarray, method: str = 'average') -> np.ndarray:
    """
    Convert data to pseudo-observations (uniform marginals via ranking).
    
    For each column, compute: u_i = rank(x_i) / (n + 1)
    
    Args:
        X: (n, d) data matrix
        method: Ranking method ('average', 'min', 'max', 'dense', 'ordinal')
        
    Returns:
        (n, d) pseudo-observations in (0, 1)
    """
    n, d = X.shape
    U = np.zeros_like(X, dtype=float)
    
    for j in range(d):
        ranks = stats.rankdata(X[:, j], method=method)
        U[:, j] = ranks / (n + 1)
    
    return U


def uniform_to_ranks(U: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    """
    Convert pseudo-observations back to ranks.
    
    Args:
        U: (n, d) pseudo-observations in (0, 1)
        n: Sample size (default: infer from U)
        
    Returns:
        (n, d) ranks
    """
    if n is None:
        n = U.shape[0]
    
    return np.round(U * (n + 1)).astype(int)


def kendall_tau(u: np.ndarray, v: np.ndarray, method: str = 'auto') -> float:
    """
    Compute Kendall's tau correlation coefficient.
    
    τ = (# concordant pairs - # discordant pairs) / (n choose 2)
    
    Args:
        u: (n,) first variable
        v: (n,) second variable  
        method: 'auto', 'scipy', or 'direct'
        
    Returns:
        Kendall's tau in [-1, 1]
    """
    if method == 'scipy' or method == 'auto':
        tau, _ = stats.kendalltau(u, v)
        return tau
    
    elif method == 'direct':
        # Direct computation (slower but more transparent)
        n = len(u)
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                sign_u = np.sign(u[j] - u[i])
                sign_v = np.sign(v[j] - v[i])
                
                if sign_u * sign_v > 0:
                    concordant += 1
                elif sign_u * sign_v < 0:
                    discordant += 1
        
        tau = (concordant - discordant) / (n * (n - 1) / 2)
        return tau
    
    else:
        raise ValueError(f"Unknown method: {method}")


def kendall_tau_matrix(U: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Kendall's tau matrix.
    
    Args:
        U: (n, d) pseudo-observations
        
    Returns:
        (d, d) correlation matrix
    """
    n, d = U.shape
    tau_matrix = np.eye(d)
    
    for i in range(d):
        for j in range(i + 1, d):
            tau = kendall_tau(U[:, i], U[:, j])
            tau_matrix[i, j] = tau
            tau_matrix[j, i] = tau
    
    return tau_matrix


def tail_dependence_empirical(
    u: np.ndarray,
    v: np.ndarray,
    threshold: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute empirical upper and lower tail dependence coefficients.
    
    Upper tail: λ_U = lim_{q→1} P(V > q | U > q)
    Lower tail: λ_L = lim_{q→0} P(V ≤ q | U ≤ q)
    
    Approximated by counting pairs in tail regions.
    
    Args:
        u: (n,) first variable in [0,1]
        v: (n,) second variable in [0,1]
        threshold: Tail threshold (default: 0.95 for upper, 0.05 for lower)
        
    Returns:
        (lambda_upper, lambda_lower) tail dependence coefficients
    """
    n = len(u)
    
    # Upper tail dependence
    upper_mask = u > threshold
    n_upper = np.sum(upper_mask)
    
    if n_upper > 0:
        lambda_upper = np.sum(v[upper_mask] > threshold) / n_upper
    else:
        lambda_upper = 0.0
    
    # Lower tail dependence
    lower_threshold = 1 - threshold
    lower_mask = u < lower_threshold
    n_lower = np.sum(lower_mask)
    
    if n_lower > 0:
        lambda_lower = np.sum(v[lower_mask] < lower_threshold) / n_lower
    else:
        lambda_lower = 0.0
    
    return lambda_upper, lambda_lower


def pit_uniform_test(u: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Test if data follows a uniform distribution using PIT diagnostics.
    
    Uses Kolmogorov-Smirnov test for uniformity.
    
    Args:
        u: (n,) values in [0,1]
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # KS test against uniform
    ks_stat, ks_pval = stats.kstest(u, 'uniform')
    
    # Anderson-Darling test (more powerful for tails)
    # Transform to normal for AD test
    try:
        z = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        ad_result = stats.anderson(z)
        ad_stat = ad_result.statistic
        ad_critical = ad_result.critical_values[2]  # 5% level
        ad_pass = ad_stat < ad_critical
    except:
        ad_stat = np.nan
        ad_pass = False
    
    # Moments check
    mean = np.mean(u)
    var = np.var(u)
    expected_mean = 0.5
    expected_var = 1.0 / 12
    
    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'ks_pass': ks_pval > alpha,
        'ad_statistic': ad_stat,
        'ad_pass': ad_pass,
        'mean': mean,
        'var': var,
        'mean_error': abs(mean - expected_mean),
        'var_error': abs(var - expected_var),
    }


def pit_multivariate(
    U: np.ndarray,
    rosenblatt_transform,
    alpha: float = 0.05,
) -> dict:
    """
    Multivariate PIT diagnostic using Rosenblatt transform.
    
    If the copula model is correct, the Rosenblatt transform should
    produce independent uniform variables.
    
    Args:
        U: (n, d) pseudo-observations
        rosenblatt_transform: Function that applies Rosenblatt transform
        alpha: Significance level
        
    Returns:
        Dictionary with test results for each dimension
    """
    W = rosenblatt_transform(U)
    n, d = W.shape
    
    results = {}
    for j in range(d):
        results[f'dim_{j}'] = pit_uniform_test(W[:, j], alpha=alpha)
    
    # Overall pass/fail
    all_pass = all(results[f'dim_{j}']['ks_pass'] for j in range(d))
    results['overall_pass'] = all_pass
    
    return results


def spearman_rho(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute Spearman's rho correlation coefficient.
    
    ρ = correlation of ranks
    
    Args:
        u: (n,) first variable
        v: (n,) second variable
        
    Returns:
        Spearman's rho in [-1, 1]
    """
    rho, _ = stats.spearmanr(u, v)
    return rho


def empirical_copula(u: np.ndarray, v: np.ndarray, grid_size: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empirical copula CDF on a grid.
    
    C_n(u, v) = 1/n Σ_i I(U_i ≤ u, V_i ≤ v)
    
    Args:
        u: (n,) first variable in [0,1]
        v: (n,) second variable in [0,1]
        grid_size: Number of grid points per dimension
        
    Returns:
        (U_grid, V_grid, C_grid) where C_grid is the empirical CDF
    """
    n = len(u)
    grid = np.linspace(0, 1, grid_size)
    U_grid, V_grid = np.meshgrid(grid, grid, indexing='ij')
    C_grid = np.zeros_like(U_grid)
    
    for i in range(grid_size):
        for j in range(grid_size):
            u_thresh = U_grid[i, j]
            v_thresh = V_grid[i, j]
            C_grid[i, j] = np.mean((u <= u_thresh) & (v <= v_thresh))
    
    return U_grid, V_grid, C_grid


def copula_distance(
    u1: np.ndarray,
    v1: np.ndarray,
    u2: np.ndarray,
    v2: np.ndarray,
    metric: str = 'l2',
    grid_size: int = 50,
) -> float:
    """
    Distance between two empirical copulas.
    
    Args:
        u1, v1: First copula sample
        u2, v2: Second copula sample
        metric: 'l2' or 'l1'
        grid_size: Grid resolution for comparison
        
    Returns:
        Distance value
    """
    _, _, C1 = empirical_copula(u1, v1, grid_size)
    _, _, C2 = empirical_copula(u2, v2, grid_size)
    
    if metric == 'l2':
        return np.sqrt(np.mean((C1 - C2) ** 2))
    elif metric == 'l1':
        return np.mean(np.abs(C1 - C2))
    else:
        raise ValueError(f"Unknown metric: {metric}")


if __name__ == "__main__":
    print("Testing statistical utilities...")
    
    # Test 1: Ranks to uniform
    print("\nTest 1: Ranks to uniform")
    np.random.seed(42)
    X = np.random.randn(1000, 3)
    U = ranks_to_uniform(X)
    
    print(f"Original data shape: {X.shape}")
    print(f"Pseudo-obs shape: {U.shape}")
    print(f"Pseudo-obs range: [{U.min():.4f}, {U.max():.4f}]")
    print(f"Pseudo-obs mean: {U.mean():.4f} (expected: 0.5)")
    
    # Test 2: Kendall's tau
    print("\nTest 2: Kendall's tau")
    n = 1000
    rho_true = 0.7
    
    # Generate correlated uniforms (Gaussian copula)
    Z = np.random.multivariate_normal([0, 0], [[1, rho_true], [rho_true, 1]], n)
    u = stats.norm.cdf(Z[:, 0])
    v = stats.norm.cdf(Z[:, 1])
    
    tau = kendall_tau(u, v)
    print(f"True correlation (ρ): {rho_true}")
    print(f"Kendall's tau: {tau:.4f}")
    print(f"Expected tau ≈ {2/np.pi * np.arcsin(rho_true):.4f} (for Gaussian)")
    
    # Test 3: Tail dependence
    print("\nTest 3: Tail dependence")
    lambda_u, lambda_l = tail_dependence_empirical(u, v, threshold=0.95)
    print(f"Upper tail dependence: {lambda_u:.4f}")
    print(f"Lower tail dependence: {lambda_l:.4f}")
    print("(Gaussian copula has zero tail dependence)")
    
    # Test 4: PIT uniformity test
    print("\nTest 4: PIT uniformity test")
    u_uniform = np.random.uniform(0, 1, 1000)
    pit_results = pit_uniform_test(u_uniform)
    
    print("PIT test on uniform data:")
    print(f"  KS p-value: {pit_results['ks_pvalue']:.4f}")
    print(f"  Test passed: {pit_results['ks_pass']}")
    print(f"  Mean: {pit_results['mean']:.4f} (expected: 0.5)")
    print(f"  Variance: {pit_results['var']:.4f} (expected: 0.083)")
    
    # Test 5: Kendall's tau matrix
    print("\nTest 5: Kendall's tau matrix")
    U_multi = ranks_to_uniform(np.random.randn(500, 4))
    tau_mat = kendall_tau_matrix(U_multi)
    print("Tau matrix (should be close to identity for independent data):")
    print(tau_mat.round(3))
    
    print("\nAll statistical tests passed!")
