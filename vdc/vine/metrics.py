"""
Evaluation metrics for vine copulas.

Includes:
- Joint log-likelihood
- PIT diagnostics
- Tail dependence
- Runtime benchmarks
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import kstest
import time


def joint_loglik(U: np.ndarray, vine_logpdf_func) -> float:
    """
    Compute average joint log-likelihood.
    
    Args:
        U: (n, d) pseudo-observations
        vine_logpdf_func: Function that computes log-density
        
    Returns:
        Average log-likelihood
    """
    loglik = vine_logpdf_func(U)
    return np.mean(loglik)


def probability_integral_transform(
    U: np.ndarray,
    vine_rosenblatt_func
) -> np.ndarray:
    """
    Apply Rosenblatt transform to get independent uniforms.
    
    Args:
        U: (n, d) pseudo-observations
        vine_rosenblatt_func: Forward Rosenblatt transform function
        
    Returns:
        W: (n, d) transformed uniforms (should be independent)
    """
    W = vine_rosenblatt_func(U)
    return W


def pit_diagnostics(W: np.ndarray) -> Dict[str, float]:
    """
    Perform PIT diagnostics on Rosenblatt-transformed data.
    
    Tests if each marginal is uniform [0,1] and if marginals are independent.
    
    Args:
        W: (n, d) Rosenblatt-transformed data
        
    Returns:
        Dictionary of diagnostic statistics
    """
    n, d = W.shape
    
    # Test uniformity of each marginal with KS test
    ks_pvalues = []
    for j in range(d):
        stat, pval = kstest(W[:, j], 'uniform')
        ks_pvalues.append(pval)
    
    # Average KS p-value
    avg_ks_pval = np.mean(ks_pvalues)
    min_ks_pval = np.min(ks_pvalues)
    
    # Test independence via correlation
    # Should be near zero if independent
    corr_matrix = np.corrcoef(W.T)
    off_diag = corr_matrix[np.triu_indices(d, k=1)]
    avg_abs_corr = np.mean(np.abs(off_diag))
    max_abs_corr = np.max(np.abs(off_diag))
    
    return {
        'avg_ks_pvalue': avg_ks_pval,
        'min_ks_pvalue': min_ks_pval,
        'avg_abs_correlation': avg_abs_corr,
        'max_abs_correlation': max_abs_corr,
    }


def tail_dependence_empirical(
    U: np.ndarray,
    threshold: float = 0.05
) -> Dict[str, float]:
    """
    Estimate empirical tail dependence coefficients.
    
    Upper tail: λ_U = lim_{u→1} P(U_2 > u | U_1 > u)
    Lower tail: λ_L = lim_{u→0} P(U_2 < u | U_1 < u)
    
    Args:
        U: (n, 2) bivariate pseudo-observations
        threshold: Quantile threshold for tail estimation
        
    Returns:
        Dictionary with upper and lower tail dependence
    """
    if U.shape[1] != 2:
        raise ValueError("Tail dependence only defined for bivariate data")
    
    n = len(U)
    u1, u2 = U[:, 0], U[:, 1]
    
    # Upper tail
    upper_threshold = 1.0 - threshold
    upper_mask = u1 > upper_threshold
    if np.sum(upper_mask) > 0:
        lambda_u = np.mean(u2[upper_mask] > upper_threshold)
    else:
        lambda_u = 0.0
    
    # Lower tail
    lower_threshold = threshold
    lower_mask = u1 < lower_threshold
    if np.sum(lower_mask) > 0:
        lambda_l = np.mean(u2[lower_mask] < lower_threshold)
    else:
        lambda_l = 0.0
    
    return {
        'lambda_upper': lambda_u,
        'lambda_lower': lambda_l,
    }


def compute_kendall_tau(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute Kendall's tau rank correlation.
    
    Args:
        u, v: (n,) arrays
        
    Returns:
        Kendall's tau
    """
    from scipy.stats import kendalltau
    tau, _ = kendalltau(u, v, nan_policy='omit')
    if np.isnan(tau):
        return 0.0
    return tau


def compute_spearman_rho(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute Spearman's rho rank correlation.
    
    Args:
        u, v: (n,) arrays
        
    Returns:
        Spearman's rho
    """
    from scipy.stats import spearmanr
    rho, _ = spearmanr(u, v, nan_policy='omit')
    if np.isnan(rho):
        return 0.0
    return rho


def benchmark_runtime(
    func,
    *args,
    n_runs: int = 10,
    warmup: int = 2,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmark function runtime.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments
        n_runs: Number of runs
        warmup: Number of warmup runs
        **kwargs: Keyword arguments
        
    Returns:
        Runtime statistics
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        times.append(end - start)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
    }


def copula_distance_metrics(
    density_pred: np.ndarray,
    density_true: np.ndarray
) -> Dict[str, float]:
    """
    Compute distance metrics between predicted and true densities.
    
    Args:
        density_pred: (m, m) predicted density grid
        density_true: (m, m) true density grid
        
    Returns:
        Dictionary of distance metrics
    """
    # L1 distance
    l1_dist = np.mean(np.abs(density_pred - density_true))
    
    # L2 distance
    l2_dist = np.sqrt(np.mean((density_pred - density_true) ** 2))
    
    # KL divergence (pred || true)
    # Avoid log(0) issues
    eps = 1e-10
    p = np.clip(density_pred.flatten(), eps, None)
    q = np.clip(density_true.flatten(), eps, None)
    
    # Normalize to probabilities
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    kl_div = np.sum(p * np.log(p / q))
    
    # Reverse KL divergence (true || pred)
    kl_div_rev = np.sum(q * np.log(q / p))
    
    # Hellinger distance
    hellinger = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
    
    return {
        'l1_distance': l1_dist,
        'l2_distance': l2_dist,
        'kl_divergence': kl_div,
        'kl_divergence_reverse': kl_div_rev,
        'hellinger_distance': hellinger,
    }


def evaluate_vine_copula(
    U_test: np.ndarray,
    vine_model,
    U_train: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a vine copula model.
    
    Args:
        U_test: Test pseudo-observations
        vine_model: Fitted vine model with logpdf, rosenblatt, simulate methods
        U_train: Training data (optional, for comparison)
        verbose: Print results
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # 1. Joint log-likelihood
    if verbose:
        print("Computing joint log-likelihood...")
    loglik = joint_loglik(U_test, vine_model.logpdf)
    metrics['joint_loglik'] = loglik
    
    # 2. PIT diagnostics
    if verbose:
        print("Computing PIT diagnostics...")
    W = probability_integral_transform(U_test, vine_model.rosenblatt)
    pit_stats = pit_diagnostics(W)
    metrics.update(pit_stats)
    
    # 3. Pairwise dependence preservation
    if verbose:
        print("Computing pairwise dependence...")
    d = U_test.shape[1]
    tau_errors = []
    for i in range(d):
        for j in range(i+1, d):
            tau_true = compute_kendall_tau(U_test[:, i], U_test[:, j])
            
            # Sample from model
            U_sim = vine_model.simulate(len(U_test))
            tau_sim = compute_kendall_tau(U_sim[:, i], U_sim[:, j])
            
            tau_errors.append(abs(tau_true - tau_sim))
    
    metrics['mean_tau_error'] = np.mean(tau_errors)
    metrics['max_tau_error'] = np.max(tau_errors)
    
    # 4. Runtime benchmarks
    if verbose:
        print("Benchmarking runtime...")
    
    # Density evaluation
    logpdf_bench = benchmark_runtime(vine_model.logpdf, U_test[:100], n_runs=5)
    metrics['logpdf_time_per_sample'] = logpdf_bench['mean_time'] / 100
    
    # Sampling
    sample_bench = benchmark_runtime(vine_model.simulate, 100, n_runs=5)
    metrics['sample_time_per_sample'] = sample_bench['mean_time'] / 100
    
    # Print summary
    if verbose:
        print("\n" + "="*60)
        print("Vine Copula Evaluation Results")
        print("="*60)
        print(f"Joint Log-Likelihood: {metrics['joint_loglik']:.4f}")
        print(f"\nPIT Diagnostics:")
        print(f"  Avg KS p-value: {metrics['avg_ks_pvalue']:.4f}")
        print(f"  Min KS p-value: {metrics['min_ks_pvalue']:.4f}")
        print(f"  Avg |correlation|: {metrics['avg_abs_correlation']:.4f}")
        print(f"  Max |correlation|: {metrics['max_abs_correlation']:.4f}")
        print(f"\nDependence Preservation:")
        print(f"  Mean τ error: {metrics['mean_tau_error']:.4f}")
        print(f"  Max τ error: {metrics['max_tau_error']:.4f}")
        print(f"\nRuntime:")
        print(f"  Log-density: {metrics['logpdf_time_per_sample']*1000:.2f} ms/sample")
        print(f"  Sampling: {metrics['sample_time_per_sample']*1000:.2f} ms/sample")
        print("="*60 + "\n")
    
    return metrics


if __name__ == "__main__":
    print("Testing evaluation metrics...")
    
    # Generate synthetic data
    np.random.seed(42)
    n, d = 1000, 5
    
    # Create correlated uniforms
    from scipy.stats import norm
    rho = 0.5
    Sigma = np.eye(d)
    for i in range(d-1):
        Sigma[i, i+1] = rho
        Sigma[i+1, i] = rho
    
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    U = norm.cdf(Z)
    
    print(f"Generated {n} samples in {d} dimensions")
    
    # Test PIT on independent uniforms (should pass)
    W_indep = np.random.uniform(0, 1, (n, d))
    pit_stats = pit_diagnostics(W_indep)
    print(f"\nPIT diagnostics on independent uniforms:")
    print(f"  Avg KS p-value: {pit_stats['avg_ks_pvalue']:.4f}")
    print(f"  Avg |correlation|: {pit_stats['avg_abs_correlation']:.4f}")
    
    # Test tail dependence
    U_pair = U[:, :2]
    tail_dep = tail_dependence_empirical(U_pair)
    print(f"\nTail dependence (bivariate):")
    print(f"  Upper: {tail_dep['lambda_upper']:.4f}")
    print(f"  Lower: {tail_dep['lambda_lower']:.4f}")
    
    # Test Kendall's tau
    tau = compute_kendall_tau(U[:, 0], U[:, 1])
    print(f"\nKendall's tau: {tau:.4f}")
    
    print("\nEvaluation metrics test completed!")
