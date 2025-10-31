"""
Per-pair copula evaluation.

Evaluates individual bivariate copula estimations.
"""

import numpy as np
import torch
from typing import Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path

from ..models.projection import copula_project
from ..losses.nll import nll_points
from ..losses.ise import ise_logdensity
from .metrics import (
    tail_dependence_empirical,
    compute_kendall_tau,
    copula_distance_metrics
)


def evaluate_pair_copula(
    model,
    hist: torch.Tensor,
    points: np.ndarray,
    true_density: Optional[np.ndarray] = None,
    device: str = 'cuda',
    m: int = 64
) -> Dict[str, float]:
    """
    Evaluate a single pair-copula estimation.
    
    Args:
        model: Trained diffusion model
        hist: (1, m, m) histogram input
        points: (n, 2) test points
        true_density: (m, m) true density (if known)
        device: Device
        m: Grid resolution
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Predict density
        hist = hist.to(device)
        t = torch.ones(1, 1, 1, 1, device=device) * 0.5
        
        logD_raw = model(hist, t)
        D_pos = torch.exp(logD_raw)
        D_hat = copula_project(D_pos)
        
        # Convert to numpy
        density_pred = D_hat[0, 0].cpu().numpy()
    
    metrics = {}
    
    # 1. NLL on test points
    points_t = torch.from_numpy(points).float().unsqueeze(0).to(device)
    nll = nll_points(D_hat, points_t)
    metrics['nll'] = nll.item()
    
    # 2. Kendall's tau
    tau_data = compute_kendall_tau(points[:, 0], points[:, 1])
    metrics['tau_data'] = tau_data
    
    # 3. Tail dependence
    tail_dep = tail_dependence_empirical(points)
    metrics['lambda_upper_data'] = tail_dep['lambda_upper']
    metrics['lambda_lower_data'] = tail_dep['lambda_lower']
    
    # 4. Distance metrics vs true density
    if true_density is not None:
        dist_metrics = copula_distance_metrics(density_pred, true_density)
        metrics.update(dist_metrics)
    
    # 5. Marginal uniformity check
    # Integrate density along each axis
    du = 1.0 / m
    marginal_u = np.sum(density_pred, axis=1) * du  # Should be ~1
    marginal_v = np.sum(density_pred, axis=0) * du  # Should be ~1
    
    metrics['marginal_u_error'] = np.mean(np.abs(marginal_u - 1.0))
    metrics['marginal_v_error'] = np.mean(np.abs(marginal_v - 1.0))
    
    # 6. Mass conservation
    total_mass = np.sum(density_pred) * du * du
    metrics['total_mass'] = total_mass
    metrics['mass_error'] = abs(total_mass - 1.0)
    
    return metrics


def plot_pair_copula_comparison(
    density_pred: np.ndarray,
    density_true: Optional[np.ndarray],
    points: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Copula Comparison"
):
    """
    Plot predicted vs true copula density with scatter points.
    
    Args:
        density_pred: (m, m) predicted density
        density_true: (m, m) true density (optional)
        points: (n, 2) data points
        save_path: Path to save figure
        title: Plot title
    """
    n_plots = 3 if density_true is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    m = density_pred.shape[0]
    extent = [0, 1, 0, 1]
    
    # Plot 1: Predicted density
    ax = axes[0] if n_plots > 1 else axes
    im1 = ax.imshow(
        density_pred.T,
        origin='lower',
        extent=extent,
        cmap='viridis',
        aspect='auto'
    )
    ax.scatter(points[:, 0], points[:, 1], c='red', s=1, alpha=0.3)
    ax.set_title('Predicted Density')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    plt.colorbar(im1, ax=ax)
    
    # Plot 2: True density (if available)
    if density_true is not None:
        im2 = axes[1].imshow(
            density_true.T,
            origin='lower',
            extent=extent,
            cmap='viridis',
            aspect='auto'
        )
        axes[1].scatter(points[:, 0], points[:, 1], c='red', s=1, alpha=0.3)
        axes[1].set_title('True Density')
        axes[1].set_xlabel('u')
        axes[1].set_ylabel('v')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot 3: Difference
        diff = np.abs(density_pred - density_true)
        im3 = axes[2].imshow(
            diff.T,
            origin='lower',
            extent=extent,
            cmap='Reds',
            aspect='auto'
        )
        axes[2].set_title('Absolute Difference')
        axes[2].set_xlabel('u')
        axes[2].set_ylabel('v')
        plt.colorbar(im3, ax=axes[2])
    else:
        # Plot 2: Log density
        log_density = np.log(np.clip(density_pred, 1e-10, None))
        im2 = axes[1].imshow(
            log_density.T,
            origin='lower',
            extent=extent,
            cmap='viridis',
            aspect='auto'
        )
        axes[1].scatter(points[:, 0], points[:, 1], c='red', s=1, alpha=0.3)
        axes[1].set_title('Log Density')
        axes[1].set_xlabel('u')
        axes[1].set_ylabel('v')
        plt.colorbar(im2, ax=axes[1])
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_marginals(
    density: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Marginal Densities"
):
    """
    Plot marginal densities (should be uniform for copulas).
    
    Args:
        density: (m, m) copula density
        save_path: Path to save figure
        title: Plot title
    """
    m = density.shape[0]
    du = 1.0 / m
    u_grid = np.linspace(du/2, 1-du/2, m)
    
    # Compute marginals
    marginal_u = np.sum(density, axis=1) * du
    marginal_v = np.sum(density, axis=0) * du
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Marginal u
    axes[0].plot(u_grid, marginal_u, 'b-', linewidth=2, label='Estimated')
    axes[0].axhline(1.0, color='r', linestyle='--', label='Uniform')
    axes[0].set_xlabel('u')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Marginal over u')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Marginal v
    axes[1].plot(u_grid, marginal_v, 'b-', linewidth=2, label='Estimated')
    axes[1].axhline(1.0, color='r', linestyle='--', label='Uniform')
    axes[1].set_xlabel('v')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Marginal over v')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Testing pair copula evaluation...")
    
    # Generate synthetic pair
    np.random.seed(42)
    n = 500
    
    from scipy.stats import norm
    rho = 0.7
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    
    Z = np.random.multivariate_normal(mean, cov, n)
    U = norm.cdf(Z)
    
    print(f"Generated {n} samples")
    
    # Compute metrics without model (just using data)
    tau = compute_kendall_tau(U[:, 0], U[:, 1])
    print(f"Kendall's tau: {tau:.4f}")
    
    tail_dep = tail_dependence_empirical(U)
    print(f"Upper tail dependence: {tail_dep['lambda_upper']:.4f}")
    print(f"Lower tail dependence: {tail_dep['lambda_lower']:.4f}")
    
    # Create dummy density for visualization
    m = 64
    density_dummy = np.ones((m, m))  # Independence copula
    
    # Plot
    plot_pair_copula_comparison(
        density_dummy,
        None,
        U,
        title="Test Pair Copula"
    )
    
    plot_marginals(density_dummy, title="Test Marginals")
    
    print("\nPair copula evaluation test completed!")
