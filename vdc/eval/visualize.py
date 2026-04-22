"""
Professional visualization utilities for copula density estimation.

Provides high-quality 2D heatmap plotting for:
- Predicted vs true copula densities
- Comparison visualizations
- Error analysis
- Publication-ready figures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import seaborn as sns

# Set publication-quality defaults
sns.set_style("whitegrid")
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9


def plot_density_heatmap(
    density: np.ndarray,
    title: str = "Copula Density",
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    add_colorbar: bool = True,
    points: Optional[np.ndarray] = None,
    point_size: float = 0.5,
    point_alpha: float = 0.3,
    point_color: str = 'red',
) -> tuple:
    """
    Plot a 2D copula density as a heatmap.
    
    Args:
        density: (m, m) density array
        title: Plot title
        ax: Matplotlib axis (creates new if None)
        cmap: Colormap name
        vmin, vmax: Color scale limits
        add_colorbar: Whether to add colorbar
        points: Optional (n, 2) scatter points to overlay
        point_size: Size of scatter points
        point_alpha: Transparency of scatter points
        point_color: Color of scatter points
        
    Returns:
        (figure, axis, image) tuple
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    
    m = density.shape[0]
    extent = [0, 1, 0, 1]
    
    # Plot heatmap
    im = ax.imshow(
        density.T,
        origin='lower',
        extent=extent,
        cmap=cmap,
        aspect='equal',
        interpolation='bilinear',
        vmin=vmin,
        vmax=vmax,
    )
    
    # Overlay scatter points if provided
    if points is not None:
        ax.scatter(
            points[:, 0], points[:, 1],
            c=point_color,
            s=point_size,
            alpha=point_alpha,
            edgecolors='none',
            rasterized=True,
        )
    
    ax.set_xlabel('u', fontweight='bold')
    ax.set_ylabel('v', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add colorbar
    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Density', fontweight='bold')
    
    return fig, ax, im


def _percentile_scale(arr: np.ndarray, lower: float = 1.0, upper: float = 99.5) -> Tuple[float, float]:
    """Return a robust (vmin, vmax) pair that is not dominated by outliers."""
    arr = np.asarray(arr)
    lo = float(np.percentile(arr, lower))
    hi = float(np.percentile(arr, upper))
    if hi - lo < 1e-12:
        lo = float(arr.min())
        hi = float(arr.max())
    if hi - lo < 1e-12:
        hi = lo + 1e-9
    return lo, hi


def _format_metric(value: float) -> str:
    """Pretty-print metrics with adaptive formatting."""
    if value == 0.0:
        return "0.0000"
    abs_val = abs(value)
    if abs_val >= 1e3 or abs_val < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _resolve_color_scales(
    density_pred: np.ndarray,
    density_true: np.ndarray,
    mode: str = "independent",
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Determine (vmin, vmax) pairs for predicted and true panels.

    Args:
        density_pred: Predicted density grid
        density_true: True density grid
        mode: 'independent' or 'shared'

    Returns:
        ((pred_vmin, pred_vmax), (true_vmin, true_vmax))
    """
    mode = mode.lower()
    if mode == "independent":
        pred_scale = _percentile_scale(density_pred)
        true_scale = _percentile_scale(density_true)
        return pred_scale, true_scale
    if mode == "shared":
        combined = np.concatenate((density_pred.ravel(), density_true.ravel()))
        shared_scale = _percentile_scale(combined)
        return shared_scale, shared_scale
    raise ValueError(f"Unknown scale mode: {mode}")


def plot_comparison(
    density_pred: np.ndarray,
    density_true: np.ndarray,
    title: str = "Copula Comparison",
    points: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    metrics: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (18, 5),
    scale_mode: str = "independent",
) -> plt.Figure:
    """
    Create a comparison plot showing predicted, true, and error panels.
    
    Args:
        density_pred: (m, m) predicted density
        density_true: (m, m) true density
        title: Overall figure title
        points: Optional (n, 2) data points
        save_path: Path to save figure
        metrics: Optional metrics dict to display
        figsize: Figure size
        
    Returns:
        Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # Color scale handling
    (pred_vmin, pred_vmax), (true_vmin, true_vmax) = _resolve_color_scales(
        density_pred, density_true, mode=scale_mode
    )

    # Plot 1: Predicted
    ax1 = fig.add_subplot(gs[0, 0])
    plot_density_heatmap(
        density_pred,
        title="Predicted Density",
        ax=ax1,
        cmap='viridis',
        vmin=pred_vmin,
        vmax=pred_vmax,
        points=points,
    )

    # Plot 2: True
    ax2 = fig.add_subplot(gs[0, 1])
    plot_density_heatmap(
        density_true,
        title="True Density",
        ax=ax2,
        cmap='viridis',
        vmin=true_vmin,
        vmax=true_vmax,
        points=points,
    )

    # Plot 3: Absolute Difference
    ax3 = fig.add_subplot(gs[0, 2])
    diff = np.abs(density_pred - density_true)
    _, diff_vmax = _percentile_scale(diff, lower=0.0, upper=99.5)
    plot_density_heatmap(
        diff,
        title="Absolute Error",
        ax=ax3,
        cmap='Reds',
        vmin=0,
        vmax=diff_vmax,
    )

    # Add metrics text if provided
    if metrics is not None and len(metrics) > 0:
        metrics_text = "\n".join([f"{k}: {_format_metric(v)}" for k, v in metrics.items()])
        fig.text(
            0.99, 0.02,
            metrics_text,
            ha='right',
            va='bottom',
            fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_log_density(
    density: np.ndarray,
    title: str = "Log Copula Density",
    ax: Optional[plt.Axes] = None,
    points: Optional[np.ndarray] = None,
    clip_value: float = 1e-10,
) -> tuple:
    """
    Plot log-density (useful for seeing tail behavior).
    
    Args:
        density: (m, m) density array
        title: Plot title
        ax: Matplotlib axis
        points: Optional scatter points
        clip_value: Minimum value before taking log
        
    Returns:
        (figure, axis, image) tuple
    """
    log_density = np.log(np.clip(density, clip_value, None))
    
    return plot_density_heatmap(
        log_density,
        title=title,
        ax=ax,
        cmap='viridis',
        points=points,
    )


def plot_marginals(
    density: np.ndarray,
    title: str = "Marginal Densities",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 4),
    row_coords: Optional[np.ndarray] = None,
    col_coords: Optional[np.ndarray] = None,
    row_widths: Optional[np.ndarray] = None,
    col_widths: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Plot marginal densities (should be uniform for copulas).
    
    Args:
        density: (m, m) copula density
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure object
    """
    m = density.shape[0]
    if row_widths is None:
        row_widths = np.full(m, 1.0 / m)
    if col_widths is None:
        col_widths = np.full(m, 1.0 / m)
    if row_coords is None:
        row_coords = np.cumsum(row_widths) - 0.5 * row_widths
    if col_coords is None:
        col_coords = np.cumsum(col_widths) - 0.5 * col_widths
    
    # Compute marginals by integration
    marginal_u = np.sum(density * col_widths[None, :], axis=1)  # ∫ c(u,v) dv
    marginal_v = np.sum(density * row_widths[:, None], axis=0)  # ∫ c(u,v) du
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Marginal over u
    axes[0].plot(row_coords, marginal_u, 'b-', linewidth=2, label='Estimated', alpha=0.8)
    axes[0].axhline(1.0, color='r', linestyle='--', linewidth=2, label='Uniform (ideal)')
    axes[0].fill_between(row_coords, marginal_u, 1.0, alpha=0.3, color='blue')
    axes[0].set_xlabel('u', fontweight='bold')
    axes[0].set_ylabel('∫ c(u,v) dv', fontweight='bold')
    axes[0].set_title('Marginal over u', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.8, 1.2])
    
    # Add error annotation
    error_u = np.mean(np.abs(marginal_u - 1.0))
    axes[0].text(
        0.02, 0.98,
        f'Mean Error: {error_u:.4f}',
        transform=axes[0].transAxes,
        va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # Marginal over v
    axes[1].plot(col_coords, marginal_v, 'g-', linewidth=2, label='Estimated', alpha=0.8)
    axes[1].axhline(1.0, color='r', linestyle='--', linewidth=2, label='Uniform (ideal)')
    axes[1].fill_between(col_coords, marginal_v, 1.0, alpha=0.3, color='green')
    axes[1].set_xlabel('v', fontweight='bold')
    axes[1].set_ylabel('∫ c(u,v) du', fontweight='bold')
    axes[1].set_title('Marginal over v', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.8, 1.2])
    
    # Add error annotation
    error_v = np.mean(np.abs(marginal_v - 1.0))
    axes[1].text(
        0.02, 0.98,
        f'Mean Error: {error_v:.4f}',
        transform=axes[1].transAxes,
        va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_multi_comparison(
    results: List[Dict],
    save_path: Optional[Path] = None,
    ncols: int = 3,
) -> plt.Figure:
    """
    Create multi-panel comparison for multiple copulas.
    
    Args:
        results: List of dicts with keys:
            - 'density_pred': (m, m) predicted density
            - 'density_true': (m, m) true density
            - 'name': copula name
            - 'metrics': dict of metrics
            - 'points': optional (n, 2) data points
        save_path: Path to save figure
        ncols: Number of columns in grid
        
    Returns:
        Figure object
    """
    n = len(results)
    nrows = (n + ncols - 1) // ncols
    
    fig = plt.figure(figsize=(6*ncols, 5*nrows))
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, result in enumerate(results):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(gs[row, col])
        
        density_pred = result['density_pred']
        density_true = result.get('density_true')
        name = result.get('name', f'Copula {idx+1}')
        metrics = result.get('metrics', {})
        points = result.get('points')
        
        # Plot prediction or difference
        if density_true is not None:
            diff = np.abs(density_pred - density_true)
            plot_density_heatmap(
                diff,
                title=f"{name}\n(Absolute Error)",
                ax=ax,
                cmap='Reds',
                vmin=0,
                points=points,
            )
            
            # Add ISE metric if available
            if 'ise' in metrics:
                ax.text(
                    0.02, 0.98,
                    f"ISE: {metrics['ise']:.4f}",
                    transform=ax.transAxes,
                    va='top',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
                )
        else:
            plot_density_heatmap(
                density_pred,
                title=name,
                ax=ax,
                cmap='viridis',
                points=points,
            )
    
    fig.suptitle('Copula Density Estimation Results', fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_metrics_summary(
    metrics_dict: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot summary of metrics across multiple test cases.
    
    Args:
        metrics_dict: Dict mapping metric names to lists of values
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure object
    """
    n_metrics = len(metrics_dict)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, (metric_name, values) in enumerate(metrics_dict.items()):
        ax = axes[idx]
        
        # Create bar plot
        x = np.arange(len(values))
        bars = ax.bar(x, values, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Color bars by value (green=good, red=bad)
        colors = plt.cm.RdYlGn_r(np.array(values) / (np.max(values) + 1e-8))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Test Case', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name}\n(mean: {np.mean(values):.4f})', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Performance Metrics Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def create_paper_figure(
    density_pred: np.ndarray,
    density_true: np.ndarray,
    points: np.ndarray,
    title: str,
    save_path: Path,
    metrics: Dict[str, float],
) -> plt.Figure:
    """
    Create publication-ready figure with all panels.
    
    Layout:
    - Top row: Predicted, True, Difference (3 panels)
    - Bottom row: Log(Predicted), Log(True), Marginals (3 panels)
    
    Args:
        density_pred: (m, m) predicted density
        density_true: (m, m) true density
        points: (n, 2) data points
        title: Figure title
        save_path: Path to save figure
        metrics: Metrics dict
        
    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.3)
    
    # Common color scale for linear densities
    vmin = min(density_pred.min(), density_true.min())
    vmax = max(density_pred.max(), density_true.max())
    
    # Row 1: Linear densities
    ax1 = fig.add_subplot(gs[0, 0])
    plot_density_heatmap(density_pred, "Predicted", ax1, vmin=vmin, vmax=vmax, points=points)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_density_heatmap(density_true, "True", ax2, vmin=vmin, vmax=vmax, points=points)
    
    ax3 = fig.add_subplot(gs[0, 2])
    diff = np.abs(density_pred - density_true)
    plot_density_heatmap(diff, "Absolute Error", ax3, cmap='Reds', vmin=0)
    
    # Row 2: Log densities + marginals
    ax4 = fig.add_subplot(gs[1, 0])
    plot_log_density(density_pred, "Log(Predicted)", ax4, points=points)
    
    ax5 = fig.add_subplot(gs[1, 1])
    plot_log_density(density_true, "Log(True)", ax5, points=points)
    
    # Marginals in rightmost panel
    ax6 = fig.add_subplot(gs[1, 2])
    m = density_pred.shape[0]
    du = 1.0 / m
    u_grid = np.linspace(du/2, 1-du/2, m)
    marginal_u = np.sum(density_pred, axis=1) * du
    marginal_v = np.sum(density_pred, axis=0) * du
    
    ax6.plot(u_grid, marginal_u, 'b-', linewidth=2, label='u marginal', alpha=0.8)
    ax6.plot(u_grid, marginal_v, 'g-', linewidth=2, label='v marginal', alpha=0.8)
    ax6.axhline(1.0, color='r', linestyle='--', linewidth=2, label='Uniform')
    ax6.set_xlabel('u / v', fontweight='bold')
    ax6.set_ylabel('Marginal Density', fontweight='bold')
    ax6.set_title('Marginal Checks', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0.8, 1.2])
    
    # Add metrics box
    metrics_text = "Metrics:\n" + "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    fig.text(
        0.99, 0.01,
        metrics_text,
        ha='right',
        va='bottom',
        fontsize=10,
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1.0)
    )
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved publication figure: {save_path}")
    
    return fig


if __name__ == "__main__":
    """Test visualization functions."""
    print("Testing visualization utilities...\n")
    
    # Create synthetic data
    m = 64
    u_grid = np.linspace(0, 1, m)
    U, V = np.meshgrid(u_grid, u_grid, indexing='ij')
    
    # Gaussian copula density (approximate)
    rho = 0.7
    from scipy.stats import norm
    z_u = norm.ppf(np.clip(U, 0.01, 0.99))
    z_v = norm.ppf(np.clip(V, 0.01, 0.99))
    density_true = np.exp(
        -(z_u**2 + z_v**2 - 2*rho*z_u*z_v) / (2*(1-rho**2))
        + (z_u**2 + z_v**2) / 2
    ) / np.sqrt(1 - rho**2)
    
    # Normalize
    du = 1.0 / m
    density_true = density_true / (np.sum(density_true) * du * du)
    
    # Create noisy prediction
    density_pred = density_true + 0.1 * np.random.randn(*density_true.shape)
    density_pred = np.clip(density_pred, 0, None)
    density_pred = density_pred / (np.sum(density_pred) * du * du)
    
    # Generate sample points
    n_points = 500
    Z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n_points)
    points = norm.cdf(Z)
    
    # Test individual plots
    print("1. Testing density heatmap...")
    fig, ax, im = plot_density_heatmap(density_pred, title="Test Density", points=points)
    plt.close(fig)
    print("Density heatmap OK\n")
    
    print("2. Testing comparison plot...")
    metrics = {'ise': 0.0234, 'nll': -0.156, 'tau_error': 0.012}
    fig = plot_comparison(density_pred, density_true, title="Test Comparison", 
                          points=points, metrics=metrics)
    plt.close(fig)
    print("Comparison plot OK\n")
    
    print("3. Testing marginals plot...")
    fig = plot_marginals(density_pred, title="Test Marginals")
    plt.close(fig)
    print("Marginals plot OK\n")
    
    print("4. Testing multi-comparison...")
    results = [
        {'density_pred': density_pred, 'density_true': density_true, 
         'name': 'Gaussian(0.7)', 'metrics': metrics, 'points': points},
        {'density_pred': density_pred * 0.9, 'density_true': density_true, 
         'name': 'Clayton(3.0)', 'metrics': {'ise': 0.045}, 'points': points},
    ]
    fig = plot_multi_comparison(results)
    plt.close(fig)
    print("Multi-comparison OK\n")
    
    print("5. Testing metrics summary...")
    metrics_dict = {
        'ISE': [0.023, 0.045, 0.019, 0.067, 0.031],
        'NLL': [-0.156, -0.234, -0.189, -0.298, -0.201],
        'Tau Error': [0.012, 0.023, 0.008, 0.034, 0.015],
    }
    fig = plot_metrics_summary(metrics_dict)
    plt.close(fig)
    print("Metrics summary OK\n")
    
    print("6. Testing paper figure...")
    fig = create_paper_figure(
        density_pred, density_true, points,
        title="Gaussian Copula (ρ=0.7)",
        save_path=Path("/tmp/test_paper_figure.png"),
        metrics=metrics
    )
    plt.close(fig)
    print("Paper figure OK\n")
    
    print("="*60)
    print("All visualization tests passed.")
    print("="*60)
