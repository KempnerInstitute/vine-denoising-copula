#!/usr/bin/env python3
"""
Example: Visualization of Copula Densities and Vine Structures.

This provides visualization utilities for:
1. Bivariate copula density heatmaps
2. Comparison of true vs estimated densities
3. Vine copula structure diagrams
4. MI edge attribution visualizations
5. h-function (conditional CDF) plots

Usage:
    python examples/visualize_copulas.py --output figures/
    python examples/visualize_copulas.py --checkpoint path/to/model.pt

Reference:
    - Schepsmeier & Stöber (2014), "Derivatives and Fisher information of
      bivariate copulas"
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Add project root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def setup_plot_style():
    """Configure publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def plot_copula_density(
    density: np.ndarray,
    ax: plt.Axes,
    title: str = "",
    cmap: str = "viridis",
    log_scale: bool = False,
    colorbar: bool = True,
    vmin: float = None,
    vmax: float = None,
):
    """
    Plot a bivariate copula density as a heatmap.
    
    Args:
        density: (m, m) density grid
        ax: Matplotlib axes
        title: Plot title
        cmap: Colormap name
        log_scale: Use log scale for colors
        colorbar: Show colorbar
        vmin, vmax: Color limits
    """
    m = density.shape[0]
    extent = [0, 1, 0, 1]
    
    if log_scale:
        norm = LogNorm(vmin=max(density.min(), 1e-3) if vmin is None else vmin,
                       vmax=density.max() if vmax is None else vmax)
        im = ax.imshow(density.T, origin='lower', extent=extent, 
                       cmap=cmap, norm=norm, aspect='equal')
    else:
        im = ax.imshow(density.T, origin='lower', extent=extent,
                       cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    
    ax.set_xlabel('$u$')
    ax.set_ylabel('$v$')
    ax.set_title(title)
    
    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    return im


def plot_copula_comparison(
    densities: Dict[str, np.ndarray],
    output_path: Path,
    copula_name: str = "Copula",
    figsize: Tuple[float, float] = None,
):
    """
    Create multi-panel comparison of copula densities.
    
    Args:
        densities: Dict mapping method names to density grids
        output_path: Path to save figure
        copula_name: Name for the figure title
        figsize: Figure size
    """
    n_panels = len(densities)
    if figsize is None:
        figsize = (4 * n_panels, 3.5)
    
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]
    
    # Find global vmin/vmax for consistent coloring
    all_densities = np.concatenate([d.flatten() for d in densities.values()])
    vmin, vmax = np.percentile(all_densities, [1, 99])
    
    for ax, (name, density) in zip(axes, densities.items()):
        plot_copula_density(density, ax, title=name, vmin=vmin, vmax=vmax,
                           colorbar=(ax == axes[-1]))
    
    fig.suptitle(f'{copula_name} Density Comparison', fontsize=12, y=1.02)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_samples_with_density(
    samples: np.ndarray,
    density: Optional[np.ndarray],
    output_path: Path,
    title: str = "",
    n_show: int = 2000,
):
    """
    Plot samples overlaid on density contours.
    
    Args:
        samples: (n, 2) samples
        density: (m, m) density grid (optional)
        output_path: Path to save figure
        title: Plot title
        n_show: Max samples to show
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot density contours if available
    if density is not None:
        m = density.shape[0]
        u = np.linspace(0, 1, m)
        U, V = np.meshgrid(u, u)
        levels = np.percentile(density, [10, 25, 50, 75, 90, 95])
        ax.contour(U, V, density.T, levels=levels, colors='gray', alpha=0.5, linewidths=0.5)
    
    # Plot samples
    idx = np.random.choice(len(samples), min(n_show, len(samples)), replace=False)
    ax.scatter(samples[idx, 0], samples[idx, 1], s=1, alpha=0.3, c='steelblue')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('$u$')
    ax.set_ylabel('$v$')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_dvine_structure(
    d: int,
    edge_labels: Optional[Dict[Tuple[int, int, int], str]] = None,
    edge_colors: Optional[Dict[Tuple[int, int, int], str]] = None,
    output_path: Path = None,
    figsize: Tuple[float, float] = None,
):
    """
    Visualize D-vine structure with edge labels.
    
    Args:
        d: Dimension (number of variables)
        edge_labels: Dict mapping (tree, i, j) to label string
        edge_colors: Dict mapping (tree, i, j) to color
        output_path: Path to save figure
        figsize: Figure size
    """
    if figsize is None:
        figsize = (2 + 1.5 * (d - 1), 2 * d)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Node positions: tree on y-axis, node index on x-axis
    node_positions = {}
    for tree in range(d):
        n_nodes = d - tree
        for node in range(n_nodes):
            x = node + tree * 0.5 + 0.5
            y = (d - 1 - tree) * 1.5
            node_positions[(tree, node)] = (x, y)
    
    # Draw edges
    for tree in range(d - 1):
        for edge in range(d - tree - 1):
            pos1 = node_positions[(tree, edge)]
            pos2 = node_positions[(tree, edge + 1)]
            
            # Get edge properties
            key = (tree, edge, edge + 1)
            color = edge_colors.get(key, 'steelblue') if edge_colors else 'steelblue'
            label = edge_labels.get(key, '') if edge_labels else ''
            
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                   color=color, linewidth=2, zorder=1)
            
            if label:
                mid_x = (pos1[0] + pos2[0]) / 2
                mid_y = (pos1[1] + pos2[1]) / 2 + 0.15
                ax.text(mid_x, mid_y, label, ha='center', va='bottom', fontsize=8)
    
    # Draw nodes
    for (tree, node), (x, y) in node_positions.items():
        if tree == 0:
            # First tree: label with variable indices
            label = f'$X_{node+1}$'
        else:
            # Higher trees: label with conditioning set
            label = f'$U_{{{node+1}|...}}$'
        
        circle = plt.Circle((x, y), 0.25, color='lightsteelblue', ec='steelblue', 
                            linewidth=1.5, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, zorder=3)
    
    # Add tree labels
    for tree in range(d):
        y = (d - 1 - tree) * 1.5
        ax.text(-0.5, y, f'Tree {tree + 1}', ha='right', va='center', 
               fontsize=10, fontweight='bold')
    
    ax.set_xlim(-1, d + 0.5)
    ax.set_ylim(-0.5, (d - 1) * 1.5 + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'{d}-Dimensional D-Vine Structure', fontsize=12, pad=20)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    return fig


def plot_hfunction(
    family: str,
    params: Dict,
    output_path: Path,
    n_v_values: int = 5,
):
    """
    Plot h-function h(u|v) for different conditioning values v.
    
    The h-function is the conditional CDF: h(u|v) = P(U ≤ u | V = v).
    
    Args:
        family: Copula family
        params: Family parameters
        output_path: Path to save figure
        n_v_values: Number of conditioning values to show
    """
    from vdc.data.conditional_copulas import H_FUNCTIONS
    
    if family not in H_FUNCTIONS:
        print(f"h-function not available for {family}")
        return
    
    h_func = H_FUNCTIONS[family]
    u = np.linspace(0.01, 0.99, 100)
    v_values = np.linspace(0.1, 0.9, n_v_values)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_v_values))
    
    for v_cond, color in zip(v_values, colors):
        v = np.full_like(u, v_cond)
        
        if family in ['gaussian', 'student']:
            rho = params.get('rho', 0.5)
            if family == 'student':
                df = int(params.get('df', params.get('nu', 5)))
                h_val = h_func(u, v, rho, df)
            else:
                h_val = h_func(u, v, rho)
        else:
            theta = params.get('theta', 2.0)
            h_val = h_func(u, v, theta)
        
        ax.plot(u, h_val, color=color, linewidth=1.5, label=f'v={v_cond:.1f}')
    
    # Reference: independence (45-degree line)
    ax.plot(u, u, 'k--', linewidth=1, alpha=0.5, label='Independence')
    
    ax.set_xlabel('$u$')
    ax.set_ylabel('$h(u|v)$')
    ax.set_title(f'h-Function: {family.title()} Copula')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_mi_edge_attribution(
    edge_mis: List[float],
    d: int,
    output_path: Path,
    title: str = "MI Edge Attribution",
):
    """
    Visualize edge-wise MI contributions in a vine.
    
    Args:
        edge_mis: List of MI values for each edge
        d: Dimension
        output_path: Path to save figure
        title: Plot title
    """
    # Create edge labels and colors based on MI
    edge_labels = {}
    edge_colors = {}
    
    mi_min = min(edge_mis)
    mi_max = max(edge_mis)
    mi_range = mi_max - mi_min + 1e-6
    
    idx = 0
    for tree in range(d - 1):
        for edge in range(d - tree - 1):
            key = (tree, edge, edge + 1)
            mi = edge_mis[idx]
            edge_labels[key] = f'{mi:.2f}'
            
            # Color by MI intensity
            intensity = (mi - mi_min) / mi_range
            edge_colors[key] = plt.cm.Reds(0.3 + 0.6 * intensity)
            
            idx += 1
    
    fig = plot_dvine_structure(d, edge_labels, edge_colors, output_path)


def demo_copula_gallery():
    """Generate a gallery of copula density visualizations."""
    print("\n" + "="*60)
    print("Generating Copula Density Gallery")
    print("="*60)
    
    from vdc.data.generators import (
        gaussian_copula_density, student_copula_density,
        clayton_copula_density, frank_copula_density,
        gumbel_copula_density, joe_copula_density
    )
    
    output_dir = REPO_ROOT / "figures" / "copula_gallery"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    m = 128
    u = np.linspace(0.5/m, 1 - 0.5/m, m)
    U, V = np.meshgrid(u, u, indexing='ij')
    
    copulas = [
        ("Gaussian (ρ=0.7)", lambda: gaussian_copula_density(U, V, 0.7)),
        ("Gaussian (ρ=-0.7)", lambda: gaussian_copula_density(U, V, -0.7)),
        ("Student-t (ρ=0.7, df=5)", lambda: student_copula_density(U, V, 0.7, 5)),
        ("Clayton (θ=3)", lambda: clayton_copula_density(U, V, 3.0)),
        ("Gumbel (θ=2.5)", lambda: gumbel_copula_density(U, V, 2.5)),
        ("Frank (θ=5)", lambda: frank_copula_density(U, V, 5.0)),
        ("Joe (θ=3)", lambda: joe_copula_density(U, V, 3.0)),
    ]
    
    # Create combined figure
    n_copulas = len(copulas)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    
    for i, (name, density_fn) in enumerate(copulas):
        density = density_fn()
        density = np.clip(density, 0, None)
        plot_copula_density(density, axes[i], title=name, colorbar=False, log_scale=True)
    
    # Hide unused axes
    for i in range(n_copulas, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    fig.savefig(output_dir / "copula_gallery.pdf")
    fig.savefig(output_dir / "copula_gallery.png")
    plt.close(fig)
    
    print(f"Saved gallery to: {output_dir}")


def demo_vine_structure():
    """Generate vine structure visualizations."""
    print("\n" + "="*60)
    print("Generating Vine Structure Diagrams")
    print("="*60)
    
    output_dir = REPO_ROOT / "figures" / "vine_structures"
    
    for d in [4, 5, 6]:
        plot_dvine_structure(d, output_path=output_dir / f"dvine_d{d}.pdf")


def demo_hfunctions():
    """Generate h-function plots for various families."""
    print("\n" + "="*60)
    print("Generating h-Function Plots")
    print("="*60)
    
    output_dir = REPO_ROOT / "figures" / "hfunctions"
    
    families = [
        ("gaussian", {"rho": 0.7}),
        ("clayton", {"theta": 3.0}),
        ("gumbel", {"theta": 2.5}),
        ("frank", {"theta": 5.0}),
    ]
    
    for family, params in families:
        plot_hfunction(family, params, output_dir / f"hfunc_{family}.pdf")


def main():
    parser = argparse.ArgumentParser(description="Copula Visualization Examples")
    parser.add_argument("--output", type=str, default="figures/",
                        help="Output directory for figures")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (for estimated densities)")
    parser.add_argument("--demo", type=str, 
                        choices=["gallery", "vine", "hfunc", "all"],
                        default="all", help="Which demo to run")
    args = parser.parse_args()
    
    setup_plot_style()
    
    print("\n" + "="*60)
    print("Copula Visualization Examples")
    print("="*60)
    
    if args.demo in ["gallery", "all"]:
        demo_copula_gallery()
    
    if args.demo in ["vine", "all"]:
        demo_vine_structure()
    
    if args.demo in ["hfunc", "all"]:
        demo_hfunctions()
    
    print("\n" + "="*60)
    print("Done! Figures saved to figures/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
