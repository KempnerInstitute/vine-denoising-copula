"""
Example: Using visualization utilities programmatically.

This demonstrates how to create custom visualizations for copula density estimation.
"""

import numpy as np
import torch
from pathlib import Path
from scipy.stats import norm

# Import visualization utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vdc.eval.visualize import (
    plot_density_heatmap,
    plot_comparison,
    plot_marginals,
    plot_multi_comparison,
    create_paper_figure,
)
from vdc.data.generators import sample_bicop, analytic_logpdf_grid
from vdc.models.unet_grid import GridUNet
from vdc.data.hist import points_to_histogram
from vdc.models.projection import copula_project


def example_1_basic_heatmap():
    """Example 1: Basic density heatmap."""
    print("Example 1: Basic 2D Density Heatmap")
    print("="*60)
    
    # Generate Gaussian copula density
    m = 64
    rho = 0.7
    log_density = analytic_logpdf_grid('gaussian', {'rho': rho}, m=m)
    density = np.exp(log_density)
    
    # Generate sample points
    n = 1000
    points = sample_bicop('gaussian', {'rho': rho}, n)
    
    # Create heatmap
    fig, ax, im = plot_density_heatmap(
        density,
        title=f"Gaussian Copula (ρ={rho})",
        points=points,
        cmap='viridis',
    )
    
    output_path = Path('examples/plots/basic_heatmap.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}\n")


def example_2_comparison():
    """Example 2: Predicted vs True comparison."""
    print("Example 2: Predicted vs True Comparison")
    print("="*60)
    
    # True density (Clayton)
    m = 64
    theta = 3.0
    log_density_true = analytic_logpdf_grid('clayton', {'theta': theta}, m=m)
    density_true = np.exp(log_density_true)
    
    # Simulated prediction (add noise for demo)
    np.random.seed(42)
    noise = 0.1 * np.random.randn(m, m)
    density_pred = density_true * np.exp(noise)
    
    # Normalize
    du = 1.0 / m
    density_pred = density_pred / (np.sum(density_pred) * du * du)
    
    # Sample points
    points = sample_bicop('clayton', {'theta': theta}, 1000)
    
    # Compute metrics
    ise = np.sum((density_pred - density_true)**2) * du * du
    mass_error = abs(np.sum(density_pred) * du * du - 1.0)
    
    metrics = {
        'ISE': ise,
        'Mass Error': mass_error,
    }
    
    # Create comparison plot
    fig = plot_comparison(
        density_pred,
        density_true,
        title=f"Clayton Copula (θ={theta})",
        points=points,
        metrics=metrics,
        save_path=Path('examples/plots/comparison.png'),
    )
    print(f"✓ Created comparison plot\n")


def example_3_marginals():
    """Example 3: Marginal density checks."""
    print("Example 3: Marginal Density Checks")
    print("="*60)
    
    # Generate Gumbel copula
    m = 64
    theta = 2.5
    log_density = analytic_logpdf_grid('gumbel', {'theta': theta}, m=m)
    density = np.exp(log_density)
    
    # Plot marginals
    fig = plot_marginals(
        density,
        title=f"Gumbel Copula (θ={theta}) - Marginals",
        save_path=Path('examples/plots/marginals.png'),
    )
    print(f"✓ Created marginals plot\n")


def example_4_multi_comparison():
    """Example 4: Multiple copulas comparison."""
    print("Example 4: Multi-Copula Comparison")
    print("="*60)
    
    m = 64
    
    # Test multiple copulas
    test_cases = [
        {'family': 'gaussian', 'params': {'rho': 0.7}, 'name': 'Gaussian(0.7)'},
        {'family': 'clayton', 'params': {'theta': 3.0}, 'name': 'Clayton(3.0)'},
        {'family': 'gumbel', 'params': {'theta': 2.5}, 'name': 'Gumbel(2.5)'},
        {'family': 'frank', 'params': {'theta': 5.0}, 'name': 'Frank(5.0)'},
    ]
    
    results = []
    for case in test_cases:
        # True density
        log_density_true = analytic_logpdf_grid(case['family'], case['params'], m=m)
        density_true = np.exp(log_density_true)
        
        # Simulated prediction
        noise = 0.08 * np.random.randn(m, m)
        density_pred = density_true * np.exp(noise)
        du = 1.0 / m
        density_pred = density_pred / (np.sum(density_pred) * du * du)
        
        # Points
        points = sample_bicop(case['family'], case['params'], 500)
        
        # Metrics
        ise = np.sum((density_pred - density_true)**2) * du * du
        
        results.append({
            'density_pred': density_pred,
            'density_true': density_true,
            'name': case['name'],
            'points': points,
            'metrics': {'ise': ise},
        })
    
    # Create multi-comparison
    fig = plot_multi_comparison(
        results,
        save_path=Path('examples/plots/multi_comparison.png'),
        ncols=2,
    )
    print(f"✓ Created multi-comparison plot\n")


def example_5_paper_figure():
    """Example 5: Publication-ready comprehensive figure."""
    print("Example 5: Publication-Ready Figure")
    print("="*60)
    
    # Student-t copula
    m = 64
    params = {'rho': 0.7, 'nu': 5.0}
    
    log_density_true = analytic_logpdf_grid('student', params, m=m)
    density_true = np.exp(log_density_true)
    
    # Simulated prediction
    noise = 0.06 * np.random.randn(m, m)
    density_pred = density_true * np.exp(noise)
    du = 1.0 / m
    density_pred = density_pred / (np.sum(density_pred) * du * du)
    
    # Sample points
    points = sample_bicop('student', params, 1500)
    
    # Comprehensive metrics
    ise = np.sum((density_pred - density_true)**2) * du * du
    mass_error = abs(np.sum(density_pred) * du * du - 1.0)
    marginal_u = np.sum(density_pred, axis=1) * du
    marginal_error = np.mean(np.abs(marginal_u - 1.0))
    
    metrics = {
        'ISE': ise,
        'Mass Error': mass_error,
        'Marginal Error': marginal_error,
    }
    
    # Create comprehensive paper figure
    fig = create_paper_figure(
        density_pred,
        density_true,
        points,
        title="Student-t Copula (ρ=0.7, ν=5)",
        save_path=Path('examples/plots/paper_figure.png'),
        metrics=metrics,
    )
    print(f"✓ Created paper figure with 6 panels\n")


def example_6_model_prediction():
    """Example 6: Using with trained model."""
    print("Example 6: Model Prediction Visualization")
    print("="*60)
    
    # This example shows how to visualize model predictions
    # (requires a trained checkpoint)
    
    print("To use with a trained model:")
    print()
    print("```python")
    print("# Load model")
    print("checkpoint = torch.load('checkpoints/model.pt')")
    print("model = GridUNet(**config).to('cuda')")
    print("model.load_state_dict(checkpoint['model_ema'])")
    print("model.eval()")
    print()
    print("# Generate test data")
    print("points = sample_bicop('gaussian', {'rho': 0.7}, 2000)")
    print("density_true = np.exp(analytic_logpdf_grid('gaussian', {'rho': 0.7}, m=64))")
    print()
    print("# Predict")
    print("hist = points_to_histogram(points, m=64)")
    print("hist_t = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).cuda()")
    print("with torch.no_grad():")
    print("    t = torch.ones(1, 1, 1, 1).cuda() * 0.5")
    print("    logD = model(hist_t, t)")
    print("    D = torch.exp(logD)")
    print("    D_proj = copula_project(D)")
    print("density_pred = D_proj[0, 0].cpu().numpy()")
    print()
    print("# Visualize")
    print("plot_comparison(density_pred, density_true, points=points,")
    print("                title='Model Prediction', save_path='prediction.png')")
    print("```")
    print()


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Copula Density Visualization Examples")
    print("="*60 + "\n")
    
    # Create output directory
    Path('examples/plots').mkdir(parents=True, exist_ok=True)
    
    # Run examples
    example_1_basic_heatmap()
    example_2_comparison()
    example_3_marginals()
    example_4_multi_comparison()
    example_5_paper_figure()
    example_6_model_prediction()
    
    print("="*60)
    print("All examples completed!")
    print(f"Plots saved to: examples/plots/")
    print("="*60 + "\n")
    
    # List generated files
    plot_files = list(Path('examples/plots').glob('*.png'))
    print(f"Generated {len(plot_files)} visualizations:")
    for f in sorted(plot_files):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
