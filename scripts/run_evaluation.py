#!/usr/bin/env python3
"""
Unified evaluation and visualization script driven by configuration.

This script provides a complete workflow:
1. Load trained model
2. Evaluate on test copulas (parametric and/or real data)
3. Generate visualizations
4. Compute metrics
5. Save results

Usage:
    python scripts/run_evaluation.py --config configs/workflow/evaluate_and_visualize.yaml

Future extensions:
    - Support for real data evaluation
    - Vine copula construction from learned bivariate copulas
    - Multi-dimensional analysis
"""
import argparse
import yaml
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from vdc.vine.copula_diffusion import DiffusionCopulaModel
from vdc.data.generators import sample_bicop, analytic_logpdf_grid
from vdc.eval.visualize import plot_comparison, plot_marginals


def load_config(config_path):
    """Load evaluation configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_parametric_copulas(model, config):
    """Evaluate model on parametric test copulas."""
    results = []
    
    test_copulas = config['evaluation']['test_copulas']
    m = config['evaluation']['grid_resolution']
    
    print(f"\nEvaluating {len(test_copulas)} parametric copulas...")
    
    for copula_spec in test_copulas:
        family = copula_spec['family']
        params = copula_spec['params']
        name = copula_spec['name']
        
        print(f"\n  Processing: {name}")
        
        # Generate samples
        samples = sample_bicop(family, params, n=5000, rotation=0)
        
        # Estimate density
        density_pred, u_coords, v_coords = model.estimate_density_from_samples(
            samples,
            m=m,
            projection_iters=config['evaluation']['projection_iters']
        )
        
        # Get true density (for comparison)
        log_density_true = analytic_logpdf_grid(family, params, m=m, rotation=0)
        density_true = np.exp(np.clip(log_density_true, -20, 20))
        du = dv = 1.0 / m
        density_true /= (density_true.sum() * du * dv)
        
        # Compute metrics
        ise = np.mean((density_pred - density_true) ** 2)
        mae = np.mean(np.abs(density_pred - density_true))
        
        # Marginal check
        u_marginal = (density_pred * dv).sum(axis=1)
        v_marginal = (density_pred * du).sum(axis=0)
        marginal_error = np.mean(np.abs(u_marginal - 1.0)) + np.mean(np.abs(v_marginal - 1.0))
        
        print(f"    ISE: {ise:.2f}")
        print(f"    MAE: {mae:.4f}")
        print(f"    Marginal error: {marginal_error:.6f}")
        
        results.append({
            'name': name,
            'family': family,
            'params': params,
            'density_pred': density_pred,
            'density_true': density_true,
            'samples': samples,
            'u_coords': u_coords,
            'v_coords': v_coords,
            'metrics': {
                'ise': float(ise),
                'mae': float(mae),
                'marginal_error': float(marginal_error)
            }
        })
    
    return results


def generate_visualizations(results, config):
    """Generate all requested visualizations."""
    output_dir = Path(config['visualization']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations in: {output_dir}")
    
    for result in results:
        name = result['name']
        
        if config['visualization']['generate_comparison_plots']:
            plot_comparison(
                result['density_pred'],
                result['density_true'],
                title=f"{name}",
                points=result['samples'],
                save_path=output_dir / f"{name}_comparison.png",
                metrics=result['metrics'],
                scale_mode='independent'
            )
            print(f"  Saved: {name}_comparison.png")
        
        if config['visualization']['generate_marginal_plots']:
            plot_marginals(
                result['density_pred'],
                title=f"{name} - Marginals",
                save_path=output_dir / f"{name}_marginals.png",
                row_coords=result['u_coords'],
                col_coords=result['v_coords']
            )
            print(f"  Saved: {name}_marginals.png")


def save_results(results, config):
    """Save evaluation results."""
    output_dir = Path(config['visualization']['output_dir'])
    
    if config['output']['save_metrics_json']:
        metrics_summary = [
            {
                'name': r['name'],
                'family': r['family'],
                'params': r['params'],
                'metrics': r['metrics']
            }
            for r in results
        ]
        
        with open(output_dir / 'metrics_summary.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\nSaved metrics to: {output_dir / 'metrics_summary.json'}")
    
    if config['output']['save_densities_npy']:
        for r in results:
            np.save(output_dir / f"{r['name']}_density_pred.npy", r['density_pred'])
            np.save(output_dir / f"{r['name']}_density_true.npy", r['density_true'])
        print(f"Saved density arrays to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run complete evaluation workflow")
    parser.add_argument('--config', required=True, help='Path to workflow config YAML')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=" * 80)
    print("Vine Diffusion Copula - Evaluation Workflow")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model: {config['model']['checkpoint']}")
    model = DiffusionCopulaModel.from_checkpoint(
        config['model']['checkpoint'],
        device=config['model']['device']
    )
    print("Model loaded successfully")
    
    # Evaluate
    results = []
    
    if config['evaluation']['data_source'] in ['parametric', 'both']:
        results.extend(evaluate_parametric_copulas(model, config))
    
    if config['evaluation']['data_source'] in ['real_data', 'both']:
        if config['evaluation']['real_data']['enabled']:
            print("\nReal data evaluation not yet implemented")
            print("See examples/use_pretrained_model.py for manual workflow")
    
    # Visualize
    if results:
        generate_visualizations(results, config)
    
    # Save results
    if results:
        save_results(results, config)
    
    # Summary
    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)
    print(f"\nResults saved to: {config['visualization']['output_dir']}")
    print(f"  - Comparison plots: {len(results)} copulas")
    print(f"  - Marginal plots: {len(results)} copulas")
    print(f"  - Metrics: metrics_summary.json")
    print(f"  - Densities: *_density_*.npy files")
    print("\nNext steps:")
    print("  - Review figures in results/evaluation/")
    print("  - Check metrics in metrics_summary.json")
    print("  - Use densities for vine copula construction (future extension)")


if __name__ == '__main__':
    main()

