#!/usr/bin/env python
"""
Comprehensive model evaluation script.

Evaluates a trained copula density model on test cases with known ground truth.
Generates professional visualizations and detailed metrics.

Usage:
    # Evaluate on default test copulas
    python scripts/evaluate_model.py --checkpoint checkpoints/light_8gpu/model_20000.pt \
                                      --output results/evaluation
    
    # Evaluate on custom copulas
    python scripts/evaluate_model.py --checkpoint path/to/model.pt \
                                      --copulas gaussian,clayton,gumbel \
                                      --n-samples 2000
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import yaml
import json
from tqdm import tqdm
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vdc.models.unet_grid import GridUNet
from vdc.models.projection import copula_project
from vdc.data.generators import sample_bicop, analytic_logpdf_grid
from vdc.data.hist import points_to_histogram
from vdc.eval.pairs import evaluate_pair_copula
from vdc.eval.visualize import (
    plot_comparison,
    plot_marginals,
    plot_multi_comparison,
    plot_metrics_summary,
    create_paper_figure,
)
from vdc.vine.copula_diffusion import DiffusionCopulaModel


# Default test copulas with ground truth
DEFAULT_TEST_COPULAS = [
    {
        'family': 'gaussian',
        'params': {'rho': 0.7},
        'name': 'Gaussian(ρ=0.7)',
        'rotation': 0,
    },
    {
        'family': 'gaussian',
        'params': {'rho': -0.7},
        'name': 'Gaussian(ρ=-0.7)',
        'rotation': 0,
    },
    {
        'family': 'clayton',
        'params': {'theta': 3.0},
        'name': 'Clayton(θ=3.0)',
        'rotation': 0,
    },
    {
        'family': 'clayton',
        'params': {'theta': 5.0},
        'name': 'Clayton(θ=5.0, 90°)',
        'rotation': 90,
    },
    {
        'family': 'gumbel',
        'params': {'theta': 2.5},
        'name': 'Gumbel(θ=2.5)',
        'rotation': 0,
    },
    {
        'family': 'gumbel',
        'params': {'theta': 4.0},
        'name': 'Gumbel(θ=4.0)',
        'rotation': 0,
    },
    {
        'family': 'frank',
        'params': {'theta': 5.0},
        'name': 'Frank(θ=5.0)',
        'rotation': 0,
    },
    {
        'family': 'frank',
        'params': {'theta': -5.0},
        'name': 'Frank(θ=-5.0)',
        'rotation': 0,
    },
    {
        'family': 'joe',
        'params': {'theta': 3.0},
        'name': 'Joe(θ=3.0)',
        'rotation': 0,
    },
    {
        'family': 'student',
        'params': {'rho': 0.7, 'nu': 5.0},
        'name': 'Student-t(ρ=0.7, ν=5)',
        'rotation': 0,
    },
    {
        'family': 'bb1',
        'params': {'theta': 0.5, 'delta': 1.5},
        'name': 'BB1(θ=0.5, δ=1.5)',
        'rotation': 0,
    },
    {
        'family': 'bb7',
        'params': {'theta': 1.5, 'delta': 0.5},
        'name': 'BB7(θ=1.5, δ=0.5)',
        'rotation': 0,
    },
]


def load_model(checkpoint_path: Path, device: str = 'cuda') -> DiffusionCopulaModel:
    """
    Load a trained diffusion model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        DiffusionCopulaModel wrapper
    """
    print(f"Loading model from: {checkpoint_path}")
    
    model = DiffusionCopulaModel.from_checkpoint(str(checkpoint_path), device=device)
    
    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    return model


def predict_density(
    model: DiffusionCopulaModel,
    points: np.ndarray,
    m: int = 64,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Predict copula density from sample points using the diffusion model.
    
    Args:
        model: DiffusionCopulaModel wrapper
        points: (n, 2) sample points in [0,1]²
        m: Grid resolution
        device: Device (unused, model has device)
        
    Returns:
        (m, m) predicted density
    """
    # Use the DiffusionCopulaModel's estimation method
    density, _, _ = model.estimate_density_from_samples(
        points, m=m, projection_iters=15
    )
    return density


def evaluate_single_copula(
    model: DiffusionCopulaModel,
    copula_spec: Dict,
    n_samples: int = 2000,
    m: int = 64,
    device: str = 'cuda',
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Evaluate model on a single copula family.
    
    Args:
        model: Trained DiffusionCopulaModel
        copula_spec: Dict with 'family', 'params', 'name', 'rotation'
        n_samples: Number of samples to generate
        m: Grid resolution
        device: Device
        output_dir: Directory to save plots
        
    Returns:
        Dict with results including metrics and densities
    """
    family = copula_spec['family']
    params = copula_spec['params']
    name = copula_spec['name']
    rotation = copula_spec.get('rotation', 0)
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")
    
    # Generate samples
    print(f"Generating {n_samples} samples...")
    points = sample_bicop(family, params, n_samples, rotation=rotation)
    
    # Compute true density
    print("Computing true density...")
    density_true = analytic_logpdf_grid(family, params, m=m, rotation=rotation)
    density_true = np.exp(density_true)
    
    # Normalize true density (should already be normalized, but ensure)
    du = 1.0 / m
    mass_true = np.sum(density_true) * du * du
    if abs(mass_true - 1.0) > 0.01:
        print(f"Warning: True density mass = {mass_true:.4f}, normalizing...")
        density_true = density_true / mass_true
    
    # Predict density
    print("Predicting density with model...")
    density_pred = predict_density(model, points, m=m, device=device)
    
    # Compute metrics directly (avoiding evaluate_pair_copula since we already have density)
    print("Computing metrics...")
    metrics = {}
    du = 1.0 / m
    
    # 1. Kendall's tau
    from vdc.utils.stats import kendall_tau as kd_tau
    metrics['tau_data'] = kd_tau(points[:, 0], points[:, 1])
    
    # 2. ISE (if ground truth available)
    if density_true is not None:
        ise = np.mean((density_pred - density_true) ** 2) * du * du
        metrics['ise'] = ise
    
    # 3. Marginal uniformity check
    marginal_u = np.sum(density_pred, axis=1) * du  # Should be ~1
    marginal_v = np.sum(density_pred, axis=0) * du  # Should be ~1
    metrics['marginal_u_error'] = np.mean(np.abs(marginal_u - 1.0))
    metrics['marginal_v_error'] = np.mean(np.abs(marginal_v - 1.0))
    
    # 4. Mass conservation
    total_mass = np.sum(density_pred) * du * du
    metrics['total_mass'] = total_mass
    metrics['mass_error'] = abs(total_mass - 1.0)
    
    # 5. NLL on test points (interpolate density and compute)
    from vdc.losses import nll_points
    D_hat_t = torch.from_numpy(density_pred).float().unsqueeze(0).unsqueeze(0).to(device)
    points_t = torch.from_numpy(points).float().unsqueeze(0).to(device)
    nll = nll_points(D_hat_t, points_t)
    metrics['nll'] = nll.item()
    
    # Print metrics
    print(f"\nMetrics:")
    print(f"  ISE:           {metrics.get('ise', -1):.6f}")
    print(f"  NLL:           {metrics.get('nll', -1):.6f}")
    print(f"  Mass Error:    {metrics.get('mass_error', -1):.6f}")
    print(f"  Marginal U:    {metrics.get('marginal_u_error', -1):.6f}")
    print(f"  Marginal V:    {metrics.get('marginal_v_error', -1):.6f}")
    print(f"  Tau Error:     {abs(metrics.get('tau_data', 0) - metrics.get('tau_pred', metrics.get('tau_data', 0))):.6f}")
    
    # Generate visualizations if output_dir provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize name for filename
        safe_name = name.replace('(', '_').replace(')', '').replace('=', '').replace(' ', '_').replace(',', '')
        
        # Comparison plot
        print("Creating comparison plot...")
        plot_comparison(
            density_pred,
            density_true,
            title=name,
            points=points,
            save_path=output_dir / f"{safe_name}_comparison.png",
            metrics=metrics,
        )
        
        # Marginals plot
        print("Creating marginals plot...")
        plot_marginals(
            density_pred,
            title=f"{name} - Marginals",
            save_path=output_dir / f"{safe_name}_marginals.png",
        )
        
        # Paper figure (comprehensive)
        print("Creating paper figure...")
        create_paper_figure(
            density_pred,
            density_true,
            points,
            title=name,
            save_path=output_dir / f"{safe_name}_paper.png",
            metrics=metrics,
        )
    
    return {
        'name': name,
        'family': family,
        'params': params,
        'rotation': rotation,
        'density_pred': density_pred,
        'density_true': density_true,
        'points': points,
        'metrics': metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained copula model')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/evaluation'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--copulas',
        type=str,
        default='all',
        help='Comma-separated list of copula families to test (or "all" for defaults)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=2000,
        help='Number of samples per copula'
    )
    parser.add_argument(
        '--m',
        type=int,
        default=64,
        help='Grid resolution'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--save-data',
        action='store_true',
        help='Save predicted and true densities as .npy files'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {args.output}")
    
    # Load model
    model = load_model(args.checkpoint, device=args.device)
    
    # Determine which copulas to test
    if args.copulas == 'all':
        test_copulas = DEFAULT_TEST_COPULAS
    else:
        # Parse comma-separated list
        families = [f.strip() for f in args.copulas.split(',')]
        test_copulas = [c for c in DEFAULT_TEST_COPULAS if c['family'] in families]
    
    print(f"\nTesting on {len(test_copulas)} copula configurations")
    
    # Evaluate each copula
    results = []
    for copula_spec in tqdm(test_copulas, desc="Evaluating copulas"):
        result = evaluate_single_copula(
            model,
            copula_spec,
            n_samples=args.n_samples,
            m=args.m,
            device=args.device,
            output_dir=args.output / 'individual',
        )
        results.append(result)
    
    # Create summary visualizations
    print(f"\n{'='*60}")
    print("Creating summary visualizations...")
    print(f"{'='*60}")
    
    # Multi-comparison plot
    plot_multi_comparison(
        results,
        save_path=args.output / 'summary_comparison.png',
        ncols=3,
    )
    
    # Metrics summary
    metrics_dict = {}
    for metric_name in ['ise', 'nll', 'mass_error', 'marginal_u_error', 'marginal_v_error']:
        metrics_dict[metric_name] = [r['metrics'].get(metric_name, 0) for r in results]
    
    plot_metrics_summary(
        metrics_dict,
        save_path=args.output / 'metrics_summary.png',
    )
    
    # Save metrics to JSON
    metrics_summary = []
    for result in results:
        metrics_summary.append({
            'name': result['name'],
            'family': result['family'],
            'params': result['params'],
            'rotation': result['rotation'],
            'metrics': result['metrics'],
        })
    
    with open(args.output / 'metrics.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2, default=float)
    
    print(f"✓ Saved metrics to: {args.output / 'metrics.json'}")
    
    # Compute and display overall statistics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    
    ise_values = [r['metrics']['ise'] for r in results]
    nll_values = [r['metrics']['nll'] for r in results]
    mass_errors = [r['metrics']['mass_error'] for r in results]
    
    print(f"\nISE (Integrated Squared Error):")
    print(f"  Mean:   {np.mean(ise_values):.6f}")
    print(f"  Median: {np.median(ise_values):.6f}")
    print(f"  Std:    {np.std(ise_values):.6f}")
    print(f"  Min:    {np.min(ise_values):.6f}")
    print(f"  Max:    {np.max(ise_values):.6f}")
    
    print(f"\nNLL (Negative Log-Likelihood):")
    print(f"  Mean:   {np.mean(nll_values):.6f}")
    print(f"  Median: {np.median(nll_values):.6f}")
    
    print(f"\nMass Conservation Error:")
    print(f"  Mean:   {np.mean(mass_errors):.6f}")
    print(f"  Median: {np.median(mass_errors):.6f}")
    print(f"  Max:    {np.max(mass_errors):.6f}")
    
    # Save data if requested
    if args.save_data:
        print(f"\nSaving density data...")
        data_dir = args.output / 'data'
        data_dir.mkdir(exist_ok=True)
        
        for result in results:
            safe_name = result['name'].replace('(', '_').replace(')', '').replace('=', '').replace(' ', '_').replace(',', '')
            np.save(data_dir / f"{safe_name}_pred.npy", result['density_pred'])
            np.save(data_dir / f"{safe_name}_true.npy", result['density_true'])
            np.save(data_dir / f"{safe_name}_points.npy", result['points'])
        
        print(f"✓ Saved data to: {data_dir}")
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
