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
from typing import List, Dict
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


def load_model(checkpoint_path: Path, device: str = 'cuda') -> GridUNet:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_config = config.get('model', {})
    else:
        # Use defaults if config not in checkpoint
        model_config = {
            'm': 64,
            'base_channels': 96,
            'channel_mults': [1, 2, 3, 4],
            'num_res_blocks': 2,
            'attention_resolutions': [16, 8],
            'dropout': 0.1,
        }
        print("Warning: Config not in checkpoint, using defaults")
    
    # Create model
    model = GridUNet(**model_config).to(device)
    
    # Load weights (handle EMA if present)
    if 'model_ema' in checkpoint:
        print("Loading EMA weights")
        model.load_state_dict(checkpoint['model_ema'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if 'step' in checkpoint:
        print(f"  Training step: {checkpoint['step']:,}")
    
    return model


def predict_density(
    model: GridUNet,
    points: np.ndarray,
    m: int = 64,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Predict copula density from sample points.
    
    Args:
        model: Trained model
        points: (n, 2) sample points in [0,1]²
        m: Grid resolution
        device: Device
        
    Returns:
        (m, m) predicted density
    """
    # Create histogram
    hist = points_to_histogram(points, m=m)
    hist_t = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Predict with model
    with torch.no_grad():
        t = torch.ones(1, 1, 1, 1, device=device) * 0.5  # Use t=0.5 for prediction
        logD_raw = model(hist_t, t)
        D_pos = torch.exp(logD_raw)
        D_hat = copula_project(D_pos)
    
    density_pred = D_hat[0, 0].cpu().numpy()
    return density_pred


def evaluate_single_copula(
    model: GridUNet,
    copula_spec: Dict,
    n_samples: int = 2000,
    m: int = 64,
    device: str = 'cuda',
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Evaluate model on a single copula family.
    
    Args:
        model: Trained model
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
    
    # Create histogram for evaluation
    hist = points_to_histogram(points, m=m)
    hist_t = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = evaluate_pair_copula(
        model,
        hist_t,
        points,
        true_density=density_true,
        device=device,
        m=m,
    )
    
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
