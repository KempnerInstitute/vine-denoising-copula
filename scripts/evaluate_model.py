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
    smooth_sigma: float = 0.0,
    num_diffusion_steps: int = 50,
    cfg_scale: float = 2.0,
    adaptive_cfg: bool = False,
    num_ensemble: int = 1,
    ensemble_mode: str = "geometric",
    use_cfg: bool = True,
    # Legacy parameter (ignored when use_cfg=True)
    noise_step: int = 300,
) -> np.ndarray:
    """
    Predict copula density from sample points using the diffusion model.
    
    Args:
        model: DiffusionCopulaModel wrapper
        points: (n, 2) sample points in [0,1]²
        m: Grid resolution
        device: Device (unused, model has device)
        smooth_sigma: Gaussian smoothing sigma (0 = no smoothing, CFG is already smooth)
        num_diffusion_steps: Number of reverse diffusion steps (50 is usually enough)
        cfg_scale: Classifier-Free Guidance scale (>1 = stronger conditioning)
        adaptive_cfg: If True, automatically adjust CFG based on histogram properties
        num_ensemble: Number of independent inferences to average (more = smoother/robust)
        ensemble_mode: How to aggregate: "geometric", "arithmetic", or "median"
        use_cfg: Whether to use CFG sampling (True for V2 models)
        
    Returns:
        (m, m) predicted density
    """
    # Use the DiffusionCopulaModel's estimation method with CFG
    # Use 100 projection iterations for better marginal uniformity
    density, _, _ = model.estimate_density_from_samples(
        points, 
        m=m, 
        projection_iters=100,
        smooth_sigma=smooth_sigma,
        num_diffusion_steps=num_diffusion_steps,
        cfg_scale=cfg_scale,
        adaptive_cfg=adaptive_cfg,
        num_ensemble=num_ensemble,
        ensemble_mode=ensemble_mode,
        use_cfg=use_cfg,
        noise_step=noise_step,
    )
    return density


def evaluate_single_copula(
    model: DiffusionCopulaModel,
    copula_spec: Dict,
    n_samples: int = 2000,
    m: int = 64,
    device: str = 'cuda',
    output_dir: Optional[Path] = None,
    smooth_sigma: float = 0.0,
    num_diffusion_steps: int = 50,
    cfg_scale: float = 2.0,
    adaptive_cfg: bool = False,
    num_ensemble: int = 1,
    ensemble_mode: str = "geometric",
    use_cfg: bool = True,
    noise_step: int = 300,
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
    density_true_raw = analytic_logpdf_grid(family, params, m=m, rotation=rotation)
    density_true_raw = np.exp(density_true_raw)
    
    # Project true density through IPFP to make it a valid copula density
    # This is necessary because copulas with singularities (Clayton, Gumbel, Joe)
    # have infinite density at corners, which breaks the uniform marginal property
    # on a finite grid. IPFP projection makes fair comparison possible.
    # Use 100 iterations for better convergence with peaked copulas.
    du = 1.0 / m
    density_true_t = torch.from_numpy(density_true_raw).float().unsqueeze(0).unsqueeze(0).to(device)
    density_true_proj = copula_project(density_true_t, iters=100)
    density_true = density_true_proj[0, 0].cpu().numpy()
    
    # Check normalization
    mass_true = np.sum(density_true) * du * du
    if abs(mass_true - 1.0) > 0.01:
        print(f"Note: True density mass = {mass_true:.4f} after projection")
        density_true = density_true / mass_true
    
    # Predict density
    cfg_info = "adaptive" if adaptive_cfg else f"cfg_scale={cfg_scale}"
    print(f"Predicting density with model (ensemble={num_ensemble}, {cfg_info})...")
    density_pred = predict_density(
        model, points, m=m, device=device,
        smooth_sigma=smooth_sigma,
        num_diffusion_steps=num_diffusion_steps,
        cfg_scale=cfg_scale,
        adaptive_cfg=adaptive_cfg,
        num_ensemble=num_ensemble,
        ensemble_mode=ensemble_mode,
        use_cfg=use_cfg,
        noise_step=noise_step,
    )
    
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
    
    # 6. Peak values comparison
    metrics['peak_pred'] = float(density_pred.max())
    metrics['peak_true'] = float(density_true.max())
    metrics['peak_ratio'] = metrics['peak_pred'] / max(metrics['peak_true'], 1e-10)
    
    # Print metrics
    print(f"\nMetrics:")
    print(f"  ISE:           {metrics.get('ise', -1):.6f}")
    print(f"  NLL:           {metrics.get('nll', -1):.6f}")
    print(f"  Mass Error:    {metrics.get('mass_error', -1):.6f}")
    print(f"  Marginal U:    {metrics.get('marginal_u_error', -1):.6f}")
    print(f"  Marginal V:    {metrics.get('marginal_v_error', -1):.6f}")
    print(f"  Peak Pred:     {metrics.get('peak_pred', -1):.2f}")
    print(f"  Peak True:     {metrics.get('peak_true', -1):.2f}")
    print(f"  Peak Ratio:    {metrics.get('peak_ratio', -1):.4f}")
    print(f"  Tau Error:     {abs(metrics.get('tau_data', 0) - metrics.get('tau_pred', metrics.get('tau_data', 0))):.6f}")
    
    # Generate visualizations if output_dir provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize name for filename
        safe_name = name.replace('(', '_').replace(')', '').replace('=', '').replace(' ', '_').replace(',', '')
        
        # Comparison plot - use shared color scale for fair visual comparison
        print("Creating comparison plot...")
        plot_comparison(
            density_pred,
            density_true,
            title=name,
            points=points,
            save_path=output_dir / f"{safe_name}_comparison.png",
            metrics=metrics,
            scale_mode="shared",  # Use same color scale for pred and true
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
    parser.add_argument(
        '--smooth-sigma',
        type=float,
        default=0.0,
        help='Gaussian smoothing sigma (0 = no smoothing, CFG is already smooth)'
    )
    parser.add_argument(
        '--num-diffusion-steps',
        type=int,
        default=50,
        help='Number of reverse diffusion steps (50 is usually enough for CFG)'
    )
    parser.add_argument(
        '--cfg-scale',
        type=float,
        default=2.0,
        help='Classifier-Free Guidance scale (>1 = stronger conditioning). Ignored if --adaptive-cfg is set.'
    )
    parser.add_argument(
        '--adaptive-cfg',
        action='store_true',
        help='Use adaptive CFG based on histogram properties (recommended)'
    )
    parser.add_argument(
        '--num-ensemble',
        type=int,
        default=1,
        help='Number of independent inferences to average (recommended: 3-5 for best quality)'
    )
    parser.add_argument(
        '--ensemble-mode',
        type=str,
        choices=['geometric', 'arithmetic', 'median'],
        default='geometric',
        help='How to aggregate ensemble: geometric (log-avg), arithmetic (mean), median'
    )
    parser.add_argument(
        '--no-cfg',
        action='store_true',
        help='Use legacy denoising method instead of CFG (for older models)'
    )
    # Legacy parameter
    parser.add_argument(
        '--noise-step',
        type=int,
        default=300,
        help='(Legacy) Starting timestep for denoising - only used with --no-cfg'
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
    
    # Print inference parameters
    use_cfg = not args.no_cfg
    print(f"\nInference parameters:")
    print(f"  Use CFG: {use_cfg}")
    if use_cfg:
        if args.adaptive_cfg:
            print(f"  CFG scale: ADAPTIVE (auto-adjusted based on histogram)")
        else:
            print(f"  CFG scale: {args.cfg_scale}")
    else:
        print(f"  Noise step (legacy): {args.noise_step}")
    print(f"  Diffusion steps: {args.num_diffusion_steps}")
    print(f"  Smooth sigma: {args.smooth_sigma}")
    print(f"  Ensemble size: {args.num_ensemble}")
    print(f"  Ensemble mode: {args.ensemble_mode}")
    
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
            smooth_sigma=args.smooth_sigma,
            num_diffusion_steps=args.num_diffusion_steps,
            cfg_scale=args.cfg_scale,
            adaptive_cfg=args.adaptive_cfg,
            num_ensemble=args.num_ensemble,
            ensemble_mode=args.ensemble_mode,
            use_cfg=use_cfg,
            noise_step=args.noise_step,
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
