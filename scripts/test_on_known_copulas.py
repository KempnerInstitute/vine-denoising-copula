#!/usr/bin/env python
"""
Quick test script for copula density estimation on known families.

Tests the model on a small set of copulas with known ground truth.
Useful for quick sanity checks during/after training.

Usage:
    # Test on checkpoint
    python scripts/test_on_known_copulas.py --checkpoint checkpoints/light_8gpu/model_10000.pt
    
    # Test specific families
    python scripts/test_on_known_copulas.py --checkpoint path/to/model.pt \
                                             --families gaussian clayton gumbel
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
from scipy.stats import kendalltau

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vdc.data.generators import sample_bicop, analytic_logpdf_grid
from vdc.data.hist import points_to_histogram
from vdc.models.unet_grid import GridUNet
from vdc.models.projection import copula_project


# Test suite: simple cases with clear ground truth
TEST_SUITE = {
    'gaussian': [
        {'params': {'rho': 0.5}, 'name': 'Gaussian(ρ=0.5)'},
        {'params': {'rho': 0.8}, 'name': 'Gaussian(ρ=0.8)'},
        {'params': {'rho': -0.5}, 'name': 'Gaussian(ρ=-0.5)'},
    ],
    'clayton': [
        {'params': {'theta': 2.0}, 'name': 'Clayton(θ=2.0)'},
        {'params': {'theta': 5.0}, 'name': 'Clayton(θ=5.0)'},
    ],
    'gumbel': [
        {'params': {'theta': 2.0}, 'name': 'Gumbel(θ=2.0)'},
        {'params': {'theta': 3.5}, 'name': 'Gumbel(θ=3.5)'},
    ],
    'frank': [
        {'params': {'theta': 5.0}, 'name': 'Frank(θ=5.0)'},
        {'params': {'theta': -5.0}, 'name': 'Frank(θ=-5.0)'},
    ],
    'joe': [
        {'params': {'theta': 2.5}, 'name': 'Joe(θ=2.5)'},
    ],
}


def load_model_simple(checkpoint_path: Path, device: str) -> GridUNet:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if 'config' in checkpoint:
        model_config = checkpoint['config'].get('model', {})
    else:
        model_config = {'m': 64, 'base_channels': 96}
    
    model = GridUNet(**model_config).to(device)
    
    # Load weights
    if 'model_ema' in checkpoint:
        model.load_state_dict(checkpoint['model_ema'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def compute_ise(pred: np.ndarray, true: np.ndarray, m: int) -> float:
    """Compute Integrated Squared Error."""
    du = 1.0 / m
    return np.sum((pred - true)**2) * du * du


def compute_metrics(
    density_pred: np.ndarray,
    density_true: np.ndarray,
    points: np.ndarray,
    m: int,
) -> dict:
    """Compute evaluation metrics."""
    du = 1.0 / m
    
    # ISE
    ise = compute_ise(density_pred, density_true, m)
    
    # Mass conservation
    mass_pred = np.sum(density_pred) * du * du
    mass_error = abs(mass_pred - 1.0)
    
    # Marginal errors
    marginal_u = np.sum(density_pred, axis=1) * du
    marginal_v = np.sum(density_pred, axis=0) * du
    marginal_error = np.mean(np.abs(marginal_u - 1.0)) + np.mean(np.abs(marginal_v - 1.0))
    
    # Kendall's tau
    tau_data, _ = kendalltau(points[:, 0], points[:, 1])
    
    return {
        'ise': ise,
        'mass_error': mass_error,
        'marginal_error': marginal_error / 2.0,  # Average of u and v
        'tau': tau_data,
    }


def test_copula(
    model: GridUNet,
    family: str,
    params: dict,
    name: str,
    n_samples: int = 1000,
    m: int = 64,
    device: str = 'cuda',
) -> dict:
    """Test model on a single copula."""
    # Generate data
    points = sample_bicop(family, params, n_samples)
    
    # True density
    log_density_true = analytic_logpdf_grid(family, params, m=m)
    density_true = np.exp(log_density_true)
    
    # Histogram
    hist = points_to_histogram(points, m=m)
    hist_t = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        t = torch.ones(1, 1, 1, 1, device=device) * 0.5
        logD = model(hist_t, t)
        D = torch.exp(logD)
        D_proj = copula_project(D)
    
    density_pred = D_proj[0, 0].cpu().numpy()
    
    # Metrics
    metrics = compute_metrics(density_pred, density_true, points, m)
    
    return {
        'name': name,
        'family': family,
        'params': params,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='Quick copula test')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Model checkpoint')
    parser.add_argument('--families', nargs='+', default=None, help='Families to test')
    parser.add_argument('--n-samples', type=int, default=1000, help='Samples per copula')
    parser.add_argument('--m', type=int, default=64, help='Grid resolution')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.checkpoint}")
    model = load_model_simple(args.checkpoint, args.device)
    print("✓ Model loaded\n")
    
    # Determine families to test
    if args.families:
        families = args.families
    else:
        families = list(TEST_SUITE.keys())
    
    # Run tests
    results = []
    print(f"{'Family':<12} {'Name':<25} {'ISE':<10} {'Mass Err':<10} {'Marg Err':<10} {'τ':<8}")
    print("="*85)
    
    for family in families:
        if family not in TEST_SUITE:
            print(f"Warning: {family} not in test suite, skipping")
            continue
        
        for test_case in TEST_SUITE[family]:
            result = test_copula(
                model,
                family,
                test_case['params'],
                test_case['name'],
                n_samples=args.n_samples,
                m=args.m,
                device=args.device,
            )
            results.append(result)
            
            # Print row
            print(f"{family:<12} {result['name']:<25} "
                  f"{result['ise']:<10.6f} {result['mass_error']:<10.6f} "
                  f"{result['marginal_error']:<10.6f} {result['tau']:<8.4f}")
    
    # Summary statistics
    if results:
        print("\n" + "="*85)
        print("SUMMARY STATISTICS")
        print("="*85)
        
        ise_values = [r['ise'] for r in results]
        mass_errors = [r['mass_error'] for r in results]
        marg_errors = [r['marginal_error'] for r in results]
        
        print(f"\nISE:                Mean={np.mean(ise_values):.6f}  "
              f"Median={np.median(ise_values):.6f}  Max={np.max(ise_values):.6f}")
        print(f"Mass Error:         Mean={np.mean(mass_errors):.6f}  "
              f"Median={np.median(mass_errors):.6f}  Max={np.max(mass_errors):.6f}")
        print(f"Marginal Error:     Mean={np.mean(marg_errors):.6f}  "
              f"Median={np.median(marg_errors):.6f}  Max={np.max(marg_errors):.6f}")
        
        # Pass/fail criteria
        print("\n" + "-"*85)
        print("QUALITY ASSESSMENT")
        print("-"*85)
        
        ise_pass = np.mean(ise_values) < 0.05
        mass_pass = np.mean(mass_errors) < 0.01
        marg_pass = np.mean(marg_errors) < 0.01
        
        print(f"ISE < 0.05:             {'✓ PASS' if ise_pass else '✗ FAIL'}")
        print(f"Mass Error < 0.01:      {'✓ PASS' if mass_pass else '✗ FAIL'}")
        print(f"Marginal Error < 0.01:  {'✓ PASS' if marg_pass else '✗ FAIL'}")
        
        all_pass = ise_pass and mass_pass and marg_pass
        print("\n" + "="*85)
        if all_pass:
            print("OVERALL: ✓ PASS - Model performs well!")
        else:
            print("OVERALL: ✗ NEEDS IMPROVEMENT - Continue training or check model")
        print("="*85)
    
    print(f"\nTested {len(results)} copula configurations")


if __name__ == '__main__':
    main()
