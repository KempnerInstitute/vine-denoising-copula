"""
Complete example: Comparing D-vine, C-vine, and R-vine on the same data.

This demonstrates:
1. Fitting all three vine types
2. Comparing their structures
3. Evaluating performance
4. Sampling and validation
"""

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

from vdc.vine.api import fit_dvine, fit_cvine, fit_rvine, VineCopulaModel
from vdc.vine.vine_types import print_vine_comparison
from vdc.models.unet_grid import GridUNet
from vdc.vine.metrics import evaluate_vine_copula


def generate_test_data(n: int, d: int, seed: int = 42):
    """Generate synthetic test data with known dependence."""
    np.random.seed(seed)
    
    from scipy.stats import norm
    
    # Create correlation matrix
    rho = 0.6
    Sigma = np.eye(d)
    for i in range(d - 1):
        Sigma[i, i + 1] = rho
        Sigma[i + 1, i] = rho
    
    # Add some longer-range correlations
    if d > 3:
        Sigma[0, d-1] = 0.4
        Sigma[d-1, 0] = 0.4
    
    # Generate data
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    
    # Convert to pseudo-observations
    U = norm.cdf(Z)
    
    return U


def load_trained_model(checkpoint_path: str, m: int = 64, device: str = 'cuda'):
    """Load trained diffusion model."""
    model = GridUNet(m=m, in_channels=1, base_channels=64)
    
    if Path(checkpoint_path).exists():
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded")
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        print("  Using untrained model (for demo purposes)")
    
    model.eval()
    model.to(device)
    
    return model


def compare_vine_types(
    U_train: np.ndarray,
    U_test: np.ndarray,
    model: torch.nn.Module,
    m: int = 64,
    device: str = 'cuda'
):
    """Compare R-vine, D-vine, and C-vine on the same data."""
    
    n, d = U_train.shape
    
    print("\n" + "="*80)
    print(f"Comparing Vine Types on {d}-dimensional data")
    print("="*80)
    
    # Store results
    results = {}
    
    # ========== R-Vine ==========
    print("\n" + "-"*80)
    print("1. Fitting R-Vine (Dißmann MST - automatic structure selection)")
    print("-"*80)
    
    try:
        rvine = fit_rvine(U_train, model, m=m, device=device)
        results['R-Vine'] = {
            'model': rvine,
            'structure': rvine.structure,
        }
        print(f"✓ R-Vine fitted successfully")
        print(f"  Order: {rvine.structure.order}")
    except Exception as e:
        print(f"✗ R-Vine fitting failed: {e}")
        results['R-Vine'] = None
    
    # ========== D-Vine (Optimized) ==========
    print("\n" + "-"*80)
    print("2. Fitting D-Vine with optimized sequential ordering")
    print("-"*80)
    
    try:
        dvine_opt = fit_dvine(U_train, model, order=None, m=m, device=device)  # None = auto-optimize
        results['D-Vine (Optimized)'] = {
            'model': dvine_opt,
            'structure': dvine_opt.structure,
        }
        print(f"✓ D-Vine (optimized) fitted successfully")
        print(f"  Order: {dvine_opt.structure.order}")
    except Exception as e:
        print(f"✗ D-Vine (optimized) fitting failed: {e}")
        results['D-Vine (Optimized)'] = None
    
    # ========== D-Vine (Custom) ==========
    print("\n" + "-"*80)
    print("3. Fitting D-Vine with custom sequential ordering")
    print("-"*80)
    
    try:
        custom_order = list(range(d))  # Simple sequential: 0,1,2,3,...
        dvine_custom = fit_dvine(U_train, model, order=custom_order, m=m, device=device)
        results['D-Vine (Sequential)'] = {
            'model': dvine_custom,
            'structure': dvine_custom.structure,
        }
        print(f"✓ D-Vine (custom) fitted successfully")
        print(f"  Order: {dvine_custom.structure.order}")
    except Exception as e:
        print(f"✗ D-Vine (custom) fitting failed: {e}")
        results['D-Vine (Sequential)'] = None
    
    # ========== C-Vine (Optimized) ==========
    print("\n" + "-"*80)
    print("4. Fitting C-Vine with optimized root selection")
    print("-"*80)
    
    try:
        cvine_opt = fit_cvine(U_train, model, order=None, m=m, device=device)  # None = auto-optimize
        results['C-Vine (Optimized)'] = {
            'model': cvine_opt,
            'structure': cvine_opt.structure,
        }
        print(f"✓ C-Vine (optimized) fitted successfully")
        print(f"  Root order: {cvine_opt.structure.order}")
    except Exception as e:
        print(f"✗ C-Vine (optimized) fitting failed: {e}")
        results['C-Vine (Optimized)'] = None
    
    # ========== C-Vine (Custom) ==========
    print("\n" + "-"*80)
    print("5. Fitting C-Vine with custom root ordering")
    print("-"*80)
    
    try:
        # Choose middle variable as first root
        custom_roots = [d//2] + [i for i in range(d) if i != d//2]
        cvine_custom = fit_cvine(U_train, model, order=custom_roots, m=m, device=device)
        results['C-Vine (Custom Roots)'] = {
            'model': cvine_custom,
            'structure': cvine_custom.structure,
        }
        print(f"✓ C-Vine (custom) fitted successfully")
        print(f"  Root order: {cvine_custom.structure.order}")
    except Exception as e:
        print(f"✗ C-Vine (custom) fitting failed: {e}")
        results['C-Vine (Custom Roots)'] = None
    
    # ========== Compare Structures ==========
    print("\n" + "="*80)
    print("Structure Comparison")
    print("="*80)
    
    structures = [(name, r['structure']) for name, r in results.items() if r is not None]
    if structures:
        print_vine_comparison(structures)
    
    # ========== Evaluate on Test Data ==========
    print("\n" + "="*80)
    print("Performance Comparison on Test Data")
    print("="*80)
    
    comparison_table = []
    
    for name, result in results.items():
        if result is None:
            continue
        
        vine_model = result['model']
        
        try:
            # Compute test log-likelihood (use subset for speed)
            test_subset = U_test[:min(100, len(U_test))]
            loglik = vine_model.logpdf(test_subset)
            avg_loglik = np.mean(loglik)
            
            comparison_table.append({
                'Vine Type': name,
                'Num Trees': len(vine_model.structure.trees),
                'Num Edges': vine_model.structure.num_edges(),
                'Avg Log-Lik': f"{avg_loglik:.4f}",
            })
        
        except Exception as e:
            print(f"  {name}: Evaluation failed - {e}")
            comparison_table.append({
                'Vine Type': name,
                'Num Trees': len(vine_model.structure.trees),
                'Num Edges': vine_model.structure.num_edges(),
                'Avg Log-Lik': 'Failed',
            })
    
    # Print comparison table
    if comparison_table:
        print("\nResults:")
        print("-" * 80)
        header = comparison_table[0].keys()
        print(" | ".join(f"{k:20s}" for k in header))
        print("-" * 80)
        for row in comparison_table:
            print(" | ".join(f"{str(v):20s}" for v in row.values()))
        print("-" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare D-vine, C-vine, and R-vine")
    parser.add_argument('--n-train', type=int, default=500, help='Training samples')
    parser.add_argument('--n-test', type=int, default=200, help='Test samples')
    parser.add_argument('--d', type=int, default=5, help='Dimension')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model checkpoint')
    parser.add_argument('--m', type=int, default=64, help='Grid resolution')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Vine Copula Type Comparison")
    print("="*80)
    print(f"Configuration:")
    print(f"  Training samples: {args.n_train}")
    print(f"  Test samples: {args.n_test}")
    print(f"  Dimension: {args.d}")
    print(f"  Grid resolution: {args.m}")
    print(f"  Device: {args.device}")
    print("="*80)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("\n⚠ CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Generate data
    print("\nGenerating synthetic data...")
    U_train = generate_test_data(args.n_train, args.d, seed=args.seed)
    U_test = generate_test_data(args.n_test, args.d, seed=args.seed + 1)
    print(f"✓ Training data: {U_train.shape}")
    print(f"✓ Test data: {U_test.shape}")
    
    # Load model
    if args.model:
        model = load_trained_model(args.model, m=args.m, device=args.device)
    else:
        print("\n⚠ No model checkpoint provided (use --model path/to/checkpoint.pt)")
        print("  Creating untrained model for demonstration...")
        model = GridUNet(m=args.m, in_channels=1, base_channels=64)
        model.eval()
        model.to(args.device)
    
    # Compare vine types
    results = compare_vine_types(
        U_train,
        U_test,
        model,
        m=args.m,
        device=args.device
    )
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("\nVine Types Fitted:")
    for name, result in results.items():
        if result is not None:
            status = "✓"
            info = f"({result['structure'].num_edges()} edges)"
        else:
            status = "✗"
            info = "(failed)"
        print(f"  {status} {name:30s} {info}")
    
    print("\n" + "="*80)
    print("Key Differences:")
    print("="*80)
    print("""
    R-Vine (Regular Vine):
      - Most flexible: can capture any dependence structure
      - Uses MST on |τ| to select edges automatically
      - Best for unknown/complex dependencies
      - More edges = potentially better fit but slower
    
    D-Vine (Drawable Vine):
      - Sequential/path structure: (1,2), (2,3), (3,4), ...
      - Requires variable ordering
      - Fast to fit and evaluate
      - Good for time series or naturally ordered data
    
    C-Vine (Canonical Vine):
      - Star structure: root variables at each level
      - First root connects to all others
      - Good when some variables are "key drivers"
      - Useful for factor models or hierarchical data
    """)
    
    print("\nNext Steps:")
    print("-" * 80)
    print("1. Train diffusion model:")
    print("   python -m vdc.train.train_grid --data_root data/synthetic")
    print("\n2. Run this comparison with trained model:")
    print(f"   python examples/compare_vines.py --model checkpoints/best.pt --d {args.d}")
    print("\n3. Try different orderings:")
    print("   - D-vine: Experiment with different sequential orders")
    print("   - C-vine: Choose roots based on domain knowledge")
    print("-" * 80)


if __name__ == "__main__":
    main()
