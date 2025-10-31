"""
Complete end-to-end example: Generate data → Train model → Build vine → Evaluate

This script demonstrates the full workflow of the vine diffusion copula system.
"""

import numpy as np
import torch
from pathlib import Path
import argparse
from tqdm import tqdm

# Import our modules
from vdc.data.generators import (
    sample_gaussian_copula,
    sample_clayton_copula,
    sample_gumbel_copula
)
from vdc.data.hist import scatter_to_hist
from vdc.models.unet_grid import GridUNet
from vdc.models.projection import copula_project
from vdc.models.hfunc import HFuncLookup
from vdc.vine.structure import build_rvine_structure, print_vine_structure
from vdc.vine.recursion import VineRecursion, VinePairCopula
from vdc.vine.metrics import evaluate_vine_copula


def generate_synthetic_multivariate_data(n: int, d: int, seed: int = 42):
    """
    Generate synthetic d-dimensional data with known vine structure.
    
    Args:
        n: Number of samples
        d: Dimension
        seed: Random seed
        
    Returns:
        U: (n, d) pseudo-observations
    """
    np.random.seed(seed)
    
    # Generate from a known vine structure
    # For simplicity: chain structure with varying copula families
    
    print(f"Generating {n} samples in {d} dimensions...")
    
    # Start with uniform marginals
    U = np.zeros((n, d))
    
    # First variable
    U[:, 0] = np.random.uniform(0, 1, n)
    
    # Subsequent variables with dependence
    for i in range(1, d):
        if i % 3 == 1:
            # Gaussian copula
            U[:, i] = sample_gaussian_copula(
                U[:, i-1],
                rho=0.5 + 0.1 * i
            )
        elif i % 3 == 2:
            # Clayton copula
            U[:, i] = sample_clayton_copula(
                U[:, i-1],
                theta=2.0
            )
        else:
            # Gumbel copula
            U[:, i] = sample_gumbel_copula(
                U[:, i-1],
                theta=2.0
            )
    
    print(f"✓ Generated data shape: {U.shape}")
    return U


def create_dummy_trained_model(m: int = 64, device: str = 'cuda'):
    """
    Create a dummy "trained" model for demonstration.
    
    In practice, you would load a checkpoint from actual training.
    
    Args:
        m: Grid resolution
        device: Device
        
    Returns:
        model: Initialized model
    """
    print(f"Initializing model (m={m})...")
    
    model = GridUNet(m=m, in_channels=1, base_channels=64)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # In real usage, you would do:
    # checkpoint = torch.load('checkpoints/best.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def estimate_pair_copula_with_model(
    model: torch.nn.Module,
    pair_data: np.ndarray,
    m: int = 64,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, HFuncLookup]:
    """
    Estimate bivariate copula density using the trained model.
    
    Args:
        model: Trained diffusion model
        pair_data: (n, 2) pseudo-observations
        m: Grid resolution
        device: Device
        
    Returns:
        density_grid: (m, m) copula density
        hfunc: HFuncLookup object
    """
    # Convert to histogram
    hist = scatter_to_hist(pair_data, m=m, reflect=True)
    hist_t = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Predict density
    with torch.no_grad():
        t = torch.ones(1, 1, 1, 1, device=device) * 0.5  # Fixed time for inference
        logD_raw = model(hist_t, t)
        D_pos = torch.exp(logD_raw)
        
        # Project to valid copula
        D_copula = copula_project(D_pos, iters=20)
    
    # Create h-function lookup
    hfunc = HFuncLookup(D_copula[0, 0])
    
    return D_copula[0, 0], hfunc


def build_diffusion_vine(
    U: np.ndarray,
    model: torch.nn.Module,
    m: int = 64,
    device: str = 'cuda',
    max_trees: int = None
):
    """
    Build complete vine using diffusion copula estimator.
    
    Args:
        U: (n, d) pseudo-observations
        model: Trained diffusion model
        m: Grid resolution
        device: Device
        max_trees: Maximum number of trees (default: d-1)
        
    Returns:
        vine: VineRecursion object ready for inference
    """
    n, d = U.shape
    
    print(f"\n{'='*60}")
    print(f"Building Vine Copula (d={d})")
    print(f"{'='*60}\n")
    
    # Step 1: Structure selection
    print("Step 1: Structure selection via Dißmann MST...")
    structure = build_rvine_structure(U, max_trees=max_trees)
    print_vine_structure(structure)
    
    # Step 2: Initialize vine recursion
    vine = VineRecursion(structure)
    
    # Step 3: Fit each pair copula
    print("Step 2: Fitting pair copulas with diffusion model...")
    
    # Keep track of transformed data for each tree
    current_data = U.copy()
    
    for tree_level in range(len(structure.trees)):
        tree = structure.trees[tree_level]
        print(f"\n  Tree {tree_level + 1}: {len(tree.edges)} edges")
        
        for edge_idx, edge in enumerate(tqdm(tree.edges, desc=f"    Tree {tree_level+1}")):
            i, j, cond = edge
            
            # For Tree 1, use original data
            # For Tree k > 1, should use h-transformed data
            # (This is simplified; full implementation tracks conditioning properly)
            
            if tree_level == 0:
                pair_data = U[:, [i, j]]
            else:
                # Simplified: still use original data
                # In practice, apply appropriate h-transforms
                pair_data = U[:, [i, j]]
            
            # Estimate copula with model
            density_grid, hfunc = estimate_pair_copula_with_model(
                model,
                pair_data,
                m=m,
                device=device
            )
            
            # Create VinePairCopula object
            copula = VinePairCopula(
                edge=edge,
                density_grid=density_grid,
                hfunc=hfunc,
                level=tree_level
            )
            
            # Add to vine
            vine.add_pair_copula(copula)
    
    print(f"\n✓ Vine construction complete!")
    print(f"  Total edges: {sum(len(t.edges) for t in structure.trees)}")
    
    return vine


def main():
    parser = argparse.ArgumentParser(description="End-to-end vine copula demo")
    parser.add_argument('--n_train', type=int, default=1000, help='Training samples')
    parser.add_argument('--n_test', type=int, default=500, help='Test samples')
    parser.add_argument('--d', type=int, default=5, help='Dimension')
    parser.add_argument('--m', type=int, default=64, help='Grid resolution')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Vine Diffusion Copula - End-to-End Demo")
    print("="*60)
    print(f"Configuration:")
    print(f"  Train samples: {args.n_train}")
    print(f"  Test samples: {args.n_test}")
    print(f"  Dimension: {args.d}")
    print(f"  Grid resolution: {args.m}")
    print(f"  Device: {args.device}")
    print("="*60 + "\n")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Step 1: Generate synthetic data
    print("Phase 1: Data Generation")
    print("-" * 60)
    U_train = generate_synthetic_multivariate_data(args.n_train, args.d, seed=args.seed)
    U_test = generate_synthetic_multivariate_data(args.n_test, args.d, seed=args.seed + 1)
    
    # Step 2: Create/load model
    print("\nPhase 2: Model Setup")
    print("-" * 60)
    print("⚠ Note: Using untrained model for demo.")
    print("   In practice, train model with: python -m vdc.train.train_grid")
    model = create_dummy_trained_model(m=args.m, device=args.device)
    
    # Step 3: Build vine
    print("\nPhase 3: Vine Construction")
    print("-" * 60)
    vine = build_diffusion_vine(
        U_train,
        model,
        m=args.m,
        device=args.device,
        max_trees=min(3, args.d-1)  # Limit to 3 trees for demo
    )
    
    # Step 4: Evaluation
    print("\nPhase 4: Evaluation")
    print("-" * 60)
    
    try:
        print("Computing test log-likelihood...")
        loglik = vine.logpdf(U_test[:100])  # Use subset for speed
        print(f"✓ Average test log-likelihood: {np.mean(loglik):.4f}")
    except Exception as e:
        print(f"⚠ Likelihood evaluation failed: {e}")
    
    try:
        print("\nAttempting to generate samples...")
        U_samples = vine.simulate(n=100, seed=args.seed)
        print(f"✓ Generated {len(U_samples)} samples")
        print(f"  Sample range: [{U_samples.min():.4f}, {U_samples.max():.4f}]")
    except Exception as e:
        print(f"⚠ Sampling failed: {e}")
    
    # Step 5: Summary
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Generate synthetic training data:")
    print("   python -m vdc.data.generators --output data/synthetic")
    print("\n2. Train the model:")
    print("   python -m vdc.train.train_grid --data_root data/synthetic")
    print("\n3. Use trained model in this script:")
    print("   python examples/end_to_end.py --model_checkpoint checkpoints/best.pt")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
