#!/usr/bin/env python3
"""
Example: Using a pretrained diffusion copula model for inference.

This shows how external users can load a trained model and use it
for copula density estimation and vine construction.
"""
import numpy as np
from pathlib import Path

# Import the main API
from vdc.vine.copula_diffusion import DiffusionCopulaModel


def main():
    """Demonstrate using a pretrained diffusion copula model."""
    
    print("=" * 80)
    print("Using Pretrained Diffusion Copula Model")
    print("=" * 80)
    
    # Step 1: Load a pretrained checkpoint
    print("\n1. Loading pretrained model...")
    checkpoint_path = "checkpoints/validate_no_probit_diffusion_m128/model_step_5000.pt"
    
    model = DiffusionCopulaModel.from_checkpoint(
        checkpoint_path,
        device='cuda'  # or 'cpu' if no GPU
    )
    print(f"✓ Loaded model from: {checkpoint_path}")
    
    # Step 2: Generate some bivariate data (already in [0,1]² after marginal CDFs)
    print("\n2. Generating pseudo-observations...")
    np.random.seed(42)
    
    # Example: Gaussian copula data (rho=0.7)
    from scipy.stats import norm
    rho = 0.7
    n_samples = 5000
    
    # Generate correlated normals
    z = np.random.randn(n_samples, 2)
    z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
    
    # Transform to uniform [0,1]²
    u = norm.cdf(z)
    
    print(f"✓ Generated {n_samples} pseudo-observations")
    print(f"  Shape: {u.shape}")
    print(f"  Range: [{u.min():.3f}, {u.max():.3f}]")
    
    # Step 3: Estimate copula density
    print("\n3. Estimating copula density...")
    density, row_coords, col_coords = model.estimate_density_from_samples(
        u,
        m=128,  # Grid resolution
        projection_iters=15
    )
    
    print(f"✓ Estimated density on {len(row_coords)}×{len(col_coords)} grid")
    print(f"  Density range: [{density.min():.6f}, {density.max():.6f}]")
    print(f"  Total mass: {(density * (1/128)**2).sum():.6f}")
    
    # Step 4: Compute h-functions for vine construction
    print("\n4. Computing h-functions...")
    h1, h2 = model.h_functions_from_grid(density)
    
    print(f"✓ Computed conditional CDFs")
    print(f"  h1(u|v) shape: {h1.shape}")
    print(f"  h2(v|u) shape: {h2.shape}")
    print(f"  h1 range: [{h1.min():.3f}, {h1.max():.3f}]")
    print(f"  h2 range: [{h2.min():.3f}, {h2.max():.3f}]")
    
    # Step 5: Sample from the estimated copula
    print("\n5. Sampling from estimated copula...")
    new_samples = model.sample_from_density(
        density,
        n_samples=1000
    )
    
    print(f"✓ Generated {len(new_samples)} new samples")
    print(f"  Sample shape: {new_samples.shape}")
    print(f"  Sample range: [{new_samples.min():.3f}, {new_samples.max():.3f}]")
    
    # Step 6: Use in a vine copula (example)
    print("\n6. Example: Integrating with vine copula...")
    print("""
    # Pseudo-code for using in a vine:
    
    from your_vine_library import VineCopula
    
    # Create a 3-variable C-vine
    vine = VineCopula(structure='C-vine', dimension=3)
    
    # Tree 1: Add bivariate copulas
    vine.add_pair_copula(
        level=1, edge=(0,1),
        copula=model  # Your diffusion copula!
    )
    
    # Tree 2: Conditional copulas using h-functions
    # h1, h2 from model.h_functions_from_grid(density)
    vine.add_conditional_copula(...)
    
    # Sample from the full vine
    vine_samples = vine.sample(n_samples=10000)
    """)
    
    print("\n" + "=" * 80)
    print("✓ Example complete!")
    print("=" * 80)
    print("\nKey takeaways:")
    print("  1. Load model with: DiffusionCopulaModel.from_checkpoint()")
    print("  2. Estimate density: model.estimate_density_from_samples()")
    print("  3. Get h-functions: model.h_functions_from_grid()")
    print("  4. Sample: model.sample_from_density()")
    print("\nSee vdc/vine/copula_diffusion.py for full API documentation.")


if __name__ == '__main__':
    main()

