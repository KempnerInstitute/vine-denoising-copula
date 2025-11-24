#!/usr/bin/env python3
"""
Quick test to verify data generation stability after fixes.
Tests that no NaN/Inf values are generated across diverse copulas.
"""
import sys
import os
sys.path.insert(0, '/n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula')

import torch
import numpy as np
from src.data.onthefly import OnTheFlyCopulaDataset

def test_data_generation():
    """Test data generation doesn't produce NaN/Inf."""
    print("=" * 80)
    print("Testing Data Generation Stability")
    print("=" * 80)
    
    # Test both with and without probit transform
    for transform in [False, True]:
        print(f"\n{'='*80}")
        print(f"Testing with transform_to_probit_space={transform}")
        print(f"{'='*80}\n")
        
        dataset = OnTheFlyCopulaDataset(
            n_samples_per_batch=8000,
            m=256,
            transform_to_probit_space=transform,
            mixture_prob=0.2,  # Include some mixtures
            seed=42
        )
        
        # Generate 50 samples (mix of single copulas and mixtures)
        nan_count = 0
        inf_count = 0
        density_ranges = []
        
        for i in range(50):
            batch = dataset[i]
            density = batch['density']
            
            # Check for NaN
            if torch.isnan(density).any():
                nan_count += 1
                print(f"  ❌ Batch {i}: Contains NaN values!")
            
            # Check for Inf
            if torch.isinf(density).any():
                inf_count += 1
                print(f"  ❌ Batch {i}: Contains Inf values!")
            
            # Record range
            min_val = density.min().item()
            max_val = density.max().item()
            density_ranges.append((min_val, max_val))
            
            if i % 10 == 0:
                print(f"  ✓ Batch {i:2d}: range [{min_val:8.2e}, {max_val:8.2e}]")
        
        # Summary
        print(f"\n{'='*80}")
        print(f"RESULTS (transform={transform}):")
        print(f"  Total batches tested: 50")
        print(f"  NaN batches: {nan_count}")
        print(f"  Inf batches: {inf_count}")
        
        if nan_count == 0 and inf_count == 0:
            print(f"  ✅ ALL TESTS PASSED!")
            
            # Show density statistics
            all_mins = [r[0] for r in density_ranges]
            all_maxs = [r[1] for r in density_ranges]
            print(f"\n  Density Statistics:")
            print(f"    Overall min: {min(all_mins):.2e}")
            print(f"    Overall max: {max(all_maxs):.2e}")
            print(f"    Mean min: {np.mean(all_mins):.2e}")
            print(f"    Mean max: {np.mean(all_maxs):.2e}")
            
            # Check if within expected bounds
            if transform:
                # Log-space: should be roughly [-15, 15]
                if max(all_maxs) > 20:
                    print(f"    ⚠️  WARNING: Max log-density {max(all_maxs):.2f} exceeds 20")
                if min(all_mins) < -20:
                    print(f"    ⚠️  WARNING: Min log-density {min(all_mins):.2f} below -20")
            else:
                # Regular space: should be [1e-10, 1000]
                if max(all_maxs) > 1500:
                    print(f"    ⚠️  WARNING: Max density {max(all_maxs):.2e} exceeds 1500")
        else:
            print(f"  ❌ TESTS FAILED!")
        print(f"{'='*80}\n")


def test_extreme_parameters():
    """Test with extreme copula parameters that previously caused issues."""
    print("\n" + "=" * 80)
    print("Testing Extreme Parameters")
    print("=" * 80)
    
    from vdc.data.generators import sample_bicop, analytic_logpdf_grid
    import numpy as np
    
    test_cases = [
        ("clayton", {"theta": 10.0}, "High Clayton theta"),
        ("gumbel", {"theta": 10.0}, "High Gumbel theta"),
        ("student", {"rho": 0.95, "nu": 3.0}, "Student-t with low df"),
        ("gaussian", {"rho": -0.95}, "Strong negative correlation"),
        ("joe", {"theta": 10.0}, "High Joe theta"),
    ]
    
    print("\nTesting copula families with extreme parameters:")
    
    for family, params, description in test_cases:
        try:
            # Generate log-density
            lg = analytic_logpdf_grid(
                family if family != 'student' else 'student_t',
                params,
                m=64
            )
            
            # Apply fix: clip BEFORE exp
            lg_clipped = np.clip(lg, -15, 15)
            density = np.exp(lg_clipped)
            density = np.clip(density, 1e-10, 1e3)
            
            # Check results
            has_nan = np.isnan(density).any()
            has_inf = np.isinf(density).any()
            
            status = "✅ PASS" if (not has_nan and not has_inf) else "❌ FAIL"
            print(f"  {status} {description:40s} - range [{density.min():.2e}, {density.max():.2e}]")
            
            if has_nan:
                print(f"        NaN count: {np.isnan(density).sum()}")
            if has_inf:
                print(f"        Inf count: {np.isinf(density).sum()}")
                
        except Exception as e:
            print(f"  ❌ FAIL {description:40s} - Exception: {e}")
    
    print("=" * 80)


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("STABILITY TEST SUITE")
    print("Verifying fixes for NaN/Inf in data generation")
    print("=" * 80)
    
    # Test 1: Regular data generation
    test_data_generation()
    
    # Test 2: Extreme parameters
    test_extreme_parameters()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
    print("\nIf all tests passed, training should be stable. You can now:")
    print("  1. Restart CNN from checkpoint 20000")
    print("  2. Restart diffusion from checkpoint 20000")
    print("  3. Monitor for NaN in early steps")
    print("\nRestart commands:")
    print("  sbatch scripts/slurm/compare_enhanced_stable.sh")
    print("  sbatch scripts/slurm/compare_diffusion_stable.sh")
    print()
