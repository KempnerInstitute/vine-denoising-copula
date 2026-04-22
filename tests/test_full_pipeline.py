#!/usr/bin/env python3
"""
End-to-end test for the vine denoising copula pipeline.

Tests:
1. Bivariate copula density estimation from samples
2. H-function computation from density grids
3. H-function inverse (required for vine sampling)
4. Vine copula structure building
5. End-to-end vine fitting (if checkpoint available)
"""

import numpy as np
import torch
import pytest
from pathlib import Path
from scipy import stats as scipy_stats

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vdc.models.hfunc import HFuncLookup
from vdc.data.generators import sample_bicop, analytic_logpdf_grid
from vdc.data.hist import scatter_to_hist
from vdc.models.projection import copula_project, check_copula_constraints
from vdc.vine.structure import build_rvine_structure, print_vine_structure
from vdc.vine.vine_types import build_dvine_structure, build_cvine_structure


class TestHFunctions:
    """Test h-function computation and inversion."""
    
    def test_independence_copula(self):
        """For independence copula c(u,v)=1, h(u|v)=u and h(v|u)=v."""
        m = 64
        D_indep = np.ones((m, m))
        hfunc = HFuncLookup(D_indep)
        
        # Test h(u|v) = u for independence
        u_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        v_test = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
        
        h_uv = hfunc.h_u_given_v(u_test, v_test)
        h_vu = hfunc.h_v_given_u(v_test, u_test)
        
        # Should be approximately equal to u_test and v_test respectively
        np.testing.assert_allclose(h_uv, u_test, atol=0.05, 
            err_msg="h(u|v) should equal u for independence copula")
        np.testing.assert_allclose(h_vu, v_test, atol=0.05,
            err_msg="h(v|u) should equal v for independence copula")
    
    def test_monotonicity(self):
        """H-functions should be monotonically increasing."""
        m = 64
        
        # Test with Gaussian copula (strong correlation)
        rho = 0.7
        u_grid = np.linspace(0.5/m, 1 - 0.5/m, m)
        v_grid = np.linspace(0.5/m, 1 - 0.5/m, m)
        U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
        
        # Approximate Gaussian copula density
        Z_u = scipy_stats.norm.ppf(np.clip(U, 0.01, 0.99))
        Z_v = scipy_stats.norm.ppf(np.clip(V, 0.01, 0.99))
        
        exponent = (Z_u**2 + Z_v**2 - 2*rho*Z_u*Z_v) / (2*(1-rho**2))
        D_gauss = np.exp(-exponent + 0.5*(Z_u**2 + Z_v**2)) / np.sqrt(1 - rho**2)
        D_gauss = np.nan_to_num(D_gauss, nan=1.0, posinf=100.0, neginf=0.0)
        D_gauss = np.clip(D_gauss, 1e-10, 1e3)
        D_gauss = D_gauss / (D_gauss.sum() / m**2)
        
        hfunc = HFuncLookup(D_gauss)
        
        # Test monotonicity along u-axis for fixed v
        v_fixed = 0.5
        u_vals = np.linspace(0.05, 0.95, 20)
        v_vals = np.full_like(u_vals, v_fixed)
        
        h_vals = hfunc.h_u_given_v(u_vals, v_vals)
        
        # Should be increasing
        diffs = np.diff(h_vals)
        assert np.all(diffs >= -1e-6), f"h(u|v) not monotonic: min diff = {diffs.min()}"
        
        # Test monotonicity along v-axis for fixed u
        u_fixed = 0.5
        v_vals2 = np.linspace(0.05, 0.95, 20)
        u_vals2 = np.full_like(v_vals2, u_fixed)
        
        h_vals2 = hfunc.h_v_given_u(v_vals2, u_vals2)
        
        diffs2 = np.diff(h_vals2)
        assert np.all(diffs2 >= -1e-6), f"h(v|u) not monotonic: min diff = {diffs2.min()}"
    
    def test_inverse_roundtrip(self):
        """Test that h_inv(h(u|v), v) = u."""
        m = 64
        
        # Use Gaussian copula
        rho = 0.5
        u_grid = np.linspace(0.5/m, 1 - 0.5/m, m)
        v_grid = np.linspace(0.5/m, 1 - 0.5/m, m)
        U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
        
        Z_u = scipy_stats.norm.ppf(np.clip(U, 0.01, 0.99))
        Z_v = scipy_stats.norm.ppf(np.clip(V, 0.01, 0.99))
        
        exponent = (Z_u**2 + Z_v**2 - 2*rho*Z_u*Z_v) / (2*(1-rho**2))
        D_gauss = np.exp(-exponent + 0.5*(Z_u**2 + Z_v**2)) / np.sqrt(1 - rho**2)
        D_gauss = np.nan_to_num(D_gauss, nan=1.0, posinf=100.0, neginf=0.0)
        D_gauss = np.clip(D_gauss, 1e-10, 1e3)
        D_gauss = D_gauss / (D_gauss.sum() / m**2)
        
        hfunc = HFuncLookup(D_gauss)
        
        # Generate random test points
        np.random.seed(42)
        n_test = 100
        u_test = np.random.uniform(0.1, 0.9, n_test)
        v_test = np.random.uniform(0.1, 0.9, n_test)
        
        # Forward: h(u|v)
        h_uv = hfunc.h_u_given_v(u_test, v_test)
        
        # Inverse: should recover u
        u_recovered = hfunc.hinv_u_given_v(h_uv, v_test)
        
        # Check roundtrip error
        error = np.abs(u_recovered - u_test).mean()
        assert error < 0.1, f"Roundtrip error too large: {error:.4f}"
        print(f"Inverse roundtrip test passed (mean error: {error:.4f})")


class TestCopulaProjection:
    """Test Sinkhorn/IPFP projection to copula constraints."""
    
    def test_uniform_marginals(self):
        """Projected density should have uniform marginals."""
        m = 64
        
        # Create a non-uniform density
        torch.manual_seed(42)
        D_raw = torch.randn(1, 1, m, m).abs() + 0.1
        
        # Project
        D_proj = copula_project(D_raw, iters=50)
        
        # Check marginals
        du = dv = 1.0 / m
        
        # Row marginal: ∫c(u,v)dv = 1 for all u
        row_marginal = (D_proj * dv).sum(dim=-1)  # (B, 1, m)
        row_error = (row_marginal - 1.0).abs().max().item()
        
        # Col marginal: ∫c(u,v)du = 1 for all v  
        col_marginal = (D_proj * du).sum(dim=-2)  # (B, 1, m)
        col_error = (col_marginal - 1.0).abs().max().item()
        
        assert row_error < 0.01, f"Row marginal error too large: {row_error}"
        assert col_error < 0.01, f"Col marginal error too large: {col_error}"
        print(f"Projection test passed (row_err={row_error:.4f}, col_err={col_error:.4f})")
    
    def test_preserves_structure(self):
        """Projection should preserve dependence structure (roughly)."""
        m = 64
        
        # Create diagonal-biased density (like positive correlation)
        u = torch.linspace(0.5/m, 1-0.5/m, m)
        v = torch.linspace(0.5/m, 1-0.5/m, m)
        U, V = torch.meshgrid(u, v, indexing='ij')
        
        D_diag = torch.exp(-20 * (U - V)**2)  # Strong diagonal
        D_diag = D_diag.unsqueeze(0).unsqueeze(0)
        
        # Project
        D_proj = copula_project(D_diag, iters=50)
        
        # Diagonal should still be high (positive correlation preserved)
        diag_vals = torch.diag(D_proj[0, 0])
        off_diag = D_proj[0, 0].clone()
        for i in range(m):
            off_diag[i, i] = 0
        
        # Diagonal should be significantly higher than off-diagonal
        diag_mean = diag_vals.mean().item()
        off_diag_mean = off_diag.sum().item() / (m*m - m)
        
        assert diag_mean > off_diag_mean, \
            f"Diagonal ({diag_mean:.4f}) should be > off-diagonal ({off_diag_mean:.4f})"
        print(f"Structure preservation test passed (diag={diag_mean:.4f} > off_diag={off_diag_mean:.4f})")


class TestVineStructure:
    """Test vine structure construction."""
    
    def test_build_rvine(self):
        """Test R-vine construction."""
        np.random.seed(42)
        n, d = 500, 5
        
        # Generate correlated data
        rho = 0.5
        Sigma = np.eye(d)
        for i in range(d - 1):
            Sigma[i, i + 1] = rho
            Sigma[i + 1, i] = rho
        
        Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
        U = scipy_stats.norm.cdf(Z)
        
        # Build structure
        structure = build_rvine_structure(U)
        
        assert structure.d == d
        # R-vine can have up to d-1 trees (may be truncated or have fewer edges)
        assert len(structure.trees) >= 1, "Should have at least 1 tree"
        assert len(structure.trees) <= d - 1, f"Should have at most {d-1} trees"
        # Check that we have some edges
        assert structure.num_edges() >= 1, "Should have at least 1 edge"
        
        print(f"R-vine structure built: {len(structure.trees)} trees, {structure.num_edges()} edges")
    
    def test_build_dvine(self):
        """Test D-vine construction."""
        np.random.seed(42)
        n, d = 500, 4
        
        # Generate data
        Z = np.random.randn(n, d)
        U = scipy_stats.norm.cdf(Z)
        
        # Build D-vine
        structure = build_dvine_structure(U)
        
        assert structure.d == d
        assert structure.order is not None
        assert len(structure.order) == d
        
        # D-vine: Tree 1 should have d-1 edges in path structure
        if len(structure.trees) > 0:
            tree1 = structure.trees[0]
            assert len(tree1.edges) == d - 1
        
        print(f"D-vine structure built with order: {structure.order}")
    
    def test_build_cvine(self):
        """Test C-vine construction."""
        np.random.seed(42)
        n, d = 500, 4
        
        Z = np.random.randn(n, d)
        U = scipy_stats.norm.cdf(Z)
        
        # Build C-vine
        structure = build_cvine_structure(U)
        
        assert structure.d == d
        assert structure.order is not None
        
        # C-vine: Tree 1 should have d-1 edges (star structure)
        if len(structure.trees) > 0:
            tree1 = structure.trees[0]
            assert len(tree1.edges) == d - 1
        
        print(f"C-vine structure built with order: {structure.order}")


class TestDensityEstimation:
    """Test density estimation from samples."""
    
    def test_histogram_to_density(self):
        """Test scatter_to_hist produces valid density."""
        # Generate samples from known copula
        samples = sample_bicop('gaussian', {'rho': 0.7}, n=5000, seed=42)
        
        m = 64
        hist = scatter_to_hist(samples, m=m, reflect=True)
        
        # Check normalization
        assert abs(hist.sum() - 1.0) < 1e-6, f"Histogram not normalized: sum={hist.sum()}"
        
        # Check non-negativity
        assert hist.min() >= 0, f"Histogram has negative values: min={hist.min()}"
        
        # Check shape
        assert hist.shape == (m, m)
        
        print("Histogram creation test passed")
    
    def test_analytic_density(self):
        """Test analytic density computation."""
        m = 64
        
        # Test different families
        families = [
            ('gaussian', {'rho': 0.7}),
            ('clayton', {'theta': 2.0}),
            ('frank', {'theta': 5.0}),
        ]
        
        for family, params in families:
            log_pdf = analytic_logpdf_grid(family, params, m=m)
            
            assert log_pdf.shape == (m, m)
            assert not np.any(np.isnan(log_pdf)), f"{family} log_pdf has NaN"
            
            # Check that most values are finite (boundary values may be extreme)
            finite_ratio = np.sum(np.isfinite(log_pdf)) / (m * m)
            assert finite_ratio > 0.95, f"{family} has too many non-finite values"
            
            # For testing purposes, use clamped density
            # (full normalization is done in the dataset generation which handles this properly)
            pdf = np.exp(np.clip(log_pdf, -15, 15))
            assert pdf.min() >= 0, f"{family} has negative pdf values"
            
        print("Analytic density test passed for multiple families")


class TestEndToEnd:
    """End-to-end tests (may require trained model)."""
    
    @pytest.mark.skipif(
        not Path("checkpoints/conditional_diffusion_m64/model_step_30000.pt").exists(),
        reason="No trained checkpoint available"
    )
    def test_density_estimation_with_model(self):
        """Test density estimation using trained diffusion model."""
        from vdc.vine.copula_diffusion import DiffusionCopulaModel
        
        # Auto-detect device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        checkpoint_path = "checkpoints/conditional_diffusion_m64/model_step_30000.pt"
        model = DiffusionCopulaModel.from_checkpoint(checkpoint_path, device=device)
        
        # Generate test data
        np.random.seed(42)
        samples = sample_bicop('gaussian', {'rho': 0.7}, n=5000)
        
        # Estimate density
        density, u_coords, v_coords = model.estimate_density_from_samples(samples, m=64)
        
        # Check basic properties
        assert density.shape == (64, 64)
        assert density.min() >= 0
        
        # Check approximate normalization
        du = dv = 1.0 / 64
        mass = (density * du * dv).sum()
        assert abs(mass - 1.0) < 0.1, f"Estimated density mass = {mass}"
        
        print("Density estimation with model passed")


def run_all_tests():
    """Run all tests manually."""
    print("=" * 60)
    print("Running Vine Denoising Copula pipeline tests")
    print("=" * 60)
    
    # H-function tests
    print("\n--- H-Function Tests ---")
    hf_tests = TestHFunctions()
    hf_tests.test_independence_copula()
    hf_tests.test_monotonicity()
    hf_tests.test_inverse_roundtrip()
    
    # Projection tests
    print("\n--- Copula Projection Tests ---")
    proj_tests = TestCopulaProjection()
    proj_tests.test_uniform_marginals()
    proj_tests.test_preserves_structure()
    
    # Vine structure tests
    print("\n--- Vine Structure Tests ---")
    vine_tests = TestVineStructure()
    vine_tests.test_build_rvine()
    vine_tests.test_build_dvine()
    vine_tests.test_build_cvine()
    
    # Density estimation tests
    print("\n--- Density Estimation Tests ---")
    density_tests = TestDensityEstimation()
    density_tests.test_histogram_to_density()
    density_tests.test_analytic_density()
    
    print("\n" + "=" * 60)
    print("All basic tests PASSED!")
    print("=" * 60)
    
    # End-to-end (optional, needs checkpoint)
    print("\n--- End-to-End Tests (Optional) ---")
    e2e_tests = TestEndToEnd()
    try:
        e2e_tests.test_density_estimation_with_model()
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")


if __name__ == "__main__":
    run_all_tests()
