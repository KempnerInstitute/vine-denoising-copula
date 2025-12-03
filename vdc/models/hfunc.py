"""
H-functions and inverse h-functions for copulas.

H-functions are conditional distribution functions:
- h_{U|V}(u|v) = ∫₀ᵘ c(s,v) ds  (CDF of U given V=v)
- h_{V|U}(v|u) = ∫₀ᵛ c(u,t) dt  (CDF of V given U=u)

These are essential for:
1. Vine copula recursion (computing conditional pseudo-observations)
2. Rosenblatt transform (for density evaluation)
3. Inverse Rosenblatt (for sampling)
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from typing import Optional, Tuple
import warnings


class HFuncLookup:
    """
    Lookup table for h-functions and their inverses from a density grid.
    
    Given a copula density c(u,v) on an m×m grid, precomputes:
    - h_{U|V}(u|v): cumulative sum along u-axis
    - h_{V|U}(v|u): cumulative sum along v-axis
    - Inverse h-functions via monotone interpolation
    
    Args:
        density_grid: (m, m) numpy array of copula density values
        u_grid: (m,) grid points for u (default: uniform on [0,1])
        v_grid: (m,) grid points for v (default: uniform on [0,1])
        interp_method: 'linear' or 'cubic' for bivariate interpolation
    """
    
    def __init__(
        self,
        density_grid: np.ndarray,
        u_grid: Optional[np.ndarray] = None,
        v_grid: Optional[np.ndarray] = None,
        interp_method: str = 'linear',
    ):
        assert density_grid.ndim == 2, "Expected 2D density grid"
        m_u, m_v = density_grid.shape
        
        self.m_u = m_u
        self.m_v = m_v
        self.density_grid = density_grid
        
        # Default uniform grids - use cell CENTERS (0.5/m, 1.5/m, ..., (m-0.5)/m)
        # This is important: copula densities are evaluated at cell centers, not edges
        if u_grid is None:
            u_grid = np.linspace(0.5/m_u, 1 - 0.5/m_u, m_u)
        if v_grid is None:
            v_grid = np.linspace(0.5/m_v, 1 - 0.5/m_v, m_v)
            
        self.u_grid = u_grid
        self.v_grid = v_grid
        self.du = u_grid[1] - u_grid[0] if m_u > 1 else 1.0
        self.dv = v_grid[1] - v_grid[0] if m_v > 1 else 1.0
        
        # Precompute h-functions
        self._compute_h_functions()
        
        # Build interpolators
        self.interp_method = interp_method
        self._build_interpolators()
        
    def _compute_h_functions(self):
        """Compute h-functions via cumulative integration."""
        # h_{U|V}(u|v) = ∫₀ᵘ c(s,v) ds
        # Integrate along u-axis (axis=0)
        self.h_u_given_v_grid = np.cumsum(self.density_grid, axis=0) * self.du
        
        # h_{V|U}(v|u) = ∫₀ᵛ c(u,t) dt
        # Integrate along v-axis (axis=1)
        self.h_v_given_u_grid = np.cumsum(self.density_grid, axis=1) * self.dv
        
        # Clamp to [0, 1] (numerical stability)
        self.h_u_given_v_grid = np.clip(self.h_u_given_v_grid, 0, 1)
        self.h_v_given_u_grid = np.clip(self.h_v_given_u_grid, 0, 1)
        
        # Ensure monotonicity (should be guaranteed by construction)
        # But numerical errors can cause small violations
        self._enforce_monotonicity()
        
    def _enforce_monotonicity(self):
        """Ensure h-functions are monotonically increasing."""
        # For h_u_given_v: should increase along axis 0 for each v
        for j in range(self.m_v):
            self.h_u_given_v_grid[:, j] = np.maximum.accumulate(
                self.h_u_given_v_grid[:, j]
            )
        
        # For h_v_given_u: should increase along axis 1 for each u
        for i in range(self.m_u):
            self.h_v_given_u_grid[i, :] = np.maximum.accumulate(
                self.h_v_given_u_grid[i, :]
            )
    
    def _build_interpolators(self):
        """Build 2D interpolators for h-functions."""
        kx = ky = 1 if self.interp_method == 'linear' else 3
        kx = min(kx, self.m_u - 1)
        ky = min(ky, self.m_v - 1)
        
        try:
            self.h_u_given_v_interp = RectBivariateSpline(
                self.u_grid, self.v_grid, self.h_u_given_v_grid,
                kx=kx, ky=ky,
            )
            self.h_v_given_u_interp = RectBivariateSpline(
                self.u_grid, self.v_grid, self.h_v_given_u_grid,
                kx=kx, ky=ky,
            )
        except Exception as e:
            warnings.warn(f"Failed to build spline interpolators: {e}. Using linear fallback.")
            self.h_u_given_v_interp = None
            self.h_v_given_u_interp = None
    
    def h_u_given_v(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Evaluate h_{U|V}(u|v).
        
        Args:
            u: Array of u values in [0,1]
            v: Array of v values in [0,1] (same shape as u)
            
        Returns:
            h values in [0,1]
        """
        u = np.asarray(u)
        v = np.asarray(v)
        
        if self.h_u_given_v_interp is not None:
            # Use spline interpolation
            result = self.h_u_given_v_interp(u.ravel(), v.ravel(), grid=False)
            return np.clip(result.reshape(u.shape), 0, 1)
        else:
            # Fallback: bilinear interpolation
            return self._bilinear_interp(
                u, v, self.h_u_given_v_grid, self.u_grid, self.v_grid
            )
    
    def h_v_given_u(self, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Evaluate h_{V|U}(v|u).
        
        Args:
            v: Array of v values in [0,1]
            u: Array of u values in [0,1] (same shape as v)
            
        Returns:
            h values in [0,1]
        """
        u = np.asarray(u)
        v = np.asarray(v)
        
        if self.h_v_given_u_interp is not None:
            result = self.h_v_given_u_interp(u.ravel(), v.ravel(), grid=False)
            return np.clip(result.reshape(u.shape), 0, 1)
        else:
            return self._bilinear_interp(
                u, v, self.h_v_given_u_grid, self.u_grid, self.v_grid
            )
    
    def hinv_u_given_v(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Inverse h-function: find u such that h_{U|V}(u|v) = q.
        
        This is needed for inverse Rosenblatt (sampling).
        
        Args:
            q: Quantile values in [0,1]
            v: Conditioning values in [0,1]
            
        Returns:
            u values in [0,1]
        """
        q = np.asarray(q)
        v = np.asarray(v)
        original_shape = q.shape
        
        q_flat = q.ravel()
        v_flat = v.ravel()
        u_flat = np.zeros_like(q_flat)
        
        # For each v, invert the h-function
        for i, (q_i, v_i) in enumerate(zip(q_flat, v_flat)):
            # Get the h-function curve for this v
            v_idx = np.searchsorted(self.v_grid, v_i)
            v_idx = np.clip(v_idx, 0, self.m_v - 1)
            
            # Get h values along u for this v
            if v_idx == 0:
                h_curve = self.h_u_given_v_grid[:, 0]
            elif v_idx >= self.m_v - 1:
                h_curve = self.h_u_given_v_grid[:, -1]
            else:
                # Linear interpolation between v grid points
                v_frac = (v_i - self.v_grid[v_idx - 1]) / (self.v_grid[v_idx] - self.v_grid[v_idx - 1])
                h_curve = (1 - v_frac) * self.h_u_given_v_grid[:, v_idx - 1] + \
                          v_frac * self.h_u_given_v_grid[:, v_idx]
            
            # Invert: find u where h_curve(u) = q_i
            # Use monotonic interpolation
            if q_i <= h_curve[0]:
                u_flat[i] = 0.0
            elif q_i >= h_curve[-1]:
                u_flat[i] = 1.0
            else:
                # Linear interpolation
                idx = np.searchsorted(h_curve, q_i)
                if idx == 0:
                    u_flat[i] = 0.0
                elif idx >= len(h_curve):
                    u_flat[i] = 1.0
                else:
                    # Linear interp between u_grid[idx-1] and u_grid[idx]
                    h0, h1 = h_curve[idx - 1], h_curve[idx]
                    u0, u1 = self.u_grid[idx - 1], self.u_grid[idx]
                    if abs(h1 - h0) < 1e-10:
                        u_flat[i] = u0
                    else:
                        frac = (q_i - h0) / (h1 - h0)
                        u_flat[i] = u0 + frac * (u1 - u0)
        
        return np.clip(u_flat.reshape(original_shape), 0, 1)
    
    def hinv_v_given_u(self, q: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Inverse h-function: find v such that h_{V|U}(v|u) = q.
        
        Args:
            q: Quantile values in [0,1]
            u: Conditioning values in [0,1]
            
        Returns:
            v values in [0,1]
        """
        q = np.asarray(q)
        u = np.asarray(u)
        original_shape = q.shape
        
        q_flat = q.ravel()
        u_flat = u.ravel()
        v_flat = np.zeros_like(q_flat)
        
        for i, (q_i, u_i) in enumerate(zip(q_flat, u_flat)):
            u_idx = np.searchsorted(self.u_grid, u_i)
            u_idx = np.clip(u_idx, 0, self.m_u - 1)
            
            if u_idx == 0:
                h_curve = self.h_v_given_u_grid[0, :]
            elif u_idx >= self.m_u - 1:
                h_curve = self.h_v_given_u_grid[-1, :]
            else:
                u_frac = (u_i - self.u_grid[u_idx - 1]) / (self.u_grid[u_idx] - self.u_grid[u_idx - 1])
                h_curve = (1 - u_frac) * self.h_v_given_u_grid[u_idx - 1, :] + \
                          u_frac * self.h_v_given_u_grid[u_idx, :]
            
            if q_i <= h_curve[0]:
                v_flat[i] = 0.0
            elif q_i >= h_curve[-1]:
                v_flat[i] = 1.0
            else:
                idx = np.searchsorted(h_curve, q_i)
                if idx == 0:
                    v_flat[i] = 0.0
                elif idx >= len(h_curve):
                    v_flat[i] = 1.0
                else:
                    h0, h1 = h_curve[idx - 1], h_curve[idx]
                    v0, v1 = self.v_grid[idx - 1], self.v_grid[idx]
                    if abs(h1 - h0) < 1e-10:
                        v_flat[i] = v0
                    else:
                        frac = (q_i - h0) / (h1 - h0)
                        v_flat[i] = v0 + frac * (v1 - v0)
        
        return np.clip(v_flat.reshape(original_shape), 0, 1)
    
    def _bilinear_interp(
        self,
        u: np.ndarray,
        v: np.ndarray,
        grid: np.ndarray,
        u_grid: np.ndarray,
        v_grid: np.ndarray,
    ) -> np.ndarray:
        """Bilinear interpolation fallback."""
        from scipy.interpolate import RegularGridInterpolator
        
        interp = RegularGridInterpolator(
            (u_grid, v_grid),
            grid,
            method='linear',
            bounds_error=False,
            fill_value=None,
        )
        
        points = np.stack([u.ravel(), v.ravel()], axis=-1)
        result = interp(points)
        return np.clip(result.reshape(u.shape), 0, 1)
    
    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Evaluate copula density c(u,v) via interpolation.
        
        Args:
            u: Array of u values
            v: Array of v values
            
        Returns:
            Density values
        """
        return self._bilinear_interp(u, v, self.density_grid, self.u_grid, self.v_grid)


if __name__ == "__main__":
    print("Testing H-function implementation...")
    
    # Create a simple copula density (e.g., independence)
    m = 64
    u_grid = np.linspace(0, 1, m)
    v_grid = np.linspace(0, 1, m)
    
    # Independence copula: c(u,v) = 1
    print("\n" + "="*60)
    print("Test 1: Independence copula")
    D_indep = np.ones((m, m))
    
    hfunc_indep = HFuncLookup(D_indep, u_grid, v_grid)
    
    # For independence, h_{U|V}(u|v) = u (independent of v)
    u_test = np.array([0.1, 0.5, 0.9])
    v_test = np.array([0.3, 0.5, 0.7])
    
    h_uv = hfunc_indep.h_u_given_v(u_test, v_test)
    h_vu = hfunc_indep.h_v_given_u(v_test, u_test)
    
    print(f"u_test: {u_test}")
    print(f"h_u_given_v: {h_uv} (should be ≈ u_test)")
    print(f"Difference: {np.abs(h_uv - u_test).max():.6f}")
    
    print(f"\nv_test: {v_test}")
    print(f"h_v_given_u: {h_vu} (should be ≈ v_test)")
    print(f"Difference: {np.abs(h_vu - v_test).max():.6f}")
    
    # Test inverse h-function
    print("\nTesting inverse h-functions...")
    q_test = np.array([0.2, 0.5, 0.8])
    v_cond = np.array([0.5, 0.5, 0.5])
    
    u_inv = hfunc_indep.hinv_u_given_v(q_test, v_cond)
    print(f"q: {q_test}")
    print(f"u from hinv: {u_inv} (should be ≈ q for independence)")
    print(f"Difference: {np.abs(u_inv - q_test).max():.6f}")
    
    # Round-trip test
    h_roundtrip = hfunc_indep.h_u_given_v(u_inv, v_cond)
    print(f"h(hinv(q)): {h_roundtrip} (should equal q)")
    print(f"Roundtrip error: {np.abs(h_roundtrip - q_test).max():.6f}")
    
    # Test with a Gaussian copula
    print("\n" + "="*60)
    print("Test 2: Gaussian copula (ρ=0.7)")
    
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
    rho = 0.7
    
    # Approximate Gaussian copula density
    # (not exact, but good enough for testing)
    from scipy.stats import norm
    Z_u = norm.ppf(np.clip(U, 0.01, 0.99))
    Z_v = norm.ppf(np.clip(V, 0.01, 0.99))
    
    exponent = (Z_u**2 + Z_v**2 - 2*rho*Z_u*Z_v) / (2*(1-rho**2))
    D_gauss = np.exp(-exponent + 0.5*(Z_u**2 + Z_v**2)) / np.sqrt(1 - rho**2)
    D_gauss = np.nan_to_num(D_gauss, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Normalize (approximate)
    D_gauss = D_gauss / (D_gauss.sum() / m**2)
    
    hfunc_gauss = HFuncLookup(D_gauss, u_grid, v_grid)
    
    u_test = np.array([0.5])
    v_test = np.array([0.5])
    
    h_uv_gauss = hfunc_gauss.h_u_given_v(u_test, v_test)
    print(f"h_{{U|V}}(0.5|0.5) = {h_uv_gauss[0]:.4f}")
    print("(For positive correlation, should be < 0.5 since mass shifts)")
    
    # Test sampling capability
    print("\n" + "="*60)
    print("Test 3: Sampling via inverse Rosenblatt")
    
    n_samples = 1000
    np.random.seed(42)
    
    # Sample from independence copula
    u1 = np.random.uniform(0, 1, n_samples)
    w = np.random.uniform(0, 1, n_samples)
    u2 = hfunc_indep.hinv_u_given_v(w, u1)
    
    # For independence, u2 should also be uniform
    print(f"u1 mean: {u1.mean():.4f}, std: {u1.std():.4f}")
    print(f"u2 mean: {u2.mean():.4f}, std: {u2.std():.4f}")
    print(f"Correlation: {np.corrcoef(u1, u2)[0,1]:.4f} (should be ≈ 0)")
    
    print("\nAll h-function tests completed!")
