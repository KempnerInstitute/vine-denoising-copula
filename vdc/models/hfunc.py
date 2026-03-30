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
        use_spline: bool = True,
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
        self.use_spline = bool(use_spline)
        if self.use_spline:
            self._build_interpolators()
        else:
            # Use lightweight on-the-fly bilinear interpolation (no SciPy spline objects).
            self.h_u_given_v_interp = None
            self.h_v_given_u_interp = None
        
    def _compute_h_functions(self):
        """Compute h-functions via cumulative integration."""
        # h_{U|V}(u|v) = ∫₀ᵘ c(s,v) ds
        # Integrate along u-axis (axis=0)
        # NOTE:
        # Our density grid is defined on *cell centers* (u=(i+0.5)/m). A naive right-Riemann cumsum
        # (sum_{r<=i} c_r * du) corresponds to integrating up to the *cell edge* (i+1)/m, which
        # produces a systematic +du/2 bias when you later evaluate h at the cell centers. This
        # bias is especially harmful for vine recursion (conditional pseudo-observations) and
        # inverse Rosenblatt sampling (tails get "pushed in" from 0/1).
        #
        # We instead use a midpoint-rule correction so that, for the independence copula
        # (c(u,v)=1), we get h(u|v)=u (up to interpolation error) at the centers.
        self.h_u_given_v_grid = (np.cumsum(self.density_grid, axis=0) - 0.5 * self.density_grid) * self.du
        
        # h_{V|U}(v|u) = ∫₀ᵛ c(u,t) dt
        # Integrate along v-axis (axis=1)
        self.h_v_given_u_grid = (np.cumsum(self.density_grid, axis=1) - 0.5 * self.density_grid) * self.dv
        
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
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        u0 = float(self.u_grid[0])
        u1 = float(self.u_grid[-1])

        # Evaluate within the grid support, then linearly extend to enforce boundary conditions:
        # h(0|v)=0 and h(1|v)=1. This avoids clamping artifacts when u is near 0/1.
        u_in = np.clip(u, u0, u1)
        if self.h_u_given_v_interp is not None:
            v_in = np.clip(v, float(self.v_grid[0]), float(self.v_grid[-1]))
            base = self.h_u_given_v_interp(u_in.ravel(), v_in.ravel(), grid=False).reshape(u.shape)
        else:
            base = self._bilinear_interp(u_in, v, self.h_u_given_v_grid, clip_min=0.0, clip_max=1.0)

        # Low end: linear from (0,0) to (u0, h(u0|v)).
        mask_lo = u < u0
        if np.any(mask_lo):
            if self.h_u_given_v_interp is not None:
                v_in = np.clip(v, float(self.v_grid[0]), float(self.v_grid[-1]))
                h0 = self.h_u_given_v_interp(
                    np.full_like(u_in, u0).ravel(), v_in.ravel(), grid=False
                ).reshape(u.shape)
                h0 = np.clip(h0, 0.0, 1.0)
            else:
                h0 = self._bilinear_interp(
                    np.full_like(u_in, u0), v, self.h_u_given_v_grid, clip_min=0.0, clip_max=1.0
                )
            u_safe = max(u0, 1e-12)
            base = np.where(mask_lo, (np.clip(u, 0.0, u_safe) / u_safe) * h0, base)

        # High end: linear from (u1, h(u1|v)) to (1,1).
        mask_hi = u > u1
        if np.any(mask_hi):
            denom = max(1.0 - u1, 1e-12)
            base = np.where(mask_hi, base + (np.clip(u, u1, 1.0) - u1) * (1.0 - base) / denom, base)

        return np.clip(base, 0.0, 1.0)
    
    def h_v_given_u(self, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Evaluate h_{V|U}(v|u).
        
        Args:
            v: Array of v values in [0,1]
            u: Array of u values in [0,1] (same shape as v)
            
        Returns:
            h values in [0,1]
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        v0 = float(self.v_grid[0])
        v1 = float(self.v_grid[-1])

        # Same boundary handling as h_u_given_v, but now the integration variable is v.
        v_in = np.clip(v, v0, v1)
        if self.h_v_given_u_interp is not None:
            u_in = np.clip(u, float(self.u_grid[0]), float(self.u_grid[-1]))
            base = self.h_v_given_u_interp(u_in.ravel(), v_in.ravel(), grid=False).reshape(u.shape)
        else:
            base = self._bilinear_interp(u, v_in, self.h_v_given_u_grid, clip_min=0.0, clip_max=1.0)

        mask_lo = v < v0
        if np.any(mask_lo):
            if self.h_v_given_u_interp is not None:
                u_in = np.clip(u, float(self.u_grid[0]), float(self.u_grid[-1]))
                h0 = self.h_v_given_u_interp(
                    u_in.ravel(), np.full_like(v_in, v0).ravel(), grid=False
                ).reshape(u.shape)
                h0 = np.clip(h0, 0.0, 1.0)
            else:
                h0 = self._bilinear_interp(u, np.full_like(v_in, v0), self.h_v_given_u_grid, clip_min=0.0, clip_max=1.0)
            v_safe = max(v0, 1e-12)
            base = np.where(mask_lo, (np.clip(v, 0.0, v_safe) / v_safe) * h0, base)

        mask_hi = v > v1
        if np.any(mask_hi):
            denom = max(1.0 - v1, 1e-12)
            base = np.where(mask_hi, base + (np.clip(v, v1, 1.0) - v1) * (1.0 - base) / denom, base)

        return np.clip(base, 0.0, 1.0)
    
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
        q = np.asarray(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
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
            
            # Invert: find u where h_curve(u) = q_i, enforcing endpoints (0,0) and (1,1).
            u_pts = np.concatenate(([0.0], self.u_grid.astype(np.float64), [1.0]))
            h_pts = np.concatenate(([0.0], np.asarray(h_curve, dtype=np.float64), [1.0]))
            h_pts = np.clip(np.maximum.accumulate(h_pts), 0.0, 1.0)

            q_i = float(np.clip(q_i, 0.0, 1.0))
            idx = int(np.searchsorted(h_pts, q_i, side="left"))
            if idx <= 0:
                u_flat[i] = 0.0
            elif idx >= h_pts.shape[0]:
                u_flat[i] = 1.0
            else:
                h0, h1 = float(h_pts[idx - 1]), float(h_pts[idx])
                u0, u1 = float(u_pts[idx - 1]), float(u_pts[idx])
                if abs(h1 - h0) < 1e-12:
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
        q = np.asarray(q, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
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
            
            v_pts = np.concatenate(([0.0], self.v_grid.astype(np.float64), [1.0]))
            h_pts = np.concatenate(([0.0], np.asarray(h_curve, dtype=np.float64), [1.0]))
            h_pts = np.clip(np.maximum.accumulate(h_pts), 0.0, 1.0)

            q_i = float(np.clip(q_i, 0.0, 1.0))
            idx = int(np.searchsorted(h_pts, q_i, side="left"))
            if idx <= 0:
                v_flat[i] = 0.0
            elif idx >= h_pts.shape[0]:
                v_flat[i] = 1.0
            else:
                h0, h1 = float(h_pts[idx - 1]), float(h_pts[idx])
                v0, v1 = float(v_pts[idx - 1]), float(v_pts[idx])
                if abs(h1 - h0) < 1e-12:
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
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> np.ndarray:
        """
        Fast bilinear interpolation on a uniform grid (cell centers).

        This intentionally avoids constructing SciPy interpolator objects, which
        can be prohibitively expensive when fitting high-dimensional vines
        (thousands of pair-copulas).
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        if grid.ndim != 2:
            raise ValueError(f"Expected 2D grid, got shape={grid.shape}")
        mu, mv = grid.shape
        if mu < 2 or mv < 2:
            # Degenerate: nearest neighbor.
            out = np.full_like(u, float(grid[0, 0]), dtype=np.float64)
            if clip_min is not None:
                out = np.maximum(out, float(clip_min))
            if clip_max is not None:
                out = np.minimum(out, float(clip_max))
            return out

        # Flatten for vectorized gather.
        u_flat = u.ravel()
        v_flat = v.ravel()

        # Map u,v to fractional indices in [0, mu-1] / [0, mv-1] for cell-center grids.
        # u_grid = (0.5/m, 1.5/m, ..., (m-0.5)/m)  => u0 = u_grid[0], du = 1/m.
        u0 = float(self.u_grid[0])
        v0 = float(self.v_grid[0])
        du = float(self.du) if float(self.du) > 0 else 1.0
        dv = float(self.dv) if float(self.dv) > 0 else 1.0

        # Clamp to the grid support.
        u_flat = np.clip(u_flat, u0, float(self.u_grid[-1]))
        v_flat = np.clip(v_flat, v0, float(self.v_grid[-1]))

        iu = (u_flat - u0) / du
        iv = (v_flat - v0) / dv
        i0 = np.floor(iu).astype(np.int64)
        j0 = np.floor(iv).astype(np.int64)

        # Keep in-bounds for i0+1 and j0+1.
        i0 = np.clip(i0, 0, mu - 2)
        j0 = np.clip(j0, 0, mv - 2)
        i1 = i0 + 1
        j1 = j0 + 1

        # Fractions
        tu = np.clip(iu - i0, 0.0, 1.0)
        tv = np.clip(iv - j0, 0.0, 1.0)

        g00 = grid[i0, j0]
        g10 = grid[i1, j0]
        g01 = grid[i0, j1]
        g11 = grid[i1, j1]

        out = (1 - tu) * (1 - tv) * g00 + tu * (1 - tv) * g10 + (1 - tu) * tv * g01 + tu * tv * g11
        out = out.reshape(u.shape)

        if clip_min is not None:
            out = np.maximum(out, float(clip_min))
        if clip_max is not None:
            out = np.minimum(out, float(clip_max))
        return out
    
    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Evaluate copula density c(u,v) via interpolation.
        
        Args:
            u: Array of u values
            v: Array of v values
            
        Returns:
            Density values
        """
        # Density can exceed 1.0; only clamp below for numerical stability.
        return self._bilinear_interp(u, v, self.density_grid, clip_min=0.0, clip_max=None)


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
