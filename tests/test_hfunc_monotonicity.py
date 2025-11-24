"""Tests for h-function monotonicity and inverse consistency."""
import numpy as np
import torch
import pytest
from vdc.models.hfunc import HFuncLookup

@pytest.mark.parametrize("m", [32, 64])
def test_hfunc_monotonicity_independence(m):
    D = np.ones((m,m))
    h = HFuncLookup(D)
    # Sample random v values
    v_vals = np.random.uniform(0,1, size=5)
    u_grid = h.u_grid
    for v in v_vals:
        # h_{U|V}(u|v) should be increasing in u
        u_vals = np.linspace(0,1, m)
        h_vals = h.h_u_given_v(u_vals, np.full_like(u_vals, v))
        assert np.all(np.diff(h_vals) >= -1e-8), "h_u_given_v not monotonic"
    # Similar for h_{V|U}
    u_vals = np.random.uniform(0,1, size=5)
    v_grid = h.v_grid
    for u in u_vals:
        v_vals2 = np.linspace(0,1,m)
        h_vals2 = h.h_v_given_u(v_vals2, np.full_like(v_vals2, u))
        assert np.all(np.diff(h_vals2) >= -1e-8), "h_v_given_u not monotonic"

@pytest.mark.parametrize("rho", [0.0, 0.7])
def test_inverse_roundtrip_gaussian(rho):
    m = 64
    u_grid = np.linspace(0,1,m)
    v_grid = np.linspace(0,1,m)
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
    from scipy.stats import norm
    Z_u = norm.ppf(np.clip(U, 0.01, 0.99))
    Z_v = norm.ppf(np.clip(V, 0.01, 0.99))
    exponent = (Z_u**2 + Z_v**2 - 2*rho*Z_u*Z_v) / (2*(1-rho**2)) if rho!=0 else (Z_u**2 + Z_v**2)/2
    D_gauss = np.exp(-exponent + 0.5*(Z_u**2 + Z_v**2)) / np.sqrt(1 - rho**2) if rho!=0 else np.ones_like(U)
    D_gauss = D_gauss / (D_gauss.sum()/m**2)
    h = HFuncLookup(D_gauss)
    np.random.seed(0)
    q = np.random.uniform(0,1,200)
    v = np.random.uniform(0,1,200)
    u = h.hinv_u_given_v(q,v)
    q_rt = h.h_u_given_v(u,v)
    err = np.abs(q_rt - q).mean()
    assert err < 0.05, f"Roundtrip error too large: {err}"