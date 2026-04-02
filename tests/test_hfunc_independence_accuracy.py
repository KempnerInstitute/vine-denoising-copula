import numpy as np
import pytest

from vdc.models.hfunc import HFuncLookup


@pytest.mark.parametrize("m", [32, 64])
@pytest.mark.parametrize("use_spline", [True, False])
def test_hfunc_independence_identity_and_inverse(m: int, use_spline: bool) -> None:
    """
    Independence copula sanity check.

    For c(u,v)=1, we should have:
      h_{U|V}(u|v)=u,   h_{V|U}(v|u)=v,
      and inverses recover the input quantile.
    """
    rng = np.random.default_rng(0)
    density = np.ones((m, m), dtype=np.float64)
    h = HFuncLookup(density, use_spline=use_spline)

    u = rng.uniform(0.0, 1.0, size=2000)
    v = rng.uniform(0.0, 1.0, size=2000)

    hu = h.h_u_given_v(u, v)
    hv = h.h_v_given_u(v, u)

    assert float(np.max(np.abs(hu - u))) < 1e-3
    assert float(np.max(np.abs(hv - v))) < 1e-3

    # Boundary conditions in the integrated variable should be exact.
    v0 = rng.uniform(0.0, 1.0, size=50)
    assert float(np.max(np.abs(h.h_u_given_v(np.zeros_like(v0), v0) - 0.0))) < 1e-12
    assert float(np.max(np.abs(h.h_u_given_v(np.ones_like(v0), v0) - 1.0))) < 1e-12

    u0 = rng.uniform(0.0, 1.0, size=50)
    assert float(np.max(np.abs(h.h_v_given_u(np.zeros_like(u0), u0) - 0.0))) < 1e-12
    assert float(np.max(np.abs(h.h_v_given_u(np.ones_like(u0), u0) - 1.0))) < 1e-12

    q = rng.uniform(0.0, 1.0, size=2000)
    u_inv = h.hinv_u_given_v(q, v)
    v_inv = h.hinv_v_given_u(q, u)

    assert float(np.max(np.abs(u_inv - q))) < 1e-3
    assert float(np.max(np.abs(v_inv - q))) < 1e-3

    q_rt = h.h_u_given_v(u_inv, v)
    assert float(np.max(np.abs(q_rt - q))) < 1e-3
