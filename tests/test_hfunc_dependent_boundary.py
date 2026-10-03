"""Regression tests for inversion inside the last conditioning-grid interval."""

import numpy as np
import pytest

from vdc.models.hfunc import HFuncLookup


@pytest.mark.parametrize("m", [4, 64])
@pytest.mark.parametrize("use_spline", [False, True])
@pytest.mark.parametrize("axis", ["u", "v"])
def test_dependent_inverse_last_interval(m, use_spline, axis):
    # Positive, asymmetric copula with exactly uniform discrete margins.
    f = np.tile([1.0, -1.0], m // 2)
    g = np.tile([-0.5, 1.0, -1.0, 0.5], m // 4)
    density = 1.0 + 0.8 * np.outer(f, g)
    h = HFuncLookup(density, use_spline=use_spline)
    grid = h.v_grid if axis == "u" else h.u_grid
    interior = grid[-2] + (grid[-1] - grid[-2]) * np.array(
        [0.001, 0.25, 0.5, 0.75, 0.999]
    )
    # Include support boundaries to preserve constant extrapolation there.
    conditioning = np.r_[0.0, grid[0], interior, grid[-1], 1.0]
    q, c = np.meshgrid(np.linspace(0.001, 0.999, 101), conditioning)
    inverse = h.hinv_u_given_v if axis == "u" else h.hinv_v_given_u
    forward = h.h_u_given_v if axis == "u" else h.h_v_given_u
    np.testing.assert_allclose(forward(inverse(q, c), c), q, atol=1e-12, rtol=0)
