#!/usr/bin/env python3
"""
Targeted correctness tests for vine recursion + Rosenblatt/sampling.

These tests intentionally use the independence copula (c(u,v)=1) so that:
  - logpdf should be ~0 everywhere
  - Rosenblatt(inverse_rosenblatt(W)) should approximately recover W

We allow moderate numerical tolerance due to grid-based h-function discretization.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest

from vdc.models.hfunc import HFuncLookup
from vdc.vine.recursion import VinePairCopula, VineRecursion
from vdc.vine.vine_types import build_cvine_structure, build_dvine_structure


def _build_independence_vine(vine_type: str, d: int, m: int = 64) -> VineRecursion:
    rng = np.random.default_rng(123)
    U_dummy = rng.uniform(0.0, 1.0, size=(128, d))

    order = list(range(d))
    if vine_type == "dvine":
        structure = build_dvine_structure(U_dummy, order=order)
    elif vine_type == "cvine":
        structure = build_cvine_structure(U_dummy, order=order)
    else:
        raise ValueError(vine_type)

    density = np.ones((m, m), dtype=np.float64)
    hfunc = HFuncLookup(density)

    vine = VineRecursion(structure, vine_type=vine_type)
    for level, tree in enumerate(structure.trees):
        for edge in tree.edges:
            vine.add_pair_copula(
                VinePairCopula(edge=edge, density_grid=density, hfunc=hfunc, level=level)
            )
    return vine


@pytest.mark.parametrize("vine_type", ["dvine", "cvine"])
def test_independence_logpdf_is_zero(vine_type: str):
    d = 5
    vine = _build_independence_vine(vine_type, d=d, m=32)

    rng = np.random.default_rng(42)
    U = rng.uniform(0.0, 1.0, size=(120, d))

    logp = vine.logpdf(U)
    assert logp.shape == (U.shape[0],)
    # Independence density is exactly 1 on the grid -> log-density ~ 0
    assert np.allclose(logp, 0.0, atol=1e-8)


@pytest.mark.parametrize("vine_type", ["dvine", "cvine"])
def test_independence_rosenblatt_roundtrip(vine_type: str):
    d = 5
    vine = _build_independence_vine(vine_type, d=d, m=32)

    rng = np.random.default_rng(7)
    W = rng.uniform(0.0, 1.0, size=(240, d))

    U = vine.inverse_rosenblatt(W)
    assert U.shape == W.shape
    assert np.isfinite(U).all()
    assert (U >= 0.0).all() and (U <= 1.0).all()

    W2 = vine.rosenblatt(U)
    assert W2.shape == W.shape
    assert np.isfinite(W2).all()

    mae = float(np.mean(np.abs(W2 - W)))
    # Grid-based h/hinv introduce discretization error; allow moderate tolerance.
    assert mae < 0.08, f"Roundtrip MAE too large: {mae:.4f}"

    # Rosenblatt output should still be (approximately) uniform marginally.
    for j in range(d):
        _, p = kstest(W2[:, j], "uniform")
        assert p > 1e-3, f"KS p-value too small for dim {j}: {p:.2e}"


@pytest.mark.parametrize("vine_type", ["dvine", "cvine"])
def test_independence_simulate_has_uniform_margins(vine_type: str):
    d = 5
    vine = _build_independence_vine(vine_type, d=d, m=32)

    U = vine.simulate(n=360, seed=123)
    assert U.shape == (360, d)
    assert np.isfinite(U).all()
    assert (U >= 0.0).all() and (U <= 1.0).all()

    for j in range(d):
        _, p = kstest(U[:, j], "uniform")
        assert p > 1e-3, f"KS p-value too small for dim {j}: {p:.2e}"

