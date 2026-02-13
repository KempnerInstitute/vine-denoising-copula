#!/usr/bin/env python3
"""Ground-truth correctness checks on a 3D Gaussian D-vine."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, norm

from vdc.models.hfunc import HFuncLookup
from vdc.vine.recursion import VinePairCopula, VineRecursion
from vdc.vine.vine_types import build_dvine_structure


def _gaussian_copula_density_grid(rho: float, m: int) -> np.ndarray:
    u = np.linspace(0.5 / m, 1.0 - 0.5 / m, m, dtype=np.float64)
    v = np.linspace(0.5 / m, 1.0 - 0.5 / m, m, dtype=np.float64)
    z_u = norm.ppf(u)[:, None]
    z_v = norm.ppf(v)[None, :]
    denom = np.sqrt(max(1.0 - rho * rho, 1e-12))
    exponent = -(z_u * z_u - 2.0 * rho * z_u * z_v + z_v * z_v) / (2.0 * max(1.0 - rho * rho, 1e-12))
    correction = 0.5 * (z_u * z_u + z_v * z_v)
    return (np.exp(exponent + correction) / denom).astype(np.float64)


def _build_3d_gaussian_dvine(m: int = 96) -> tuple[VineRecursion, np.ndarray]:
    # Correlation matrix for a valid 3D Gaussian copula.
    r12 = 0.55
    r23 = 0.45
    r13 = 0.30
    corr = np.array(
        [
            [1.0, r12, r13],
            [r12, 1.0, r23],
            [r13, r23, 1.0],
        ],
        dtype=np.float64,
    )
    # Conditional correlation for pair-copula c_{13|2}.
    r13_2 = (r13 - r12 * r23) / np.sqrt((1.0 - r12 * r12) * (1.0 - r23 * r23))

    dummy = np.random.default_rng(0).uniform(0.0, 1.0, size=(128, 3))
    structure = build_dvine_structure(dummy, order=[0, 1, 2])
    vine = VineRecursion(structure, vine_type="dvine")

    d12 = _gaussian_copula_density_grid(r12, m=m)
    d23 = _gaussian_copula_density_grid(r23, m=m)
    d13_2 = _gaussian_copula_density_grid(float(r13_2), m=m)

    h12 = HFuncLookup(d12, use_spline=False)
    h23 = HFuncLookup(d23, use_spline=False)
    h13_2 = HFuncLookup(d13_2, use_spline=False)

    for level, tree in enumerate(structure.trees):
        for edge in tree.edges:
            pair = set(edge[:2])
            if level == 0 and pair == {0, 1}:
                vine.add_pair_copula(VinePairCopula(edge=edge, density_grid=d12, hfunc=h12, level=level))
            elif level == 0 and pair == {1, 2}:
                vine.add_pair_copula(VinePairCopula(edge=edge, density_grid=d23, hfunc=h23, level=level))
            elif level == 1 and pair == {0, 2}:
                vine.add_pair_copula(VinePairCopula(edge=edge, density_grid=d13_2, hfunc=h13_2, level=level))
            else:
                raise RuntimeError(f"Unexpected edge for 3D D-vine: level={level}, edge={edge}")

    return vine, corr


def _gaussian_copula_logpdf(U: np.ndarray, corr: np.ndarray) -> np.ndarray:
    z = norm.ppf(np.clip(U, 1e-12, 1.0 - 1e-12))
    inv_corr = np.linalg.inv(corr)
    quad = np.einsum("ni,ij,nj->n", z, inv_corr - np.eye(corr.shape[0]), z)
    return -0.5 * np.log(np.linalg.det(corr)) - 0.5 * quad


def test_3d_gaussian_dvine_logpdf_matches_ground_truth() -> None:
    vine, corr = _build_3d_gaussian_dvine(m=96)
    rng = np.random.default_rng(5)
    U = rng.uniform(1e-4, 1.0 - 1e-4, size=(2000, 3))

    logp_vine = vine.logpdf(U)
    logp_true = _gaussian_copula_logpdf(U, corr)
    abs_err = np.abs(logp_vine - logp_true)

    # Mean error should remain small; a few tail points may have larger interpolation error.
    assert float(np.mean(abs_err)) < 0.03
    assert float(np.quantile(abs_err, 0.95)) < 0.08


def test_3d_gaussian_dvine_sampling_recovers_pairwise_kendall_tau() -> None:
    vine, corr = _build_3d_gaussian_dvine(m=96)
    U = vine.simulate(n=5000, seed=11)

    expected_taus = {
        (0, 1): float((2.0 / np.pi) * np.arcsin(corr[0, 1])),
        (1, 2): float((2.0 / np.pi) * np.arcsin(corr[1, 2])),
        (0, 2): float((2.0 / np.pi) * np.arcsin(corr[0, 2])),
    }
    for (i, j), tau_true in expected_taus.items():
        tau_emp = float(kendalltau(U[:, i], U[:, j]).correlation)
        assert abs(tau_emp - tau_true) < 0.03, f"tau({i},{j}) mismatch: emp={tau_emp:.4f}, true={tau_true:.4f}"

