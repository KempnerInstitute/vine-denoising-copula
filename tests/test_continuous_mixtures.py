"""Continuous mixture source of the linear-grid models: families, BB7 tails and margins."""
import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import kstest

from vdc.data.continuous_mixtures import (
    FAMILIES,
    STRENGTH_BINS,
    ContinuousMixture,
    bb7_cdf_stable,
    bb7_h_stable,
    bb7_pdf_stable,
    component,
    draw_spec,
)


def test_bb7_density_against_library_and_high_precision_tail():
    mp = pytest.importorskip("mpmath")
    import pyvinecopulib as pv
    rng = np.random.default_rng(75)
    x = rng.uniform(.05, .95, (500, 2))
    params = np.array([[4.1372649785605615], [16.765563120911576]])
    for rotation in (0, 90, 180, 270):
        c = pv.Bicop(family=pv.BicopFamily.bb7, parameters=params, rotation=rotation)
        np.testing.assert_allclose(bb7_pdf_stable(x, params, rotation), c.pdf(x), rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(bb7_cdf_stable(x, params, rotation), c.cdf(x), rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(bb7_h_stable(x, params, rotation), c.hfunc2(x), rtol=1e-7, atol=1e-10)
    # Regression: the installed pyvine 0.7.5 returns NaN here. Independently
    # differentiate the analytical CDF using 80-decimal arithmetic.
    point = np.array([[8.952966891229153e-05, .9998822594352532]])
    with mp.workdps(80):
        theta, delta = (mp.mpf(str(p)) for p in params.ravel())
        def cdf(u, v):
            s = (1-(1-u)**theta)**(-delta)+(1-(1-v)**theta)**(-delta)-1
            return 1-(1-s**(-1/delta))**(1/theta)
        expected = float(mp.diff(cdf, (mp.mpf(1-float(point[0, 0])), mp.mpf(float(point[0, 1]))), (1, 1)))
    np.testing.assert_allclose(bb7_pdf_stable(point, params, 90)[0], expected, rtol=1e-10)


def test_bb7_strong_tail_cell_masses():
    spec = {"components": [{"family": "bb7", "parameters": [[5.844695249652971], [11.397005658822758]], "rotation": 180, "weight": 1.}]}
    truth = ContinuousMixture(spec)
    for edges in (np.linspace(0, 1, 65), np.r_[0., ndtr(np.linspace(-3.5, 3.5, 65))[1:-1], 1.]):
        masses = truth.cell_masses(edges)
        assert masses.min() >= 0
        np.testing.assert_allclose(masses.sum(0), np.diff(edges), atol=1e-12)
        np.testing.assert_allclose(masses.sum(1), np.diff(edges), atol=1e-12)


def test_component_coverage_and_reserved_families():
    rng = np.random.default_rng(71)
    for family in FAMILIES:
        for _ in range(12):
            c = component(rng, family)
            lo, hi = STRENGTH_BINS[c["strength_bin"]]
            assert lo-1e-7 <= abs(c["component_tau"]) <= hi+1e-7
    specs = [draw_spec(rng) for _ in range(200)]
    assert {"between_family_mixture", "within_family_mixture", "single", "independence"} <= {s["regime"] for s in specs}
    assert 8 in {len(s["components"]) for s in specs}
    assert {c["family"] for s in specs for c in s["components"]}.isdisjoint({"bb6", "bb8", "tawn"})


def test_continuous_samples_match_mixture_cdf_and_uniform_margins():
    rng = np.random.default_rng(73)
    cases = [draw_spec(rng) for _ in range(12)]
    probes = np.array([[.2, .25], [.5, .7], [.8, .9]])
    for i, spec in enumerate(cases):
        truth = ContinuousMixture(spec)
        x = truth.simulate(12000, seed=100+i)
        assert np.isfinite(x).all() and ((x > 0) & (x < 1)).all()
        assert np.isfinite(truth.pdf(x)).all()
        for j in range(2):
            assert kstest(x[:, j], "uniform").statistic < .035
        assert kstest(truth.h_u_given_v(x), "uniform").statistic < .035
        cdf = truth.cdf(probes)
        empirical = np.array([(x <= p).all(1).mean() for p in probes])
        assert np.all(np.abs(cdf-empirical) < 5*np.sqrt(cdf*(1-cdf)/len(x))+.002)
        for edges in (np.linspace(0, 1, 65), np.r_[0., ndtr(np.linspace(-3.5, 3.5, 65))[1:-1], 1.]):
            mass = truth.cell_masses(edges)
            np.testing.assert_allclose(mass.sum(0), np.diff(edges), atol=1e-8)
            np.testing.assert_allclose(mass.sum(1), np.diff(edges), atol=1e-8)
        np.testing.assert_array_equal(truth.simulate(100, seed=11), truth.simulate(100, seed=11))
