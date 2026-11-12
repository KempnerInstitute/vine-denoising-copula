"""Distribution checks independent of the sampling implementation."""
import numpy as np
import pytest
from scipy.stats import kstest

from vdc.data.conditional_copulas import bb1_density, bb7_density, h_frank, h_joe, hinv_frank
from vdc.data.generators import analytic_logpdf_grid, sample_bicop
from vdc.data.onthefly import OnTheFlyCopulaDataset


def cdf(family, u, v, theta, delta=1):
    if family == "joe":
        a, b = (1-u)**theta, (1-v)**theta
        return 1 - (a+b-a*b)**(1/theta)
    if family == "bb1":
        return (1 + ((u**(-theta)-1)**delta + (v**(-theta)-1)**delta)**(1/delta))**(-1/theta)
    if family == "bb7":
        s = (1-(1-u)**theta)**(-delta) + (1-(1-v)**theta)**(-delta) - 1
        return 1 - (1-s**(-1/delta))**(1/theta)
    raise ValueError(family)


@pytest.mark.parametrize("family,theta,delta", [("joe", 2., 1.), ("joe", 6., 1.),
                                              ("bb1", 1.2, 1.5), ("bb7", 2., 1.5)])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_samples_have_uniform_margins_and_correct_joint_cdf(family, theta, delta, rotation):
    params = {"theta": theta, "delta": delta}
    uv = sample_bicop(family, params, 16000, rotation=rotation, seed=20391)
    for margin in uv.T:
        assert kstest(margin, "uniform").statistic < 0.02
    # Undo rotation and compare empirical joint probabilities to closed-form CDF.
    base = uv.copy()
    if rotation in (90, 180):
        base[:, 0] = 1-base[:, 0]
    if rotation in (180, 270):
        base[:, 1] = 1-base[:, 1]
    for u, v in [(0.2, 0.4), (0.5, 0.5), (0.8, 0.6)]:
        empirical = np.mean((base[:, 0] <= u) & (base[:, 1] <= v))
        assert abs(empirical-cdf(family, u, v, theta, delta)) < 0.02
    np.testing.assert_array_equal(uv, sample_bicop(family, params, 16000,
                                                rotation=rotation, seed=20391))


@pytest.mark.parametrize("family,fn", [("bb1", bb1_density), ("bb7", bb7_density)])
def test_density_matches_mixed_derivative_of_cdf(family, fn):
    u = np.array([.15, .35, .6, .85])[:, None]
    v = np.array([.2, .5, .8])[None, :]
    eps = 1e-4
    args = (2., 1.5)
    derivative = (cdf(family, u+eps, v+eps, *args) - cdf(family, u+eps, v-eps, *args)
                  - cdf(family, u-eps, v+eps, *args) + cdf(family, u-eps, v-eps, *args))/(4*eps**2)
    np.testing.assert_allclose(fn(u, v, *args), derivative, rtol=1e-5, atol=1e-6)


def test_joe_conditional_is_cdf_derivative():
    u, v = np.array([.1, .3, .7, .9]), np.array([.2, .6, .4, .8])
    eps = 1e-6
    expected = (cdf("joe", u, v+eps, 2.5)-cdf("joe", u, v-eps, 2.5))/(2*eps)
    np.testing.assert_allclose(h_joe(u, v, 2.5), expected, rtol=1e-7)


@pytest.mark.parametrize("theta", [-8., 0., 2., 8.])
def test_frank_conditional_roundtrip(theta):
    w = np.linspace(.01, .99, 101)
    v = np.linspace(.95, .05, 101)
    np.testing.assert_allclose(h_frank(hinv_frank(w, v, theta), v, theta), w, atol=1e-10)


def test_bb_rotation_applies_to_both_training_input_and_target():
    ds = OnTheFlyCopulaDataset(n_samples_per_batch=5000, m=64, families={"bb1": 1}, seed=48)
    ds._sample_family = lambda: ("bb1", {"theta": 1.2, "delta": 1.5}, 90)
    item = ds[0]
    expected = np.exp(analytic_logpdf_grid("bb1", {"theta": 1.2, "delta": 1.5}, 64, rotation=90))
    np.testing.assert_allclose(item["density"][0], expected, rtol=1e-6)
    uv = item["samples"].numpy()
    assert np.corrcoef(uv.T)[0, 1] < -.5


def test_mixture_counts_small_samples_and_seeded_replay():
    kwargs = dict(n_samples_per_batch=2, m=16, families={"bb1": 1, "bb7": 1, "joe": 1},
                  mixture_prob=1, n_mixture_components=(4, 4), seed=123)
    a, b = OnTheFlyCopulaDataset(**kwargs), OnTheFlyCopulaDataset(**kwargs)
    for i in range(4):
        x, y = a[i], b[i]
        assert x["samples"].shape == (2, 2)
        np.testing.assert_array_equal(x["samples"], y["samples"])
        np.testing.assert_array_equal(x["density"], y["density"])
        assert abs(x["density"].mean().item()-1) < 1e-6


@pytest.mark.parametrize("families", [{"conditional_gaussian": 1}, {"bb6": 1}, {"joe": -1}, {}])
def test_invalid_training_configuration_fails_early(families):
    with pytest.raises(ValueError):
        OnTheFlyCopulaDataset(families=families)


def test_complex_only_mixture_is_rejected():
    with pytest.raises(ValueError, match="Mixtures require"):
        OnTheFlyCopulaDataset(families={"complex": 1}, mixture_prob=.5)
