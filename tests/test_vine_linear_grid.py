"""Opt-in linear-grid vines: defaults unchanged, consistent fits, persistence and guards."""
import pickle

import numpy as np
import pytest
import torch
from scipy.stats import norm

from vdc.models.copula_denoiser import CopulaDenoiser
from vdc.models.hfunc import HFuncLookup
from vdc.models.linear_grid_copula import LinearGridCopula
from vdc.pretrained import LoadedPretrainedModel, linear_grid_settings_for
from vdc.vine.api import VineCopulaModel
from vdc.vine.recursion import VineRecursion
from vdc.vine.vine_types import build_dvine_structure


def network(m=16):
    torch.manual_seed(1)
    model = CopulaDenoiser(m=m, input_channels=4, base_channels=8, depth=2, blocks_per_level=1,
                           time_emb_dim=16, dropout=0.1).eval()
    return model


def gaussian_data(n, d, rho, seed):
    rng = np.random.default_rng(seed)
    cov = np.array([[rho**abs(i-j) for j in range(d)] for i in range(d)])
    return norm.cdf(rng.multivariate_normal(np.zeros(d), cov, size=n))


LINEAR = dict(m=16, device="cpu", projection_iters=30, hfunc_use_spline=False, batch_edges=True,
              edge_batch_size=64, pair_estimator="linear_grid", copula_projection_solver="hybrid",
              conditional_clip_eps=1e-12, forward_batch_size=4)


def test_defaults_are_unchanged():
    vine = VineCopulaModel(vine_type="dvine")
    assert vine.pair_estimator == "legacy" and vine.conditional_clip_eps == 1e-6
    assert vine.forward_batch_size is None and vine.copula_projection_solver == "ipfp"
    assert isinstance(vine._make_hfunc(np.ones((8, 8))), HFuncLookup)
    structure = build_dvine_structure(np.random.default_rng(0).random((20, 3)), order=[0, 1, 2])
    recursion = VineRecursion(structure, vine_type="dvine")
    assert recursion.clip_eps == 1e-6
    np.testing.assert_array_equal(recursion._clip01(np.array([0., 1.])), [1e-6, 1-1e-6])
    del recursion.clip_eps  # objects pickled by earlier versions lack the attribute
    np.testing.assert_array_equal(recursion._clip01(np.array([0., 1.])), [1e-6, 1-1e-6])


@pytest.mark.parametrize("kwargs", [dict(pair_estimator="spline"), dict(copula_projection_solver="hybrid"),
                                    dict(forward_batch_size=16), dict(conditional_clip_eps=0.),
                                    dict(pair_estimator="linear_grid", hfunc_vectorized_inverse=True),
                                    dict(pair_estimator="linear_grid", forward_batch_size=0)])
def test_invalid_combinations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        VineCopulaModel(vine_type="dvine", **kwargs)


def test_linear_grid_dvine_is_consistent_and_batching_invariant(tmp_path):
    x = gaussian_data(300, 4, .6, 7)
    probes = gaussian_data(80, 4, .6, 8)
    batched = VineCopulaModel(vine_type="dvine", **LINEAR)
    batched.fit(x, diffusion_model=network(), verbose=False)
    single = VineCopulaModel(vine_type="dvine", **{**LINEAR, "batch_edges": False})
    single.fit(x, diffusion_model=network(), verbose=False)
    for a, b in zip(batched.vine.pair_copulas, single.vine.pair_copulas):
        for ca, cb in zip(a, b):
            assert isinstance(ca.hfunc, LinearGridCopula)
            np.testing.assert_array_equal(ca.density_grid, cb.density_grid)
    assert len(batched.normalization_diagnostics) == 6
    assert max(e for e, _ in batched.normalization_diagnostics) < 1e-11
    lp = batched.logpdf(probes)
    assert np.isfinite(lp).all()
    np.testing.assert_array_equal(lp, single.logpdf(probes))
    w = np.random.default_rng(9).uniform(1e-6, 1-1e-6, (60, 4))
    np.testing.assert_allclose(batched.rosenblatt(batched.inverse_rosenblatt(w)), w, atol=1e-9)
    assert batched.vine.clip_eps == 1e-12

    path = tmp_path / "vine.pkl"
    batched.save(path)
    loaded = VineCopulaModel.load(path)
    assert (loaded.pair_estimator, loaded.conditional_clip_eps, loaded.forward_batch_size) == ("linear_grid", 1e-12, 4)
    np.testing.assert_array_equal(loaded.logpdf(probes), lp)

    prepared = batched.prepare_sampler()
    assert prepared.backend == "numpy" and "HFuncLookup" in prepared.fallback_reason
    np.testing.assert_array_equal(prepared.inverse_rosenblatt(w), batched.inverse_rosenblatt(w))
    with pytest.raises(ValueError):
        batched.prepare_sampler(backend="compiled")


@pytest.mark.parametrize("vine_type", ["cvine", "rvine"])
def test_other_structures_fit_with_linear_grid(vine_type):
    x = gaussian_data(200, 4, .5, 11)
    vine = VineCopulaModel(vine_type=vine_type, **LINEAR)
    vine.fit(x, diffusion_model=network(), verbose=False)
    assert np.isfinite(vine.logpdf(x[:20])).all()
    assert all(isinstance(c.hfunc, LinearGridCopula) for level in vine.vine.pair_copulas for c in level)


def test_linear_grid_rejects_iterative_diffusion():
    vine = VineCopulaModel(vine_type="dvine", **LINEAR)
    with pytest.raises(ValueError):
        vine.fit(gaussian_data(50, 3, .5, 1), diffusion_model=network(), diffusion=object(), verbose=False)


def test_from_pretrained_applies_manifest_settings():
    model = network()
    config = {"data": {"m": 16}, "model": {"type": "denoiser"}}
    inference = {"pair_estimator": "linear_grid", "grid_size": 16, "projection_iters": 30,
                 "copula_projection_solver": "hybrid", "conditional_clip_eps": 1e-12,
                 "forward_batch_size": 16, "edge_batch_size": 64}
    bundle = LoadedPretrainedModel("test-v2", None, {"inference": inference}, model, None, config,
                                   torch.device("cpu"))
    vine = VineCopulaModel.from_pretrained(bundle)
    assert (vine.vine_type, vine.m, vine.pair_estimator, vine.forward_batch_size) == ("dvine", 16, "linear_grid", 16)
    assert (vine.edge_batch_size, vine.projection_iters, vine.conditional_clip_eps) == (64, 30, 1e-12)
    assert VineCopulaModel.from_pretrained(bundle, projection_iters=15).projection_iters == 15
    legacy = LoadedPretrainedModel("test-v1", None, {}, model, None, {"data": {"m": 64}}, torch.device("cpu"))
    assert linear_grid_settings_for(legacy) is None
    vine = VineCopulaModel.from_pretrained(legacy, vine_type="cvine")
    assert (vine.m, vine.pair_estimator, vine.conditional_clip_eps) == (64, "legacy", 1e-6)


def test_research_checkpoint_config_is_recognized():
    config = {"data": {"m": 128}, "estimator_study": {"representation": "linear_nodes", "loss": "nll",
                                                      "spacing": "uniform"}}
    bundle = LoadedPretrainedModel("custom", None, {}, network(), None, config, torch.device("cpu"))
    settings = linear_grid_settings_for(bundle)
    assert settings.grid_size == 128 and settings.forward_batch_size == 16


def test_compiled_sampler_rejects_nondefault_clipping():
    from vdc.vine.compiled_sampler import CompiledDVineSampler
    structure = build_dvine_structure(np.random.default_rng(3).random((30, 3)), order=[0, 1, 2])
    vine = VineRecursion(structure, vine_type="dvine", clip_eps=1e-12)
    with pytest.raises(ValueError, match="clipping"):
        CompiledDVineSampler(vine, "unused-kernel-path")


def test_pickled_linear_grid_copula_round_trip():
    density = np.exp(np.random.default_rng(2).normal(0, .5, (8, 8)))
    copula = LinearGridCopula(density, (np.arange(8)+.5)/8, solver="hybrid")
    restored = pickle.loads(pickle.dumps(copula))
    u = np.random.default_rng(3).uniform(size=(2, 50))
    np.testing.assert_array_equal(restored.pdf(*u), copula.pdf(*u))
    np.testing.assert_array_equal(restored.hinv_v_given_u(*u), copula.hinv_v_given_u(*u))
