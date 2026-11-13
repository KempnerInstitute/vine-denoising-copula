"""Linear-grid estimator inputs, projection, fixed-shape batching and settings."""
import json
from importlib import resources

import numpy as np
import pytest
import torch
from scipy.special import ndtr, ndtri

from vdc.models.copula_denoiser import CopulaDenoiser
from vdc.models.linear_grid_copula import LinearGridCopula, basis_weights, project_density
from vdc.models.linear_grid_estimator import (
    FixedBatchForward,
    LinearGridEstimator,
    LinearGridSettings,
    bilinear_density_torch,
    estimate_linear_grid_densities,
    float32_network_execution,
    grid_geometry,
    project_density_torch,
)


def tiny_network(m=16, seed=0):
    torch.manual_seed(seed)
    return CopulaDenoiser(m=m, input_channels=4, base_channels=8, depth=2, blocks_per_level=1,
                          time_emb_dim=16, dropout=0.1).eval()


class Recorder(torch.nn.Module):
    """Returns a fixed log-density pattern and records its inputs."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x, t):
        self.inputs.append((x.clone(), t.clone()))
        m = x.shape[-1]
        z = torch.linspace(-1, 1, m)
        return {"log_density": (z[:, None]*z[None, :])[None, None].expand(len(x), 1, m, m) + x[:, :1]*0}


def test_geometry_uniform_and_probit():
    edges, nodes = grid_geometry(64)
    np.testing.assert_array_equal(edges, np.linspace(0, 1, 65))
    np.testing.assert_allclose(nodes, (np.arange(64)+.5)/64, rtol=0, atol=1e-16)
    np.testing.assert_allclose(basis_weights(nodes), np.full(64, 1/64), atol=1e-16)
    edges, nodes = grid_geometry(16, "probit")
    assert edges[0] == 0 and edges[-1] == 1
    np.testing.assert_allclose(edges[1:-1], ndtr(np.linspace(-3.5, 3.5, 17))[1:-1])
    with pytest.raises(ValueError):
        grid_geometry(16, "log")


def test_differentiable_nonuniform_density_agrees_with_runtime():
    _, nodes = grid_geometry(16, "probit")
    rng = np.random.default_rng(2026)
    base = torch.tensor(rng.uniform(.1, 3, (2, 16, 16)), dtype=torch.float64, requires_grad=True)
    w = torch.tensor(basis_weights(nodes))
    d = project_density_torch(base, w, 100)
    points = torch.tensor(rng.uniform(0, 1, (2, 200, 2)), dtype=torch.float64)
    got = bilinear_density_torch(d, torch.tensor(nodes), points)
    for i in range(2):
        ref = LinearGridCopula(d[i].detach().numpy(), nodes, project=False)
        np.testing.assert_allclose(got[i].detach().numpy(), ref.pdf(*points[i].numpy().T), rtol=1e-12)
        reference, _, _ = project_density(base[i].detach().numpy(), nodes)
        np.testing.assert_allclose(d[i].detach().numpy(), reference, rtol=1e-9)
    (-got.log().mean()).backward()
    assert torch.isfinite(base.grad).all() and base.grad.abs().max() > 0


def test_network_inputs_match_the_evaluated_recipe():
    rng = np.random.default_rng(4)
    pairs = [rng.uniform(0, 1, (n, 2)) for n in (50, 333, 1000)]
    pairs[0][0] = [0., 1.]  # boundary points fall in the outer cells
    model = Recorder()
    grids = estimate_linear_grid_densities(model, pairs, grid_size=8, device="cpu", edge_batch_size=2,
                                           projection_iters=30)
    assert len(model.inputs) == 2 and [len(x) for x, _ in model.inputs] == [2, 1]
    x = torch.cat([x for x, _ in model.inputs])
    edges, nodes = grid_geometry(8)
    area = np.outer(np.diff(edges), np.diff(edges))
    for k, pair in enumerate(pairs):
        counts = np.histogram2d(pair[:, 0], pair[:, 1], bins=(edges, edges))[0]
        assert counts.sum() == len(pair)
        np.testing.assert_array_equal(x[k, 0].numpy(), (counts/(len(pair)*area)).astype(np.float32))
        np.testing.assert_array_equal(x[k, 1].numpy(), np.full((8, 8), np.float32(np.log(len(pair)))))
        z = ndtri(nodes).astype(np.float32)
        np.testing.assert_array_equal(x[k, 2].numpy(), np.repeat(z[:, None], 8, 1))
        np.testing.assert_array_equal(x[k, 3].numpy(), np.repeat(z[None, :], 8, 0))
    assert all(float(t.abs().max()) == 0 for _, t in model.inputs)
    w = basis_weights(nodes)
    for grid in grids:
        assert grid.dtype == np.float32 and grid.shape == (8, 8)
        np.testing.assert_allclose(grid @ w, 1, atol=1e-5)
        np.testing.assert_allclose(w @ grid, 1, atol=1e-5)


@pytest.mark.parametrize("bad", [np.array([[.5, 1.2]]), np.array([[np.nan, .5]]), np.ones((3, 3)) / 2])
def test_invalid_pairs_are_rejected(bad):
    with pytest.raises(ValueError):
        estimate_linear_grid_densities(Recorder(), [bad], grid_size=8, device="cpu")


def test_fixed_batch_forward_pads_and_preserves_order():
    network = tiny_network()
    network.vdc_use_log_n = True
    rng = np.random.default_rng(5)
    x = torch.tensor(rng.normal(size=(7, 4, 16, 16)), dtype=torch.float32)
    t = torch.zeros(7)
    wrapped = FixedBatchForward(network, 4)
    assert wrapped.vdc_use_log_n and wrapped.in_conv is network.in_conv
    out = wrapped(x, t)["log_density"]
    with torch.no_grad():
        first = network(x[:4], t[:4])["log_density"]
        last = network(torch.cat([x[4:], x[4:5]]), torch.zeros(4))["log_density"][:3]
    np.testing.assert_array_equal(out.numpy(), torch.cat([first, last]).numpy())
    with pytest.raises(ValueError):
        wrapped(x[:0], t[:0])
    with pytest.raises(ValueError):
        FixedBatchForward(network, 0)


def test_results_do_not_depend_on_grouping_with_fixed_forward_batch():
    network = tiny_network()
    settings = LinearGridSettings(grid_size=16, forward_batch_size=4, edge_batch_size=3)
    estimator = LinearGridEstimator(network, settings, "cpu")
    rng = np.random.default_rng(6)
    pairs = [rng.uniform(1e-12, 1-1e-12, (n, 2)) for n in (40, 80, 160, 320, 640)]
    together = estimator.estimate_grids(pairs)
    for pair, grid in zip(pairs, together):
        np.testing.assert_array_equal(estimator.estimate_grids([pair])[0], grid)
    copulas = estimator.copulas(pairs[:2])
    assert all(isinstance(c, LinearGridCopula) and c.marginal_error < 1e-11 for c in copulas)


def test_float32_execution_restores_backend_flags():
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        with float32_network_execution():
            assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cudnn.benchmark
            assert not torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32 and torch.backends.cudnn.benchmark
    finally:
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark) = saved


@pytest.mark.parametrize("model_id,m", [("vdc-denoiser-m64-v2", 64), ("vdc-denoiser-m128-v2", 128)])
def test_packaged_settings_are_the_evaluated_ones(model_id, m):
    name = model_id.replace("-", "_") + ".json"
    manifest = json.loads(resources.files("vdc.resources.pretrained").joinpath(name).read_text())
    settings = LinearGridSettings.from_manifest(manifest["inference"])
    assert settings == LinearGridSettings(grid_size=m)
    assert (settings.projection_iters, settings.forward_batch_size, settings.edge_batch_size) == (30, 16, 64)
    assert settings.copula_projection_solver == "hybrid" and settings.conditional_clip_eps == 1e-12
    options = settings.vine_options()
    assert options["pair_estimator"] == "linear_grid" and options["m"] == m and options["batch_edges"]
    with pytest.raises(ValueError):
        LinearGridSettings.from_manifest({"pair_estimator": "legacy"})


@pytest.mark.parametrize("kwargs", [dict(grid_spacing="log"), dict(copula_projection_solver="newton"),
                                    dict(forward_batch_size=0), dict(conditional_clip_eps=0.),
                                    dict(edge_batch_size=0)])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        LinearGridSettings(grid_size=64, **kwargs)
