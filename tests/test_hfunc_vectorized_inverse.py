"""Qualify opt-in batched inversion against the corrected scalar lookup."""
import pickle
import numpy as np
import pytest
from vdc.models.hfunc import HFuncLookup
from vdc.vine.api import VineCopulaModel


@pytest.mark.parametrize('shape', [(4, 8), (64, 64), (128, 128)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('spline', [False, True])
def test_vectorized_inverse_matches_scalar(shape, dtype, spline):
    rng = np.random.default_rng(135)
    density = np.exp(rng.normal(0, 3, shape)).astype(dtype)
    scalar = HFuncLookup(density, use_spline=spline)
    fast = HFuncLookup(density, use_spline=spline, vectorized_inverse=True)
    q = np.r_[0., 1., 1e-12, 1-1e-12, rng.random(513)]
    for axis in ('u', 'v'):
        grid = fast.v_grid if axis == 'u' else fast.u_grid
        conditioning = np.resize(np.r_[0., grid[0], grid[-2], (grid[-2]+grid[-1])/2,
                                       grid[-1], 1., rng.random(53)], q.shape)
        old = getattr(scalar, 'hinv_'+axis+'_given_'+('v' if axis=='u' else 'u'))
        new = getattr(fast, 'hinv_'+axis+'_given_'+('v' if axis=='u' else 'u'))
        np.testing.assert_allclose(new(q, conditioning), old(q, conditioning), atol=2e-14, rtol=0)
    # The optimization cannot change any fitted grid or forward operation.
    np.testing.assert_array_equal(fast.pdf(q, q), scalar.pdf(q, q))
    np.testing.assert_array_equal(fast.h_u_given_v(q, q), scalar.h_u_given_v(q, q))


def test_vectorized_inverse_plateaus_boundaries_and_shapes():
    scalar = HFuncLookup(np.ones((4, 4)), use_spline=False)
    fast = HFuncLookup(np.ones((4, 4)), use_spline=False, vectorized_inverse=True)
    table = np.repeat(np.array([0., .5, .5, 1.])[:,None],4,axis=1)
    scalar.h_u_given_v_grid = table.copy(); fast.h_u_given_v_grid = table.copy()
    q = np.array([[-.1, 0., .5], [.5+1e-13, 1., 1.1]])
    c = np.full_like(q, .4)
    np.testing.assert_array_equal(fast.hinv_u_given_v(q,c), scalar.hinv_u_given_v(q,c))
    assert fast.hinv_u_given_v(np.array([]),np.array([])).shape == (0,)
    assert fast.hinv_u_given_v(.5,.3).shape == ()
    restored = pickle.loads(pickle.dumps(fast))
    np.testing.assert_array_equal(restored.hinv_u_given_v(q,c), fast.hinv_u_given_v(q,c))


def test_vine_builder_forwards_opt_in_and_preserves_default():
    default = VineCopulaModel(device='cpu',hfunc_use_spline=False)
    fast = VineCopulaModel(device='cpu',hfunc_use_spline=False,hfunc_vectorized_inverse=True)
    assert not default._make_hfunc(np.ones((4,4))).vectorized_inverse
    assert fast._make_hfunc(np.ones((4,4))).vectorized_inverse
def test_production_vine_preservation_and_saved_option(tmp_path):
    import torch
    from vdc.vine.api import VineCopulaModel

    class ConstantModel(torch.nn.Module):
        def forward(self, x, t):
            z = torch.linspace(-1, 1, x.shape[-1], device=x.device)
            return {'log_density': (z[:, None] * z[None, :])[None, None].expand(len(x), 1, -1, -1)}

    train = np.random.default_rng(77).random((101, 4))
    probes = np.random.default_rng(78).random((53, 4))
    opts = dict(vine_type='dvine', m=8, device='cpu', projection_iters=15,
                hfunc_use_spline=False, batch_edges=True, edge_batch_size=16)
    old = VineCopulaModel(**opts)
    fast = VineCopulaModel(**opts, hfunc_vectorized_inverse=True)
    for vine in (old, fast):
        vine.fit(train, diffusion_model=ConstantModel(), verbose=False)
    np.testing.assert_array_equal(old.logpdf(probes), fast.logpdf(probes))
    np.testing.assert_array_equal(old.rosenblatt(probes), fast.rosenblatt(probes))
    np.testing.assert_allclose(old.inverse_rosenblatt(probes), fast.inverse_rosenblatt(probes), atol=2e-14, rtol=0)
    path = tmp_path / 'vine.pkl'
    fast.save(path)
    loaded = VineCopulaModel.load(path)
    assert loaded.hfunc_vectorized_inverse
    np.testing.assert_array_equal(loaded.inverse_rosenblatt(probes), fast.inverse_rosenblatt(probes))
