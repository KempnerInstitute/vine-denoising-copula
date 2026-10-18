"""The opt-in execution path preserves structure, tables and full-vine scores."""
import pickle
import numpy as np
import pytest
import torch
from vdc.models.hfunc import HFuncLookup
from vdc.vine.vine_types import build_dvine_structure
from vdc.vine.api import VineCopulaModel
from test_compiled_sampler import kernel


@pytest.mark.parametrize('ties', [False, True])
@pytest.mark.parametrize('order', [None, [3, 1, 0, 4, 2]])
def test_cached_tau_structure_identity(ties, order):
    x=np.random.default_rng(2).random((157,5))
    if ties:x=np.round(x,1)
    reference=build_dvine_structure(x,order=order)
    fast=build_dvine_structure(x,order=order,reuse_tau=True)
    assert fast.order==reference.order
    for old,new in zip(reference.trees,fast.trees):
        assert new.edges==old.edges
        np.testing.assert_array_equal(new.tau_values,old.tau_values)


@pytest.mark.parametrize('dtype', [np.float32,np.float64])
def test_vectorized_table_identity(dtype):
    d=np.exp(np.random.default_rng(3).normal(0,4,(64,128))).astype(dtype)
    reference=HFuncLookup(d,use_spline=False)
    fast=HFuncLookup(d,use_spline=False,vectorized_tables=True)
    for key in ['density_grid','h_u_given_v_grid','h_v_given_u_grid']:
        np.testing.assert_array_equal(getattr(reference,key),getattr(fast,key))


def test_full_vine_cpu_identity_and_saved_options(tmp_path, kernel):
    class ConstantModel(torch.nn.Module):
        def forward(self,x,t):
            m=x.shape[-1]
            z=torch.linspace(-1,1,m,device=x.device)
            return {'log_density':(z[:,None]*z[None,:])[None,None].expand(len(x),1,m,m)}
    x=np.random.default_rng(5).random((101,4))
    probes=np.random.default_rng(6).random((53,4))
    model=ConstantModel()
    common=dict(vine_type='dvine',m=8,device='cpu',hfunc_use_spline=False,
                projection_iters=15,batch_edges=True,edge_batch_size=16,hfunc_vectorized_inverse=True)
    old=VineCopulaModel(**common);fast=VineCopulaModel(**common,fast_fit=True)
    for vine in (old,fast):vine.fit(x,diffusion_model=model,verbose=False)
    np.testing.assert_array_equal(old.logpdf(probes),fast.logpdf(probes))
    np.testing.assert_array_equal(old.rosenblatt(probes),fast.rosenblatt(probes))
    np.testing.assert_array_equal(old.inverse_rosenblatt(probes),fast.inverse_rosenblatt(probes))
    path=tmp_path/'vine.pkl';fast.save(path);loaded=VineCopulaModel.load(path)
    assert loaded.fast_fit and loaded.hfunc_vectorized_inverse
    np.testing.assert_array_equal(loaded.logpdf(probes),fast.logpdf(probes))
    expected=fast.inverse_rosenblatt(probes)
    for obj in (fast,loaded):
        compiled=obj.prepare_sampler(kernel=kernel,num_threads=4)
        assert compiled.backend=='compiled'
        np.testing.assert_array_equal(compiled.inverse_rosenblatt(probes),expected)
        fallback=obj.prepare_sampler(max_table_bytes=0)
        assert fallback.backend=='numpy' and 'limit is 0' in fallback.fallback_reason
        np.testing.assert_array_equal(fallback.inverse_rosenblatt(probes),expected)
        missing=obj.prepare_sampler(compiler='vdc-missing-compiler',cache_dir=tmp_path/'empty')
        assert missing.backend=='numpy' and 'compiler unavailable' in missing.fallback_reason
        np.testing.assert_array_equal(missing.inverse_rosenblatt(probes),expected)
    with path.open('rb') as f: state=pickle.load(f)
    del state['fast_fit']
    with path.open('wb') as f: pickle.dump(state,f)
    legacy=VineCopulaModel.load(path)
    assert legacy.fast_fit is False
    np.testing.assert_array_equal(legacy.logpdf(probes),old.logpdf(probes))
    np.testing.assert_array_equal(legacy.prepare_sampler(kernel=kernel).inverse_rosenblatt(probes),expected)
