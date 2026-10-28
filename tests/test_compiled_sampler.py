"""Strict preservation gates for the optional C++ sampler, before timing it."""
import os
from pathlib import Path
import numpy as np
import pytest
from vdc.models.hfunc import HFuncLookup
from vdc.vine.recursion import VineRecursion,VinePairCopula
from vdc.vine.vine_types import build_dvine_structure
from vdc.vine.compiled_sampler import CompiledDVineSampler,build_kernel


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    supplied=os.environ.get('VDC_SAMPLER_KERNEL')
    if supplied: return Path(supplied)
    path=tmp_path_factory.mktemp('kernel')/'sampler.so'
    build_kernel(path)
    return path


def make_vine(m=64,d=7,truncated=False,dtype=np.float64):
    rng=np.random.default_rng(61013)
    order=rng.permutation(d).tolist()
    structure=build_dvine_structure(rng.random((40,d)),order=order,truncation_level=2 if truncated else None)
    vine=VineRecursion(structure,vine_type='dvine')
    for level,tree in enumerate(structure.trees):
        for index,(i,j,cond) in enumerate(tree.edges):
            density=np.exp(rng.normal(0,1.5,(m,m)))
            # Strongly heterogeneous edges, without deep-vine saturation everywhere.
            for _ in range(20):
                density/=density.mean(0,keepdims=True)
                density/=density.mean(1,keepdims=True)
            density=density.astype(dtype)
            if index%2: i,j=j,i;density=density.T.copy()
            h=HFuncLookup(density,use_spline=False,vectorized_inverse=True)
            vine.add_pair_copula(VinePairCopula((i,j,cond),density,h,level))
    return vine


@pytest.mark.parametrize('m',[4,64,128])
@pytest.mark.parametrize('truncated',[False,True])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_bitwise_equivalence(kernel,m,truncated,dtype):
    vine=make_vine(m=m,truncated=truncated,dtype=dtype)
    rng=np.random.default_rng(313)
    w=rng.random((131,vine.d))
    w[:7]=np.asarray([0,1,1e-15,1-1e-15,.5,.5/m,1-.5/m])[:,None]
    reference=vine.inverse_rosenblatt(w)
    density_before=vine.logpdf(w)
    fast=CompiledDVineSampler(vine,kernel)
    for threads,block in ((1,1),(1,64),(4,32),(4,128)):
        answer=fast.inverse_rosenblatt(w,threads=threads,block_size=block)
        np.testing.assert_array_equal(answer,reference)
    np.testing.assert_array_equal(vine.logpdf(w),density_before)
    np.testing.assert_array_equal(vine.inverse_rosenblatt(w),reference)
    np.testing.assert_array_equal(fast.inverse_rosenblatt(w[::2]),reference[::2])
    assert fast.inverse_rosenblatt(w[:0]).shape==(0,vine.d)


def test_plateaus_and_tied_quantiles(kernel):
    vine=make_vine(m=4,d=3)
    for level in vine.pair_copulas:
        for pair in level:
            pair.hfunc.h_u_given_v_grid=np.tile(np.array([0.,.5,.5,1.])[:,None],(1,4))
            pair.hfunc.h_v_given_u_grid=np.tile(np.array([0.,.5,.5,1.])[None,:],(4,1))
    w=np.resize(np.array([0.,1.,.5,.5+1e-13,.5-1e-13,1e-6,1-1e-6]),(71,3))
    fast=CompiledDVineSampler(vine,kernel)
    np.testing.assert_array_equal(fast.inverse_rosenblatt(w,threads=4),vine.inverse_rosenblatt(w))


def test_snapshot_input_guards_and_rng(kernel):
    vine=make_vine(m=4,d=3)
    fast=CompiledDVineSampler(vine,kernel)
    w=np.random.default_rng(45).random((27,3))
    state=np.random.get_state()
    expected=fast.inverse_rosenblatt(w)
    after=np.random.get_state()
    assert state[0]==after[0] and state[2:]==after[2:]
    np.testing.assert_array_equal(state[1],after[1])
    vine.pair_copulas[0][0].hfunc.h_u_given_v_grid[:]=0
    np.testing.assert_array_equal(fast.inverse_rosenblatt(w),expected)
    for bad in (np.zeros((3,4)),np.array([[np.nan,0,0]]),np.zeros(3)):
        with pytest.raises(ValueError): fast.inverse_rosenblatt(bad)
    with pytest.raises(ValueError): fast.inverse_rosenblatt(w,threads=0)
    with pytest.raises(ValueError): fast.inverse_rosenblatt(w,block_size=0)
    assert not fast.tables.flags.writeable
    vine.vine_type='cvine'
    with pytest.raises(ValueError): CompiledDVineSampler(vine,kernel)


def test_reject_unsupported_hfunctions(kernel):
    for attribute,value in [('use_spline',True),('vectorized_inverse',False)]:
        vine=make_vine(m=4,d=3)
        setattr(vine.pair_copulas[0][0].hfunc,attribute,value)
        with pytest.raises(ValueError): CompiledDVineSampler(vine,kernel)
