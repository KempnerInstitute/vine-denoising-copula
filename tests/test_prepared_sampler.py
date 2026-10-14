"""Production entry point, automatic fallback, memory and RNG contracts."""
import numpy as np
import pytest
import vdc.vine.compiled_sampler as cs
from vdc.vine.api import VineCopulaModel
from test_compiled_sampler import kernel,make_vine


def model():
    vine=make_vine(m=64,d=5)
    obj=VineCopulaModel(vine_type='dvine',device='cpu',hfunc_use_spline=False,hfunc_vectorized_inverse=True)
    obj.vine=vine;obj.structure=vine.structure;obj.fitted=True
    return obj


@pytest.mark.parametrize('backend',['auto','numpy','compiled'])
def test_public_api_identity_and_rng(kernel,backend,tmp_path):
    obj=model();sampler=obj.prepare_sampler(backend=backend,kernel=kernel,num_threads=4)
    assert sampler.backend==('numpy' if backend=='numpy' else 'compiled')
    expected=obj.simulate(137,seed=18);after=np.random.random(7)
    np.testing.assert_array_equal(sampler.simulate(137,seed=18),expected)
    np.testing.assert_array_equal(np.random.random(7),after)
    uniforms=np.random.default_rng(4).random((129,5))
    np.testing.assert_array_equal(sampler.inverse_rosenblatt(uniforms),obj.inverse_rosenblatt(uniforms))
    assert sampler.simulate(0,seed=3).shape==(0,5)
    path=tmp_path/'model.pkl';obj.save(path)
    loaded=VineCopulaModel.load(path)
    np.testing.assert_array_equal(loaded.prepare_sampler(backend=backend,kernel=kernel).simulate(137,seed=18),expected)


def test_missing_compiler_falls_back_without_rng_changes(tmp_path):
    obj=model()
    np.random.seed(91);state=np.random.get_state()
    sampler=obj.prepare_sampler(compiler='vdc-missing-compiler',cache_dir=tmp_path)
    assert sampler.backend=='numpy' and 'compiler unavailable' in sampler.fallback_reason
    after=np.random.get_state()
    np.testing.assert_array_equal(state[1],after[1]);assert state[2:]==after[2:]
    np.testing.assert_array_equal(sampler.simulate(151,seed=9),obj.simulate(151,seed=9))
    with pytest.raises(FileNotFoundError): obj.prepare_sampler(backend='compiled',compiler='vdc-missing-compiler')


def test_memory_cap_precedes_compilation(monkeypatch):
    def forbidden(*a,**kw): raise AssertionError('Must not compile')
    monkeypatch.setattr(cs,'cached_kernel',forbidden)
    obj=model();sampler=obj.prepare_sampler(max_table_bytes=0)
    assert sampler.backend=='numpy' and 'limit is 0' in sampler.fallback_reason
    assert sampler.compiled is None
    np.testing.assert_array_equal(sampler.simulate(13,seed=8),obj.simulate(13,seed=8))
    with pytest.raises(MemoryError): obj.prepare_sampler(backend='compiled',max_table_bytes=0)


def test_failed_build_and_failed_validation_fall_back(monkeypatch,tmp_path,kernel):
    import subprocess
    obj=model()
    def fail_build(*a,**kw): raise subprocess.CalledProcessError(1,['g++'])
    monkeypatch.setattr(cs,'build_kernel',fail_build)
    sampler=obj.prepare_sampler(cache_dir=tmp_path)
    assert sampler.backend=='numpy' and 'CalledProcessError' in sampler.fallback_reason
    def fail_validation(*a,**kw): raise RuntimeError('identity failure')
    monkeypatch.setattr(cs,'validate_kernel',fail_validation)
    sampler=obj.prepare_sampler(kernel=kernel)
    assert sampler.backend=='numpy' and 'identity failure' in sampler.fallback_reason


def test_unsupported_grids_and_load_errors_fall_back(kernel,tmp_path):
    obj=model();obj.vine.pair_copulas[0][0].hfunc.use_spline=True
    assert obj.prepare_sampler(kernel=kernel).backend=='numpy'
    obj=model();bad=tmp_path/'broken.so';bad.write_text('not a shared library')
    sampler=obj.prepare_sampler(kernel=bad)
    assert sampler.backend=='numpy' and 'OSError' in sampler.fallback_reason


def test_runtime_failure_uses_same_uniforms(kernel,monkeypatch):
    obj=model();sampler=obj.prepare_sampler(kernel=kernel)
    def fail(*a,**kw): raise RuntimeError('kernel failure')
    monkeypatch.setattr(sampler.compiled,'inverse_rosenblatt',fail)
    np.testing.assert_array_equal(sampler.simulate(129,seed=6),obj.simulate(129,seed=6))
    assert sampler.backend=='numpy' and sampler.compiled is None


def test_cache_reuse_and_input_errors(tmp_path,monkeypatch):
    obj=model();sampler=obj.prepare_sampler(cache_dir=tmp_path)
    assert sampler.backend=='compiled'
    def forbidden(*a,**kw): raise AssertionError('Unexpected rebuild')
    monkeypatch.setattr(cs,'build_kernel',forbidden)
    assert obj.prepare_sampler(cache_dir=tmp_path).backend=='compiled'
    for kwargs in ({'backend':'wrong'},{'num_threads':0},{'block_size':0},{'max_table_bytes':-1}):
        with pytest.raises(ValueError): obj.prepare_sampler(**kwargs)
    with pytest.raises(ValueError): sampler.inverse_rosenblatt([[np.nan]*5])
    with pytest.raises(ValueError): sampler.inverse_rosenblatt(np.zeros((1,4)))
    with pytest.raises(RuntimeError): VineCopulaModel(device='cpu').prepare_sampler()
