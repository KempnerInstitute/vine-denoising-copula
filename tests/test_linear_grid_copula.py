# ruff: noqa: E702  (verbatim port of the research implementation)
"""Consistent bilinear pair copulas, weighted projection and opt-in vine clipping."""
import numpy as np
import pytest
from scipy.special import ndtr

from vdc.models.linear_grid_copula import LinearGridCopula, basis_weights


@pytest.mark.parametrize("spacing", ["uniform", "probit"])
@pytest.mark.parametrize("m", [8, 64, 128])
def test_hybrid_solver_preserves_solution_and_scaling(spacing, m):
    from vdc.models.linear_grid_copula import project_density
    nodes = (np.arange(m)+.5)/m if spacing == "uniform" else ndtr(np.linspace(-3,3,m))
    w = basis_weights(nodes)
    rng = np.random.default_rng(805)
    d = np.exp(rng.normal(0,2,(m,m)))
    reference,_,_ = project_density(d,nodes)
    unchanged,_,_ = project_density(d,nodes,solver="hybrid")
    np.testing.assert_array_equal(reference,unchanged)
    solved,error,iterations = project_density(d,nodes,max_iterations=2,solver="hybrid")
    assert iterations>2 and error<1e-11
    np.testing.assert_allclose(solved,reference,rtol=2e-9,atol=1e-10)
    np.testing.assert_allclose(solved@w,1,atol=1e-11)
    np.testing.assert_allclose(w@solved,1,atol=1e-11)
    logscale=np.log(solved)-np.log(d)
    np.testing.assert_allclose(logscale-logscale[:,0,None]-logscale[None,0,:]+logscale[0,0],0,atol=1e-12)


def test_hybrid_solver_resolves_nearly_disconnected_mass_blocks():
    from vdc.models.linear_grid_copula import project_density
    m=64;nodes=(np.arange(m)+.5)/m
    d=np.full((m,m),1e-9);d[:m//3,:m//2]=1;d[m//3:,m//2:]=1
    solved,error,_=project_density(d,nodes,max_iterations=2,solver="hybrid")
    assert np.isfinite(solved).all() and solved.min()>0 and error<1e-11


@pytest.mark.parametrize("spacing", ["uniform", "probit"])
@pytest.mark.parametrize("m", [8, 64, 128])
def test_integrated_pdf_h_inverse_and_margins(spacing, m):
    e = np.linspace(0, 1, m+1) if spacing == "uniform" else ndtr(np.linspace(-3.5, 3.5, m+1))
    e[0], e[-1] = 0., 1.
    nodes = .5*(e[:-1]+e[1:])
    rng = np.random.default_rng(123)
    cop = LinearGridCopula(np.exp(rng.normal(0, 2, (m,m))), nodes)
    w = basis_weights(nodes)
    np.testing.assert_allclose(w.sum(), 1, atol=1e-15)
    np.testing.assert_allclose(cop.density_grid@w, 1, atol=1e-11)
    u, v = rng.uniform(1e-5, 1-1e-5, (2, 1000))
    q = np.r_[0., 1e-12, .01, .99, 1-1e-12, 1., u]
    condition = rng.uniform(0, 1, len(q))
    for h, inv in [(cop.h_u_given_v, cop.hinv_u_given_v), (cop.h_v_given_u, cop.hinv_v_given_u)]:
        np.testing.assert_allclose(h(inv(q, condition), condition), q, atol=2e-11)
        np.testing.assert_allclose(h(np.zeros_like(v), v), 0, atol=1e-12)
        np.testing.assert_allclose(h(np.ones_like(v), v), 1, atol=1e-11)
    eps = 1e-8
    derivative = (cop.h_u_given_v(u+eps,v)-cop.h_u_given_v(u-eps,v))/(2*eps)
    np.testing.assert_allclose(derivative, cop.pdf(u,v), atol=2e-7, rtol=2e-6)


def test_independence_and_invalid_grid():
    nodes = np.array([.001,.03,.2,.7,.99])
    c = LinearGridCopula(np.ones((5,5)), nodes)
    q = np.linspace(0,1,101)
    np.testing.assert_allclose(c.hinv_u_given_v(q, .999), q, atol=1e-15)
    np.testing.assert_allclose(c.pdf(q, 1-q), 1, atol=1e-15)
    with pytest.raises(ValueError):
        LinearGridCopula(np.ones((5,5)), nodes[::-1])


def test_vine_jacobian_matches_density_and_clip_is_opt_in():
    from vdc.vine.recursion import VinePairCopula, VineRecursion
    from vdc.vine.vine_types import build_dvine_structure
    rng = np.random.default_rng(76)
    structure = build_dvine_structure(rng.uniform(size=(50,3)), order=[0,1,2])
    vine = VineRecursion(structure, vine_type="dvine", clip_eps=1e-12)
    old = VineRecursion(structure, vine_type="dvine")
    np.testing.assert_array_equal(old._clip01(np.array([0.,1.])), [1e-6,1-1e-6])
    np.testing.assert_array_equal(vine._clip01(np.array([0.,1.])), [1e-12,1-1e-12])
    nodes = ndtr(np.linspace(-3,3,16))
    for level, tree in enumerate(structure.trees):
        for edge in tree.edges:
            h = LinearGridCopula(np.exp(rng.normal(0,.8,(16,16))), nodes)
            vine.add_pair_copula(VinePairCopula(edge, h.density_grid, h, level))
    u = rng.uniform(.1,.9,(20,3)); eps=1e-7
    jac = np.empty((len(u),3,3))
    for axis in range(3):
        offset = np.zeros_like(u); offset[:,axis]=eps
        jac[:,:,axis]=(vine.rosenblatt(u+offset)-vine.rosenblatt(u-offset))/(2*eps)
    np.testing.assert_allclose(np.abs(np.linalg.det(jac)),np.exp(vine.logpdf(u)),rtol=2e-7,atol=1e-8)
    np.testing.assert_allclose(vine.rosenblatt(vine.inverse_rosenblatt(u)),u,atol=1e-10)
