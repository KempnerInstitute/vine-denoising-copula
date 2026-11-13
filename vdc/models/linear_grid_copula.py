# ruff: noqa: E702  (verbatim port of the research implementation)
"""Positive bilinear copulas with matching PDF, h and inverse-h functions.

A density is stored at grid nodes inside the unit square and extended as a
constant from the outer nodes to the boundary. Between nodes it is bilinear.
The conditional CDFs (h-functions) are exact integrals of that same density,
and their inverses solve the resulting piecewise-quadratic equations, so the
PDF, h and inverse-h functions describe one distribution. Margins are made
uniform with the integrated basis weights of this representation
(:func:`basis_weights`), not with cell-average masses.

This is an opt-in representation (``VineCopulaModel(pair_estimator=
"linear_grid")`` and the ``linear_grid`` pretrained models). The legacy
:class:`vdc.models.hfunc.HFuncLookup` behavior is unchanged. Nonuniform nodes
are supported; the released linear-grid models use uniform cell centers.
"""
import numpy as np


def basis_weights(nodes):
    """Integrals of the bilinear basis functions along one axis.

    For nodes ``x_1 < ... < x_m`` in (0, 1), with constant extension to the
    boundary, ``int_0^1 f(x) dx = sum_i w_i f(x_i)``. Uniform cell centers give
    ``w_i = 1/m``. A density ``D`` on the node grid has uniform margins when
    ``D @ w == 1`` and ``w @ D == 1``.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1 or len(nodes) < 2 or not np.all(np.diff(nodes) > 0) or nodes[0] <= 0 or nodes[-1] >= 1:
        raise ValueError("Nodes must be strictly increasing inside (0,1)")
    extended = np.r_[0., nodes, 1.]
    w = .5*(extended[2:]-extended[:-2])
    w[0] += .5*nodes[0]
    w[-1] += .5*(1-nodes[-1])
    return w


def _newton_projection(density, weights, tolerance, iterations=100):
    """Finish matrix scaling with damped Newton steps on the column potentials.

    Row constraints are eliminated analytically. Both IPFP and this dual solve
    apply only positive row/column scaling, preserving all odds ratios. The
    singular constant-potential direction is removed by a pseudoinverse.
    """
    from scipy.special import logsumexp
    logbase = np.log(density)
    logw = np.log(weights)
    potential = np.zeros(len(weights))

    def evaluate(y):
        logmass = logbase + logw[None, :] + y[None, :]
        mass = np.exp(logmass-logsumexp(logmass, axis=1)[:, None])
        d = mass/weights[None, :]
        residual = weights @ d-1
        return d, mass, residual

    for iteration in range(iterations):
        d, mass, residual = evaluate(potential)
        error = float(max(np.max(np.abs(d @ weights-1)), np.max(np.abs(residual))))
        if error < tolerance:
            return d, error, iteration+1
        gradient = weights*residual
        hessian = np.diag(weights*(residual+1))-mass.T @ (weights[:, None]*mass)
        direction = np.linalg.lstsq(hessian, -gradient, rcond=1e-14)[0]
        direction -= weights @ direction
        direction *= min(1., 10./max(np.max(np.abs(direction)), 1e-20))
        merit = float(weights @ residual**2)
        for backtrack in range(40):
            step = 2.**(-backtrack)
            candidate = potential+step*direction
            trial, _, next_residual = evaluate(candidate)
            if np.isfinite(trial).all() and np.min(trial)>0 and float(weights @ next_residual**2)<merit:
                potential = candidate
                break
        else:
            raise ValueError(f"Newton marginal projection stalled: marginal error {error}")
    raise ValueError(f"Newton marginal projection failed: marginal error {error}")


def project_density(density, nodes, tolerance=1e-11, max_iterations=10000, solver="ipfp"):
    """Scale a positive node-value grid to uniform margins in float64.

    Alternates row and column scaling under :func:`basis_weights` until the
    largest marginal error is below ``tolerance``. With ``solver="hybrid"``,
    grids that do not converge within ``max_iterations`` are completed by
    damped Newton steps on the dual potentials; grids that converge earlier are
    returned bitwise identical to ``solver="ipfp"``. Returns
    ``(density, marginal_error, iterations)``; raises ``ValueError`` on failure.
    """
    if solver not in ("ipfp", "hybrid"):
        raise ValueError("Projection solver must be ipfp or hybrid")
    d = np.asarray(density, dtype=np.float64).copy()
    w = basis_weights(nodes)
    if d.shape != (len(w), len(w)) or not np.isfinite(d).all() or np.min(d) <= 0:
        raise ValueError("Density must be finite, positive and match nodes")
    for iteration in range(max_iterations):
        d /= (d @ w)[:, None]
        d /= (w @ d)[None, :]
        error = max(np.max(np.abs(d @ w-1)), np.max(np.abs(w @ d-1)))
        if error < tolerance:
            return d, float(error), iteration+1
    if solver == "hybrid":
        d, error, steps = _newton_projection(d, w, tolerance)
        return d, error, max_iterations+steps
    raise ValueError(f"Weighted projection failed: marginal error {error}")


class LinearGridCopula:
    """Bilinear copula density with consistent conditional CDFs and inverses.

    Args:
        density_grid: ``(m, m)`` positive density values at ``nodes x nodes``
            (first axis ``u``, second axis ``v``).
        nodes: Strictly increasing node coordinates in (0, 1).
        project: Scale to uniform margins (:func:`project_density`). If False,
            the grid must already have uniform margins (error <= 1e-9).
        solver: ``"ipfp"`` or ``"hybrid"`` (see :func:`project_density`).

    Attributes ``marginal_error`` and ``projection_iterations`` record the
    projection. Methods accept points in [0, 1] and raise ``ValueError``
    otherwise. ``h_u_given_v(u, v)`` is ``P(U <= u | V = v)`` and
    ``h_v_given_u(v, u)`` is ``P(V <= v | U = u)``; the inverse functions
    take the probability first. The interface matches ``HFuncLookup``, so the
    object can be used as a vine pair copula.
    """

    def __init__(self, density_grid, nodes, project=True, solver="ipfp"):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        w = basis_weights(self.nodes)
        d = np.asarray(density_grid, dtype=np.float64)
        if project:
            d, self.marginal_error, self.projection_iterations = project_density(d, self.nodes, solver=solver)
        else:
            if d.shape != (len(w), len(w)) or not np.isfinite(d).all() or np.min(d) <= 0:
                raise ValueError("Invalid density")
            self.marginal_error = float(max(np.max(np.abs(d @ w-1)), np.max(np.abs(w @ d-1))))
            self.projection_iterations = 0
            if self.marginal_error > 1e-9:
                raise ValueError("Unprojected copula margins are not uniform")
        self.density_grid = d
        self.knots = np.r_[0., self.nodes, 1.]
        self._d = np.pad(d, 1, mode="edge")
        self._prefix = [self._integrate(self._d), self._integrate(self._d.T)]

    def _integrate(self, d):
        mass = .5*(d[1:]+d[:-1])*np.diff(self.knots)[:, None]
        return np.vstack([np.zeros((1, d.shape[1])), mass.cumsum(0)])

    def _indices(self, x):
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
            raise ValueError("Points must lie in [0,1]")
        i = np.clip(np.searchsorted(self.knots, x, side="right")-1, 0, len(self.knots)-2)
        return i, (x-self.knots[i])/(self.knots[i+1]-self.knots[i])

    def pdf(self, u, v):
        """Copula density at ``(u, v)``."""
        u, v = np.broadcast_arrays(u, v)
        i, a = self._indices(u); j, b = self._indices(v)
        d = self._d
        return (1-a)*((1-b)*d[i,j]+b*d[i,j+1])+a*((1-b)*d[i+1,j]+b*d[i+1,j+1])

    def _h(self, u, v, transpose=False):
        u, v = np.broadcast_arrays(u, v)
        i, a = self._indices(u); j, b = self._indices(v)
        d, p = (self._d.T if transpose else self._d), self._prefix[int(transpose)]
        left = (1-b)*d[i,j]+b*d[i,j+1]
        right = (1-b)*d[i+1,j]+b*d[i+1,j+1]
        return (1-b)*p[i,j]+b*p[i,j+1]+(u-self.knots[i])*(left+.5*a*(right-left))

    def h_u_given_v(self, u, v):
        """Conditional CDF ``P(U <= u | V = v)``."""
        return self._h(u, v)

    def h_v_given_u(self, v, u):
        """Conditional CDF ``P(V <= v | U = u)``."""
        return self._h(v, u, True)

    def _hinv(self, q, v, transpose=False):
        q, v = np.broadcast_arrays(np.asarray(q, dtype=float), np.asarray(v, dtype=float))
        shape = q.shape
        q, v = q.ravel(), v.ravel()
        self._indices(q)
        j, b = self._indices(v)
        p, d = self._prefix[int(transpose)], self._d.T if transpose else self._d
        cumulative = (1-b)[None]*p[:,j]+b[None]*p[:,j+1]
        i = np.clip((cumulative <= q[None]).sum(0)-1, 0, len(self.knots)-2)
        left = (1-b)*d[i,j]+b*d[i,j+1]
        right = (1-b)*d[i+1,j]+b*d[i+1,j+1]
        width = self.knots[i+1]-self.knots[i]
        mass = np.maximum(q-cumulative[i, np.arange(len(q))], 0)
        slope = (right-left)/width
        # Stable root of a*x + .5*b*x*x = mass, including b=0.
        delta = 2*mass/(left+np.sqrt(np.maximum(left*left+2*slope*mass, 0)))
        result = self.knots[i]+np.clip(delta, 0, width)
        result = np.where(q == 0, 0., np.where(q == 1, 1., result))
        return result.reshape(shape)

    def hinv_u_given_v(self, q, v):
        """Solve ``h_u_given_v(u, v) = q`` for ``u``."""
        return self._hinv(q, v)

    def hinv_v_given_u(self, q, u):
        """Solve ``h_v_given_u(v, u) = q`` for ``v``."""
        return self._hinv(q, u, True)
