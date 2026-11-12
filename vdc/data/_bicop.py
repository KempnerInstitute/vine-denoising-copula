"""Validated parametric sampling and density targets for training data."""
from __future__ import annotations

import numpy as np


def bicop(family: str, parameters):
    """Construct a pyvinecopulib copula; invalid parameters raise an error."""
    import pyvinecopulib as pv

    return pv.Bicop(family=getattr(pv.BicopFamily, family),
                    parameters=np.asarray(parameters, dtype=float).reshape(-1, 1))


def sample(family: str, parameters, n: int, seed: int | None = None):
    if n < 0:
        raise ValueError("n must be nonnegative")
    copula = bicop(family, parameters)
    if n == 0:
        return np.empty((0, 2), dtype=float)
    # Respect the public generators' NumPy seeding convention when no explicit
    # seed is passed; an explicit seed never changes NumPy's global RNG state.
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    return copula.simulate(int(n), seeds=[int(seed)])


def density(family: str, parameters, u, v):
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    points = np.column_stack([u.ravel(), v.ravel()])
    return bicop(family, parameters).pdf(points).reshape(u.shape)
