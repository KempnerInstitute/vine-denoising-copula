"""Information-theoretic estimators used for paper comparisons.

We keep these as lightweight, dependency-minimal utilities to support
"literature-style" evaluation beyond density metrics.

Currently implemented
---------------------
- KSG kNN mutual information estimator (continuous, kNN-based).

Notes
-----
For bivariate copulas, mutual information can also be computed directly from the
estimated copula density grid:

    I(U;V) = ∬ c(u,v) log c(u,v) du dv

See `vdc.utils.metrics.mutual_information_from_density_grid`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


def ksg_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 5,
    seed: Optional[int] = None,
) -> float:
    """KSG mutual information estimator (kNN, continuous).

    Implements the classic Kraskov–Stögbauer–Grassberger estimator using the
    Chebyshev (L∞) metric, which is standard for KSG.

    Args:
        x: (n,) or (n,dx)
        y: (n,) or (n,dy)
        k: number of neighbors (typical: 3–10). Must satisfy 1 <= k < n.
        seed: optional RNG seed for tiny tie-breaking jitter.

    Returns:
        Estimated mutual information in nats.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must have same n, got {x.shape[0]} vs {y.shape[0]}")
    n = int(x.shape[0])
    if not (1 <= int(k) < n):
        raise ValueError(f"k must satisfy 1 <= k < n, got k={k}, n={n}")

    # Jitter to break exact ties (common with discretized / rounded data).
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        scale = 1e-10
        x = x + scale * rng.standard_normal(size=x.shape)
        y = y + scale * rng.standard_normal(size=y.shape)

    xy = np.concatenate([x, y], axis=1)
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    # Distance to k-th neighbor in joint space (exclude self -> query k+1).
    dists, _ = tree_xy.query(xy, k=int(k) + 1, p=np.inf)
    eps = dists[:, int(k)]

    # Use a strict inequality radius: eps - tiny to exclude boundary points.
    # This is standard in KSG implementations to handle ties consistently.
    eps = np.maximum(eps - 1e-15, 0.0)

    nx = np.empty(n, dtype=np.int64)
    ny = np.empty(n, dtype=np.int64)
    for i in range(n):
        nx[i] = len(tree_x.query_ball_point(x[i], eps[i], p=np.inf)) - 1
        ny[i] = len(tree_y.query_ball_point(y[i], eps[i], p=np.inf)) - 1

    # KSG-1 estimator:
    # I = ψ(k) + ψ(n) - ⟨ψ(nx+1) + ψ(ny+1)⟩
    mi = digamma(int(k)) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(mi)

