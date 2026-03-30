#!/usr/bin/env python3
"""Robustness tests for vine structure construction under degenerate inputs."""

from __future__ import annotations

import numpy as np

from vdc.vine.vine_types import build_cvine_structure


def test_build_cvine_with_constant_features_does_not_crash() -> None:
    rng = np.random.default_rng(0)
    n, d = 200, 6
    U = rng.uniform(0.0, 1.0, size=(n, d))

    # Create degeneracy: two fully constant columns and one nearly-constant column.
    U[:, 1] = 0.5
    U[:, 4] = 0.5
    U[:, 5] = np.round(U[:, 5], 2)

    structure = build_cvine_structure(U, order=None)

    assert structure.order is not None
    assert len(structure.order) == d
    assert set(structure.order) == set(range(d))
    assert len(structure.trees) == d - 1
