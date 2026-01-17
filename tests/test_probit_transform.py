"""Unit tests for probit-space Jacobian transforms.

These transforms underpin the "estimate in probit space, map back" idea used in
nonparametric copula estimators (normal-score transform).
"""
from __future__ import annotations

import torch

from vdc.utils.probit_transform import (
    copula_density_to_probit_density,
    probit_density_to_copula_density,
    copula_logdensity_to_probit_logdensity,
    probit_logdensity_to_copula_logdensity,
)


def test_probit_density_roundtrip_identity():
    m = 32
    c = torch.ones(1, 1, m, m)  # simple positive grid
    fz = copula_density_to_probit_density(c, m)
    c2 = probit_density_to_copula_density(fz, m)
    assert torch.allclose(c2, c, atol=1e-6, rtol=1e-6)


def test_probit_logdensity_roundtrip_identity():
    m = 32
    c = torch.ones(1, 1, m, m)
    logc = torch.log(c)
    logfz = copula_logdensity_to_probit_logdensity(logc, m)
    logc2 = probit_logdensity_to_copula_logdensity(logfz, m)
    assert torch.allclose(logc2, logc, atol=1e-6, rtol=1e-6)

