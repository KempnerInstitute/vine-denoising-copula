import math

import numpy as np
import torch

from vdc.data.generators import analytic_logpdf_grid
from vdc.utils.metrics import mutual_information_from_density_grid


def _correct_grid_mi(family: str, params: dict[str, float], m: int = 512) -> float:
    log_d = analytic_logpdf_grid(family, params, m=m)
    d = torch.from_numpy(np.exp(log_d)).float()
    p_mass = d / d.sum()
    du = 1.0 / m
    density = p_mass / (du * du)
    return float((p_mass * torch.log(density.clamp_min(1e-12))).sum().item())


def test_mutual_information_from_density_grid_matches_gaussian_reference() -> None:
    rho = 0.7
    expected = -0.5 * math.log(1.0 - rho * rho)
    log_d = analytic_logpdf_grid("gaussian", {"rho": rho}, m=512)
    d = torch.from_numpy(np.exp(log_d)).float()
    estimate = float(mutual_information_from_density_grid(d).item())
    assert abs(estimate - expected) < 0.01


def test_mutual_information_from_density_grid_positive_for_clayton() -> None:
    log_d = analytic_logpdf_grid("clayton", {"theta": 3.0}, m=512)
    d = torch.from_numpy(np.exp(log_d)).float()
    estimate = float(mutual_information_from_density_grid(d).item())
    expected = _correct_grid_mi("clayton", {"theta": 3.0}, m=512)
    assert estimate > 0.5
    assert abs(estimate - expected) < 1e-6
