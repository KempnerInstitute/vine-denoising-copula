"""Tests for log-domain IPFP projection accuracy and convergence."""

import pytest
import torch

from vdc.utils.ipfp_log import ipfp_project_log, marginal_deviation


@pytest.mark.parametrize("m", [32, 64])
def test_ipfp_marginal_accuracy(m):
    batch = 3
    torch.manual_seed(0)
    raw = torch.rand(batch, 1, m, m) * torch.rand(1, 1, 1, 1) + 0.1
    proj = ipfp_project_log(raw, iters=25)
    # Compute marginals
    du = dv = 1.0 / m
    mass = (proj * du * dv).sum(dim=(2, 3), keepdim=True)
    proj_n = proj / mass
    rows = (proj_n * dv).sum(dim=3)
    cols = (proj_n * du).sum(dim=2)
    assert torch.allclose(rows, torch.ones_like(rows), atol=5e-3), f"Row marginals dev: {rows.mean().item()}"
    assert torch.allclose(cols, torch.ones_like(cols), atol=5e-3), f"Col marginals dev: {cols.mean().item()}"


@pytest.mark.parametrize("iters_fast,iters_full", [(3, 20), (5, 30)])
def test_ipfp_convergence(iters_fast, iters_full):
    m = 64
    batch = 2
    torch.manual_seed(42)
    raw = torch.rand(batch, 1, m, m) + 0.05
    fast = ipfp_project_log(raw, iters=iters_fast)
    full = ipfp_project_log(raw, iters=iters_full)
    dev_fast = marginal_deviation(fast)
    dev_full = marginal_deviation(full)
    assert dev_full <= dev_fast + 1e-6, "Full iterations should not worsen marginal deviation"
