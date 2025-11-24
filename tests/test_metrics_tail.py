"""Tests for tail dependence and Kendall tau metrics."""
import torch
from src.utils.metrics import kendall_tau, tail_dependence_from_grid

def test_kendall_tau_independence():
    # Uniform independent samples should yield tau near 0
    torch.manual_seed(0)
    samples = torch.rand(500,2)
    tau = kendall_tau(samples)
    assert abs(tau.item()) < 0.05, f"Tau too large for independence: {tau.item()}"

def test_tail_dependence_independence():
    m = 64
    # Independent uniform density -> lambda_U ~ (1-q_high), lambda_L ~ q_low approximately
    u = torch.ones(1,1,m,m)
    lambda_U, lambda_L = tail_dependence_from_grid(u, q_high=0.95, q_low=0.05)
    # For independence: P(U>q,V>q)= (1-q)^2 ; P(V>q)=1-q => lambda_U=(1-q)
    qh = 0.95
    ql = 0.05
    assert abs(lambda_U.item() - (1-qh)) < 0.02, f"lambda_U off: {lambda_U.item()}"
    # Lower tail: P(U<q,V<q)= q^2 ; P(V<q)=q => lambda_L=q
    assert abs(lambda_L.item() - ql) < 0.02, f"lambda_L off: {lambda_L.item()}"
