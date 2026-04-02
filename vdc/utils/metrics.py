# ruff: noqa: N806, N812
"""Evaluation metrics for copula density models.

Includes:
    - KL divergence (discrete grid approximation)
    - Integrated Squared Error (ISE)
    - Marginal uniformity deviation (L2 from 1)
    - Tail density approximation (average density in corner regions)
    - Kendall's tau estimate (sample-based or grid-based fallback)
    - Upper / Lower tail dependence coefficients λ_U, λ_L (grid-based approximation)
"""
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _safe_normalize(d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    total = d.sum(dim=(-2,-1), keepdim=True).clamp(min=eps)
    return d / total

def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(p||q) with safety clamps; p,q shape (B,1,m,m)."""
    p_n = _safe_normalize(p, eps)
    q_n = _safe_normalize(q, eps)
    ratio = (p_n + eps) / (q_n + eps)
    kl = (p_n * torch.log(ratio)).sum(dim=(-2,-1))
    return kl.mean()

def ise(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return ((p - q) ** 2).mean()

def marginal_uniformity(d: torch.Tensor) -> torch.Tensor:
    """Return mean L2 deviation of marginals from uniform. d shape (B,1,m,m)."""
    d_n = _safe_normalize(d)
    B, _, m, _ = d_n.shape
    # Marginals: integrate along each axis
    marg_u = d_n.sum(dim=-1)  # (B,1,m) - should each be ~1/m for uniform marginals
    marg_v = d_n.sum(dim=-2)  # (B,1,m) - should each be ~1/m for uniform marginals
    # Target is 1/m for each marginal value (since after normalization, sum over all = 1)
    target = torch.full_like(marg_u, 1.0 / m)
    dev = F.mse_loss(marg_u, target) + F.mse_loss(marg_v, target)
    return dev * 0.5

def tail_density(d: torch.Tensor, tail_region: float = 0.1) -> torch.Tensor:
    """Average density in four corner squares of side floor(m*tail_region)."""
    B, _, m, _ = d.shape
    k = max(1, int(m * tail_region))
    regions = [
        d[..., :k, :k],
        d[..., :k, -k:],
        d[..., -k:, :k],
        d[..., -k:, -k:],
    ]
    vals = [r.mean() for r in regions]
    return torch.stack(vals).mean()

def kendall_tau(samples: torch.Tensor, max_points: int = 2000) -> torch.Tensor:
    """Approximate Kendall's tau from samples (N,2) or (B,N,2).
    Subsamples to max_points for O(n^2) concordance computation.

    Returns Kendall's tau in [-1, 1].
    """
    if samples.dim() == 2:
        samples = samples.unsqueeze(0)
    B, N, _ = samples.shape
    if N > max_points:
        idx = torch.randperm(N, device=samples.device)[:max_points]
        samples = samples[:, idx]
        N = max_points
    u = samples[..., 0]  # (B, N)
    v = samples[..., 1]  # (B, N)

    # (B, N, N) differences
    du = u.unsqueeze(-1) - u.unsqueeze(-2)
    dv = v.unsqueeze(-1) - v.unsqueeze(-2)
    sign = torch.sign(du * dv)

    # Ignore zeros (ties) by masking
    mask = (du != 0) & (dv != 0)

    # Apply mask and sum per batch
    # Concordant: sign > 0, Discordant: sign < 0
    concordant = ((sign > 0) & mask).float().sum(dim=(-2, -1))  # (B,)
    discordant = ((sign < 0) & mask).float().sum(dim=(-2, -1))  # (B,)
    total_pairs = mask.float().sum(dim=(-2, -1)).clamp_min(1.0)  # (B,)

    # Kendall's tau = (concordant - discordant) / total_pairs
    tau = (concordant - discordant) / total_pairs
    return tau.mean()

def tail_dependence_from_grid(d: torch.Tensor, q_high: float = 0.95, q_low: float = 0.05) -> torch.Tensor:
    """Approximate λ_U and λ_L using grid mass regions.
    d: (B,1,m,m) raw (not necessarily normalized) density.
    Returns: (lambda_U, lambda_L)
    """
    B, _, m, _ = d.shape
    du = dv = 1.0 / m
    mass = (d * du * dv)
    total = mass.sum(dim=(2,3), keepdim=True).clamp_min(1e-12)
    p = mass / total
    u = torch.linspace(0.5/m, 1-0.5/m, m, device=d.device)
    v = torch.linspace(0.5/m, 1-0.5/m, m, device=d.device)
    U, V = torch.meshgrid(u, v, indexing='ij')
    U = U.unsqueeze(0)
    V = V.unsqueeze(0)
    high_mask = (U > q_high) & (V > q_high)
    low_mask = (U < q_low) & (V < q_low)
    # Conditional probabilities
    prob_V_high = (p * (V > q_high)).sum(dim=(2,3)).clamp_min(1e-12)
    prob_V_low = (p * (V < q_low)).sum(dim=(2,3)).clamp_min(1e-12)
    lambda_U = (p * high_mask).sum(dim=(2,3)) / prob_V_high
    lambda_L = (p * low_mask).sum(dim=(2,3)) / prob_V_low
    return lambda_U.mean(), lambda_L.mean()

def aggregate_metrics(pred: torch.Tensor, target: torch.Tensor, samples: Optional[torch.Tensor] = None) -> Dict[str, float]:
    metrics = {
        'kl': float(kl_divergence(target, pred).item()),
        'ise': float(ise(pred, target).item()),
        'marginal_dev': float(marginal_uniformity(pred).item()),
        'tail_mean': float(tail_density(pred).item()),
    }
    try:
        lambda_U, lambda_L = tail_dependence_from_grid(pred)
        metrics['lambda_U'] = float(lambda_U.item())
        metrics['lambda_L'] = float(lambda_L.item())
    except Exception:
        pass
    if samples is not None:
        try:
            tau_est = kendall_tau(samples)
            metrics['tau_est'] = float(tau_est.item())
        except Exception:
            pass
    return metrics


def mutual_information_from_density_grid(d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Approximate mutual information for a (bi)variate copula density on a grid.

    For a copula density c(u,v) with uniform marginals:
        I(U;V) = ∫∫ c(u,v) log c(u,v) du dv

    This function expects a *density grid* (not log-density). If `d` is not perfectly
    normalized, it will be normalized internally.

    Args:
        d: Tensor shaped (m,m), (B,m,m), or (B,1,m,m).
        eps: Clamp for log stability.

    Returns:
        Scalar tensor (mean over batch if B>1).
    """
    if d.dim() == 2:
        d_ = d.unsqueeze(0).unsqueeze(0)  # (1,1,m,m)
    elif d.dim() == 3:
        d_ = d.unsqueeze(1)  # (B,1,m,m)
    elif d.dim() == 4:
        d_ = d
    else:
        raise ValueError(f"d must have 2/3/4 dims, got shape={tuple(d.shape)}")

    B, _, m, _ = d_.shape
    du = 1.0 / m
    p_mass = _safe_normalize(d_.clamp_min(0.0), eps=eps)
    density = p_mass / (du * du)
    mi = (p_mass * torch.log(density.clamp_min(eps))).sum(dim=(-2, -1))  # (B,1)
    return mi.mean()


def copula_entropy_from_density_grid(d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Copula entropy H_c = -∫ c log c = -I(U;V) for bivariate copulas."""
    return -mutual_information_from_density_grid(d, eps=eps)
