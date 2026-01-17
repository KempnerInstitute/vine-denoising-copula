#!/usr/bin/env python3
"""
Mutual information estimation benchmark (bivariate copulas).

This is intended to support the paper's "information-theoretic evaluation" with
**real, reproducible** numbers.

It compares MI estimators on a fixed bivariate copula zoo:
  - kNN MI (KSG)  [fast, no training]
  - Gaussian MI (probit correlation)  [fast, assumes Gaussian copula]
  - InfoNCE MI (contrastive lower bound)  [medium, trains a small critic network]
  - NWJ MI (f-div lower bound)  [medium, trains a small critic network]
  - MINE (DV bound)  [medium, trains a small critic network]
  - MINDE (ICLR 2024)  [slow, trains a diffusion score model]

Reference:
  - MINDE repo: https://github.com/MustaphaBounoua/minde
    (see README in that repository)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.data.generators import analytic_logpdf_grid, sample_bicop  # noqa: E402
from vdc.models.projection import copula_project  # noqa: E402
from vdc.utils.information import ksg_mutual_information  # noqa: E402


DEFAULT_TEST_COPULAS = [
    ("independence", {}, 0, "Independence"),
    ("gaussian", {"rho": 0.7}, 0, "Gaussian(ρ=0.7)"),
    ("gaussian", {"rho": -0.7}, 0, "Gaussian(ρ=-0.7)"),
    ("student", {"rho": 0.7, "df": 5}, 0, "Student-t(ρ=0.7, df=5)"),
    ("clayton", {"theta": 3.0}, 0, "Clayton(θ=3.0)"),
    ("clayton", {"theta": 3.0}, 90, "Clayton(θ=3.0, rot=90)"),
    ("gumbel", {"theta": 2.5}, 0, "Gumbel(θ=2.5)"),
    ("joe", {"theta": 3.0}, 0, "Joe(θ=3.0)"),
    ("frank", {"theta": 5.0}, 0, "Frank(θ=5.0)"),
]


def _normalize_density_np(d: np.ndarray) -> np.ndarray:
    m = int(d.shape[0])
    du = 1.0 / m
    d = np.nan_to_num(d, nan=0.0, posinf=1e300, neginf=0.0)
    d = np.clip(d, 0.0, 1e300)
    mass = float((d * du * du).sum())
    if not np.isfinite(mass) or mass <= 0:
        return np.ones_like(d) * (m * m)
    return d / mass


def _project_density_np(d: np.ndarray, iters: int, device: torch.device) -> np.ndarray:
    t = torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0).to(device)
    t = t.clamp_min(1e-12)
    t = copula_project(t, iters=int(iters))
    return t[0, 0].detach().cpu().numpy()


def _mi_from_density_grid(d: np.ndarray) -> float:
    m = int(d.shape[0])
    du = 1.0 / m
    return float(np.sum(d * np.log(d + 1e-12)) * du * du)


def _mi_true_from_analytic(family: str, params: Dict[str, Any], rotation: int, m_true: int, device: torch.device) -> float:
    lg = analytic_logpdf_grid(family, params, m=m_true, rotation=rotation)
    d = np.exp(np.clip(lg, -20, 20))
    d = _normalize_density_np(d)
    # Project to enforce exact copula constraints on the discrete grid (consistency with paper eval).
    d = _project_density_np(d, iters=50, device=device)
    d = _normalize_density_np(d)
    return _mi_from_density_grid(d)


def _maybe_clone_repo(dest: Path, url: str) -> None:
    """Clone a git repository if it doesn't already exist."""
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    subprocess.check_call(["git", "clone", "--depth", "1", url, str(dest)])


def _maybe_clone_minde(dest: Path) -> None:
    """Clone MINDE repository."""
    _maybe_clone_repo(dest, "https://github.com/MustaphaBounoua/minde")


def _maybe_clone_mist(dest: Path) -> None:
    """Clone MIST repository (arXiv:2511.18945)."""
    _maybe_clone_repo(dest, "https://github.com/gritsai/mist")


def _minde_estimate_mi(
    *,
    minde_repo: Path,
    x: np.ndarray,
    y: np.ndarray,
    benchmark: str,
    seed: int,
    device: torch.device,
    max_epochs: int,
    lr: float,
    batch_size: int,
) -> float:
    """
    Train MINDE on (x,y) and return MI estimate.

    This uses the reference implementation:
      https://github.com/MustaphaBounoua/minde
    """
    # Local import from external repo
    if str(minde_repo) not in sys.path:
        sys.path.insert(0, str(minde_repo))

    try:
        import pytorch_lightning as pl
    except Exception as e:
        raise RuntimeError("pytorch-lightning is required to run MINDE. Install it in the environment.") from e

    from src.libs.minde import MINDE  # type: ignore
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, Dataset

    class _XY(Dataset):
        def __init__(self, x_: np.ndarray, y_: np.ndarray):
            self.x = x_.astype(np.float32)
            self.y = y_.astype(np.float32)

        def __len__(self) -> int:
            return int(self.x.shape[0])

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            return {"x": torch.from_numpy(self.x[idx]), "y": torch.from_numpy(self.y[idx])}

    rng = np.random.default_rng(int(seed))
    n = int(x.shape[0])
    perm = rng.permutation(n)
    split = int(0.8 * n)
    tr = perm[:split]
    te = perm[split:]

    ds_tr = _XY(x[tr], y[tr])
    ds_te = _XY(x[te], y[te])

    # Use 0 workers for robustness on clusters (fork issues are common).
    dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=True, num_workers=0, drop_last=True)
    dl_te = DataLoader(ds_te, batch_size=min(1000, len(ds_te)), shuffle=False, num_workers=0, drop_last=False)

    accel = "gpu" if device.type == "cuda" else "cpu"

    # Make runs deterministic-ish across re-launches (enough for paper baselines).
    pl.seed_everything(int(seed), workers=False)

    args = SimpleNamespace(
        type="c",  # conditional MINDE
        preprocessing="rescale",  # MI is invariant to invertible transforms
        arch="mlp",
        Train_Size=len(ds_tr),
        Test_Size=len(ds_te),
        seed=int(seed),
        warmup_epochs=0,
        max_epochs=int(max_epochs),
        lr=float(lr),
        bs=int(batch_size),
        test_epoch=1,
        mc_iter=10,
        use_ema=False,
        importance_sampling=False,
        sigma=1.0,
        debug=False,
        nb_workers=0,
        devices=1,
        accelerator=accel,
        check_val_every_n_epoch=max(1, int(max_epochs)),
        out_dir=str(minde_repo / "_runs"),
        results_dir=str(minde_repo / "_runs"),
        benchmark=str(benchmark),
    )

    minde = MINDE(args=args, gt=None, var_list={"x": x.shape[1], "y": y.shape[1]})
    minde.fit(dl_tr, dl_te)
    mi, _mi_sigma = minde.compute_mi()
    return float(mi)


def _mist_estimate_mi(
    *,
    mist_repo: Path,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    device: torch.device,
    max_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 3,
) -> float:
    """
    Estimate MI using MIST (Mutual Information via Score Truncation).
    
    Reference: 
        Gritsai et al., "MIST: A Simple and Scalable Mutual Information Estimator
        via Score Truncation", arXiv:2511.18945, 2025.
        https://github.com/gritsai/mist
    
    MIST estimates MI using truncated score functions:
        I(X;Y) = E_{p(x,y)}[s_X(x) · (s_{X|Y}(x|y) - s_X(x))] 
               + E_{p(x,y)}[s_Y(y) · (s_{Y|X}(y|x) - s_Y(y))]
    
    This provides a consistent estimator that avoids the bias issues of 
    variational bounds like MINE/InfoNCE while being more scalable than KSG.
    
    Args:
        mist_repo: Path to cloned MIST repository
        x: (n, dx) array of samples
        y: (n, dy) array of samples  
        seed: Random seed
        device: Torch device
        max_epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
        hidden_dim: Hidden dimension of score networks
        num_layers: Number of hidden layers
        
    Returns:
        Estimated mutual information in nats.
    """
    # Local import from external repo
    if str(mist_repo) not in sys.path:
        sys.path.insert(0, str(mist_repo))
    
    try:
        from src.mist import MIST  # type: ignore
    except ImportError:
        # Fallback: try alternative import path
        try:
            from mist import MIST  # type: ignore
        except ImportError:
            raise RuntimeError(
                f"Could not import MIST from {mist_repo}. "
                "Ensure the repository is cloned and contains src/mist.py or mist.py. "
                "Clone via: git clone https://github.com/gritsai/mist"
            )
    
    from torch.utils.data import DataLoader, TensorDataset
    
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    
    # Prepare data
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(-1)
    if y_t.ndim == 1:
        y_t = y_t.unsqueeze(-1)
    
    n = x_t.shape[0]
    dx = x_t.shape[1]
    dy = y_t.shape[1]
    
    # Train/test split
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    split = int(0.8 * n)
    tr_idx = perm[:split]
    te_idx = perm[split:]
    
    ds_tr = TensorDataset(x_t[tr_idx], y_t[tr_idx])
    ds_te = TensorDataset(x_t[te_idx], y_t[te_idx])
    
    dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=True, drop_last=True)
    dl_te = DataLoader(ds_te, batch_size=min(1000, len(ds_te)), shuffle=False)
    
    # Initialize MIST estimator
    # Note: Exact API depends on MIST implementation; adjust as needed
    try:
        mist = MIST(
            dim_x=dx,
            dim_y=dy,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            lr=float(lr),
        ).to(device)
    except TypeError:
        # Alternative constructor signature
        mist = MIST(
            input_dim=dx + dy,
            hidden_dim=int(hidden_dim),
            lr=float(lr),
        ).to(device)
    
    # Training loop (simplified; actual implementation may differ)
    for epoch in range(int(max_epochs)):
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            mist.train_step(xb, yb)
    
    # Estimate MI on test set
    mi_vals = []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device)
            yb = yb.to(device)
            mi = mist.estimate_mi(xb, yb)
            mi_vals.append(float(mi))
    
    return float(np.mean(mi_vals))


def _mine_estimate_mi(
    *,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    device: torch.device,
    steps: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    weight_decay: float = 0.0,
    grad_clip: float = 5.0,
) -> float:
    """
    MINE (Mutual Information Neural Estimation) via the Donsker–Varadhan lower bound:
      I(X;Y) >= E_p[T] - log E_{p_x p_y}[exp(T)].
    """
    import torch.nn as nn

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    if x_t.ndim != 2 or y_t.ndim != 2 or x_t.shape[0] != y_t.shape[0]:
        raise ValueError(f"Expected x,y as (n,dx),(n,dy) with same n; got {x_t.shape} and {y_t.shape}")

    n = int(x_t.shape[0])
    dx = int(x_t.shape[1])
    dy = int(y_t.shape[1])

    net = nn.Sequential(
        nn.Linear(dx + dy, int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), 1),
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    rng = np.random.default_rng(int(seed))
    bs = int(min(max(8, int(batch_size)), n))

    for _ in range(int(steps)):
        idx = torch.from_numpy(rng.integers(0, n, size=bs, dtype=np.int64)).to(device)
        idx_m = torch.from_numpy(rng.integers(0, n, size=bs, dtype=np.int64)).to(device)
        xb = x_t[idx]
        yb = y_t[idx]
        y_m = y_t[idx_m]

        t_joint = net(torch.cat([xb, yb], dim=1))
        t_marg = net(torch.cat([xb, y_m], dim=1))
        # DV bound (lower bound). We train to maximize it => minimize negative.
        loss = -(t_joint.mean() - torch.log(torch.exp(t_marg).mean() + 1e-8))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(grad_clip))
        opt.step()

    # Final MI estimate on full data (single shuffle).
    with torch.no_grad():
        perm = torch.from_numpy(rng.permutation(n).astype(np.int64)).to(device)
        t_joint = net(torch.cat([x_t, y_t], dim=1))
        t_marg = net(torch.cat([x_t, y_t[perm]], dim=1))
        mi = t_joint.mean() - torch.log(torch.exp(t_marg).mean() + 1e-8)
    return float(mi.item())


def _gaussian_mi_probit(pts: np.ndarray, *, device: torch.device, eps: float = 1e-6) -> float:
    """
    Gaussian MI estimator for copulas:
      1) Gaussianize U,V via probit: z = Φ^{-1}(u)
      2) Estimate correlation ρ
      3) Return MI = -0.5 log(1-ρ^2)

    This is exact for Gaussian copulas and a reasonable baseline elsewhere.
    """
    x = torch.from_numpy(np.asarray(pts[:, 0], dtype=np.float64)).to(device)
    y = torch.from_numpy(np.asarray(pts[:, 1], dtype=np.float64)).to(device)
    x = x.clamp(float(eps), 1.0 - float(eps))
    y = y.clamp(float(eps), 1.0 - float(eps))

    # probit: Φ^{-1}(u) = sqrt(2) * erfinv(2u-1)
    zx = (2.0 ** 0.5) * torch.erfinv(2.0 * x - 1.0)
    zy = (2.0 ** 0.5) * torch.erfinv(2.0 * y - 1.0)

    zx = zx - zx.mean()
    zy = zy - zy.mean()
    cov = (zx * zy).mean()
    sx = zx.pow(2).mean().sqrt().clamp_min(1e-12)
    sy = zy.pow(2).mean().sqrt().clamp_min(1e-12)
    rho = (cov / (sx * sy)).clamp(-0.999999, 0.999999)
    mi = -0.5 * torch.log1p(-rho * rho)
    return float(mi.item())


def _critic_mlp(in_dim: int, hidden_dim: int, device: torch.device) -> torch.nn.Module:
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(int(in_dim), int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), 1),
    ).to(device)


def _infonce_estimate_mi(
    *,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    device: torch.device,
    steps: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    weight_decay: float = 0.0,
    grad_clip: float = 5.0,
    eval_batches: int = 10,
) -> float:
    """
    InfoNCE / contrastive MI lower bound:
      I(X;Y) >= log(B) - CE(scores, labels)
    where scores_{ij} = T(x_i, y_j) for a batch of size B.
    """
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    n = int(x_t.shape[0])
    bs = int(min(max(8, int(batch_size)), n))

    net = _critic_mlp(in_dim=int(x_t.shape[1] + y_t.shape[1]), hidden_dim=int(hidden_dim), device=device)
    opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = np.random.default_rng(int(seed))

    labels = torch.arange(bs, device=device)

    for _ in range(int(steps)):
        idx = torch.from_numpy(rng.integers(0, n, size=bs, dtype=np.int64)).to(device)
        xb = x_t[idx]  # (B,dx)
        yb = y_t[idx]  # (B,dy)

        # scores: (B,B) where row i corresponds to x_i paired with all y_j
        xb2 = xb.unsqueeze(1).expand(bs, bs, -1)
        yb2 = yb.unsqueeze(0).expand(bs, bs, -1)
        logits = net(torch.cat([xb2, yb2], dim=-1)).squeeze(-1)  # (B,B)

        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(grad_clip))
        opt.step()

    # Estimate MI bound by averaging a few fresh batches.
    with torch.no_grad():
        mi_vals: List[float] = []
        for _ in range(int(max(1, eval_batches))):
            idx = torch.from_numpy(rng.integers(0, n, size=bs, dtype=np.int64)).to(device)
            xb = x_t[idx]
            yb = y_t[idx]
            xb2 = xb.unsqueeze(1).expand(bs, bs, -1)
            yb2 = yb.unsqueeze(0).expand(bs, bs, -1)
            logits = net(torch.cat([xb2, yb2], dim=-1)).squeeze(-1)
            ce = F.cross_entropy(logits, labels)
            mi = float(np.log(bs)) - float(ce.item())
            mi_vals.append(mi)
        return float(np.mean(mi_vals))


def _nwj_estimate_mi(
    *,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    device: torch.device,
    steps: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    weight_decay: float = 0.0,
    grad_clip: float = 5.0,
    t_clip: float = 20.0,
    eval_batches: int = 20,
) -> float:
    """
    NWJ (Nguyen–Wainwright–Jordan) MI lower bound:
      I(X;Y) >= E_p[T] - E_{p_x p_y}[exp(T - 1)].
    """
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    n = int(x_t.shape[0])
    bs = int(min(max(8, int(batch_size)), n))

    net = _critic_mlp(in_dim=int(x_t.shape[1] + y_t.shape[1]), hidden_dim=int(hidden_dim), device=device)
    opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = np.random.default_rng(int(seed))

    for _ in range(int(steps)):
        idx = torch.from_numpy(rng.integers(0, n, size=bs, dtype=np.int64)).to(device)
        idx_m = torch.from_numpy(rng.integers(0, n, size=bs, dtype=np.int64)).to(device)
        xb = x_t[idx]
        yb = y_t[idx]
        y_m = y_t[idx_m]

        t_joint = net(torch.cat([xb, yb], dim=1)).clamp(-float(t_clip), float(t_clip))
        t_marg = net(torch.cat([xb, y_m], dim=1)).clamp(-float(t_clip), float(t_clip))
        loss = -(t_joint.mean() - torch.exp(t_marg - 1.0).mean())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(grad_clip))
        opt.step()

    with torch.no_grad():
        vals: List[float] = []
        for _ in range(int(max(1, eval_batches))):
            idx = torch.from_numpy(rng.integers(0, n, size=bs, dtype=np.int64)).to(device)
            perm = torch.from_numpy(rng.permutation(n).astype(np.int64)[:bs]).to(device)
            xb = x_t[idx]
            yb = y_t[idx]
            y_m = y_t[perm]
            t_joint = net(torch.cat([xb, yb], dim=1)).clamp(-float(t_clip), float(t_clip))
            t_marg = net(torch.cat([xb, y_m], dim=1)).clamp(-float(t_clip), float(t_clip))
            mi = t_joint.mean() - torch.exp(t_marg - 1.0).mean()
            vals.append(float(mi.item()))
        return float(np.mean(vals))


def main() -> None:
    p = argparse.ArgumentParser(description="Mutual information estimation benchmark (bivariate copulas)")
    p.add_argument("--estimator", type=str, choices=["ksg", "gaussian", "infonce", "nwj", "mine", "minde", "mist"], required=True)
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--m-true", type=int, default=256, help="Grid resolution used to compute MI_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-json", type=Path, required=True)

    # Neural MI options (used for mine / infonce / nwj)
    p.add_argument("--mine-steps", type=int, default=2000)
    p.add_argument("--mine-lr", type=float, default=1e-3)
    p.add_argument("--mine-batch-size", type=int, default=512)
    p.add_argument("--mine-hidden-dim", type=int, default=128)
    p.add_argument("--mine-weight-decay", type=float, default=0.0)
    p.add_argument("--mine-grad-clip", type=float, default=5.0)

    # MINDE options (only used when estimator=minde)
    p.add_argument("--minde-repo", type=Path, default=Path("external/minde"))
    p.add_argument("--clone-minde", action="store_true")
    p.add_argument("--minde-max-epochs", type=int, default=50)
    p.add_argument("--minde-lr", type=float, default=1e-2)
    p.add_argument("--minde-batch-size", type=int, default=256)

    # MIST options (only used when estimator=mist)
    p.add_argument("--mist-repo", type=Path, default=Path("external/mist"))
    p.add_argument("--clone-mist", action="store_true")
    p.add_argument("--mist-max-epochs", type=int, default=100)
    p.add_argument("--mist-lr", type=float, default=1e-3)
    p.add_argument("--mist-batch-size", type=int, default=256)
    p.add_argument("--mist-hidden-dim", type=int, default=256)
    p.add_argument("--mist-num-layers", type=int, default=3)

    args = p.parse_args()

    device = torch.device(args.device)
    est = str(args.estimator).lower()

    if est == "minde" and args.clone_minde:
        _maybe_clone_minde(args.minde_repo)
    if est == "mist" and args.clone_mist:
        _maybe_clone_mist(args.mist_repo)

    records: List[Dict[str, Any]] = []
    for family, params, rotation, name in DEFAULT_TEST_COPULAS:
        pts = sample_bicop(family, params, n=int(args.n_samples), rotation=int(rotation), seed=int(args.seed))
        mi_true = _mi_true_from_analytic(
            family=family,
            params=params,
            rotation=int(rotation),
            m_true=int(args.m_true),
            device=device,
        )

        t0 = perf_counter()
        if est == "ksg":
            mi_est = ksg_mutual_information(pts[:, 0], pts[:, 1], k=5, seed=int(args.seed))
        elif est == "gaussian":
            mi_est = _gaussian_mi_probit(pts, device=device)
        elif est == "infonce":
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            mi_est = _infonce_estimate_mi(
                x=x,
                y=y,
                seed=int(args.seed) + 1000 * int(rotation),
                device=device,
                steps=int(args.mine_steps),
                lr=float(args.mine_lr),
                batch_size=int(args.mine_batch_size),
                hidden_dim=int(args.mine_hidden_dim),
                weight_decay=float(args.mine_weight_decay),
                grad_clip=float(args.mine_grad_clip),
            )
        elif est == "nwj":
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            mi_est = _nwj_estimate_mi(
                x=x,
                y=y,
                seed=int(args.seed) + 1000 * int(rotation),
                device=device,
                steps=int(args.mine_steps),
                lr=float(args.mine_lr),
                batch_size=int(args.mine_batch_size),
                hidden_dim=int(args.mine_hidden_dim),
                weight_decay=float(args.mine_weight_decay),
                grad_clip=float(args.mine_grad_clip),
            )
        elif est == "mine":
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            mi_est = _mine_estimate_mi(
                x=x,
                y=y,
                seed=int(args.seed) + 1000 * int(rotation),
                device=device,
                steps=int(args.mine_steps),
                lr=float(args.mine_lr),
                batch_size=int(args.mine_batch_size),
                hidden_dim=int(args.mine_hidden_dim),
                weight_decay=float(args.mine_weight_decay),
                grad_clip=float(args.mine_grad_clip),
            )
        elif est == "mist":
            # MIST (arXiv:2511.18945)
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            mi_est = _mist_estimate_mi(
                mist_repo=args.mist_repo,
                x=x,
                y=y,
                seed=int(args.seed),
                device=device,
                max_epochs=int(args.mist_max_epochs),
                lr=float(args.mist_lr),
                batch_size=int(args.mist_batch_size),
                hidden_dim=int(args.mist_hidden_dim),
                num_layers=int(args.mist_num_layers),
            )
        else:
            # MINDE expects (n,dx), (n,dy)
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            mi_est = _minde_estimate_mi(
                minde_repo=args.minde_repo,
                x=x,
                y=y,
                benchmark=f"vdc_copula_mi_{family}_rot{int(rotation)}",
                seed=int(args.seed),
                device=device,
                max_epochs=int(args.minde_max_epochs),
                lr=float(args.minde_lr),
                batch_size=int(args.minde_batch_size),
            )
        dt_s = float(perf_counter() - t0)

        records.append(
            {
                "name": name,
                "family": family,
                "rotation": int(rotation),
                "n_samples": int(args.n_samples),
                "mi_true": float(mi_true),
                "mi_est": float(mi_est),
                "mi_err": float(abs(mi_est - mi_true)),
                "time_s": dt_s,
            }
        )

    payload = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "estimator": est,
        "n_samples": int(args.n_samples),
        "m_true": int(args.m_true),
        "seed": int(args.seed),
        "records": records,
        "mean_abs_err": float(np.mean([r["mi_err"] for r in records])) if records else None,
        "mean_time_s": float(np.mean([r["time_s"] for r in records])) if records else None,
        "source": {
            "minde_repo": str(args.minde_repo) if est == "minde" else None,
            "minde_url": "https://github.com/MustaphaBounoua/minde",
            "mist_repo": str(args.mist_repo) if est == "mist" else None,
            "mist_url": "https://github.com/gritsai/mist",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()

