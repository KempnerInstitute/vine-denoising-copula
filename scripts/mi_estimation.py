#!/usr/bin/env python3
"""
Mutual information estimation benchmark (bivariate copulas).

This is intended to support the paper's "information-theoretic evaluation" with
**real, reproducible** numbers.

It compares MI estimators on a fixed bivariate copula zoo:
  - kNN MI (KSG)  [fast, no training]
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


def _maybe_clone_minde(dest: Path) -> None:
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/MustaphaBounoua/minde", str(dest)])


def _minde_estimate_mi(
    *,
    minde_repo: Path,
    x: np.ndarray,
    y: np.ndarray,
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
        import pytorch_lightning as pl  # noqa: F401
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
        benchmark="vdc_copula_mi",
    )

    minde = MINDE(args=args, gt=None, var_list={"x": x.shape[1], "y": y.shape[1]})
    minde.fit(dl_tr, dl_te)
    mi, _mi_sigma = minde.compute_mi()
    return float(mi)


def main() -> None:
    p = argparse.ArgumentParser(description="Mutual information estimation benchmark (bivariate copulas)")
    p.add_argument("--estimator", type=str, choices=["ksg", "minde"], required=True)
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--m-true", type=int, default=256, help="Grid resolution used to compute MI_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-json", type=Path, required=True)

    # MINDE options (only used when estimator=minde)
    p.add_argument("--minde-repo", type=Path, default=Path("external/minde"))
    p.add_argument("--clone-minde", action="store_true")
    p.add_argument("--minde-max-epochs", type=int, default=50)
    p.add_argument("--minde-lr", type=float, default=1e-2)
    p.add_argument("--minde-batch-size", type=int, default=256)

    args = p.parse_args()

    device = torch.device(args.device)
    est = str(args.estimator).lower()

    if est == "minde" and args.clone_minde:
        _maybe_clone_minde(args.minde_repo)

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
        else:
            # MINDE expects (n,dx), (n,dy)
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            mi_est = _minde_estimate_mi(
                minde_repo=args.minde_repo,
                x=x,
                y=y,
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
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()

