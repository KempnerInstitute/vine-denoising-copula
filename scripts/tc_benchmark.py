#!/usr/bin/env python3
"""
Total correlation (TC) benchmark for the paper (E1b).

We benchmark TC estimation on synthetic Gaussian AR(1) copulas:
  Sigma[i,j] = rho^{|i-j|}

For a Gaussian copula with correlation matrix Sigma, the true total correlation is:
  TC = -0.5 * log det(Sigma)

We compare:
  - KSG (kNN MI) via chain rule: TC = sum_{i=2..d} I(U_i ; U_{1:i-1})
  - DCD-Vine (ours): fit a D-vine with the pretrained bivariate estimator and
    estimate TC via mean log copula density on held-out samples:
      TC = E[log c(U)]  (since marginals are uniform in copula space)

Outputs a JSON consumable by:
  - drafts/scripts/fig_information_estimation.py (fig_tc_vs_dim.pdf)
  - drafts/scripts/paper_artifacts.py (table generation)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _true_tc_gaussian_ar1(d: int, rho: float) -> float:
    """True TC for Gaussian AR(1) copula: -0.5 log det(Sigma)."""
    Sigma = np.array([[rho ** abs(i - j) for j in range(d)] for i in range(d)], dtype=np.float64)
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        raise RuntimeError(f"Non-positive definite Sigma? sign={sign}, d={d}, rho={rho}")
    return float(-0.5 * logdet)


def _tc_ksg_chain_rule(U: np.ndarray, *, k: int, seed: int) -> float:
    """Estimate TC via chain rule using KSG MI between growing blocks."""
    from vdc.utils.information import ksg_mutual_information

    U = np.asarray(U, dtype=np.float64)
    n, d = U.shape
    if d < 2:
        return 0.0
    tc = 0.0
    # TC = sum_{i=2..d} I(U_i ; U_{1:i-1})
    for i in range(1, d):
        x = U[:, :i]
        y = U[:, i]
        mi = ksg_mutual_information(x, y, k=int(k), seed=int(seed) + 10_000 * i)
        tc += max(0.0, float(mi))
    return float(tc)


def _tc_mine(
    U: np.ndarray,
    *,
    device: str,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    hidden: int,
    layers: int,
) -> Tuple[float, float]:
    """Estimate TC = KL(p||Uniform) using a DV/MINE-style critic."""
    import torch
    import torch.nn as nn

    U = np.asarray(U, dtype=np.float32)
    n, d = U.shape
    if n < 10:
        return 0.0, 0.0

    torch.manual_seed(int(seed))
    dev = torch.device(device)

    p_t = torch.from_numpy(U).to(dev)

    # Simple MLP critic T(u)
    mods: List[nn.Module] = [nn.Linear(d, int(hidden)), nn.ReLU(inplace=True)]
    for _ in range(max(0, int(layers) - 1)):
        mods += [nn.Linear(int(hidden), int(hidden)), nn.ReLU(inplace=True)]
    mods += [nn.Linear(int(hidden), 1)]
    net = nn.Sequential(*mods).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=float(lr))

    t0 = perf_counter()
    bs = max(32, int(batch_size))
    for _s in range(int(steps)):
        idx = torch.randint(0, n, (bs,), device=dev)
        p_b = p_t[idx]
        q_b = torch.rand(bs, d, device=dev)  # uniform on (0,1)

        Tp = net(p_b).mean()
        Tq = net(q_b).squeeze(-1)
        log_mean_exp_Tq = torch.logsumexp(Tq, dim=0) - float(np.log(bs))
        loss = -(Tp - log_mean_exp_Tq)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Final estimate on a fresh uniform batch
    with torch.no_grad():
        Tp = net(p_t).mean()
        q_eval = torch.rand(min(n, 20000), d, device=dev)
        Tq = net(q_eval).squeeze(-1)
        log_mean_exp_Tq = torch.logsumexp(Tq, dim=0) - float(np.log(Tq.shape[0]))
        tc_hat = float((Tp - log_mean_exp_Tq).item())

    return float(tc_hat), float(perf_counter() - t0)


@dataclass(frozen=True)
class _LoadedModel:
    model: Any
    model_type: str
    config: Dict[str, Any]


def _load_checkpoint_model(ckpt_path: Path, device: str) -> _LoadedModel:
    import torch
    from vdc.train.unified_trainer import build_model

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(ckpt)}")
    config = ckpt.get("config", {})
    if not isinstance(config, dict):
        raise RuntimeError("Checkpoint missing config dict.")

    # Train configs use model.type; keep fallback for older checkpoints.
    model_type = str(config.get("model", {}).get("type", "diffusion_unet"))
    dev = torch.device(device)
    model = build_model(model_type, config, dev)
    model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
    model.eval()
    return _LoadedModel(model=model, model_type=model_type, config=config)


def _tc_dcd_vine(
    U: np.ndarray,
    *,
    loaded: _LoadedModel,
    device: str,
    seed: int,
    test_frac: float,
    batch_edges: bool,
    edge_batch_size: int,
    hfunc_use_spline: bool,
) -> Tuple[float, float]:
    """
    Returns (tc_hat, total_time_s) using a held-out evaluation split.
    """
    from vdc.vine.api import VineCopulaModel
    import torch

    rng = np.random.default_rng(int(seed))
    n = int(U.shape[0])
    perm = rng.permutation(n)
    n_test = max(1, int(round(float(test_frac) * n)))
    te = perm[:n_test]
    tr = perm[n_test:]
    if len(tr) < 2:
        tr = perm  # fallback
        te = perm

    U_tr = U[tr]
    U_te = U[te]

    m = int(loaded.config.get("data", {}).get("m", 64))
    proj_iters = int(loaded.config.get("training", {}).get("projection_iters", 30))

    if str(loaded.model_type) == "diffusion_unet":
        raise RuntimeError(
            "tc_benchmark currently expects a single-pass checkpoint (denoiser/enhanced_cnn). "
            "For diffusion_unet, add an iterative sampling path (slower) or use a denoiser checkpoint."
        )

    vine = VineCopulaModel(
        vine_type="dvine",
        m=m,
        device=str(device),
        projection_iters=int(proj_iters),
        hfunc_use_spline=bool(hfunc_use_spline),
        batch_edges=bool(batch_edges),
        edge_batch_size=int(edge_batch_size),
    )

    loaded.model.to(str(device))
    loaded.model.eval()

    # Sync before timing if on GPU.
    try:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    t0 = perf_counter()
    vine.fit(U_tr, diffusion_model=loaded.model, diffusion=None, verbose=False)
    tc_hat = float(np.mean(vine.logpdf(U_te)))
    t1 = perf_counter()

    try:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    return float(tc_hat), float(t1 - t0)


def _mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray(list(xs), dtype=np.float64)
    return float(np.mean(x)), float(np.std(x))


def main() -> None:
    p = argparse.ArgumentParser(description="TC benchmark (Gaussian AR(1) copulas).")
    p.add_argument("--checkpoint", type=Path, required=True, help="Single-pass checkpoint (denoiser/enhanced_cnn).")
    p.add_argument("--device", type=str, default="cuda", help="torch device (default: cuda).")
    p.add_argument("--rho", type=float, default=0.7, help="AR(1) correlation rho.")
    p.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10, 20, 50], help="Dimensions to benchmark.")
    p.add_argument("--n", type=int, default=5000, help="Sample size per trial for the scaling curve.")
    p.add_argument("--n-trials", type=int, default=3, help="Random trials per dimension.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ksg-k", type=int, default=5)
    p.add_argument("--include-mine", action="store_true", help="Include a MINE/DV baseline (KL vs Uniform).")
    p.add_argument("--mine-steps", type=int, default=800)
    p.add_argument("--mine-batch-size", type=int, default=512)
    p.add_argument("--mine-lr", type=float, default=1e-4)
    p.add_argument("--mine-hidden", type=int, default=256)
    p.add_argument("--mine-layers", type=int, default=3)
    p.add_argument("--test-frac", type=float, default=0.25, help="Held-out fraction for DCD-Vine evaluation.")
    p.add_argument("--batch-edges", action="store_true", help="Enable tree-level edge batching (ours, faster).")
    p.add_argument("--edge-batch-size", type=int, default=256)
    p.add_argument("--hfunc-use-spline", action="store_true", help="Use spline h-functions (slower, more accurate).")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--table-d", type=int, default=10, help="Dimension used for the main TC estimation table.")
    p.add_argument("--table-n", type=int, nargs="+", default=[500, 5000], help="Sample sizes for the table.")
    args = p.parse_args()

    from vdc.data.generators import generate_gaussian_vine

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    loaded = _load_checkpoint_model(ckpt, device=str(args.device))

    dims = [int(d) for d in args.dims]
    rho = float(args.rho)
    n = int(args.n)
    n_trials = int(args.n_trials)
    seed = int(args.seed)
    k = int(args.ksg_k)

    tc_true: List[float] = []
    tc_ksg_means: List[float] = []
    tc_ksg_stds: List[float] = []
    tc_dcd_means: List[float] = []
    tc_dcd_stds: List[float] = []
    tc_mine_means: List[float] = []
    tc_mine_stds: List[float] = []

    raw_by_dim: List[Dict[str, Any]] = []

    for d in dims:
        true_val = _true_tc_gaussian_ar1(d, rho)
        tc_true.append(true_val)

        ksg_vals: List[float] = []
        dcd_vals: List[float] = []
        mine_vals: List[float] = []

        for t in range(n_trials):
            trial_seed = seed + 10_000 * d + 1_000 * t
            U = generate_gaussian_vine(n=n, d=d, rho=rho, seed=trial_seed)

            t0 = perf_counter()
            tc_ksg = _tc_ksg_chain_rule(U, k=k, seed=trial_seed)
            ksg_time = float(perf_counter() - t0)

            mine_time = None
            tc_mine = None
            if bool(args.include_mine):
                tc_mine, mine_time = _tc_mine(
                    U,
                    device=str(args.device),
                    seed=trial_seed,
                    steps=int(args.mine_steps),
                    batch_size=int(args.mine_batch_size),
                    lr=float(args.mine_lr),
                    hidden=int(args.mine_hidden),
                    layers=int(args.mine_layers),
                )
                mine_vals.append(float(tc_mine))

            tc_dcd, dcd_time = _tc_dcd_vine(
                U,
                loaded=loaded,
                device=str(args.device),
                seed=trial_seed,
                test_frac=float(args.test_frac),
                batch_edges=bool(args.batch_edges),
                edge_batch_size=int(args.edge_batch_size),
                hfunc_use_spline=bool(args.hfunc_use_spline),
            )

            ksg_vals.append(float(tc_ksg))
            dcd_vals.append(float(tc_dcd))

            raw_by_dim.append(
                {
                    "d": int(d),
                    "trial": int(t),
                    "n": int(n),
                    "rho": float(rho),
                    "tc_true": float(true_val),
                    "tc_ksg": float(tc_ksg),
                    "tc_mine": float(tc_mine) if tc_mine is not None else None,
                    "tc_dcd": float(tc_dcd),
                    "time_ksg_s": float(ksg_time),
                    "time_mine_s": float(mine_time) if mine_time is not None else None,
                    "time_dcd_s": float(dcd_time),
                }
            )

        m_ksg, s_ksg = _mean_std(ksg_vals)
        m_dcd, s_dcd = _mean_std(dcd_vals)
        tc_ksg_means.append(m_ksg)
        tc_ksg_stds.append(s_ksg)
        tc_dcd_means.append(m_dcd)
        tc_dcd_stds.append(s_dcd)
        if bool(args.include_mine) and mine_vals:
            m_mine, s_mine = _mean_std(mine_vals)
            tc_mine_means.append(m_mine)
            tc_mine_stds.append(s_mine)

    # Table entries (d fixed, n varies): absolute error + time.
    table_rows: List[Dict[str, Any]] = []
    d_table = int(args.table_d)
    for n_table in [int(x) for x in args.table_n]:
        # One fresh sample (single trial) per n for simplicity; increase n_trials if needed.
        trial_seed = seed + 99_000 + 10 * n_table
        U = generate_gaussian_vine(n=n_table, d=d_table, rho=rho, seed=trial_seed)
        tc_t = _true_tc_gaussian_ar1(d_table, rho)

        t0 = perf_counter()
        tc_k = _tc_ksg_chain_rule(U, k=k, seed=trial_seed)
        t_ksg = float(perf_counter() - t0)

        tc_m = None
        t_mine = None
        if bool(args.include_mine):
            tc_m, t_mine = _tc_mine(
                U,
                device=str(args.device),
                seed=trial_seed + 999,
                steps=int(args.mine_steps),
                batch_size=int(args.mine_batch_size),
                lr=float(args.mine_lr),
                hidden=int(args.mine_hidden),
                layers=int(args.mine_layers),
            )

        tc_d, t_dcd = _tc_dcd_vine(
            U,
            loaded=loaded,
            device=str(args.device),
            seed=trial_seed,
            test_frac=float(args.test_frac),
            batch_edges=bool(args.batch_edges),
            edge_batch_size=int(args.edge_batch_size),
            hfunc_use_spline=bool(args.hfunc_use_spline),
        )

        table_rows.extend(
            [
                {
                    "estimator": "KSG",
                    "n": int(n_table),
                    "d": int(d_table),
                    "rho": float(rho),
                    "tc_true": float(tc_t),
                    "tc_hat": float(tc_k),
                    "abs_err": float(abs(tc_k - tc_t)),
                    "time_s": float(t_ksg),
                },
                {
                    "estimator": "MINE",
                    "n": int(n_table),
                    "d": int(d_table),
                    "rho": float(rho),
                    "tc_true": float(tc_t),
                    "tc_hat": float(tc_m) if tc_m is not None else None,
                    "abs_err": float(abs(float(tc_m) - tc_t)) if tc_m is not None else None,
                    "time_s": float(t_mine) if t_mine is not None else None,
                },
                {
                    "estimator": "DCD-Vine",
                    "n": int(n_table),
                    "d": int(d_table),
                    "rho": float(rho),
                    "tc_true": float(tc_t),
                    "tc_hat": float(tc_d),
                    "abs_err": float(abs(tc_d - tc_t)),
                    "time_s": float(t_dcd),
                },
            ]
        )

    payload: Dict[str, Any] = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "checkpoint": str(ckpt),
        "model_type": str(loaded.model_type),
        "rho": float(rho),
        "n_samples": int(n),
        "n_trials": int(n_trials),
        "ksg_k": int(k),
        "dimensions": dims,
        "tc_true": tc_true,
        "tc_ksg_mean": tc_ksg_means,
        "tc_ksg_std": tc_ksg_stds,
        "tc_mine_mean": tc_mine_means if bool(args.include_mine) else None,
        "tc_mine_std": tc_mine_stds if bool(args.include_mine) else None,
        "tc_dcd_mean": tc_dcd_means,
        "tc_dcd_std": tc_dcd_stds,
        "table": {
            "d": int(d_table),
            "rho": float(rho),
            "rows": table_rows,
        },
        "raw_records": raw_by_dim,
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

