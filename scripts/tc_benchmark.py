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
import importlib.util
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


def _load_mi_estimation_module(repo_root: Path):
    """Load scripts/mi_estimation.py as a module for neural MI baselines."""
    mod_path = repo_root / "scripts" / "mi_estimation.py"
    mod_name = "vdc_mi_estimation_tc_runtime"
    spec = importlib.util.spec_from_file_location(mod_name, str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load MI estimation module: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tc_neural_chain_rule(
    U: np.ndarray,
    *,
    estimator: str,
    mi_mod: Any,
    device: str,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    hidden: int,
    weight_decay: float,
    grad_clip: float,
    eval_batches: int = 10,
    t_clip: float = 20.0,
) -> Tuple[float, float]:
    """Estimate TC via chain rule using a neural MI estimator on growing blocks."""
    import torch

    U = np.asarray(U, dtype=np.float32)
    n, d = U.shape
    if n < 10 or d < 2:
        return 0.0, 0.0

    dev = torch.device(device)
    t0 = perf_counter()
    tc = 0.0

    for i in range(1, d):
        x = np.asarray(U[:, :i], dtype=np.float32)
        y = np.asarray(U[:, i : i + 1], dtype=np.float32)
        step_seed = int(seed) + 10_000 * i
        if estimator == "mine":
            mi = mi_mod._mine_estimate_mi(
                x=x,
                y=y,
                seed=step_seed,
                device=dev,
                steps=int(steps),
                lr=float(lr),
                batch_size=int(batch_size),
                hidden_dim=int(hidden),
                weight_decay=float(weight_decay),
                grad_clip=float(grad_clip),
            )
        elif estimator == "infonce":
            mi = mi_mod._infonce_estimate_mi(
                x=x,
                y=y,
                seed=step_seed,
                device=dev,
                steps=int(steps),
                lr=float(lr),
                batch_size=int(batch_size),
                hidden_dim=int(hidden),
                weight_decay=float(weight_decay),
                grad_clip=float(grad_clip),
                eval_batches=int(eval_batches),
            )
        elif estimator == "nwj":
            mi = mi_mod._nwj_estimate_mi(
                x=x,
                y=y,
                seed=step_seed,
                device=dev,
                steps=int(steps),
                lr=float(lr),
                batch_size=int(batch_size),
                hidden_dim=int(hidden),
                weight_decay=float(weight_decay),
                grad_clip=float(grad_clip),
                t_clip=float(t_clip),
                eval_batches=int(eval_batches),
            )
        else:
            raise ValueError(f"Unknown estimator for TC chain rule: {estimator}")
        tc += max(0.0, float(mi))

    return float(tc), float(perf_counter() - t0)


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
    model_type_raw = str(config.get("model", {}).get("type", "diffusion_unet"))
    model_type = "diffusion_unet" if model_type_raw.startswith("diffusion_unet") else model_type_raw
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
    diffusion_steps: int,
    diffusion_cfg_scale: float,
    diffusion_pred_noise_clip: Optional[float],
    truncation_level: Optional[int],
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

    diffusion_obj = None
    if str(loaded.model_type) == "diffusion_unet":
        from vdc.models.copula_diffusion import CopulaAwareDiffusion

        diff_cfg = loaded.config.get("diffusion", {})
        diffusion_obj = CopulaAwareDiffusion(
            timesteps=int(diff_cfg.get("timesteps", 1000)),
            beta_schedule=str(diff_cfg.get("noise_schedule", "cosine")),
        ).to(str(device))

    vine = VineCopulaModel(
        vine_type="dvine",
        truncation_level=int(truncation_level) if truncation_level is not None else None,
        m=m,
        device=str(device),
        projection_iters=int(proj_iters),
        hfunc_use_spline=bool(hfunc_use_spline),
        batch_edges=bool(batch_edges),
        edge_batch_size=int(edge_batch_size),
        diffusion_steps=int(diffusion_steps),
        cfg_scale=float(diffusion_cfg_scale),
        pred_noise_clip=diffusion_pred_noise_clip,
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
    vine.fit(U_tr, diffusion_model=loaded.model, diffusion=diffusion_obj, verbose=False)
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
    p.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint (denoiser/enhanced_cnn/diffusion_unet).")
    p.add_argument("--device", type=str, default="cuda", help="torch device (default: cuda).")
    p.add_argument("--rho", type=float, default=0.7, help="AR(1) correlation rho.")
    p.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10, 20, 50], help="Dimensions to benchmark.")
    p.add_argument("--n", type=int, default=5000, help="Sample size per trial for the scaling curve.")
    p.add_argument("--n-trials", type=int, default=3, help="Random trials per dimension.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ksg-k", type=int, default=5)
    p.add_argument("--include-mine", action="store_true", help="Include a MINE chain-rule TC baseline.")
    p.add_argument("--include-infonce", action="store_true", help="Include an InfoNCE chain-rule TC baseline.")
    p.add_argument("--include-nwj", action="store_true", help="Include an NWJ chain-rule TC baseline.")
    p.add_argument("--mine-steps", type=int, default=800)
    p.add_argument("--mine-batch-size", type=int, default=512)
    p.add_argument("--mine-lr", type=float, default=1e-4)
    p.add_argument("--mine-hidden", type=int, default=256)
    p.add_argument("--mine-weight-decay", type=float, default=0.0)
    p.add_argument("--mine-grad-clip", type=float, default=5.0)
    p.add_argument("--mine-eval-batches", type=int, default=10)
    p.add_argument("--nwj-t-clip", type=float, default=20.0)
    p.add_argument("--test-frac", type=float, default=0.25, help="Held-out fraction for DCD-Vine evaluation.")
    p.add_argument("--batch-edges", action="store_true", help="Enable tree-level edge batching (ours, faster).")
    p.add_argument("--edge-batch-size", type=int, default=256)
    p.add_argument("--hfunc-use-spline", action="store_true", help="Use spline h-functions (slower, more accurate).")
    p.add_argument("--diffusion-steps", type=int, default=16, help="DDIM steps for diffusion checkpoints.")
    p.add_argument("--diffusion-cfg-scale", type=float, default=1.0, help="CFG scale for histogram-conditioned diffusion.")
    p.add_argument(
        "--diffusion-pred-noise-clip",
        type=float,
        default=1.0,
        help="Clip predicted diffusion noise to [-clip,clip]; <=0 disables clipping.",
    )
    p.add_argument(
        "--truncation-level",
        type=int,
        default=None,
        help="Optional D-vine truncation level (1..d-1). If unset, fit full vine.",
    )
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--table-d", type=int, default=10, help="Dimension used for the main TC estimation table.")
    p.add_argument("--table-n", type=int, nargs="+", default=[500, 5000], help="Sample sizes for the table.")
    p.add_argument("--skip-table", action="store_true", help="Skip fixed-d table generation (faster).")
    args = p.parse_args()

    from vdc.data.generators import generate_gaussian_vine
    import torch

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        args.device = "cpu"
    pred_noise_clip = None if float(args.diffusion_pred_noise_clip) <= 0 else float(args.diffusion_pred_noise_clip)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    loaded = _load_checkpoint_model(ckpt, device=str(args.device))
    mi_mod = None
    if bool(args.include_mine) or bool(args.include_infonce) or bool(args.include_nwj):
        mi_mod = _load_mi_estimation_module(REPO_ROOT)

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
    tc_infonce_means: List[float] = []
    tc_infonce_stds: List[float] = []
    tc_nwj_means: List[float] = []
    tc_nwj_stds: List[float] = []

    raw_by_dim: List[Dict[str, Any]] = []

    for d in dims:
        print(f"[dim={d}] starting {n_trials} trial(s) with n={n}", flush=True)
        true_val = _true_tc_gaussian_ar1(d, rho)
        tc_true.append(true_val)

        ksg_vals: List[float] = []
        dcd_vals: List[float] = []
        mine_vals: List[float] = []
        infonce_vals: List[float] = []
        nwj_vals: List[float] = []

        for t in range(n_trials):
            trial_seed = seed + 10_000 * d + 1_000 * t
            U = generate_gaussian_vine(n=n, d=d, rho=rho, seed=trial_seed)

            t0 = perf_counter()
            tc_ksg = _tc_ksg_chain_rule(U, k=k, seed=trial_seed)
            ksg_time = float(perf_counter() - t0)

            mine_time = None
            tc_mine = None
            if bool(args.include_mine):
                tc_mine, mine_time = _tc_neural_chain_rule(
                    U,
                    estimator="mine",
                    mi_mod=mi_mod,
                    device=str(args.device),
                    seed=trial_seed,
                    steps=int(args.mine_steps),
                    batch_size=int(args.mine_batch_size),
                    lr=float(args.mine_lr),
                    hidden=int(args.mine_hidden),
                    weight_decay=float(args.mine_weight_decay),
                    grad_clip=float(args.mine_grad_clip),
                    eval_batches=int(args.mine_eval_batches),
                )
                mine_vals.append(float(tc_mine))

            infonce_time = None
            tc_infonce = None
            if bool(args.include_infonce):
                tc_infonce, infonce_time = _tc_neural_chain_rule(
                    U,
                    estimator="infonce",
                    mi_mod=mi_mod,
                    device=str(args.device),
                    seed=trial_seed,
                    steps=int(args.mine_steps),
                    batch_size=int(args.mine_batch_size),
                    lr=float(args.mine_lr),
                    hidden=int(args.mine_hidden),
                    weight_decay=float(args.mine_weight_decay),
                    grad_clip=float(args.mine_grad_clip),
                    eval_batches=int(args.mine_eval_batches),
                )
                infonce_vals.append(float(tc_infonce))

            nwj_time = None
            tc_nwj = None
            if bool(args.include_nwj):
                tc_nwj, nwj_time = _tc_neural_chain_rule(
                    U,
                    estimator="nwj",
                    mi_mod=mi_mod,
                    device=str(args.device),
                    seed=trial_seed,
                    steps=int(args.mine_steps),
                    batch_size=int(args.mine_batch_size),
                    lr=float(args.mine_lr),
                    hidden=int(args.mine_hidden),
                    weight_decay=float(args.mine_weight_decay),
                    grad_clip=float(args.mine_grad_clip),
                    eval_batches=int(args.mine_eval_batches),
                    t_clip=float(args.nwj_t_clip),
                )
                nwj_vals.append(float(tc_nwj))

            print(
                f"    fitting DCD-Vine (ddim={int(args.diffusion_steps)}, cfg={float(args.diffusion_cfg_scale):.2f})...",
                flush=True,
            )
            tc_dcd, dcd_time = _tc_dcd_vine(
                U,
                loaded=loaded,
                device=str(args.device),
                seed=trial_seed,
                test_frac=float(args.test_frac),
                batch_edges=bool(args.batch_edges),
                edge_batch_size=int(args.edge_batch_size),
                hfunc_use_spline=bool(args.hfunc_use_spline),
                diffusion_steps=int(args.diffusion_steps),
                diffusion_cfg_scale=float(args.diffusion_cfg_scale),
                diffusion_pred_noise_clip=pred_noise_clip,
                truncation_level=args.truncation_level,
            )
            print(
                f"  trial={t+1}/{n_trials} tc_true={true_val:.4f} "
                f"ksg={tc_ksg:.4f} dcd={tc_dcd:.4f} "
                f"time_s(ksg,dcd)=({ksg_time:.2f},{dcd_time:.2f})",
                flush=True,
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
                    "tc_infonce": float(tc_infonce) if tc_infonce is not None else None,
                    "tc_nwj": float(tc_nwj) if tc_nwj is not None else None,
                    "tc_dcd": float(tc_dcd),
                    "time_ksg_s": float(ksg_time),
                    "time_mine_s": float(mine_time) if mine_time is not None else None,
                    "time_infonce_s": float(infonce_time) if infonce_time is not None else None,
                    "time_nwj_s": float(nwj_time) if nwj_time is not None else None,
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
        if bool(args.include_infonce) and infonce_vals:
            m_infonce, s_infonce = _mean_std(infonce_vals)
            tc_infonce_means.append(m_infonce)
            tc_infonce_stds.append(s_infonce)
        if bool(args.include_nwj) and nwj_vals:
            m_nwj, s_nwj = _mean_std(nwj_vals)
            tc_nwj_means.append(m_nwj)
            tc_nwj_stds.append(s_nwj)

    # Table entries (d fixed, n varies): absolute error + time.
    table_rows: List[Dict[str, Any]] = []
    d_table = int(args.table_d)
    if not bool(args.skip_table):
        for n_table in [int(x) for x in args.table_n]:
            print(f"[table] d={d_table} n={n_table}", flush=True)
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
                tc_m, t_mine = _tc_neural_chain_rule(
                    U,
                    estimator="mine",
                    mi_mod=mi_mod,
                    device=str(args.device),
                    seed=trial_seed + 999,
                    steps=int(args.mine_steps),
                    batch_size=int(args.mine_batch_size),
                    lr=float(args.mine_lr),
                    hidden=int(args.mine_hidden),
                    weight_decay=float(args.mine_weight_decay),
                    grad_clip=float(args.mine_grad_clip),
                    eval_batches=int(args.mine_eval_batches),
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
            diffusion_steps=int(args.diffusion_steps),
            diffusion_cfg_scale=float(args.diffusion_cfg_scale),
            diffusion_pred_noise_clip=pred_noise_clip,
            truncation_level=args.truncation_level,
        )
            print(
                f"  table tc_true={tc_t:.4f} ksg={tc_k:.4f} dcd={tc_d:.4f} "
                f"time_s(ksg,dcd)=({t_ksg:.2f},{t_dcd:.2f})",
                flush=True,
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
        "diffusion_steps": int(args.diffusion_steps),
        "diffusion_cfg_scale": float(args.diffusion_cfg_scale),
        "diffusion_pred_noise_clip": pred_noise_clip,
        "truncation_level": int(args.truncation_level) if args.truncation_level is not None else None,
        "dimensions": dims,
        "tc_true": tc_true,
        "tc_ksg_mean": tc_ksg_means,
        "tc_ksg_std": tc_ksg_stds,
        "tc_mine_mean": tc_mine_means if bool(args.include_mine) else None,
        "tc_mine_std": tc_mine_stds if bool(args.include_mine) else None,
        "tc_infonce_mean": tc_infonce_means if bool(args.include_infonce) else None,
        "tc_infonce_std": tc_infonce_stds if bool(args.include_infonce) else None,
        "tc_nwj_mean": tc_nwj_means if bool(args.include_nwj) else None,
        "tc_nwj_std": tc_nwj_stds if bool(args.include_nwj) else None,
        "tc_dcd_mean": tc_dcd_means,
        "tc_dcd_std": tc_dcd_stds,
        "table": {
            "d": int(d_table),
            "rho": float(rho),
            "rows": table_rows,
        }
        if not bool(args.skip_table)
        else None,
        "raw_records": raw_by_dim,
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
