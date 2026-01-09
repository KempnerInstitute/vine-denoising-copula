#!/usr/bin/env python
"""
Model selection / comparison harness for copula density estimators.

This script evaluates one or more checkpoints (diffusion_unet, denoiser, enhanced_cnn, baseline_cnn)
on a fixed bivariate benchmark suite with ground-truth copula densities.

Why this exists
--------------
Your repo now supports multiple credible training formulations:
- Conditional diffusion (iterative sampling, potentially best fidelity)
- Noise-conditioned single-pass denoiser (fast inference)
- Strong CNN baseline

We need an apples-to-apples evaluation harness to pick the best method empirically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from time import perf_counter

# Project imports
from scripts.train_unified import build_model, build_coordinates  # type: ignore
from vdc.inference.density import sample_density_grid
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project
from vdc.data.generators import (
    sample_bicop,
    analytic_logpdf_grid,
    generate_gaussian_vine,
    generate_student_vine,
    generate_clayton_vine,
    generate_mixed_vine,
)
from vdc.losses import nll_points
from vdc.vine.api import VineCopulaModel


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

DEFAULT_VINE_TASKS = [
    ("gaussian_ar1", {"rho": 0.6}, 5),
    ("student_ar1", {"rho": 0.6, "df": 5}, 5),
    ("clayton_dvine", {"theta": 2.0}, 5),
    ("mixed_dvine", {}, 5),
]


def _normalize_density_np(d: np.ndarray) -> np.ndarray:
    m = d.shape[0]
    du = 1.0 / m
    mass = float((d * du * du).sum())
    if mass <= 0:
        return np.ones_like(d) * (m * m)
    return d / mass


def _project_density_np(d: np.ndarray, iters: int, device: torch.device) -> np.ndarray:
    t = torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0).to(device)
    t = t.clamp_min(1e-12)
    t = copula_project(t, iters=int(iters))
    return t[0, 0].detach().cpu().numpy()


def _tail_dependence_from_density_grid(d: np.ndarray, q_upper: float = 0.95, q_lower: float = 0.05) -> Dict[str, float]:
    """
    Crude grid-based tail dependence proxies for a copula density:
      λ_U(q) ≈ P(U>q, V>q) / (1-q)
      λ_L(q) ≈ P(U<q, V<q) / q
    For true tail dependence coefficients, take limits q→1 or q→0; here we use fixed q for benchmarking.
    """
    m = int(d.shape[0])
    du = 1.0 / m
    centers = (np.arange(m) + 0.5) * du

    up_mask = centers >= q_upper
    lo_mask = centers <= q_lower

    mass_uu = float((d[np.ix_(up_mask, up_mask)] * du * du).sum())
    mass_ll = float((d[np.ix_(lo_mask, lo_mask)] * du * du).sum())

    lam_u = mass_uu / max(1e-12, (1.0 - q_upper))
    lam_l = mass_ll / max(1e-12, q_lower)
    return {"tail_u": float(lam_u), "tail_l": float(lam_l)}

def _pit_metrics_uniform(W: np.ndarray) -> Dict[str, float]:
    """Compute simple PIT uniformity diagnostics for Rosenblatt outputs W in [0,1]^d."""
    from scipy.stats import kstest

    d = W.shape[1]
    ks = [float(kstest(W[:, k], "uniform").statistic) for k in range(d)]
    # Independence proxy: mean absolute off-diagonal correlation
    corr = np.corrcoef(W.T)
    mask = ~np.eye(d, dtype=bool)
    mean_abs_corr = float(np.mean(np.abs(corr[mask])))
    return {
        "pit_ks_mean": float(np.mean(ks)),
        "pit_ks_max": float(np.max(ks)),
        "pit_mean_abs_corr": mean_abs_corr,
    }


def _generate_vine_data(task: str, n: int, d: int, params: Dict[str, Any], seed: int = 123) -> np.ndarray:
    if task == "gaussian_ar1":
        return generate_gaussian_vine(n=n, d=d, rho=float(params.get("rho", 0.6)), seed=seed)
    if task == "student_ar1":
        return generate_student_vine(
            n=n, d=d, rho=float(params.get("rho", 0.6)), df=int(params.get("df", 5)), seed=seed
        )
    if task == "clayton_dvine":
        return generate_clayton_vine(n=n, d=d, theta=float(params.get("theta", 2.0)), seed=seed)
    if task == "mixed_dvine":
        return generate_mixed_vine(n=n, d=d, seed=seed)
    raise ValueError(f"Unknown vine task: {task}")


@torch.no_grad()
def evaluate_vine_tasks(
    model: torch.nn.Module,
    model_type: str,
    diffusion: Any,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Small vine-level sanity suite:
    - fit a D-vine on synthetic d-dimensional data
    - report held-out NLL and PIT (Rosenblatt) uniformity diagnostics
    """
    m = int(config.get("data", {}).get("m", 64))
    proj_iters = int(config.get("training", {}).get("projection_iters", 30))
    diff_steps = int(config.get("diffusion", {}).get("sampling_steps", 50))
    cfg_scale = float(config.get("diffusion", {}).get("cfg_scale", 1.0))

    # Keep this small so diffusion evaluation is tractable.
    n_train = int(config.get("evaluation", {}).get("vine_n_train", 2000))
    n_test = int(config.get("evaluation", {}).get("vine_n_test", 2000))

    rows: List[Dict[str, Any]] = []

    for task, params, d in DEFAULT_VINE_TASKS:
        U = _generate_vine_data(task, n=n_train + n_test, d=int(d), params=params, seed=123)
        U_train = U[:n_train]
        U_test = U[n_train:]

        vine = VineCopulaModel(
            vine_type="dvine",
            m=m,
            device=str(device),
            diffusion_steps=diff_steps,
            cfg_scale=cfg_scale,
            projection_iters=proj_iters,
        )

        t0 = perf_counter()
        vine.fit(U_train, diffusion_model=model, diffusion=diffusion, verbose=False)
        fit_s = float(perf_counter() - t0)

        t1 = perf_counter()
        logp = vine.logpdf(U_test)
        ll_s = float(perf_counter() - t1)
        mean_nll = float(-np.mean(logp))

        t2 = perf_counter()
        W = vine.rosenblatt(U_test)
        pit_s = float(perf_counter() - t2)
        pit = _pit_metrics_uniform(W)

        rows.append(
            {
                "task": task,
                "d": int(d),
                "n_train": n_train,
                "n_test": n_test,
                "mean_nll": mean_nll,
                "fit_s": fit_s,
                "logpdf_s": ll_s,
                "rosenblatt_s": pit_s,
                **pit,
            }
        )

    summary = {
        "vine_tasks": rows,
        "vine_mean_nll": float(np.mean([r["mean_nll"] for r in rows])),
        "vine_mean_pit_ks_mean": float(np.mean([r["pit_ks_mean"] for r in rows])),
        "vine_mean_fit_s": float(np.mean([r["fit_s"] for r in rows])),
    }
    return summary


@torch.no_grad()
def estimate_density_single_pass(
    model: torch.nn.Module,
    model_type: str,
    samples: np.ndarray,
    m: int,
    device: torch.device,
    projection_iters: int,
    use_coordinates: bool,
    use_probit_coords: bool,
) -> np.ndarray:
    """
    Single-pass estimator: histogram -> model -> density -> IPFP.
    """
    # Build histogram density on grid
    from vdc.inference.density import scatter_to_hist as scatter_to_hist_density

    hist = scatter_to_hist_density(samples, m=m, reflect=True)  # density integrating to 1
    x = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,m,m)

    if use_coordinates:
        coords = build_coordinates(1, m, device, probit=use_probit_coords)
        x = torch.cat([x, coords], dim=1)

    # Forward model
    if model_type == "denoiser":
        t = torch.zeros(1, device=device)
        out = model(x, t)
    else:
        # enhanced_cnn supports optional time conditioning but we keep it off for eval unless configured
        out = model(x) if model_type != "enhanced_cnn" else model(x)

    if isinstance(out, dict):
        if "density" in out:
            d = out["density"]
        elif "log_density" in out:
            d = torch.exp(out["log_density"].clamp(min=-20, max=20))
        elif "residual" in out:
            # Interpret residual as log-density residual on top of log(hist)
            d = torch.exp((torch.log(x[:, :1].clamp_min(1e-12)) + out["residual"]).clamp(min=-20, max=20))
        else:
            raise ValueError(f"Unknown output keys: {list(out.keys())}")
    else:
        d = out

    d = d.clamp(min=1e-12, max=1e6)
    # Normalize to integrate 1 then project
    du = 1.0 / m
    d = d / ((d * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
    if projection_iters > 0:
        d = copula_project(d, iters=int(projection_iters))
        d = d / ((d * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
    return d[0, 0].detach().cpu().numpy()


@torch.no_grad()
def evaluate_checkpoint(ckpt_path: Path, device: torch.device, n_samples: int) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_type = str(config.get("model", {}).get("type", "diffusion_unet"))

    # Build model
    model = build_model(model_type, config, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Diffusion object if needed
    diffusion = None
    if model_type == "diffusion_unet":
        diff_cfg = config.get("diffusion", {})
        diffusion = CopulaAwareDiffusion(
            timesteps=int(diff_cfg.get("timesteps", 1000)),
            beta_schedule=str(diff_cfg.get("noise_schedule", "cosine")),
        ).to(device)

    m = int(config.get("data", {}).get("m", 64))
    proj_iters = int(config.get("training", {}).get("projection_iters", 30))
    use_coords = bool(config.get("model", {}).get("use_coordinates", False))
    use_probit_coords = bool(config.get("model", {}).get("use_probit_coords", False))

    # Diffusion inference params
    diff_steps = int(config.get("diffusion", {}).get("sampling_steps", 50))
    cfg_scale = float(config.get("diffusion", {}).get("cfg_scale", 1.0))

    rows: List[Dict[str, Any]] = []

    for family, params, rotation, name in DEFAULT_TEST_COPULAS:
        pts = sample_bicop(family, params, n=n_samples, rotation=rotation, seed=123)

        # Predict
        if model_type == "diffusion_unet":
            assert diffusion is not None
            use_hist_cond = bool(getattr(model, "conv_in").in_channels > 1)
            d_pred = sample_density_grid(
                model=model,
                diffusion=diffusion,
                samples=pts,
                m=m,
                device=device,
                num_steps=diff_steps,
                cfg_scale=cfg_scale,
                use_histogram_conditioning=use_hist_cond,
                projection_iters=proj_iters,
            )
        else:
            d_pred = estimate_density_single_pass(
                model=model,
                model_type=model_type,
                samples=pts,
                m=m,
                device=device,
                projection_iters=proj_iters,
                use_coordinates=use_coords,
                use_probit_coords=use_probit_coords,
            )

        # True density (projected for fair discrete comparison)
        lg_true = analytic_logpdf_grid(family, params, m=m, rotation=rotation)
        d_true = np.exp(np.clip(lg_true, -20, 20))
        d_true = _normalize_density_np(d_true)
        d_true = _project_density_np(d_true, iters=max(50, proj_iters), device=device)
        d_true = _normalize_density_np(d_true)

        du = 1.0 / m
        ise = float(np.mean((d_pred - d_true) ** 2) * du * du)

        # Differential cross-entropy H(p_true, p_pred) ≈ -∫ p_true log p_pred
        ce = float(-np.sum(d_true * np.log(d_pred + 1e-12)) * du * du)

        tail_pred = _tail_dependence_from_density_grid(d_pred)
        tail_true = _tail_dependence_from_density_grid(d_true)
        tail_u_err = float(abs(tail_pred["tail_u"] - tail_true["tail_u"]))
        tail_l_err = float(abs(tail_pred["tail_l"] - tail_true["tail_l"]))

        d_pred_t = torch.from_numpy(d_pred).float().unsqueeze(0).unsqueeze(0).to(device)
        pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
        nll = float(nll_points(d_pred_t, pts_t).item())

        # Marginal uniformity errors
        row_m = d_pred.sum(axis=1) * du
        col_m = d_pred.sum(axis=0) * du
        row_err = float(np.mean(np.abs(row_m - 1.0)))
        col_err = float(np.mean(np.abs(col_m - 1.0)))

        rows.append(
            {
                "name": name,
                "family": family,
                "rotation": rotation,
                "ise": ise,
                "ce": ce,
                "nll": nll,
                "row_marg_err": row_err,
                "col_marg_err": col_err,
                "tail_u": float(tail_pred["tail_u"]),
                "tail_l": float(tail_pred["tail_l"]),
                "tail_u_err": tail_u_err,
                "tail_l_err": tail_l_err,
                "peak": float(np.max(d_pred)),
            }
        )

    # Aggregate
    summary = {
        "checkpoint": str(ckpt_path),
        "model_type": model_type,
        "m": m,
        "n_samples": n_samples,
        "mean_ise": float(np.mean([r["ise"] for r in rows])),
        "mean_ce": float(np.mean([r["ce"] for r in rows])),
        "mean_nll": float(np.mean([r["nll"] for r in rows])),
        "mean_row_marg_err": float(np.mean([r["row_marg_err"] for r in rows])),
        "mean_col_marg_err": float(np.mean([r["col_marg_err"] for r in rows])),
        "mean_tail_u_err": float(np.mean([r["tail_u_err"] for r in rows])),
        "mean_tail_l_err": float(np.mean([r["tail_l_err"] for r in rows])),
        "per_copula": rows,
    }

    # Vine-level sanity suite (skip by default on CPU to avoid very long runs)
    include_vine = bool(config.get("evaluation", {}).get("include_vine", device.type == "cuda"))
    if include_vine:
        summary.update(evaluate_vine_tasks(model, model_type, diffusion, config, device=device))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare copula density estimator checkpoints")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="One or more model_step_*.pt files")
    parser.add_argument("--n-samples", type=int, default=2000, help="Samples per test copula")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", type=Path, default=None, help="Write full results JSON to this path")
    parser.add_argument("--out-csv", type=Path, default=None, help="Write per-copula results CSV to this path")
    args = parser.parse_args()

    device = torch.device(args.device)
    results = []
    for ckpt in args.checkpoints:
        results.append(evaluate_checkpoint(ckpt, device=device, n_samples=int(args.n_samples)))

    # Print compact table
    print("\n=== MODEL SELECTION SUMMARY ===")
    for r in results:
        vine_str = ""
        if "vine_mean_nll" in r:
            vine_str = f" | vine NLL={r['vine_mean_nll']:.3f} | PIT-KS={r['vine_mean_pit_ks_mean']:.3f}"
        print(
            f"- {Path(r['checkpoint']).name:30s} | {r['model_type']:14s} | "
            f"mean ISE={r['mean_ise']:.4e} | mean CE={r['mean_ce']:.3f} | mean NLL={r['mean_nll']:.3f} | "
            f"tail_err(U,L)=({r['mean_tail_u_err']:.3f},{r['mean_tail_l_err']:.3f}) | "
            f"marg_err=({r['mean_row_marg_err']:.3e},{r['mean_col_marg_err']:.3e})"
            f"{vine_str}"
        )

    # Optional outputs
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "device": str(device),
            "n_samples": int(args.n_samples),
            "results": results,
        }
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON: {args.out_json}")

    if args.out_csv is not None:
        import csv

        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        # Flatten per-copula rows (bivariate) and attach checkpoint metadata
        rows_out: List[Dict[str, Any]] = []
        for r in results:
            for row in r.get("per_copula", []):
                rows_out.append(
                    {
                        "checkpoint": r.get("checkpoint"),
                        "model_type": r.get("model_type"),
                        **row,
                    }
                )
        if rows_out:
            fieldnames = list(rows_out[0].keys())
            with args.out_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows_out)
            print(f"Wrote CSV: {args.out_csv}")
        else:
            print("No per-copula rows to write to CSV.")


if __name__ == "__main__":
    main()

