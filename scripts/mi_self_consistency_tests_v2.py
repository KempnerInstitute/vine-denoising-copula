#!/usr/bin/env python3
"""
MI self-consistency tests for the paper (Appendix).

This is a **working** implementation aligned with the current codebase
(train_unified checkpoints + VineCopulaModel inference).

Tests:
  1) Data Processing Inequality (DPI): I(X;Y) >= I(X;Z) for X->Y->Z
  2) Additivity under independence:
       if (X1,Y1) ⟂ (X2,Y2) then I((X1,X2);(Y1,Y2)) = I(X1;Y1)+I(X2;Y2)
  3) Invariance to monotone transforms of marginals.

Estimators:
  - KSG (kNN MI, continuous)
  - VDC (ours): bivariate MI from the predicted copula density grid;
    group MI via TC on the 4D copula (D-vine) for the additivity test.

Outputs:
  - JSON: results/mi_self_consistency.json (for paper_artifacts.py)
  - LaTeX table: drafts/tables/tab_self_consistency.tex (optional convenience)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _pseudo_obs_1d(x: np.ndarray) -> np.ndarray:
    """Rank-transform to pseudo-observations in (0,1)."""
    x = np.asarray(x)
    n = x.shape[0]
    # Stable rank via argsort twice; ties get arbitrary ordering (fine for our use).
    r = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(np.float64)
    return (r + 0.5) / n


def _pseudo_obs(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return _pseudo_obs_1d(X)[:, None]
    cols = []
    for j in range(X.shape[1]):
        cols.append(_pseudo_obs_1d(X[:, j]))
    return np.column_stack(cols)


def _mi_from_density_grid(d: np.ndarray) -> float:
    m = int(d.shape[0])
    du = 1.0 / m
    d = np.asarray(d, dtype=np.float64)
    d = np.clip(d, 1e-12, 1e12)
    return float(np.sum(d * np.log(d)) * du * du)


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

    model_type_raw = str(config.get("model", {}).get("type", "diffusion_unet"))
    model_type = "diffusion_unet" if model_type_raw.startswith("diffusion_unet") else model_type_raw
    dev = torch.device(device)
    model = build_model(model_type, config, dev)
    model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
    model.eval()
    return _LoadedModel(model=model, model_type=model_type, config=config)


class _DCDVineMI:
    """Bivariate MI via predicted copula density grid; 4D additivity via TC (D-vine)."""

    def __init__(
        self,
        ckpt: Path,
        device: str,
        *,
        diffusion_steps: int = 4,
        diffusion_cfg_scale: float = 1.0,
        diffusion_pred_noise_clip: Optional[float] = 1.0,
        truncation_level: Optional[int] = 1,
    ):
        import torch

        self.device = str(device)
        self.loaded = _load_checkpoint_model(Path(ckpt), device=self.device)
        self.m = int(self.loaded.config.get("data", {}).get("m", 64))
        self.proj_iters = int(self.loaded.config.get("training", {}).get("projection_iters", 30))
        self.diffusion_steps = int(diffusion_steps)
        self.diffusion_cfg_scale = float(diffusion_cfg_scale)
        self.diffusion_pred_noise_clip = (
            None if diffusion_pred_noise_clip is None or float(diffusion_pred_noise_clip) <= 0 else float(diffusion_pred_noise_clip)
        )
        self.truncation_level = int(truncation_level) if truncation_level is not None else None

        # A lightweight VineCopulaModel instance gives us the robust single-pass
        # "samples -> density grid" path used in actual vine fitting.
        from vdc.vine.api import VineCopulaModel

        self.diffusion = None
        self.use_histogram_conditioning = False
        if str(self.loaded.model_type) == "diffusion_unet":
            from vdc.models.copula_diffusion import CopulaAwareDiffusion

            diff_cfg = self.loaded.config.get("diffusion", {})
            self.diffusion = CopulaAwareDiffusion(
                timesteps=int(diff_cfg.get("timesteps", 1000)),
                beta_schedule=str(diff_cfg.get("noise_schedule", "cosine")),
            ).to(self.device)
            conv_in = getattr(self.loaded.model, "conv_in", None)
            if conv_in is not None and hasattr(conv_in, "in_channels"):
                self.use_histogram_conditioning = int(conv_in.in_channels) > 1

        self.loaded.model.to(self.device)
        self.loaded.model.eval()

        self._vine_helper = VineCopulaModel(
            vine_type="dvine",
            truncation_level=self.truncation_level,
            m=self.m,
            device=self.device,
            diffusion_steps=self.diffusion_steps,
            cfg_scale=self.diffusion_cfg_scale,
            projection_iters=self.proj_iters,
            pred_noise_clip=self.diffusion_pred_noise_clip,
            hfunc_use_spline=False,
            batch_edges=False,
        )

    def mi_bivariate(self, x: np.ndarray, y: np.ndarray) -> float:
        U = _pseudo_obs(np.column_stack([x, y]))
        d = self._vine_helper._estimate_pair_density_from_samples(
            model=self.loaded.model,
            diffusion=self.diffusion,
            pair_data=U,
            use_histogram_conditioning=bool(self.use_histogram_conditioning),
        )
        return _mi_from_density_grid(d)

    def tc_dvine(self, U: np.ndarray, *, seed: int, test_frac: float = 0.25) -> float:
        """Estimate TC on d-dim copula sample via held-out mean logpdf under a D-vine."""
        from vdc.vine.api import VineCopulaModel

        U = np.asarray(U, dtype=np.float64)
        n = int(U.shape[0])
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(n)
        n_test = max(1, int(round(float(test_frac) * n)))
        te = perm[:n_test]
        tr = perm[n_test:]
        if len(tr) < 2:
            tr = perm
            te = perm

        vine = VineCopulaModel(
            vine_type="dvine",
            truncation_level=self.truncation_level,
            m=self.m,
            device=self.device,
            diffusion_steps=self.diffusion_steps,
            cfg_scale=self.diffusion_cfg_scale,
            projection_iters=self.proj_iters,
            pred_noise_clip=self.diffusion_pred_noise_clip,
            hfunc_use_spline=False,
            batch_edges=True,
            edge_batch_size=256,
        )
        vine.fit(U[tr], diffusion_model=self.loaded.model, diffusion=self.diffusion, verbose=False)
        return float(np.mean(vine.logpdf(U[te])))


def _mi_ksg(x: np.ndarray, y: np.ndarray, *, k: int, seed: int) -> float:
    from vdc.utils.information import ksg_mutual_information

    U = _pseudo_obs(np.column_stack([x, y]))
    return float(ksg_mutual_information(U[:, 0], U[:, 1], k=int(k), seed=int(seed)))


def _mi_ksg_blocks(x: np.ndarray, y: np.ndarray, *, k: int, seed: int) -> float:
    from vdc.utils.information import ksg_mutual_information

    XU = _pseudo_obs(x)
    YU = _pseudo_obs(y)
    return float(ksg_mutual_information(XU, YU, k=int(k), seed=int(seed)))


def _dpi_test(
    mi_fn: Callable[[np.ndarray, np.ndarray, int], float],
    *,
    rng: np.random.Generator,
    n: int,
    n_trials: int,
    noise_levels: List[float],
) -> Dict[str, Any]:
    from vdc.data.generators import sample_bicop
    from scipy.stats import norm

    copulas = [
        ("gaussian", {"rho": 0.7}, 0),
        ("student", {"rho": 0.7, "df": 5}, 0),
        ("clayton", {"theta": 3.0}, 0),
        ("frank", {"theta": 5.0}, 0),
    ]

    violations = []
    per = []

    for fam, params, rot in copulas:
        for _t in range(n_trials):
            seed = int(rng.integers(0, 2**30))
            uv = sample_bicop(fam, params, n=n, rotation=rot, seed=seed)
            x = uv[:, 0]
            y = uv[:, 1]
            mi0 = mi_fn(x, y, seed)

            for sigma in noise_levels[1:]:
                z = norm.ppf(np.clip(y, 1e-6, 1 - 1e-6))
                z_noisy = z + float(sigma) * rng.standard_normal(size=z.shape)
                y_noisy = norm.cdf(z_noisy)
                mi1 = mi_fn(x, y_noisy, seed + 17)
                violations.append(float(mi1 > mi0 + 1e-3))
                per.append({"family": fam, "sigma": float(sigma), "mi_base": float(mi0), "mi_noisy": float(mi1)})

    viol_rate = float(np.mean(violations)) if violations else 0.0
    return {"summary": {"violation_rate": viol_rate}, "records": per}


def _additivity_test(
    mi_bi_fn: Callable[[np.ndarray, np.ndarray, int], float],
    mi_block_fn: Callable[[np.ndarray, np.ndarray, int], float],
    mi_tc4_fn: Optional[Callable[[np.ndarray, int], float]],
    *,
    rng: np.random.Generator,
    n: int,
    n_trials: int,
) -> Dict[str, Any]:
    from vdc.data.generators import sample_bicop

    # Two independent copulas (can differ) → independence across blocks.
    copulas = [
        ("gaussian", {"rho": 0.7}, 0),
        ("clayton", {"theta": 3.0}, 0),
        ("frank", {"theta": 5.0}, 0),
    ]

    errs_abs: List[float] = []
    errs_rel: List[float] = []
    recs: List[Dict[str, Any]] = []

    for _t in range(n_trials):
        seed = int(rng.integers(0, 2**30))
        (f1, p1, r1) = copulas[int(seed) % len(copulas)]
        (f2, p2, r2) = copulas[int(seed // 7) % len(copulas)]
        uv1 = sample_bicop(f1, p1, n=n, rotation=r1, seed=seed)
        uv2 = sample_bicop(f2, p2, n=n, rotation=r2, seed=seed + 11)

        x1, y1 = uv1[:, 0], uv1[:, 1]
        x2, y2 = uv2[:, 0], uv2[:, 1]

        mi1 = mi_bi_fn(x1, y1, seed)
        mi2 = mi_bi_fn(x2, y2, seed + 1)
        rhs = float(mi1 + mi2)

        # Block MI: I((X1,X2);(Y1,Y2))
        X = np.column_stack([x1, x2])
        Y = np.column_stack([y1, y2])

        if mi_tc4_fn is not None:
            # For DCD-Vine we estimate I(A;B) = TC(U1,U2,V1,V2) since U1 ⟂ U2 and V1 ⟂ V2.
            U4 = _pseudo_obs(np.column_stack([x1, x2, y1, y2]))
            lhs = float(mi_tc4_fn(U4, seed + 2))
        else:
            lhs = float(mi_block_fn(X, Y, seed + 2))

        err = float(abs(lhs - rhs))
        errs_abs.append(err)
        errs_rel.append(float(err / max(1e-6, abs(rhs))))
        recs.append(
            {
                "copula1": f1,
                "copula2": f2,
                "mi1": float(mi1),
                "mi2": float(mi2),
                "rhs_sum": float(rhs),
                "lhs_block": float(lhs),
                "abs_error": float(err),
            }
        )

    return {
        "summary": {
            "mean_abs_error": float(np.mean(errs_abs)) if errs_abs else 0.0,
            "mean_relative_error": float(np.mean(errs_rel)) if errs_rel else 0.0,
        },
        "records": recs,
    }


def _monotone_invariance_test(
    mi_fn: Callable[[np.ndarray, np.ndarray, int], float],
    *,
    rng: np.random.Generator,
    n: int,
    n_trials: int,
) -> Dict[str, Any]:
    from vdc.data.generators import sample_bicop

    copulas = [
        ("gaussian", {"rho": 0.7}, 0),
        ("student", {"rho": 0.7, "df": 5}, 0),
        ("clayton", {"theta": 3.0}, 0),
    ]

    def t_cube(u: np.ndarray) -> np.ndarray:
        return np.clip(u, 1e-6, 1 - 1e-6) ** 3

    def t_exp(u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return (np.exp(u) - 1.0) / (np.e - 1.0)

    transforms = [("cube", t_cube), ("exp", t_exp)]

    errs_abs: List[float] = []
    errs_rel: List[float] = []
    recs: List[Dict[str, Any]] = []

    for fam, params, rot in copulas:
        for _t in range(n_trials):
            seed = int(rng.integers(0, 2**30))
            uv = sample_bicop(fam, params, n=n, rotation=rot, seed=seed)
            x, y = uv[:, 0], uv[:, 1]
            mi0 = mi_fn(x, y, seed)

            for name, f in transforms:
                mi1 = mi_fn(f(x), f(y), seed + 19)
                err = float(abs(mi1 - mi0))
                errs_abs.append(err)
                errs_rel.append(float(err / max(1e-6, abs(mi0))))
                recs.append({"family": fam, "transform": name, "mi_base": float(mi0), "mi_trans": float(mi1), "abs_error": err})

    return {
        "summary": {
            "mean_abs_error": float(np.mean(errs_abs)) if errs_abs else 0.0,
            "mean_relative_error": float(np.mean(errs_rel)) if errs_rel else 0.0,
        },
        "records": recs,
    }


def _write_latex_table(rows: List[Dict[str, Any]], out_tex: Path) -> None:
    lines = [
        "% AUTO-GENERATED by scripts/mi_self_consistency_tests_v2.py",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & DPI Viol. (\\%) $\\downarrow$ & Add. Err. (nats) $\\downarrow$ & Mono. Err. (nats) $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        name = str(r.get("estimator", "?"))
        dpi = float(r.get("dpi", {}).get("summary", {}).get("violation_rate", 0.0)) * 100.0
        add = float(r.get("additivity", {}).get("summary", {}).get("mean_abs_error", 0.0))
        mono = float(r.get("monotone_invariance", {}).get("summary", {}).get("mean_abs_error", 0.0))
        lines.append(f"{name} & {dpi:.1f} & {add:.3f} & {mono:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="MI self-consistency tests (KSG vs VDC).")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint for VDC MI (optional).")
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--n_trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--ksg-k", type=int, default=5)
    ap.add_argument("--dcd-diffusion-steps", type=int, default=4)
    ap.add_argument("--dcd-diffusion-cfg-scale", type=float, default=1.0)
    ap.add_argument("--dcd-pred-noise-clip", type=float, default=1.0)
    ap.add_argument("--dcd-truncation-level", type=int, default=1)
    ap.add_argument("--output", type=Path, default=None, help="Write LaTeX table to this path (optional).")
    ap.add_argument("--json_output", type=Path, required=True, help="Write JSON results to this path.")
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    n = int(args.n_samples)
    n_trials = int(args.n_trials)
    k = int(args.ksg_k)

    results: List[Dict[str, Any]] = []

    # KSG
    def ksg_bi(x: np.ndarray, y: np.ndarray, seed: int) -> float:
        return _mi_ksg(x, y, k=k, seed=seed)

    def ksg_blk(x: np.ndarray, y: np.ndarray, seed: int) -> float:
        return _mi_ksg_blocks(x, y, k=k, seed=seed)

    t0 = perf_counter()
    ksg_res = {
        "estimator": "KSG",
        "dpi": _dpi_test(ksg_bi, rng=rng, n=n, n_trials=n_trials, noise_levels=[0.0, 0.1, 0.3, 0.5]),
        "additivity": _additivity_test(ksg_bi, ksg_blk, None, rng=rng, n=min(n, 4000), n_trials=n_trials),
        "monotone_invariance": _monotone_invariance_test(ksg_bi, rng=rng, n=n, n_trials=n_trials),
        "wall_time_s": float(perf_counter() - t0),
    }
    results.append(ksg_res)

    # VDC (ours)
    if args.checkpoint is not None and Path(args.checkpoint).exists():
        pred_noise_clip = None if float(args.dcd_pred_noise_clip) <= 0 else float(args.dcd_pred_noise_clip)
        ours = _DCDVineMI(
            Path(args.checkpoint),
            device=str(args.device),
            diffusion_steps=int(args.dcd_diffusion_steps),
            diffusion_cfg_scale=float(args.dcd_diffusion_cfg_scale),
            diffusion_pred_noise_clip=pred_noise_clip,
            truncation_level=int(args.dcd_truncation_level) if args.dcd_truncation_level is not None else None,
        )

        def ours_bi(x: np.ndarray, y: np.ndarray, seed: int) -> float:
            _ = seed
            return ours.mi_bivariate(x, y)

        def ours_tc4(U4: np.ndarray, seed: int) -> float:
            return ours.tc_dvine(U4, seed=seed, test_frac=0.25)

        t0 = perf_counter()
        ours_res = {
            "estimator": "VDC",
            "checkpoint": str(Path(args.checkpoint)),
            "dpi": _dpi_test(ours_bi, rng=rng, n=n, n_trials=n_trials, noise_levels=[0.0, 0.1, 0.3, 0.5]),
            "additivity": _additivity_test(ours_bi, ours_bi, ours_tc4, rng=rng, n=min(n, 4000), n_trials=n_trials),
            "monotone_invariance": _monotone_invariance_test(ours_bi, rng=rng, n=n, n_trials=n_trials),
            "wall_time_s": float(perf_counter() - t0),
        }
        results.append(ours_res)

    out_json = Path(args.json_output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote JSON: {out_json}")

    if args.output is not None:
        _write_latex_table(results, Path(args.output))
        print(f"Wrote LaTeX: {Path(args.output)}")


if __name__ == "__main__":
    main()
