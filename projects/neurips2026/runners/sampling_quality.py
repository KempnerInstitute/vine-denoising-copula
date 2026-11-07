#!/usr/bin/env python3
"""Unconditional sampling-quality protocol for the paper appendix.

Fit VDC and TLL on the same training pseudo-observations; compare simulated
samples and a training-data bootstrap with held-out Kendall-tau and tail
concentration. The paper's historical dataset labelled power here was CCPP;
this public runner calls it ccpp. S&P100 prices are not redistributed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _repo_root() -> Path:
    return Path(os.environ["VDC_LIBRARY_ROOT"]).resolve()


@contextmanager
def time_limit(seconds: int):
    def _h(s, f):
        raise TimeoutError(f"timed out after {seconds}s")

    if seconds <= 0:
        yield
        return
    old = signal.signal(signal.SIGALRM, _h)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _empirical_pit(X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train-rank PIT (same convention as e23_sp100_density.py)."""
    from scipy.stats import rankdata

    n_tr, d = X_tr.shape
    U_tr = np.zeros_like(X_tr, dtype=np.float64)
    U_te = np.zeros_like(X_te, dtype=np.float64)
    for j in range(d):
        ranks = rankdata(X_tr[:, j], method="average")
        U_tr[:, j] = ranks / (n_tr + 1)
        sorted_x = np.sort(X_tr[:, j])
        for i in range(X_te.shape[0]):
            r = np.searchsorted(sorted_x, X_te[i, j], side="right")
            U_te[i, j] = (r + 0.5) / (n_tr + 1)
    return np.clip(U_tr, 1e-6, 1 - 1e-6), np.clip(U_te, 1e-6, 1 - 1e-6)


def _load_vdc_checkpoint(ckpt_path: Path, device: str):
    import torch
    from vdc.train.unified_trainer import build_model

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    mt = str(config.get("model", {}).get("type", "diffusion_unet"))
    mt_norm = "diffusion_unet" if mt.startswith("diffusion_unet") else mt
    model = build_model(mt_norm, config, torch.device(device))
    model.load_state_dict(ckpt.get("model_state_dict", {}), strict=True)
    model.eval()
    return model


def select_pairs(d: int, max_pairs: int, pair_seed: int) -> List[Tuple[int, int]]:
    """All variable pairs, or a fixed random subset of `max_pairs` if d is large.

    Selection is deterministic in (d, max_pairs, pair_seed) so all methods and
    all run seeds are scored on the same pairs.
    """
    all_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    if len(all_pairs) <= int(max_pairs):
        return all_pairs
    rng = np.random.default_rng(int(pair_seed))
    idx = rng.choice(len(all_pairs), size=int(max_pairs), replace=False)
    return [all_pairs[k] for k in sorted(idx)]


def tau_vector(U: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
    from scipy.stats import kendalltau

    out = np.zeros(len(pairs), dtype=np.float64)
    for p, (i, j) in enumerate(pairs):
        t = kendalltau(U[:, i], U[:, j])[0]
        out[p] = float(t) if np.isfinite(t) else 0.0
    return out


def tail_lambda_vectors(
    U: np.ndarray, pairs: List[Tuple[int, int]], q: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical per-pair tail concentration at level q.

    lambda_L = P(U_i <= q, U_j <= q) / q ; lambda_U = P(U_i > 1-q, U_j > 1-q) / q.
    """
    lo = U <= q
    hi = U > (1.0 - q)
    lamL = np.zeros(len(pairs), dtype=np.float64)
    lamU = np.zeros(len(pairs), dtype=np.float64)
    for p, (i, j) in enumerate(pairs):
        lamL[p] = float(np.mean(lo[:, i] & lo[:, j])) / q
        lamU[p] = float(np.mean(hi[:, i] & hi[:, j])) / q
    return lamL, lamU


def metrics_vs_test(
    U_sim: np.ndarray,
    pairs: List[Tuple[int, int]],
    tau_te: np.ndarray,
    lamL_te: np.ndarray,
    lamU_te: np.ndarray,
    q: float,
) -> Dict[str, float]:
    tau_sim = tau_vector(U_sim, pairs)
    lamL_sim, lamU_sim = tail_lambda_vectors(U_sim, pairs, q)
    return {
        # ||tau_sim - tau_test||_F / sqrt(#pairs)
        "tau_rmse": float(np.sqrt(np.mean((tau_sim - tau_te) ** 2))),
        "lambdaL_mae": float(np.mean(np.abs(lamL_sim - lamL_te))),
        "lambdaU_mae": float(np.mean(np.abs(lamU_sim - lamU_te))),
    }


def load_dataset(
    name: str,
    seed: int,
    sp100_path: Path,
    train_frac: float,
    max_train: int,
    max_test: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (U_tr, U_te) pseudo-observations for a dataset."""
    rng = np.random.default_rng(int(seed))

    if name == "sp100":
        X = np.load(sp100_path).astype(np.float64)
        n = X.shape[0]
        perm = rng.permutation(n)
        n_tr = int(float(train_frac) * n)
        X_tr = X[perm[:n_tr]]
        X_te = X[perm[n_tr:]]
        U_tr, U_te = _empirical_pit(X_tr, X_te)
        return U_tr, U_te

    from vdc.data.paths import data_root as _data_root
    from vdc.data.tabular import EmpiricalMarginals, maybe_load_uci

    X_tr, X_te = maybe_load_uci(name, _data_root())
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)
    if X_tr.shape[0] > int(max_train):
        idx = rng.choice(X_tr.shape[0], size=int(max_train), replace=False)
        X_tr = X_tr[idx]
    if X_te.shape[0] > int(max_test):
        idx = rng.choice(X_te.shape[0], size=int(max_test), replace=False)
        X_te = X_te[idx]
    marg = EmpiricalMarginals.fit(X_tr)
    U_tr = np.clip(marg.transform(X_tr), 1e-6, 1 - 1e-6)
    U_te = np.clip(marg.transform(X_te), 1e-6, 1 - 1e-6)
    return U_tr, U_te


def sample_vdc(
    U_tr: np.ndarray,
    model,
    m_grid: int,
    device: str,
    n_sim: int,
    seed: int,
    vectorized_inverse: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """Fit a VDC D-vine and draw n_sim unconditional samples (copula space)."""
    from vdc.vine.api import VineCopulaModel

    t0 = time.perf_counter()
    vine = VineCopulaModel(
        vine_type="dvine",
        m=int(m_grid),
        device=str(device),
        projection_iters=15,
        hfunc_use_spline=False,
        batch_edges=True,
        edge_batch_size=256,
        hfunc_vectorized_inverse=vectorized_inverse,
    )
    vine.fit(U_tr, diffusion_model=model, diffusion=None, verbose=False)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    # Same sampling entry point as e4_imputation_benchmark.py's candidate pool.
    U_sim = vine.simulate(int(n_sim), seed=int(seed) + 123)
    sim_s = time.perf_counter() - t0
    return np.asarray(U_sim, dtype=np.float64), float(fit_s), float(sim_s)


def sample_pyvine(
    U_tr: np.ndarray,
    n_sim: int,
    seed: int,
    num_threads: int,
    timeout_s: int,
    grid_size: int = 30,
) -> Tuple[np.ndarray, float, float]:
    """Fit pyvine-TLL and draw n_sim unconditional samples (copula space)."""
    from vdc.baselines.pyvinecopulib import fit_vinecop, is_available

    if not is_available():
        raise RuntimeError("pyvinecopulib unavailable")
    t0 = time.perf_counter()
    with time_limit(int(timeout_s)):
        vine = fit_vinecop(
            U_tr,
            mode="nonparametric",
            num_threads=int(num_threads),
            nonparametric_grid_size=int(grid_size),
        )
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    U_sim = np.asarray(
        vine.simulate(int(n_sim), num_threads=int(num_threads), seeds=[int(seed) + 123]),
        dtype=np.float64,
    )
    sim_s = time.perf_counter() - t0
    return U_sim, float(fit_s), float(sim_s)


METHOD_LABEL = {
    "vdc": "VDC (Ours)",
    "pyvine_tll": "pyvine-TLL",
    "data_ref": "Data reference (train bootstrap)",
}

DATASET_LABEL = {
    "gas": "Gas",
    "power": "Power",
    "sp100": "S\\&P100",
}


def emit_tex(summary: Dict[str, Dict[str, Any]], datasets: List[str], out_path: Path, n_sim: int, q: float) -> None:
    rows = [
        f"% E34 sampling quality: n_sim={n_sim}, tail q={q}; mean $\\pm$ sd over seeds.",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Dataset & Method & $\\tau$-RMSE $\\downarrow$ & $\\lambda_L$ MAE $\\downarrow$ & $\\lambda_U$ MAE $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for di, ds in enumerate(datasets):
        ds_summary = summary.get(ds, {})
        for method in ["vdc", "pyvine_tll", "data_ref"]:
            s = ds_summary.get(method)
            label = METHOD_LABEL[method]
            ds_label = DATASET_LABEL.get(ds, ds) if method == "vdc" else ""
            if s is None or s.get("n_seeds", 0) == 0:
                status = (s or {}).get("status", "no result")
                rows.append(f"{ds_label} & {label} & \\multicolumn{{3}}{{c}}{{\\textit{{{status}}}}} \\\\")
                continue

            def _fmt(key: str) -> str:
                m = s[key]["mean"]
                sd = s[key]["sd"]
                if s["n_seeds"] > 1 and sd == sd:  # sd is not NaN
                    return f"{m:.4f} $\\pm$ {sd:.4f}"
                return f"{m:.4f}"

            rows.append(
                f"{ds_label} & {label} & {_fmt('tau_rmse')} & {_fmt('lambdaL_mae')} & {_fmt('lambdaU_mae')} \\\\"
            )
        if di < len(datasets) - 1:
            rows.append("\\midrule")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    out_path.write_text("\n".join(rows) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="E34 unconditional sampling quality benchmark.")
    p.add_argument("--datasets", type=str, nargs="+", default=["ccpp", "sp100"],
                   choices=["gas", "ccpp", "sp100"])
    p.add_argument("--sp100-data", type=Path,
                   default=None)
    p.add_argument("--train-frac", type=float, default=0.8, help="Train fraction for the S&P100 split (as e23).")
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 42])
    p.add_argument("--n-sim", type=int, default=5000, help="Samples drawn from each fitted model.")
    p.add_argument("--max-pairs", type=int, default=200, help="Max variable pairs scored (fixed subsample if d large).")
    p.add_argument("--pair-seed", type=int, default=12345, help="Fixed seed for pair subsampling (shared across seeds/methods).")
    p.add_argument("--tail-q", type=float, default=0.05, help="Tail-concentration level q.")
    p.add_argument("--max-train", type=int, default=20000, help="Cap on UCI training rows.")
    p.add_argument("--max-test", type=int, default=20000, help="Cap on UCI test rows.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Released bivariate checkpoint.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--m-grid", type=int, default=64)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--pyvine-timeout-s", type=int, default=3600, help="Per-fit time limit for pyvine-TLL.")
    p.add_argument("--methods", type=str, nargs="+", default=["vdc", "pyvine_tll", "data_ref"],
                   choices=["vdc", "pyvine_tll", "data_ref"])
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-tex", type=Path, default=None)
    p.add_argument("--vectorized-inverse", action="store_true", help="Use the qualified faster inverse-h sampler; fitting is unchanged.")
    args = p.parse_args()

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    os.environ["DATA_ROOT"] = str(args.data_root.resolve())
    if "sp100" in args.datasets and args.sp100_data is None:
        p.error("--sp100-data is required for sp100")
    import hashlib
    ckpt = args.checkpoint
    expected = "8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f"
    with ckpt.open("rb") as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    if h.hexdigest() != expected:
        p.error("--checkpoint must be the released paper checkpoint")
    if args.out_json.exists():
        p.error("--out-json must be a new file")
    print(f"[e34] checkpoint: {ckpt}", flush=True)

    model = None
    if "vdc" in args.methods:
        model = _load_vdc_checkpoint(ckpt, str(args.device))

    q = float(args.tail_q)
    records: List[Dict[str, Any]] = []

    for name in [str(x).lower() for x in args.datasets]:
        for seed in [int(s) for s in args.seeds]:
            print(f"\n=== dataset={name} seed={seed} ===", flush=True)
            U_tr, U_te = load_dataset(
                name, seed,
                sp100_path=args.sp100_data,
                train_frac=float(args.train_frac),
                max_train=int(args.max_train),
                max_test=int(args.max_test),
            )
            d = int(U_tr.shape[1])
            pairs = select_pairs(d, int(args.max_pairs), int(args.pair_seed))
            print(f"[e34] n_tr={U_tr.shape[0]}, n_te={U_te.shape[0]}, d={d}, pairs={len(pairs)}", flush=True)

            tau_te = tau_vector(U_te, pairs)
            lamL_te, lamU_te = tail_lambda_vectors(U_te, pairs, q)

            samples: Dict[str, Tuple[Optional[np.ndarray], float, float, str]] = {}

            if "vdc" in args.methods:
                try:
                    U_sim, fit_s, sim_s = sample_vdc(
                        U_tr, model=model, m_grid=int(args.m_grid),
                        device=str(args.device), n_sim=int(args.n_sim), seed=seed,
                        vectorized_inverse=bool(args.vectorized_inverse),
                    )
                    samples["vdc"] = (U_sim, fit_s, sim_s, "ok")
                except Exception as e:
                    samples["vdc"] = (None, float("nan"), float("nan"), f"error:{type(e).__name__}:{e}")

            if "pyvine_tll" in args.methods:
                try:
                    U_sim, fit_s, sim_s = sample_pyvine(
                        U_tr, n_sim=int(args.n_sim), seed=seed,
                        num_threads=int(args.num_threads),
                        timeout_s=int(args.pyvine_timeout_s),
                    )
                    samples["pyvine_tll"] = (U_sim, fit_s, sim_s, "ok")
                except TimeoutError:
                    samples["pyvine_tll"] = (None, float(args.pyvine_timeout_s), float("nan"),
                                             f"timeout({args.pyvine_timeout_s}s)")
                except Exception as e:
                    samples["pyvine_tll"] = (None, float("nan"), float("nan"), f"error:{type(e).__name__}:{e}")

            if "data_ref" in args.methods:
                rng_boot = np.random.default_rng(seed + 777)
                idx = rng_boot.choice(U_tr.shape[0], size=int(args.n_sim), replace=True)
                samples["data_ref"] = (U_tr[idx], 0.0, 0.0, "ok")

            for method, (U_sim, fit_s, sim_s, status) in samples.items():
                rec: Dict[str, Any] = {
                    "dataset": name,
                    "seed": seed,
                    "method": method,
                    "d": d,
                    "n_train": int(U_tr.shape[0]),
                    "n_test": int(U_te.shape[0]),
                    "n_sim": int(args.n_sim),
                    "n_pairs": len(pairs),
                    "tail_q": q,
                    "fit_s": fit_s,
                    "sim_s": sim_s,
                    "status": status,
                }
                if U_sim is not None and status == "ok":
                    rec.update(metrics_vs_test(U_sim, pairs, tau_te, lamL_te, lamU_te, q))
                    print(
                        f"[e34] {method:>10}: tau_rmse={rec['tau_rmse']:.4f} "
                        f"lamL_mae={rec['lambdaL_mae']:.4f} lamU_mae={rec['lambdaU_mae']:.4f} "
                        f"(fit={fit_s:.1f}s, sim={sim_s:.1f}s)",
                        flush=True,
                    )
                else:
                    print(f"[e34] {method:>10}: {status}", flush=True)
                records.append(rec)

    # Aggregate: mean +/- sd over seeds per dataset x method.
    summary: Dict[str, Dict[str, Any]] = {}
    for name in [str(x).lower() for x in args.datasets]:
        summary[name] = {}
        for method in args.methods:
            ok = [r for r in records if r["dataset"] == name and r["method"] == method and r["status"] == "ok"]
            if not ok:
                bad = [r for r in records if r["dataset"] == name and r["method"] == method]
                summary[name][method] = {"n_seeds": 0, "status": bad[0]["status"] if bad else "missing"}
                continue
            entry: Dict[str, Any] = {"n_seeds": len(ok), "status": "ok"}
            for key in ["tau_rmse", "lambdaL_mae", "lambdaU_mae"]:
                vals = np.array([r[key] for r in ok], dtype=np.float64)
                entry[key] = {
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                }
            entry["fit_s_mean"] = float(np.mean([r["fit_s"] for r in ok]))
            entry["sim_s_mean"] = float(np.mean([r["sim_s"] for r in ok]))
            summary[name][method] = entry

    out_json = args.out_json
    out_tex = args.out_tex
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_tex is not None:
        out_tex.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "checkpoint": str(ckpt),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "records": records,
        "summary": summary,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    if out_tex is not None:
        emit_tex(summary, [str(x).lower() for x in args.datasets], out_tex, int(args.n_sim), q)
    print(f"\nWrote {out_json}\nWrote {out_tex}", flush=True)


if __name__ == "__main__":
    main()
