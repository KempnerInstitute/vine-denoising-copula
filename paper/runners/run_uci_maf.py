#!/usr/bin/env python3
"""UCI copula-space density benchmark on the MAF splits, with paired bootstrap CIs.

Portable version of the runner behind the paper's UCI tables (Appendix tables of held-out NLL at
5,000 and 50,000 training rows). It fits VDC with the released checkpoint, pyvine parametric and
TLL vines, a RealNVP flow and a Gaussian copula on the same rows, saves the per-row test
log-densities, and computes per-method bootstrap CIs for held-out NLL (bits/dim) and paired
bootstrap CIs for VDC minus each baseline on the matched test rows.

The numerical procedure is unchanged from the paper's runner; only input and output handling
differ: explicit paths, a checkpoint hash check, a content check of the MAF arrays, and a
provenance record. Data are the preprocessed MAF splits written by
`scripts/download_uci_datasets.py --out <data-root>` to <data-root>/uci/<name>.npz (keys
train/validate/test; validation rows are not used); they are loaded with the library's
`maybe_load_uci`, as in the paper's runs, and their train and test arrays are checked against the
paper's before use.

Writes into a new --output directory:
  results.json                 settings, per-dataset NLL with CIs, paired differences, provenance
  per_row_logpdf.npz           test log-densities per dataset and method
  tab_vine_nll_ci_maf.tex      LaTeX table in the layout of the paper's raw table
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kstest, norm
from torch.utils.data import DataLoader, TensorDataset


LOG_2PI = float(math.log(2.0 * math.pi))

CHECKPOINT_SHA256 = "8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f"  # vdc-denoiser-m64-v1
# SHA-256 of the float64 little-endian bytes of the train and test arrays used in the paper
# (preprocessed MAF splits of Zenodo record 1161203).
MAF_ARRAY_SHA256 = {
    "power": {"train": "a3c6eee0ffee8de3059925f1778afc1e09a384c14503b1e5e1dc5f186c46dcad",
              "test": "68dab933584464a90156b733f20d907e83462fb06c100a1c2ee8b70068ec7d69"},
    "gas": {"train": "70136e10e909ed59bfdead8e8251ecf9925ec5215abd1e8852c1f2ebd3ddab05",
            "test": "c52361fc20ff1dc236cae616dc689f8f1789843bd461d4d89e22d5212459f211"},
    "hepmass": {"train": "7eb15894622c90eb1e1b4433521feb3d81ffae834e7e07030b6918f0601a9a96",
                "test": "613a470cdc390f3ca3bd4aa22c3d1b9eb3cf755e2fd6a8010b6370491134d423"},
    "miniboone": {"train": "aafbb8e1cb55685729ce36329d159f6b8f5bf869c4b61a1651cae41a0fba0f19",
                  "test": "ff61c966cd4f99049eecf6e2ecb880ba52afe3605f6b925f406efcf418ef30b7"},
}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).astype("<f8", copy=False).tobytes()).hexdigest()


def _clip_unit(U: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(U, dtype=np.float64), eps, 1.0 - eps)


def _pit_ks_stat(W: np.ndarray) -> float:
    W = np.asarray(W, dtype=np.float64)
    stats = []
    for j in range(W.shape[1]):
        stat, _p = kstest(W[:, j], "uniform")
        stats.append(float(stat))
    return float(np.mean(stats)) if stats else float("nan")


def _nll_bits_from_logpdf_rows(logpdf_rows: np.ndarray, d: int) -> float:
    return float((-float(np.mean(logpdf_rows))) / (max(1, int(d)) * np.log(2.0)))


def _nearest_corr(corr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    corr = np.asarray(corr, dtype=np.float64)
    corr = 0.5 * (corr + corr.T)
    w, v = np.linalg.eigh(corr)
    w = np.clip(w, eps, None)
    corr_psd = (v * w[None, :]) @ v.T
    diag = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.maximum(1e-12, diag[:, None] * diag[None, :])
    np.fill_diagonal(corr_psd, 1.0)
    return corr_psd


def _gaussian_copula_logpdf(U: np.ndarray, corr: np.ndarray) -> np.ndarray:
    Z = norm.ppf(_clip_unit(U))
    inv_corr = np.linalg.inv(corr)
    sign, logdet = np.linalg.slogdet(corr)
    if sign <= 0:
        raise RuntimeError("Correlation matrix is not SPD.")
    quad = np.einsum("ni,ij,nj->n", Z, inv_corr - np.eye(Z.shape[1]), Z, optimize=True)
    return -0.5 * logdet - 0.5 * quad


# -------- minimal RealNVP, copied from e10 with per-row eval --------

class _MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = dim
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2 * dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CouplingLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_layers: int, mask: torch.Tensor, scale_clip: float = 3.0) -> None:
        super().__init__()
        self.net = _MLP(dim, hidden_dim, hidden_layers)
        self.register_buffer("mask", mask.view(1, dim))
        self.scale_clip = float(scale_clip)

    def _st(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x * self.mask)
        s, t = torch.chunk(h, 2, dim=-1)
        s = torch.tanh(s) * self.scale_clip
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        return s, t

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s, t = self._st(x)
        y = x * self.mask + (1.0 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(dim=-1)


class RealNVP(nn.Module):
    def __init__(self, dim: int, num_layers: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        masks = []
        for i in range(int(num_layers)):
            mask = torch.zeros(dim)
            mask[i % 2 :: 2] = 1.0
            masks.append(mask)
        self.layers = nn.ModuleList(
            [_CouplingLayer(dim, hidden_dim, hidden_layers, m) for m in masks]
        )
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_abs_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer(z)
            log_abs_det = log_abs_det + ld
        return z, log_abs_det

    def log_prob_z(self, x: torch.Tensor) -> torch.Tensor:
        z, ladj = self.forward(x)
        prior = -0.5 * (z.pow(2).sum(dim=-1) + self.dim * LOG_2PI)
        return prior + ladj

    def log_prob_u(self, u: torch.Tensor) -> torch.Tensor:
        u = u.clamp(1e-6, 1.0 - 1e-6)
        z = torch.distributions.Normal(0.0, 1.0).icdf(u)
        log_jac = (0.5 * z.pow(2) + 0.5 * LOG_2PI).sum(dim=-1)
        return self.log_prob_z(z) + log_jac


def _fit_flow(
    Z_tr: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    num_layers: int,
    hidden_dim: int,
    hidden_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    patience: int,
) -> Tuple[RealNVP, float]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    n = int(Z_tr.shape[0])
    n_val = max(1, int(round(float(val_fraction) * n)))
    idx = np.random.permutation(n)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    Z_tr_t = torch.tensor(Z_tr[tr_idx], dtype=torch.float32)
    Z_val_t = torch.tensor(Z_tr[val_idx], dtype=torch.float32)
    model = RealNVP(int(Z_tr.shape[1]), num_layers, hidden_dim, hidden_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loader = DataLoader(
        TensorDataset(Z_tr_t),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pat = 0
    t0 = perf_counter()
    for ep in range(int(epochs)):
        model.train()
        for (zb,) in loader:
            zb = zb.to(device, non_blocking=True)
            opt.zero_grad()
            lp = model.log_prob_z(zb)
            loss = -lp.mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            zv = Z_val_t.to(device, non_blocking=True)
            v = -model.log_prob_z(zv).mean().item()
        if v + 1e-6 < best_val:
            best_val = v
            best_state = {k: v_.detach().clone() for k, v_ in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= int(patience):
                break
    elapsed = float(perf_counter() - t0)
    model.load_state_dict(best_state)
    model.eval()
    return model, elapsed


def _eval_flow_logpdf_u(model: RealNVP, U: np.ndarray, *, device: torch.device, batch_size: int) -> np.ndarray:
    u_t = torch.tensor(_clip_unit(U), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(u_t),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    vals: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (ub,) in loader:
            ub = ub.to(device, non_blocking=True)
            lp = model.log_prob_u(ub).detach().cpu().numpy()
            vals.append(lp)
    return np.concatenate(vals, axis=0)


# -------- bootstrap helpers --------

def _bootstrap_ci_mean(
    x: np.ndarray, n_boot: int, alpha: float, rng: np.random.Generator
) -> Tuple[float, float, float]:
    n = int(x.shape[0])
    means = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        means[b] = float(np.mean(x[idx]))
    mean_pt = float(np.mean(x))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return mean_pt, lo, hi


def _bootstrap_paired_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int, alpha: float, rng: np.random.Generator
) -> Tuple[float, float, float, float]:
    """Bootstrap CI for E[a - b] under matched-row resampling. Returns (mean_diff, lo, hi, p_two_sided)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    n = int(a.shape[0])
    diffs = a - b
    boot_means = np.empty(int(n_boot), dtype=np.float64)
    for k in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boot_means[k] = float(np.mean(diffs[idx]))
    mean_pt = float(np.mean(diffs))
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    # two-sided "p-value" approximated as 2 * min(p_left, p_right) under the bootstrap distribution
    p_left = float(np.mean(boot_means <= 0.0))
    p_right = float(np.mean(boot_means >= 0.0))
    p_two = float(min(1.0, 2.0 * min(p_left, p_right)))
    return mean_pt, lo, hi, p_two


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True,
                   help="output directory of scripts/download_uci_datasets.py (default there: <data root>/uci_maf), "
                        "holding uci/<name>.npz")
    p.add_argument("--checkpoint", type=Path, required=True, help="vdc-denoiser-m64-v1 checkpoint file")
    p.add_argument("--expected-checkpoint-sha256", default=CHECKPOINT_SHA256,
                   help="refuse other checkpoints; pass 'any' to run a different model")
    p.add_argument("--skip-data-check", action="store_true",
                   help="do not require the paper's MAF arrays (results are then not the paper's benchmark)")
    p.add_argument("--output", type=Path, required=True, help="new output directory")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["power", "gas", "hepmass", "miniboone"],
    )
    p.add_argument("--max-train", type=int, required=True, help="training rows per dataset (paper: 5000 and 50000)")
    p.add_argument("--max-test", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--num-threads", type=int, default=16)
    # flow hyperparams (match e10 paper budget)
    p.add_argument("--flow-num-layers", type=int, default=8)
    p.add_argument("--flow-hidden-dim", type=int, default=128)
    p.add_argument("--flow-hidden-layers", type=int, default=2)
    p.add_argument("--flow-epochs", type=int, default=25)
    p.add_argument("--flow-batch-size", type=int, default=2048)
    p.add_argument("--flow-eval-batch-size", type=int, default=4096)
    p.add_argument("--flow-lr", type=float, default=1e-3)
    p.add_argument("--flow-val-fraction", type=float, default=0.1)
    p.add_argument("--flow-patience", type=int, default=5)
    args = p.parse_args()

    import vdc
    from vdc.data.tabular import EmpiricalMarginals, maybe_load_uci
    from vdc.vine.api import VineCopulaModel
    from vdc.train.unified_trainer import build_model

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    ckpt_sha = _file_sha256(ckpt)
    if args.expected_checkpoint_sha256 != "any" and ckpt_sha != args.expected_checkpoint_sha256:
        raise SystemExit(f"Checkpoint sha256 {ckpt_sha} differs from the expected {args.expected_checkpoint_sha256}")
    datasets = [str(x).lower() for x in args.datasets]
    data_files = {name: Path(args.data_root) / "uci" / f"{name}.npz" for name in datasets}
    data_hashes: Dict[str, Dict[str, str]] = {}
    for name, path in data_files.items():
        if not path.exists():
            raise SystemExit(f"Missing {path}; run scripts/download_uci_datasets.py first")
        X_tr, X_te = maybe_load_uci(name, Path(args.data_root))
        data_hashes[name] = {"train": _array_sha256(X_tr), "test": _array_sha256(X_te)}
        del X_tr, X_te
        if not args.skip_data_check and data_hashes[name] != MAF_ARRAY_SHA256.get(name):
            raise SystemExit(f"{path} does not hold the paper's MAF {name} arrays (pass --skip-data-check to run anyway)")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=False)

    # load model
    cobj = torch.load(str(ckpt), map_location="cpu")
    cfg = cobj.get("config", {})
    model_type_raw = str(cfg.get("model", {}).get("type", "diffusion_unet"))
    model_type = "diffusion_unet" if model_type_raw.startswith("diffusion_unet") else model_type_raw
    model = build_model(model_type, cfg, torch.device(args.device))
    load_result = model.load_state_dict(cobj.get("model_state_dict", {}), strict=False)
    if load_result.missing_keys:
        raise SystemExit(f"Checkpoint lacks {len(load_result.missing_keys)} model parameters, e.g. {load_result.missing_keys[:3]}")
    model.eval()
    m_model = int(cfg.get("data", {}).get("m", 64))
    proj_iters = int(cfg.get("training", {}).get("projection_iters", 30))

    pyvine_available = False
    try:
        from vdc.baselines.pyvinecopulib import fit_vinecop, is_available as _pyvine_avail

        pyvine_available = bool(_pyvine_avail())
    except Exception:
        pyvine_available = False

    out_json = out_dir / "results.json"
    out_npz = out_dir / "per_row_logpdf.npz"
    out_tex = out_dir / "tab_vine_nll_ci_maf.tex"

    rng = np.random.default_rng(int(args.seed))
    boot_rng = np.random.default_rng(int(args.seed) + 1)
    device = torch.device(args.device)

    records: List[Dict[str, Any]] = []
    npz_payload: Dict[str, np.ndarray] = {}

    for name in datasets:
        print(f"\n=== Dataset: {name} ===", flush=True)
        X_tr, X_te = maybe_load_uci(name, Path(args.data_root))
        X_tr = np.asarray(X_tr, dtype=np.float64)
        X_te = np.asarray(X_te, dtype=np.float64)
        if X_tr.shape[0] > int(args.max_train):
            idx = rng.choice(X_tr.shape[0], size=int(args.max_train), replace=False)
            X_tr = X_tr[idx]
        if X_te.shape[0] > int(args.max_test):
            idx = rng.choice(X_te.shape[0], size=int(args.max_test), replace=False)
            X_te = X_te[idx]
        marg = EmpiricalMarginals.fit(X_tr)
        U_tr = _clip_unit(marg.transform(X_tr))
        U_te = _clip_unit(marg.transform(X_te))
        d = int(U_tr.shape[1])
        n_test = int(U_te.shape[0])

        per_row: Dict[str, np.ndarray] = {}
        timings: Dict[str, float] = {}
        pit_ks_map: Dict[str, float] = {}

        # ---- VDC ----
        print("  fitting VDC...", flush=True)
        vine = VineCopulaModel(
            vine_type="dvine",
            m=int(m_model),
            device=str(args.device),
            projection_iters=int(proj_iters),
            hfunc_use_spline=False,
            batch_edges=True,
            edge_batch_size=256,
        )
        t0 = perf_counter()
        vine.fit(U_tr, diffusion_model=model, diffusion=None, verbose=False)
        timings["ours"] = float(perf_counter() - t0)
        lp_vdc = np.asarray(vine.logpdf(U_te), dtype=np.float64)
        per_row["ours"] = lp_vdc
        try:
            pit_ks_map["ours"] = _pit_ks_stat(np.asarray(vine.rosenblatt(U_te), dtype=np.float64))
        except Exception:
            pit_ks_map["ours"] = float("nan")
        print(f"    VDC NLL bits/dim = {_nll_bits_from_logpdf_rows(lp_vdc, d):.4f} (fit={timings['ours']:.2f}s)", flush=True)

        # ---- Gaussian copula ----
        print("  fitting Gaussian copula...", flush=True)
        t0 = perf_counter()
        Z_tr = norm.ppf(_clip_unit(U_tr))
        corr = _nearest_corr(np.corrcoef(Z_tr, rowvar=False), eps=1e-6)
        timings["gaussian_copula"] = float(perf_counter() - t0)
        per_row["gaussian_copula"] = _gaussian_copula_logpdf(U_te, corr)

        # ---- pyvine baselines ----
        if pyvine_available:
            for label, mode in (("pyvine_param", "parametric"), ("pyvine_nonpar", "nonparametric")):
                print(f"  fitting {label}...", flush=True)
                t0 = perf_counter()
                pv = fit_vinecop(
                    U_tr,
                    mode=mode,
                    num_threads=int(args.num_threads),
                    nonparametric_grid_size=int(m_model) if mode == "nonparametric" else 30,
                )
                timings[label] = float(perf_counter() - t0)
                pdf = np.asarray(pv.pdf(U_te), dtype=np.float64)
                per_row[label] = np.log(np.maximum(pdf, 1e-300))
                try:
                    pit_ks_map[label] = _pit_ks_stat(
                        np.asarray(pv.rosenblatt(U_te), dtype=np.float64)
                    )
                except Exception:
                    pit_ks_map[label] = float("nan")

        # ---- RealNVP flow ----
        print("  fitting RealNVP...", flush=True)
        Z_tr_for_flow = norm.ppf(U_tr)
        flow, fit_s_flow = _fit_flow(
            Z_tr_for_flow,
            device=device,
            seed=int(args.seed),
            num_layers=int(args.flow_num_layers),
            hidden_dim=int(args.flow_hidden_dim),
            hidden_layers=int(args.flow_hidden_layers),
            epochs=int(args.flow_epochs),
            batch_size=int(args.flow_batch_size),
            lr=float(args.flow_lr),
            val_fraction=float(args.flow_val_fraction),
            patience=int(args.flow_patience),
        )
        timings["flow_realnvp"] = float(fit_s_flow)
        per_row["flow_realnvp"] = _eval_flow_logpdf_u(
            flow, U_te, device=device, batch_size=int(args.flow_eval_batch_size)
        )

        # store per-row
        for method, arr in per_row.items():
            npz_payload[f"{name}__{method}"] = np.asarray(arr, dtype=np.float64)

        # bootstrap CIs and paired diffs (paired against pyvine_nonpar if available)
        bits_per_unit = 1.0 / (max(1, int(d)) * np.log(2.0))
        method_records: List[Dict[str, Any]] = []
        for method, arr in per_row.items():
            mean_pt, lo, hi = _bootstrap_ci_mean(
                -arr * bits_per_unit, int(args.n_boot), float(args.alpha), boot_rng
            )
            method_records.append(
                {
                    "method": method,
                    "nll_bits_per_dim": mean_pt,
                    "nll_lo": lo,
                    "nll_hi": hi,
                    "fit_s": float(timings.get(method, float("nan"))),
                    "pit_ks": pit_ks_map.get(method),
                }
            )

        # paired diffs
        baseline_methods = [m for m in ("pyvine_nonpar", "pyvine_param", "flow_realnvp", "gaussian_copula") if m in per_row]
        diff_records: List[Dict[str, Any]] = []
        for base in baseline_methods:
            ours = per_row["ours"]
            other = per_row[base]
            n = min(ours.shape[0], other.shape[0])
            mean_pt, lo, hi, p_two = _bootstrap_paired_diff(
                -ours[:n] * bits_per_unit,
                -other[:n] * bits_per_unit,
                int(args.n_boot),
                float(args.alpha),
                boot_rng,
            )
            diff_records.append(
                {
                    "vs": base,
                    "diff_nll_bits_per_dim": mean_pt,  # >0 means VDC is worse
                    "diff_lo": lo,
                    "diff_hi": hi,
                    "p_two_sided": p_two,
                }
            )
            print(
                f"  Δ NLL VDC vs {base}: {mean_pt:+.5f}  [{lo:+.5f}, {hi:+.5f}]  p≈{p_two:.3f}",
                flush=True,
            )

        records.append(
            {
                "dataset": name,
                "d": d,
                "n_train": int(U_tr.shape[0]),
                "n_test": n_test,
                "methods": method_records,
                "paired_diffs": diff_records,
            }
        )

    payload = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "model_type": str(model_type),
        "settings": {
            "datasets": list(args.datasets),
            "n_boot": int(args.n_boot),
            "alpha": float(args.alpha),
            "max_train": int(args.max_train),
            "max_test": int(args.max_test),
            "seed": int(args.seed),
            "m_model": int(m_model),
            "projection_iters": int(proj_iters),
            "flow": {
                "num_layers": int(args.flow_num_layers),
                "hidden_dim": int(args.flow_hidden_dim),
                "hidden_layers": int(args.flow_hidden_layers),
                "epochs": int(args.flow_epochs),
            },
        },
        "records": records,
        "provenance": {
            "runner_sha256": _file_sha256(Path(__file__)),
            "vdc_module": str(Path(vdc.__file__).resolve()),
            "vdc_version": str(getattr(vdc, "__version__", "unknown")),
            "data_array_sha256": data_hashes,
            "data_check": not bool(args.skip_data_check),
            "unexpected_checkpoint_keys": list(load_result.unexpected_keys),
            "device": str(args.device),
            "torch_threads": int(torch.get_num_threads()),
            "num_threads_pyvine": int(args.num_threads),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "torch": torch.__version__},
            "hostname": platform.node(),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    np.savez(str(out_npz), **npz_payload)
    print(f"\nWrote: {out_json}")
    print(f"Wrote: {out_npz}")

    # ---- emit a CI-augmented LaTeX table mirroring tab_vine_nll style ----
    label_map = {
        "ours": "VDC (Ours)",
        "pyvine_param": "pyvine-param",
        "pyvine_nonpar": "pyvine-TLL",
        "flow_realnvp": "Flow (RealNVP)",
        "gaussian_copula": "Gaussian copula",
    }
    method_order = ["ours", "pyvine_param", "pyvine_nonpar", "flow_realnvp", "gaussian_copula"]
    dim_map = {"power": 6, "gas": 8, "hepmass": 21, "miniboone": 43}  # MAF fallback only
    ds_order = {"power": 0, "gas": 1, "hepmass": 2, "credit": 3, "miniboone": 4}

    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Method & NLL (bits/dim) [95\% CI] $\downarrow$ & Fit (s) $\downarrow$ & PIT-KS $\downarrow$ \\",
        r"\midrule",
    ]
    for i_ds, rec in enumerate(sorted(records, key=lambda r: ds_order.get(str(r["dataset"]).lower(), 99))):
        ds = str(rec["dataset"]).lower()
        # Use the dimension of the data actually loaded (the old hard-coded map described the
        # placeholder arrays: power d=5, miniboone d=50; MAF POWER is d=6, MINIBOONE d=43).
        ds_label = f"{ds.capitalize()} ($d={rec.get('d', dim_map.get(ds, '--'))}$)"
        by_method = {m["method"]: m for m in rec["methods"]}
        # pick best NLL by point estimate
        candidates = [(m, by_method[m]) for m in method_order if m in by_method]
        if candidates:
            best_m = min(candidates, key=lambda kv: kv[1]["nll_bits_per_dim"])[0]
        else:
            best_m = None
        first = True
        for m in method_order:
            if m not in by_method:
                continue
            mr = by_method[m]
            label = label_map[m]
            ds_cell = ds_label if first else ""
            first = False
            nll = mr["nll_bits_per_dim"]
            lo = mr["nll_lo"]
            hi = mr["nll_hi"]
            fit_s = mr.get("fit_s")
            pit = mr.get("pit_ks")
            cell = f"{nll:.4f} [{lo:.4f}, {hi:.4f}]"
            if best_m == m:
                cell = rf"\textbf{{{cell}}}"
            fit_cell = "--" if (fit_s is None or not np.isfinite(fit_s)) else f"{fit_s:.2f}"
            pit_cell = "--" if (pit is None or not np.isfinite(pit)) else f"{pit:.3f}"
            lines.append(f"{ds_cell} & {label} & {cell} & {fit_cell} & {pit_cell} \\\\")
        if i_ds < len(records) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {out_tex}")


if __name__ == "__main__":
    main()
