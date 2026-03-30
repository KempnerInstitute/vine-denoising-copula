#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Verify that the official pretrained checkpoint is usable and paper-aligned."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.data.generators import analytic_logpdf_grid, sample_bicop  # noqa: E402
from vdc.models.projection import copula_project  # noqa: E402
from vdc.pretrained import (  # noqa: E402
    DEFAULT_PRETRAINED_MODEL_ID,
    estimate_pair_density_from_samples,
    load_pretrained_model,
)


DEFAULT_CASES: List[Tuple[str, Dict[str, float], int, str]] = [
    ("independence", {}, 0, "Independence"),
    ("gaussian", {"rho": 0.7}, 0, "Gaussian(ρ=0.7)"),
    ("gaussian", {"rho": -0.7}, 0, "Gaussian(ρ=-0.7)"),
    ("clayton", {"theta": 3.0}, 0, "Clayton(θ=3.0)"),
    ("frank", {"theta": 5.0}, 0, "Frank(θ=5.0)"),
    ("gumbel", {"theta": 2.5}, 0, "Gumbel(θ=2.5)"),
]


def _normalize_density_np(d: np.ndarray) -> np.ndarray:
    m = int(d.shape[0])
    du = 1.0 / float(m)
    arr = np.asarray(d, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mass = float((arr * du * du).sum())
    if (not np.isfinite(mass)) or mass <= 0:
        return np.ones_like(arr, dtype=np.float64)
    return arr / mass


def _project_density_np(d: np.ndarray, *, iters: int, device: torch.device) -> np.ndarray:
    t = torch.from_numpy(np.asarray(d, dtype=np.float64)).float().unsqueeze(0).unsqueeze(0).to(device)
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-12)
    t = copula_project(t, iters=int(iters))
    return t[0, 0].detach().cpu().numpy()


def _mi_from_density_grid(d: np.ndarray) -> float:
    m = int(d.shape[0])
    du = 1.0 / float(m)
    arr = np.clip(np.asarray(d, dtype=np.float64), 1e-12, 1e12)
    return float(np.sum(arr * np.log(arr)) * du * du)


def _ise(d_pred: np.ndarray, d_true: np.ndarray, m: int) -> float:
    du = 1.0 / float(m)
    return float(np.mean((d_pred - d_true) ** 2) * du * du)


def _run_qualitative_figure(checkpoint: Path, out_dir: Path, device: str) -> Dict[str, Any]:
    out_pdf = out_dir / "fig_copula_example_main_verify.pdf"
    out_png = out_dir / "fig_copula_example_main_verify.png"
    out_json = out_dir / "fig_copula_example_main_verify_metrics.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "drafts" / "scripts" / "generate_fig_copula_example_main.py"),
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
        "--out-pdf",
        str(out_pdf),
        "--out-png",
        str(out_png),
        "--out-json",
        str(out_json),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return json.loads(out_json.read_text())


def _write_summary_markdown(summary: Dict[str, Any], out_path: Path) -> None:
    lines = []
    lines.append("# Pretrained model verification")
    lines.append("")
    lines.append(f"- Model id: `{summary['model_id']}`")
    lines.append(f"- Checkpoint: `{summary['checkpoint_path']}`")
    lines.append(f"- Device: `{summary['device']}`")
    lines.append(f"- Mean ISE: `{summary['aggregate']['mean_ise']:.6e}`")
    lines.append(f"- Mean MI absolute error: `{summary['aggregate']['mean_mi_abs_err']:.6f}`")
    lines.append(f"- Mean mass error: `{summary['aggregate']['mean_mass_err']:.6e}`")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| Case | ISE | MI true | MI est | abs error | mass |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary["cases"]:
        lines.append(
            f"| {row['label']} | {row['ise']:.6e} | {row['mi_true']:.4f} | {row['mi_est']:.4f} | {row['mi_abs_err']:.4f} | {row['mass']:.6f} |"
        )
    lines.append("")
    if "qualitative_figure" in summary:
        fig = summary["qualitative_figure"]
        lines.append("## Qualitative figure")
        lines.append("")
        lines.append(f"- PNG: `{fig['png']}`")
        lines.append(f"- PDF: `{fig['pdf']}`")
        lines.append(f"- DCD ISE: `{fig['metrics'].get('ise_dcd_one_shot', float('nan'))}`")
        lines.append(f"- pyvine ISE: `{fig['metrics'].get('ise_pyvine_selected', float('nan'))}`")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the official pretrained VDC checkpoint")
    parser.add_argument("--model-id", default=DEFAULT_PRETRAINED_MODEL_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--m-true", type=int, default=128)
    parser.add_argument("--projection-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "reports" / "pretrained_release")
    parser.add_argument("--skip-qualitative-figure", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_pretrained_model(args.model_id, device=args.device)
    device_t = torch.device(args.device)

    records: List[Dict[str, Any]] = []
    fig, axes = plt.subplots(2, len(DEFAULT_CASES), figsize=(2.5 * len(DEFAULT_CASES), 4.4))
    for idx, (family, params, rotation, label) in enumerate(DEFAULT_CASES):
        samples = sample_bicop(family, params, n=args.n_samples, rotation=rotation, seed=args.seed + idx)
        d_pred = estimate_pair_density_from_samples(
            bundle,
            samples,
            projection_iters=args.projection_iters,
        )
        log_true = analytic_logpdf_grid(family, params, m=args.m_true, rotation=rotation)
        d_true = np.exp(np.clip(log_true, -20, 20))
        d_true = _normalize_density_np(d_true)
        d_true = _project_density_np(d_true, iters=50, device=device_t)
        d_true = _normalize_density_np(d_true)

        if d_pred.shape != d_true.shape:
            # Resize via nearest-neighbor by evaluating truth at bundle resolution.
            log_true_eval = analytic_logpdf_grid(family, params, m=d_pred.shape[0], rotation=rotation)
            d_true_eval = np.exp(np.clip(log_true_eval, -20, 20))
            d_true_eval = _normalize_density_np(d_true_eval)
            d_true_eval = _project_density_np(d_true_eval, iters=50, device=device_t)
            d_true_eval = _normalize_density_np(d_true_eval)
        else:
            d_true_eval = d_true

        mi_true = _mi_from_density_grid(d_true_eval)
        mi_est = _mi_from_density_grid(d_pred)
        mass = float(d_pred.sum() * (1.0 / d_pred.shape[0]) ** 2)
        rec = {
            "family": family,
            "params": params,
            "rotation": rotation,
            "label": label,
            "ise": _ise(d_pred, d_true_eval, d_pred.shape[0]),
            "mi_true": mi_true,
            "mi_est": mi_est,
            "mi_abs_err": abs(mi_est - mi_true),
            "mass": mass,
            "mass_err": abs(mass - 1.0),
            "density_min": float(d_pred.min()),
            "density_max": float(d_pred.max()),
        }
        records.append(rec)

        axes[0, idx].imshow(d_true_eval, origin="lower", extent=[0, 1, 0, 1], cmap="magma")
        axes[0, idx].set_title(label, fontsize=8)
        axes[0, idx].set_xticks([])
        axes[0, idx].set_yticks([])
        axes[1, idx].imshow(d_pred, origin="lower", extent=[0, 1, 0, 1], cmap="magma")
        axes[1, idx].set_xticks([])
        axes[1, idx].set_yticks([])
        axes[1, idx].set_xlabel(f"ISE={rec['ise']:.1e}\n|MI err|={rec['mi_abs_err']:.3f}", fontsize=7)

    axes[0, 0].set_ylabel("True", fontsize=9)
    axes[1, 0].set_ylabel("VDC", fontsize=9)
    fig.tight_layout()
    pair_grid_png = out_dir / "pretrained_bivariate_verification.png"
    pair_grid_pdf = out_dir / "pretrained_bivariate_verification.pdf"
    fig.savefig(pair_grid_png, dpi=200)
    fig.savefig(pair_grid_pdf)
    plt.close(fig)

    summary: Dict[str, Any] = {
        "model_id": bundle.model_id,
        "checkpoint_path": str(bundle.checkpoint_path),
        "device": str(bundle.device),
        "cases": records,
        "aggregate": {
            "mean_ise": float(np.mean([r["ise"] for r in records])),
            "mean_mi_abs_err": float(np.mean([r["mi_abs_err"] for r in records])),
            "mean_mass_err": float(np.mean([r["mass_err"] for r in records])),
        },
        "artifacts": {
            "bivariate_grid_png": str(pair_grid_png),
            "bivariate_grid_pdf": str(pair_grid_pdf),
        },
    }

    if not args.skip_qualitative_figure:
        qual = _run_qualitative_figure(bundle.checkpoint_path, out_dir, args.device)
        summary["qualitative_figure"] = {
            "pdf": str(out_dir / "fig_copula_example_main_verify.pdf"),
            "png": str(out_dir / "fig_copula_example_main_verify.png"),
            "metrics": qual,
        }

    json_path = out_dir / "pretrained_release_verification.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_summary_markdown(summary, out_dir / "PRETRAINED_RELEASE_VERIFICATION.md")
    print(json_path)


if __name__ == "__main__":
    main()
