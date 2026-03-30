#!/usr/bin/env python3
"""
Fast sweep over *diffusion inference* hyperparameters for E1 bivariate evaluation.

This is meant for the exact situation you described:
DDIM outputs can look good in samples but have "patchy" high-frequency artifacts
on the density grid. We can quickly test a small grid of inference settings and
pick the one with the best likelihood / NLL (or ISE) **without retraining**.

Implementation strategy
----------------------
We reuse the canonical evaluator `scripts/model_selection.py` (so metrics match
paper tables), and run it repeatedly with different diffusion flags:
  --diffusion-steps
  --diffusion-cfg-scale
  --diffusion-ensemble
  --diffusion-ensemble-mode
  --diffusion-smooth-sigma
  --diffusion-pred-noise-clip

Outputs:
  - prints the best setting
  - optionally writes a JSON summary of all tried settings
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    # scripts/ -> repo root
    return Path(__file__).resolve().parents[1]


def _run_model_selection(cmd: List[str]) -> Dict[str, Any]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "model_selection failed:\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )
    try:
        return json.loads(p.stdout)
    except Exception as e:
        raise RuntimeError(
            "model_selection did not emit JSON on stdout as expected.\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        ) from e


def _extract_metric(res: Dict[str, Any], metric: str) -> float:
    """
    model_selection writes {"results":[{... per copula ...}], "summary":{...}}
    The summary contains mean_* fields; we prefer that.
    """
    metric_l = str(metric).strip()
    summary = res.get("summary", {})
    if isinstance(summary, dict) and metric_l in summary:
        try:
            return float(summary[metric_l])
        except Exception:
            pass
    # fallback: mean over per-copula entries
    rows = res.get("results", [])
    vals = []
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            if metric_l in r:
                try:
                    vals.append(float(r[metric_l]))
                except Exception:
                    pass
    if not vals:
        raise KeyError(f"Metric '{metric_l}' not found in model_selection output.")
    return float(sum(vals) / max(1, len(vals)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep diffusion inference hyperparameters for E1 and select best by metric.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Diffusion checkpoint (model_step_*.pt)")
    parser.add_argument("--suite", type=str, default="standard", choices=["standard", "complex"], help="Evaluation suite")
    parser.add_argument("--device", type=str, default="cuda", help="Device for evaluation (cuda/cpu)")
    parser.add_argument("--n-samples", type=int, default=1000, help="Samples per copula (lower=faster)")
    parser.add_argument("--metric", type=str, default="mean_nll", help="Selection metric in model_selection output (lower is better).")

    parser.add_argument("--steps", type=int, nargs="*", default=[50, 100, 200], help="Candidate diffusion steps")
    parser.add_argument("--cfg-scales", type=float, nargs="*", default=[2.0, 4.0, 6.0], help="Candidate CFG scales")
    parser.add_argument("--smooth-sigmas", type=float, nargs="*", default=[0.0, 0.25, 0.5], help="Candidate smoothing sigmas")
    parser.add_argument("--ensembles", type=int, nargs="*", default=[1], help="Candidate ensemble sizes")
    parser.add_argument("--ensemble-mode", type=str, default="geometric", choices=["geometric", "arithmetic", "median"])
    parser.add_argument("--pred-noise-clip", type=float, default=10.0, help="Predicted-noise clip (<=0 disables)")

    parser.add_argument("--out-json", type=Path, default=None, help="Optional path to write sweep results JSON")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    repo = _repo_root()
    model_sel = repo / "scripts" / "model_selection.py"
    if not model_sel.exists():
        raise SystemExit(f"Missing: {model_sel}")

    metric = str(args.metric).strip()

    trials: List[Dict[str, Any]] = []
    best: Optional[Tuple[float, Dict[str, Any]]] = None

    combos = list(itertools.product(args.steps, args.cfg_scales, args.smooth_sigmas, args.ensembles))
    for steps, cfg, smooth, ens in combos:
        # model_selection uses --out-json/--out-csv paths; we don't want to write into run dirs here.
        with tempfile.TemporaryDirectory() as td:
            out_json = Path(td) / "tmp.json"
            cmd = [
                sys.executable,
                str(model_sel),
                "--suite",
                str(args.suite),
                "--n-samples",
                str(int(args.n_samples)),
                "--device",
                str(args.device),
                "--checkpoints",
                str(ckpt),
                "--out-json",
                str(out_json),
                "--diffusion-steps",
                str(int(steps)),
                "--diffusion-cfg-scale",
                str(float(cfg)),
                "--diffusion-ensemble",
                str(int(ens)),
                "--diffusion-ensemble-mode",
                str(args.ensemble_mode),
                "--diffusion-smooth-sigma",
                str(float(smooth)),
                "--diffusion-pred-noise-clip",
                str(float(args.pred_noise_clip)),
            ]

            # Run and load the JSON that model_selection wrote.
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if p.returncode != 0:
                trials.append(
                    {
                        "steps": steps,
                        "cfg_scale": cfg,
                        "smooth_sigma": smooth,
                        "ensemble": ens,
                        "ensemble_mode": args.ensemble_mode,
                        "pred_noise_clip": args.pred_noise_clip,
                        "ok": False,
                        "error": p.stderr[-4000:],
                    }
                )
                continue
            try:
                res = json.loads(out_json.read_text())
            except Exception:
                trials.append(
                    {
                        "steps": steps,
                        "cfg_scale": cfg,
                        "smooth_sigma": smooth,
                        "ensemble": ens,
                        "ensemble_mode": args.ensemble_mode,
                        "pred_noise_clip": args.pred_noise_clip,
                        "ok": False,
                        "error": "Failed to parse model_selection JSON output.",
                    }
                )
                continue

            try:
                val = _extract_metric(res, metric)
            except Exception as e:
                trials.append(
                    {
                        "steps": steps,
                        "cfg_scale": cfg,
                        "smooth_sigma": smooth,
                        "ensemble": ens,
                        "ensemble_mode": args.ensemble_mode,
                        "pred_noise_clip": args.pred_noise_clip,
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue

            rec = {
                "steps": steps,
                "cfg_scale": cfg,
                "smooth_sigma": smooth,
                "ensemble": ens,
                "ensemble_mode": args.ensemble_mode,
                "pred_noise_clip": args.pred_noise_clip,
                "ok": True,
                metric: val,
            }
            trials.append(rec)
            if best is None or val < best[0]:
                best = (val, rec)

    out = {"checkpoint": str(ckpt), "suite": args.suite, "metric": metric, "trials": trials, "best": best[1] if best else None}
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, indent=2))

    if best is None:
        raise SystemExit("No successful trials. Try reducing the search grid or checking the checkpoint/device.")

    bval, brec = best
    print("Best setting:")
    print(json.dumps({"metric": metric, "value": bval, "setting": brec}, indent=2))


if __name__ == "__main__":
    main()

