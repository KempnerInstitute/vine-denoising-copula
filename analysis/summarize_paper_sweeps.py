#!/usr/bin/env python3
"""Summarize E3/E4/E5 sweep jobs listed in a manifest."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_OUTPUT_BASE = Path("/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _f3(x: float) -> str:
    if not math.isfinite(x):
        return "--"
    return f"{x:.3f}"


def _f2(x: float) -> str:
    if not math.isfinite(x):
        return "--"
    return f"{x:.2f}"


def _pct(x: float) -> str:
    if not math.isfinite(x):
        return "--"
    return f"{x:+.2f}%"


def _job_artifact(job: Dict[str, Any], output_base: Path) -> Optional[Path]:
    jid = int(job["job_id"])
    exp = str(job["experiment"])
    if exp == "e3_var":
        pats = [f"vdc_paper_e3_var_*_{jid}/results/e3_var_results.json"]
    elif exp == "e4_imputation":
        pats = [f"vdc_paper_e4_imputation_*_{jid}/results/e4_imputation_results.json"]
    elif exp == "e5_anomaly":
        pats = [f"vdc_paper_e5_anomaly_*_{jid}/results/e5_anomaly_results.json"]
    else:
        return None

    for pat in pats:
        hits = sorted(output_base.glob(pat))
        if hits:
            return hits[-1]
    return None


def _summarize_e3(payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = list(payload.get("rows", []))
    out: Dict[str, Any] = {
        "n_refits": int(payload.get("n_refits", 0) or 0),
        "mean_fit_s": _safe_float(payload.get("mean_fit_s")),
    }
    for alpha in (0.01, 0.05):
        for method in ("VDC (Ours)", "Historical", "Gaussian"):
            key = (method, alpha)
            for r in rows:
                if str(r.get("method")) == method and abs(float(r.get("alpha", -1)) - alpha) < 1e-12:
                    n = int(r.get("n", 0))
                    x = int(r.get("violations", 0))
                    vr = (100.0 * x / n) if n > 0 else float("nan")
                    out[f"{method}_alpha_{alpha}_viol_pct"] = vr
                    out[f"{method}_alpha_{alpha}_kupiec_p"] = _safe_float(r.get("kupiec_p"))
                    break
    return out


def _summarize_e4(payload: Dict[str, Any]) -> Dict[str, Any]:
    records = list(payload.get("records", []))
    deltas: List[float] = []
    by_dataset: List[Dict[str, Any]] = []
    for r in records:
        ours = _safe_float(r.get("rmse_ours"))
        baselines: List[Tuple[str, float]] = []
        for key, name in (
            ("rmse_mean", "mean"),
            ("rmse_median", "median"),
            ("rmse_knn", "knn"),
            ("rmse_iterative", "iterative"),
        ):
            v = _safe_float(r.get(key))
            if math.isfinite(v) and v > 0:
                baselines.append((name, v))
        if not baselines or not math.isfinite(ours):
            continue
        bname, best = min(baselines, key=lambda t: t[1])
        delta_pct = 100.0 * (ours - best) / best
        deltas.append(delta_pct)
        by_dataset.append(
            {
                "dataset": str(r.get("dataset")),
                "rmse_ours": ours,
                "best_baseline": bname,
                "best_baseline_rmse": best,
                "delta_pct": delta_pct,
            }
        )
    return {
        "mean_delta_pct": mean(deltas) if deltas else float("nan"),
        "std_delta_pct": pstdev(deltas) if len(deltas) > 1 else 0.0,
        "n_datasets": len(deltas),
        "datasets": by_dataset,
    }


def _summarize_e5(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    method_summary = summary.get("method_summary", {}) if isinstance(summary, dict) else {}
    ours = method_summary.get("VDC (Ours)", {}) if isinstance(method_summary, dict) else {}
    ours_auroc = _safe_float(ours.get("mean_auroc"))
    ours_ap = _safe_float(ours.get("mean_average_precision"))
    score_mode = None
    records = list(payload.get("records", []))
    for r in records:
        if str(r.get("method")) == "VDC (Ours)":
            score_mode = str(r.get("score_mode", ""))
            break

    best_baseline_auroc = float("nan")
    best_baseline_method = ""
    if isinstance(method_summary, dict):
        for method, vals in method_summary.items():
            if method == "VDC (Ours)" or not isinstance(vals, dict):
                continue
            auroc = _safe_float(vals.get("mean_auroc"))
            if math.isfinite(auroc) and (not math.isfinite(best_baseline_auroc) or auroc > best_baseline_auroc):
                best_baseline_auroc = auroc
                best_baseline_method = str(method)

    gap = ours_auroc - best_baseline_auroc if (math.isfinite(ours_auroc) and math.isfinite(best_baseline_auroc)) else float("nan")
    return {
        "score_mode": score_mode,
        "ours_mean_auroc": ours_auroc,
        "ours_mean_ap": ours_ap,
        "best_baseline_method": best_baseline_method,
        "best_baseline_auroc": best_baseline_auroc,
        "gap_vs_best_baseline_auroc": gap,
    }


def _aggregate_group(values: List[float]) -> Dict[str, Any]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize sweep jobs from a manifest.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    ap.add_argument("--out-json", type=Path, default=Path("analysis/paper_sweep_results_latest.json"))
    ap.add_argument("--out-md", type=Path, default=Path("analysis/paper_sweep_results_latest.md"))
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    jobs = list(manifest.get("jobs", []))

    completed: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []

    for job in jobs:
        artifact = _job_artifact(job, args.output_base)
        rec = dict(job)
        rec["artifact"] = str(artifact) if artifact else None
        if artifact is None:
            pending.append(rec)
            continue
        payload = json.loads(artifact.read_text())
        exp = str(job["experiment"])
        if exp == "e3_var":
            rec["summary"] = _summarize_e3(payload)
        elif exp == "e4_imputation":
            rec["summary"] = _summarize_e4(payload)
        elif exp == "e5_anomaly":
            rec["summary"] = _summarize_e5(payload)
        else:
            rec["summary"] = {}
        completed.append(rec)

    e4_by_label: Dict[str, List[float]] = defaultdict(list)
    e5_by_label_auroc: Dict[str, List[float]] = defaultdict(list)
    e5_by_label_gap: Dict[str, List[float]] = defaultdict(list)
    for rec in completed:
        exp = str(rec["experiment"])
        label = str(rec.get("label", ""))
        s = rec.get("summary", {})
        if exp == "e4_imputation":
            e4_by_label[label].append(_safe_float(s.get("mean_delta_pct")))
        elif exp == "e5_anomaly":
            e5_by_label_auroc[label].append(_safe_float(s.get("ours_mean_auroc")))
            e5_by_label_gap[label].append(_safe_float(s.get("gap_vs_best_baseline_auroc")))

    aggregated = {
        "e4_by_label": {k: _aggregate_group(v) for k, v in sorted(e4_by_label.items())},
        "e5_by_label_ours_auroc": {k: _aggregate_group(v) for k, v in sorted(e5_by_label_auroc.items())},
        "e5_by_label_gap_vs_best": {k: _aggregate_group(v) for k, v in sorted(e5_by_label_gap.items())},
    }

    out_payload = {
        "manifest": str(args.manifest),
        "output_base": str(args.output_base),
        "n_jobs": len(jobs),
        "n_completed_with_artifact": len(completed),
        "n_pending_or_missing_artifact": len(pending),
        "completed": completed,
        "pending_or_missing": pending,
        "aggregated": aggregated,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out_payload, indent=2))

    md_lines: List[str] = []
    md_lines.append("# Paper Sweep Status")
    md_lines.append("")
    md_lines.append(f"- Jobs listed: `{len(jobs)}`")
    md_lines.append(f"- Completed (artifact found): `{len(completed)}`")
    md_lines.append(f"- Pending/missing artifact: `{len(pending)}`")
    md_lines.append("")

    if aggregated["e4_by_label"]:
        md_lines.append("## E4 (Imputation) by config")
        md_lines.append("")
        md_lines.append("| Config | Mean Δ% vs best baseline | Std | N |")
        md_lines.append("|---|---:|---:|---:|")
        for label, stats in aggregated["e4_by_label"].items():
            md_lines.append(
                f"| `{label}` | {_pct(stats['mean'])} | {_f2(stats['std'])} | {int(stats['n'])} |"
            )
        md_lines.append("")

    if aggregated["e5_by_label_ours_auroc"]:
        md_lines.append("## E5 (Anomaly) by config")
        md_lines.append("")
        md_lines.append("| Config | Ours AUROC mean | AUROC std | Gap vs best baseline (mean) | N |")
        md_lines.append("|---|---:|---:|---:|---:|")
        for label in sorted(aggregated["e5_by_label_ours_auroc"].keys()):
            au = aggregated["e5_by_label_ours_auroc"][label]
            gp = aggregated["e5_by_label_gap_vs_best"][label]
            md_lines.append(
                f"| `{label}` | {_f3(au['mean'])} | {_f3(au['std'])} | {_f3(gp['mean'])} | {int(au['n'])} |"
            )
        md_lines.append("")

    if pending:
        md_lines.append("## Pending / Missing Artifacts")
        md_lines.append("")
        for rec in pending:
            md_lines.append(
                f"- `{rec['experiment']}` job `{rec['job_id']}` label `{rec.get('label','')}` seed `{rec.get('seed','')}`"
            )
        md_lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_lines) + "\n")

    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_md}")
    print(f"Completed artifacts: {len(completed)}/{len(jobs)}")


if __name__ == "__main__":
    main()
