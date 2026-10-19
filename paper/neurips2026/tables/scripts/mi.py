#!/usr/bin/env python3
"""Bivariate MI benchmark table (tab_mi) without the Joe case.

Inputs (mi/): results/mi_benchmark_multiseed_summary.json lists, per estimator, the saved per-case
result files (three seeds for VDC, KSG and the Gaussian baseline; one seed for the neural estimators),
all written by the library's scripts/mi_estimation.py. The Joe(theta=3) case is dropped because its
samples came from a defective sampler while it was scored against the analytic Joe MI; the table
reports the mean absolute MI error over the remaining eight cases and the mean time per case
(mean +- standard deviation over seeds where there are several).
"""
from __future__ import annotations

import json

import numpy as np

from common import parse, write

NAMES = {"dcd": "VDC (Ours)", "mine": "MINE", "ksg": "KSG", "infonce": "InfoNCE", "nwj": "NWJ",
         "gaussian": "Gaussian", "minde": "MINDE"}


def mae_without(records, drop="joe"):
    keep = [r for r in records if str(r.get("family", "")).lower() != drop]
    assert len(keep) == len(records) - 1, (len(records), len(keep))
    return float(np.mean([abs(r["mi_err"]) for r in keep])), keep


def main() -> None:
    a = parse(__doc__)
    root = a.inputs / "mi"
    src = json.loads((root / "results/mi_benchmark_multiseed_summary.json").read_text())
    methods = {}
    for method, v in src["methods"].items():
        maes, times = [], []
        for f in v["files"]:
            m, keep = mae_without(json.loads((root / f).read_text())["records"])
            maes.append(m)
            times.append(float(np.mean([r["time_s"] for r in keep])))
        methods[method] = dict(mae_mean=float(np.mean(maes)),
                               mae_std=float(np.std(maes)) if len(maes) > 1 else float("nan"),
                               time_mean_s=float(np.mean(times)),
                               time_std_s=float(np.std(times)) if len(times) > 1 else float("nan"))
    ms = sorted(methods.items(), key=lambda kv: kv[1]["mae_mean"])
    best_t = min(v["time_mean_s"] for _, v in ms)

    def pm(mu, sd, fmt):
        return f"{mu:{fmt}} $\\pm$ {sd:{fmt}}" if np.isfinite(sd) else f"{mu:{fmt}} (1 seed)"

    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Method & MAE (nats) & Time (s) & Rank \\", r"\midrule"]
    for rank, (k, v) in enumerate(ms, 1):
        mae, tt = pm(v["mae_mean"], v["mae_std"], ".4f"), pm(v["time_mean_s"], v["time_std_s"], ".2f")
        if rank == 1:
            mae, lab, rk = rf"\textbf{{{mae}}}", rf"\textbf{{{NAMES[k]}}}", r"\textbf{1}"
        else:
            lab, rk = NAMES[k], str(rank)
        if np.isclose(v["time_mean_s"], best_t):
            tt = rf"\textbf{{{tt}}}"
        lines.append(f"{lab} & {mae} & {tt} & {rk} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(a.out, "tab_mi.tex", "\n".join(lines) + "\n", "mi.py")


if __name__ == "__main__":
    main()
