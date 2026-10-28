#!/usr/bin/env python3
"""Sampling and imputation tables: tab_sampling_quality, tab_imputation_exact and tab_imputation.

Inputs (sampling/):
  sampling.json                  sampling-quality replay with the corrected sampler (three seeds):
                                 Kendall-tau RMSE and tail-concentration errors of simulated versus
                                 held-out data, per dataset and method, and a train-bootstrap reference
  compiled_sampler_summary.json  timings of the compiled sampler (eight threads, table packing
                                 included) and of eight-thread TLL; the simulation-time column uses
                                 these, with the VDC time rounded upwards
  imputation_approximate.json    imputation RMSE of the approximate conditional procedure
  imputation_exact.json          imputation RMSE of exact inverse-Rosenblatt conditional sampling
The data set labelled "power" in these files is the Combined Cycle Power Plant (CCPP) data.
"""
from __future__ import annotations

import math

from common import load, parse, write


def sampling_table(s, compiled) -> str:
    assert compiled["status"] == "complete" and compiled["all_six_paper_quality_rows_reproduced_exactly"]
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Dataset & Method & $\tau$-RMSE $\downarrow$ & $\lambda_L$ MAE $\downarrow$ & $\lambda_U$ MAE $\downarrow$ & Sim.\ time (s) \\",
             r"\midrule"]
    for k, (dataset, label) in enumerate([("power", "CCPP (power plant)"), ("sp100", r"S\&P100")]):
        if k:
            lines.append(r"\midrule")
        data = compiled["datasets"][dataset]
        for i, (method, name) in enumerate([("vdc", "VDC (Ours)"), ("pyvine_tll", "pyvine-TLL"),
                                            ("data_ref", "Data reference (train bootstrap)")]):
            row = s[dataset][method]
            assert row["n_seeds"] == 3 and row["status"] == "ok"
            vals = [f"{row[x]['mean']:.4f} $\\pm$ {row[x]['sd']:.4f}" for x in ("tau_rmse", "lambdaL_mae", "lambdaU_mae")]
            if method == "data_ref":
                elapsed = "--"
            else:
                seconds = (data["native_eight_thread_seconds_including_packing"] if method == "vdc"
                           else data["timings_seconds"]["tll_8"]["mean"])
                decimals = 4 if seconds < .01 else (3 if seconds < .1 else 2)
                if method == "vdc":  # displayed VDC latency includes preparation, rounded upwards
                    seconds = math.ceil(seconds * 10 ** decimals) / 10 ** decimals
                elapsed = f"{seconds:.{decimals}f}"
            lines.append(" & ".join([label if i == 0 else "", name, *vals, elapsed]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    a = parse(__doc__)
    s = load(a.inputs, "sampling/sampling.json")["summary"]
    compiled = load(a.inputs, "sampling/compiled_sampler_summary.json")
    write(a.out, "tab_sampling_quality.tex", sampling_table(s, compiled), "sampling.py")
    approx = load(a.inputs, "sampling/imputation_approximate.json")["records"]
    exact = load(a.inputs, "sampling/imputation_exact.json")["records"]
    assert len(approx) == len(exact) == 1 and approx[0]["dataset"] == exact[0]["dataset"] == "power"
    ex, ap = exact[0], approx[0]
    assert ex["rmse_vdc_approx"] == ap["rmse_ours"]
    b, r, v = ex["rmse_best_baseline"], ex["rmse_vdc_exact"], ap["rmse_ours"]
    exact_table = (r"""\begin{tabular}{lccccc}
\toprule
Dataset & RMSE (VDC-exact) & RMSE (VDC-approx) & RMSE (Best Baseline) & $\Delta$ vs.\ approx & $\Delta$ vs.\ best baseline \\
\midrule
""" + f"CCPP (power plant) & {r:.3f} & {v:.3f} & {b:.3f} (knn) & {100 * (v - r) / v:.1f}\\% & {100 * (b - r) / b:.1f}\\%"
                   + r""" \\
\bottomrule
\end{tabular}
""")
    write(a.out, "tab_imputation_exact.tex", exact_table, "sampling.py")
    approx_table = (r"""\begin{tabular}{lccccc}
\toprule
Dataset & $d$ & RMSE (Ours) & RMSE (Best Baseline) & Baseline & Improvement \\
\midrule
""" + f"CCPP (power plant) & 5 & {v:.3f} & {b:.3f} & kNN & {100 * (b - v) / b:.1f}\\%" + r""" \\
\bottomrule
\end{tabular}
""")
    write(a.out, "tab_imputation.tex", approx_table, "sampling.py")


if __name__ == "__main__":
    main()
