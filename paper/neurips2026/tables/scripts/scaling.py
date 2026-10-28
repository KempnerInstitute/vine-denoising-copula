#!/usr/bin/env python3
"""Scaling tables: tab_highdim_scaling, tab_tc_scaling and tab_fast_fit_timing.

Inputs (scaling/):
  e22_highdim_scaling_seed_merged.json   synthetic Clayton-vine benchmark at d = 100, 200, 500
                                         (seeds 7, 17, 42; mean and seed SD of held-out NLL and fit time)
  tc_benchmark_seed_merged.json          TC of a Gaussian AR(1) process by dimension (same seeds)
  fast_fit_timing_rows.csv               fixed-thread timing run: one complete fit per method,
                                         dimension and data seed (VDC original and optimized execution
                                         on one H100 with 16 CPU threads; TLL with 8 and 16 threads)
"""
from __future__ import annotations

import csv
import math
import statistics

from common import load, parse, write


def fmt_with_std(v, s, prec_v=4, prec_s=4, units=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return r"\textit{timeout}"
    if s is None or s == 0.0:
        return f"{v:+.{prec_v}f}{units}"
    return f"{v:+.{prec_v}f}{units} {{\\scriptsize $\\pm$ {s:.{prec_s}f}}}"


def fmt_t_with_std(v, s):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return r"\textit{>2400 s}"
    if s is None or s == 0.0:
        return f"{v:.1f}"
    return f"{v:.1f} {{\\scriptsize $\\pm$ {s:.1f}}}"


def highdim(m) -> str:
    order = ["VDC", "pyvine_tll", "flow_realnvp", "gaussian_copula"]
    labels = {"VDC": "VDC (Ours)", "pyvine_tll": "pyvine-TLL", "flow_realnvp": "Flow (RealNVP)",
              "gaussian_copula": "Gaussian copula"}
    by_key = {(r["method"], r["d"]): r for r in m["records"]}
    lines = [f"% seeds = {sorted(m.get('merged_seeds', []))}", r"\begin{tabular}{llcc}", r"\toprule",
             r"$d$ & Method & NLL (bits/dim) $\downarrow$ & Fit time (s) $\downarrow$ \\"]
    for d in (100, 200, 500):
        lines.append(r"\midrule")
        for k in order:
            r = by_key.get((k, d), {})
            nll = fmt_with_std(r.get("nll_bits_per_dim"), r.get("nll_bits_per_dim_std"), 4, 4)
            ft = fmt_t_with_std(r.get("fit_s"), r.get("fit_s_std"))
            if k == "pyvine_tll" and d == 500:
                if any("timeout" in psr.get("status", "") for psr in r.get("per_seed_records", [])):
                    nll, ft = r"\textit{timeout}", r"\textit{>2400 s}"
            prefix = f"{d}" if k == order[0] else ""
            lines.append(f"{prefix} & {labels[k]} & {nll} & {ft} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def tc(t) -> str:
    dims, tc_true, tc_dcd, tc_ksg = t["dimensions"], t["tc_true"], t["tc_dcd_mean"], t["tc_ksg_mean"]
    tc_dcd_std = t.get("tc_dcd_seedstd", [0.0] * len(dims))
    tc_ksg_std = t.get("tc_ksg_seedstd", [0.0] * len(dims))

    def fmt(v, s, p=3):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "--"
        if s is None or s == 0.0:
            return f"{v:.{p}f}"
        return f"{v:.{p}f} {{\\scriptsize $\\pm$ {s:.3f}}}"

    lines = [f"% seeds = {sorted(t.get('merged_seeds', []))}", r"\begin{tabular}{lcccc}", r"\toprule",
             r"$d$ & TC (True) & TC (VDC) & TC (KSG) & $|$Err$_\text{VDC}|$ \\", r"\midrule"]
    for i, d in enumerate(dims):
        if d == 2:  # the d=2 point is not part of the table
            continue
        v = tc_dcd[i]
        s = tc_dcd_std[i] if i < len(tc_dcd_std) else 0.0
        ks = tc_ksg_std[i] if i < len(tc_ksg_std) else 0.0
        err = abs(v - tc_true[i]) if v is not None and not math.isnan(v) else float("nan")
        lines.append(f"{d} & {tc_true[i]:.3f} & {fmt(v, s)} & {fmt(tc_ksg[i], ks)} & {err:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def fast_fit(path) -> str:
    methods = ["vdc_production", "vdc_fast", "tll_8", "tll_16"]
    dims, seeds = [100, 200, 500], [7, 17, 42]
    with open(path, newline="") as f:
        rows = {(int(r["dimension"]), int(r["data_seed"]), r["method"]): float(r["fit_seconds"])
                for r in csv.DictReader(f)}
    assert len(rows) == 36
    summary = []
    for d in dims:
        times = {m: [rows[(d, s, m)] for s in seeds] for m in methods}
        summary.append({m: (statistics.mean(v), statistics.stdev(v)) for m, v in times.items()})
    labels = dict(vdc_production="VDC original execution", vdc_fast="VDC optimized execution",
                  tll_8="TLL (8 threads)", tll_16="TLL (16 threads)")
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Method & $d=100$ & $d=200$ & $d=500$ \\", r"\midrule"]
    for m in methods:
        lines.append(labels[m] + " & " + " & ".join(f"{row[m][0]:.2f} $\\pm$ {row[m][1]:.2f}" for row in summary) + r" \\")
    lines += [r"\midrule"]
    for m in ["tll_8", "tll_16"]:
        lines.append(labels[m] + " / optimized VDC & "
                     + " & ".join(f"{row[m][0] / row['vdc_fast'][0]:.2f}" + r"$\times$" for row in summary) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    a = parse(__doc__)
    write(a.out, "tab_highdim_scaling.tex", highdim(load(a.inputs, "scaling/e22_highdim_scaling_seed_merged.json")), "scaling.py")
    write(a.out, "tab_tc_scaling.tex", tc(load(a.inputs, "scaling/tc_benchmark_seed_merged.json")), "scaling.py")
    write(a.out, "tab_fast_fit_timing.tex", fast_fit(a.inputs / "scaling/fast_fit_timing_rows.csv"), "scaling.py")


if __name__ == "__main__":
    main()
