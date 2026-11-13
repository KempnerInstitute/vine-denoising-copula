#!/usr/bin/env python3
"""Neural-copula comparison with ACNet (tab_neural_copula).

Inputs: bivariate/e8_neural_copula_results_seed{0,1,2}.json, three ACNet training seeds on the same
synthetic Archimedean cases (n = 2000). Each cell is the mean over the seeds; the speedup is ACNet's
mean fit time over VDC's. The Joe case is left out because its samples came from a defective sampler,
and the Mean rows average the remaining cases before rounding. The saved ISE values carry an extra
1/m^2 factor (m = 64), so they are multiplied by m^2 = 4096. Bold marks the lower value of each pair.
"""
from __future__ import annotations

import math

import numpy as np

from common import load, parse, write

ISE_SCALE = 64 ** 2
SEEDS = (0, 1, 2)
ORDER = ["Clayton(θ=3.0)", "Frank(θ=5.0)", "Gumbel(θ=2.5)"]
METRICS = [("ise", "{:.3e}", ISE_SCALE), ("hfunc_mae", "{:.3f}", 1.0), ("mi_err", "{:.3f}", 1.0)]


def speedup(x: float) -> str:
    return f"{x / 1e3:.1f}k$\\times$" if x >= 1e3 else f"{x:.1f}$\\times$"


def cells(vdc: dict, acnet: dict) -> tuple[list, list]:
    out_v, out_a = [], []
    for key, fmt, scale in METRICS:
        v, a = vdc[key] * scale, acnet[key] * scale
        sv, sa = fmt.format(v), fmt.format(a)
        out_v.append(rf"\textbf{{{sv}}}" if v <= a else sv)
        out_a.append(rf"\textbf{{{sa}}}" if a < v else sa)
    return out_v, out_a


def main() -> None:
    a = parse(__doc__)
    runs = [load(a.inputs, f"bivariate/e8_neural_copula_results_seed{s}.json") for s in SEEDS]
    assert [r["settings"]["seed"] for r in runs] == list(SEEDS)
    rows = {}
    for name in ORDER:
        recs = [next(x for x in r["records"] if x["name"] == name) for r in runs]
        row = {m: {k: float(np.mean([rec[m][k] for rec in recs])) for k in ("ise", "hfunc_mae", "mi_err", "fit_s")}
               for m in ("vdc", "acnet")}
        row["speedup"] = row["acnet"]["fit_s"] / row["vdc"]["fit_s"]
        rows[name] = row
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Case & Method & ISE $\downarrow$ & $h$-MAE $\downarrow$ & MI err $\downarrow$ & Speedup \\", r"\midrule"]
    for i, name in enumerate(ORDER):
        v, ac = cells(rows[name]["vdc"], rows[name]["acnet"])
        label = name.replace("θ", r"$\theta$")
        lines.append(" & ".join([label, "VDC"] + v + [speedup(rows[name]["speedup"])]) + r" \\")
        lines.append(" & ".join(["", "ACNet"] + ac + [""]) + r" \\")
        if i < len(ORDER) - 1:
            lines.append(r"\midrule")
    mean = {m: {k: float(np.mean([rows[n][m][k] for n in ORDER])) for k in ("ise", "hfunc_mae", "mi_err")}
            for m in ("vdc", "acnet")}
    mean_speedup = float(np.mean([rows[n]["speedup"] for n in ORDER]))
    assert all(math.isfinite(x) for x in [mean_speedup] + [mean[m][k] for m in mean for k in mean[m]])
    v, ac = cells(mean["vdc"], mean["acnet"])
    lines += [r"\midrule",
              " & ".join([r"\textbf{Mean}", "VDC"] + v + [rf"\textbf{{{speedup(mean_speedup)}}}"]) + r" \\",
              " & ".join(["", "ACNet"] + ac + [""]) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    write(a.out, "tab_neural_copula.tex", "\n".join(lines) + "\n", "neural_copula.py")


if __name__ == "__main__":
    main()
