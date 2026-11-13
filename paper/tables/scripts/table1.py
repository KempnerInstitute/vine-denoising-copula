#!/usr/bin/env python3
"""Table 1 (tab_bivariate) and its dispersion table (tab_bivariate_dispersion).

Input: table1/table1_v2.json, the per-fit records of the bivariate benchmark (nine cases of the
in-distribution suite, five evaluation seeds, n=2000 training samples and 20,000 independent test
samples per fit; frozen checkpoint with K=30). Each record holds, per method, the ISE of the
cell-averaged density against the exact cell averages of the true copula, the held-out KL
(NLL of the method minus NLL of the true density), the exact grid Kendall-tau error and the upper
tail error. Table 1 reports means over the 45 fits; the dispersion table reports the mean and
standard deviation over the nine per-case means.

The time column is not measured in this benchmark: it holds the per-fit times of the same five
procedures from the original timing run at n=2000 (VDC on one H100 GPU, pyvine fits on one CPU thread,
histogram and KDE in a job allowing up to 16 OpenMP/MKL threads).
"""
from __future__ import annotations

import numpy as np

from common import load, parse, write

METHODS = [("hist", "Histogram"), ("kde", "KDE"), ("param", "pyvine-param"), ("tll", "pyvine-TLL"),
           ("vdc", "VDC (one-shot)")]
ORIG_TIME_MS = {"hist": 2.2, "kde": 85.5, "param": 591.0, "tll": 16.2, "vdc": 6.0}
DIGITS = {"ise": None, "kl": 4, "tau_err": 3, "lambdaU_err": 3, "time_ms": 1}
COLS = [("ise", True), ("kl", False), ("tau_err", False), ("lambdaU_err", False), ("time_ms", False)]


def fmt(x: float, best: bool, col: str) -> str:
    d = DIGITS[col]
    s = f"{x:.2e}" if d is None else f"{x:.{d}f}"
    return f"\\textbf{{{s}}}" if best else s


def main() -> None:
    a = parse(__doc__)
    j = load(a.inputs, "table1/table1_v2.json")
    recs = j["records"]
    assert len(recs) == j["n_fits"] == 45
    mean = {k: {c: float(np.mean([r["methods"][k][c] for r in recs])) for c, _ in COLS} for k, _ in METHODS}
    cases = list(dict.fromkeys(r["case"] for r in recs))
    case_means = {k: {c: [float(np.mean([r["methods"][k][c] for r in recs if r["case"] == cn])) for cn in cases]
                      for c, _ in COLS} for k, _ in METHODS}
    for k, _ in METHODS:
        for c, _ in COLS[:4]:
            assert np.isclose(mean[k][c], j["mean"][k][c], rtol=0, atol=0), (k, c)
        mean[k]["time_ms"] = ORIG_TIME_MS[k]
    best = {c: min(mean[k][c] for k, _ in METHODS) for c, _ in COLS}
    head = ("\\begin{tabular}{lccccc}\n\\toprule\n"
            "Method & ISE $\\downarrow$ & KL $\\downarrow$ & $|\\Delta\\tau|$ $\\downarrow$ & "
            "$|\\Delta\\lambda_U|$ $\\downarrow$ & Time (ms) $\\downarrow$ \\\\\n\\midrule\n")
    body = "".join(f"{lab} & " + " & ".join(fmt(mean[k][c], np.isclose(mean[k][c], best[c]), c) for c, _ in COLS)
                   + " \\\\\n" for k, lab in METHODS)
    write(a.out, "tab_bivariate.tex", head + body + "\\bottomrule\n\\end{tabular}\n", "table1.py")
    body_d = "".join(f"{lab} & " + " & ".join(
        (f"{np.mean(case_means[k][c]):.2e} $\\pm$ {np.std(case_means[k][c]):.1e}" if sci else
         f"{np.mean(case_means[k][c]):.3f} $\\pm$ {np.std(case_means[k][c]):.3f}") for c, sci in COLS[:4]) + " \\\\\n"
        for k, lab in METHODS)
    head_d = ("\\begin{tabular}{lcccc}\n\\toprule\nMethod & ISE & KL (nats) & $|\\Delta\\tau|$ & "
              "$|\\Delta\\lambda_U|$ \\\\\n\\midrule\n")
    write(a.out, "tab_bivariate_dispersion.tex", head_d + body_d + "\\bottomrule\n\\end{tabular}\n", "table1.py")


if __name__ == "__main__":
    main()
