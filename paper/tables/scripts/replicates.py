#!/usr/bin/env python3
"""Paired bivariate replicates (tab_bivariate_replicates).

Input: bivariate/e31_bivariate_replicates.json. Eight in-library cases, 100 paired replicates each
(n = 2000 training and 2000 test points per replicate): VDC and pyvine-TLL fitted to the same sample.
Cells are mean +- standard deviation over replicates, the paired VDC win rate with its 95% Wilson
interval, and a suite row (mean of the case means; win rates pooled over all 800 replicates). Bold
marks the lower mean of each pair.

The saved ISE values carry an extra 1/m^2 factor (m = 64), so they are multiplied by m^2 = 4096
before rounding; the MI error and the win rates are used as saved.
"""
from __future__ import annotations

import math

from common import load, parse, write

ISE_SCALE = 64 ** 2


def mean_sd(stats: dict, spec: str, bold: bool, scale: float = 1.0) -> str:
    m, s = stats.get("mean"), stats.get("std")
    if m is None or not math.isfinite(m):
        return "--"
    m = m * scale
    txt = f"{format(m, spec)} $\\pm$ {format(s * scale, spec)}" if (s is not None and math.isfinite(s)) \
        else format(m, spec)
    return f"\\textbf{{{txt}}}" if bold else txt


def win(paired: dict) -> str:
    p = paired.get("win_rate")
    if p is None or not math.isfinite(p):
        return "--"
    lo, hi = paired["win_rate_ci95"]
    return f"{100.0 * p:.0f}\\% [{100.0 * lo:.0f}, {100.0 * hi:.0f}]"


def pair_cells(v: dict, t: dict, paired: dict, spec: str, scale: float) -> list:
    v_bold = v["mean"] <= t["mean"]
    return [mean_sd(v, spec, v_bold, scale), mean_sd(t, spec, not v_bold, scale), win(paired)]


def main() -> None:
    a = parse(__doc__)
    j = load(a.inputs, "bivariate/e31_bivariate_replicates.json")
    assert j["status"] == "complete" and j["settings"]["m_grid"] == 64
    suite = j["suite"]
    lines = []
    # two stacked blocks (ISE, then MI error) so the table fits the text width at a legible size
    for k, title, spec, scale in [("ise", r"ISE $\downarrow$", ".1e", ISE_SCALE),
                                  ("mi_err", r"MI error $\downarrow$", ".3f", 1.0)]:
        if lines:
            lines.append(r"\par\medskip")
        lines += [r"\begin{tabular}{lccc}", r"\toprule",
                  r"\multicolumn{4}{l}{" + title + r"} \\",
                  r"Case & VDC & pyvine-TLL & VDC win\% [95\% CI] \\", r"\midrule"]
        for c in j["cases"]:
            s, p = c["stats"], c["paired"]
            cells = [c["tex"]] + pair_cells(s["vdc"][k], s["pyvine_tll"][k], p[k], spec, scale)
            lines.append(" & ".join(cells) + r" \\")
        v, t = suite["vdc"][k]["mean_of_case_means"], suite["pyvine_tll"][k]["mean_of_case_means"]
        cells = [f"Suite ({suite['num_cases']} cases, {suite['paired_pooled']['ise']['n']} paired reps)"]
        cells += pair_cells({"mean": v}, {"mean": t}, suite["paired_pooled"][k], spec, scale)
        lines += [r"\midrule", " & ".join(cells) + r" \\", r"\bottomrule", r"\end{tabular}"]
    write(a.out, "tab_bivariate_replicates.tex", "\n".join(lines) + "\n", "replicates.py")


if __name__ == "__main__":
    main()
