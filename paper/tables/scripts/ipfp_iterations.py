#!/usr/bin/env python3
"""IPFP iteration ablation (tab_ipfp_iter_ablation).

Input: ipfp/ipfp_iter_ablation.json, the maximum row and column marginal errors (deviation from
uniform) and the wall-clock time of the projection for K in {0, 1, 2, 5, 10, 15, 20, 30, 50, 100},
on raw outputs of the deployed checkpoint (m=64) for four representative families (Gaussian,
Clayton, Frank, Gumbel). The deployed K=30 row is starred and bold.
"""
from __future__ import annotations

import math

from common import load, parse, write

DEPLOYED = 30


def sci(x: float, digits: int) -> str:
    e = math.floor(math.log10(abs(x)))
    m = x / 10 ** e
    if round(m, digits - 1) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{digits - 1}f}\\!\\times\\!10^{{{e}}}"


def err(x: float, digits: int) -> str:
    return f"{x:.2f}" if x >= 1 else sci(x, digits)


def main() -> None:
    a = parse(__doc__)
    j = load(a.inputs, "ipfp/ipfp_iter_ablation.json")
    assert j["m"] == 64 and j["families"] == ["gaussian", "clayton", "frank", "gumbel"]
    lines = [r"\begin{tabular}{rccc}", r"\toprule",
             r"Iterations $K$ & Max row err. $\downarrow$ & Max col err. $\downarrow$ & Time (ms) \\", r"\midrule"]
    for r in j["results"]:
        row = err(r["max_row_err"], 3)
        col = err(r["max_col_err"], 2)
        t = r["elapsed_ms"]
        ms = f"{t:.1f}" if t >= 10 else f"{t:.2f}"
        if r["iters"] == DEPLOYED:
            cells = [rf"\textbf{{{r['iters']}}}$^\star$", rf"$\mathbf{{{row}}}$", rf"$\mathbf{{{col}}}$", rf"$\mathbf{{{ms}}}$"]
            lines.append(" & ".join(cells) + r" \\")
        else:
            # values printed without an exponent are padded so the columns line up in the source
            row_c, col_c = f"${row}$", f"${col}$"
            if r["max_row_err"] >= 1:
                row_c = row_c.ljust(13)
            if r["max_col_err"] >= 1:
                col_c = col_c.ljust(13)
            lines.append(f"{str(r['iters']).ljust(4)} & {row_c} & {col_c} & ${ms}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(a.out, "tab_ipfp_iter_ablation.tex", "\n".join(lines) + "\n", "ipfp_iterations.py")


if __name__ == "__main__":
    main()
