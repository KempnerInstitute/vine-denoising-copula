#!/usr/bin/env python3
"""Raw-space composition on S&P100 (tab_raw_composition).

Input: sp100/e37_raw_composition.json. Chronological S&P100 split (1,356 training and 339 test days,
d=100): per-dimension Student-t marginals shared by every copula model, combined with each copula,
against models fitted directly to raw returns. Reported: held-out raw-space NLL in bits/dim with a
95% bootstrap interval over test days, and the copula-space NLL where the model has a copula part.
The best raw NLL is bold.
"""
from __future__ import annotations

from common import load, parse, write

ROWS = [("vdc_plus_t", r"\ours{} copula + $t$ marginals"),
        ("tll_plus_t", "pyvine-TLL copula + $t$ marginals"),
        ("gausscop_plus_t", "Gaussian copula + $t$ marginals"),
        ("flowcop_plus_t", "RealNVP copula + $t$ marginals"),
        ("independent_t", "Independence + $t$ marginals"),
        ("flow_realnvp_raw", "RealNVP on raw returns (monolithic)"),
        ("mv_gaussian_raw", "Multivariate Gaussian (raw)")]


def main() -> None:
    a = parse(__doc__)
    j = load(a.inputs, "sp100/e37_raw_composition.json")
    assert j["settings"]["split"] == "chronological" and j["settings"]["d"] == 100
    by = {r["method"]: r for r in j["records"]}
    assert all(by[m]["status"] == "ok" for m, _ in ROWS)
    best = min((m for m, _ in ROWS), key=lambda m: by[m]["raw_nll_bits_per_dim"])
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"Model & Raw NLL (bits/dim) [95\% CI] $\downarrow$ & Copula NLL (bits/dim) \\", r"\midrule"]
    for m, label in ROWS:
        r = by[m]
        cell = f"{r['raw_nll_bits_per_dim']:.3f} [{r['ci_self']['lo']:.3f}, {r['ci_self']['hi']:.3f}]"
        if m == best:
            cell = rf"\textbf{{{cell}}}"
        cop = r.get("copula_nll_bits_per_dim")
        lines.append(f"{label} & {cell} & {'---' if cop is None else f'{cop:.3f}'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(a.out, "tab_raw_composition.tex", "\n".join(lines) + "\n", "raw_composition.py")


if __name__ == "__main__":
    main()
