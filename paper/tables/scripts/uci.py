#!/usr/bin/env python3
"""UCI tables on the MAF splits: tab_vine_nll_ci, tab_vine_nll_ci_50k, tab_uci_matched_structure and
tab_ipfp_uci_sensitivity.

Inputs (uci/):
  e21_maf5k.json, e21_maf50k.json   held-out copula NLL with 95% bootstrap intervals over test rows,
                                    fit time and mean marginal PIT-KS, per dataset and method
                                    (n_train 5,000 and 50,000; all 29,556 rows for MINIBOONE)
  matched_structure_5k.json         paired NLL differences, VDC minus pyvine-TLL, with the vine
                                    structures as in the main table, both D-vine and both regular vine
  ipfp_sensitivity_5k.json          NLL, PIT-KS and maximum marginal error for IPFP K in {15, 30, 100}
"""
from __future__ import annotations

import math

from common import load, parse, write

LABELS = {"ours": "VDC (Ours)", "pyvine_param": "pyvine-param", "pyvine_nonpar": "pyvine-TLL",
          "flow_realnvp": "Flow (RealNVP)", "gaussian_copula": "Gaussian copula"}
ORDER = ["ours", "pyvine_param", "pyvine_nonpar", "flow_realnvp", "gaussian_copula"]
DS_ORDER = {"power": 0, "gas": 1, "hepmass": 2, "miniboone": 3}
MATCHED_COLS = ["vdc_dvine_minus_tll_rvine (e21 table)", "vdc_dvine_minus_tll_dvine (matched D-vine)",
                "vdc_rvine_minus_tll_rvine (both Dissmann)", "vdc_rvine_minus_vdc_dvine (VDC structure effect)"]
NAMES = {"power": "POWER", "gas": "GAS", "hepmass": "HEPMASS", "miniboone": "MINIBOONE"}


def nll_table(j) -> str:
    recs = sorted(j["records"], key=lambda r: DS_ORDER[r["dataset"].lower()])
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Dataset & Method & NLL (bits/dim) [95\% CI] $\downarrow$ & Fit (s) $\downarrow$ & PIT-KS $\downarrow$ \\",
             r"\midrule"]
    for i, rec in enumerate(recs):
        by = {m["method"]: m for m in rec["methods"]}
        best = min((m for m in ORDER if m in by), key=lambda m: by[m]["nll_bits_per_dim"])
        first = True
        for m in ORDER:
            if m not in by:
                continue
            r = by[m]
            cell = f"{r['nll_bits_per_dim']:.4f} [{r['nll_lo']:.4f}, {r['nll_hi']:.4f}]"
            if m == best:
                cell = rf"\textbf{{{cell}}}"
            fit, pit = r.get("fit_s"), r.get("pit_ks")
            fit_cell = "--" if fit is None or not math.isfinite(fit) else f"{fit:.2f}"
            pit_cell = "--" if pit is None or not math.isfinite(pit) else f"{pit:.3f}"
            ds_cell = f"{rec['dataset'].upper()} ($d={rec['d']}$)" if first else ""
            first = False
            lines.append(f"{ds_cell} & {LABELS[m]} & {cell} & {fit_cell} & {pit_cell} \\\\")
        if i < len(recs) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def matched_table(j) -> str:
    def cell(v):
        return f"{v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"

    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \multicolumn{3}{c}{\ours{} $-$ pyvine-TLL} & \ours{} regular $-$ D-vine \\",
             r"\cmidrule(lr){2-4}",
             r"Dataset & as in Table~\ref{tab:vine_nll} & both D-vine & both regular vine & \\",
             r"\midrule"]
    for r in j["records"]:
        lines.append(f"{r['dataset'].upper()} ($d={r['d']}$) & " + " & ".join(cell(r["paired"][c]) for c in MATCHED_COLS)
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def ipfp_table(j) -> str:
    recs = j["records"]
    ks = (15, 30, 100)
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Dataset & IPFP $K$ & NLL (bits/dim) $\downarrow$ & PIT-KS $\downarrow$ & Max marg.\ err. $\downarrow$ \\",
             r"\midrule"]
    for i, ds in enumerate(["power", "gas", "hepmass", "miniboone"]):
        rows = sorted((r for r in recs if r["dataset"] == ds and int(r["ipfp_iters"]) in ks), key=lambda r: r["ipfp_iters"])
        if i:
            lines.append(r"\midrule")
        for r in rows:
            err = max(r["max_row_error"], r["max_col_error"])
            lines.append(f"{NAMES[ds]} ($d{{=}}{r['d']}$) & {r['ipfp_iters']} & {r['nll_bits_per_dim']:.4f} & "
                         f"{r['pit_ks']:.3f} & {err:.1e} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    a = parse(__doc__)
    write(a.out, "tab_vine_nll_ci.tex", nll_table(load(a.inputs, "uci/e21_maf5k.json")), "uci.py")
    write(a.out, "tab_vine_nll_ci_50k.tex", nll_table(load(a.inputs, "uci/e21_maf50k.json")), "uci.py")
    write(a.out, "tab_uci_matched_structure.tex", matched_table(load(a.inputs, "uci/matched_structure_5k.json")), "uci.py")
    write(a.out, "tab_ipfp_uci_sensitivity.tex", ipfp_table(load(a.inputs, "uci/ipfp_sensitivity_5k.json")), "uci.py")


if __name__ == "__main__":
    main()
