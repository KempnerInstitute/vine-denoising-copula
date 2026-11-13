#!/usr/bin/env python3
"""Out-of-distribution families table (tab_ood_families).

Input: ood/e26_rescored.json, the per-case summaries of the re-scored OOD benchmark (nine cases from
families absent from the training zoo, three seeds, n=2000, pyvinecopulib samples, frozen checkpoint):
ISE of cell-averaged densities against the exact cell averages of the truth (TLL on its own cell
averages), exact grid Kendall-tau error, MI error against a Monte Carlo estimate of the true MI, and
held-out NLL (VDC through the library's bilinear edge density, TLL through its own pdf). The file is
a compact extract of the full re-scoring output (per-replicate records dropped); see README.md.
"""
from __future__ import annotations

from common import load, parse, write

LABELS = {
    "BB6(theta=2.0, delta=1.5)": r"BB6 ($\theta{=}2.0,\,\delta{=}1.5$)",
    "BB6(theta=3.0, delta=2.0)": r"BB6 ($\theta{=}3.0,\,\delta{=}2.0$)",
    "BB8(theta=3.0, delta=0.8)": r"BB8 ($\theta{=}3.0,\,\delta{=}0.8$)",
    "BB8(theta=5.0, delta=0.6)": r"BB8 ($\theta{=}5.0,\,\delta{=}0.6$)",
    "Tawn(theta=2.5, psi1=0.7, psi2=1.0)": r"Tawn ($\theta{=}2.5,\,\psi_1{=}0.7,\,\psi_2{=}1.0$)",
    "Plackett(theta=5.0)": r"Plackett ($\theta{=}5.0$)",
    "Plackett(theta=20.0)": r"Plackett ($\theta{=}20.0$)",
    "BB6-90(theta=2.0, delta=1.5)": r"BB6$^{90}$ ($\theta{=}2.0,\,\delta{=}1.5$)",
    "BB8-90(theta=3.0, delta=0.8)": r"BB8$^{90}$ ($\theta{=}3.0,\,\delta{=}0.8$)",
}
KEYS = [("vdc_ise_cellavg", "tll_cellavg_ise_cellavg"), ("vdc_tau_err_exact", "tll_tau_err_exact"),
        ("vdc_mi_err_mc", "tll_mi_err_mc"), ("vdc_nll_lib", "tll_nll_native")]
DIGITS = [4, 3, 3, 3]


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", r"\_"))


def row(vals):
    out = []
    for (kv, kt), dg in zip(KEYS, DIGITS):
        fv, ft = f"{vals[kv]:.{dg}f}", f"{vals[kt]:.{dg}f}"
        if float(fv) < float(ft):  # bold only when strictly better at the displayed precision
            fv = rf"\textbf{{{fv}}}"
        elif float(ft) < float(fv):
            ft = rf"\textbf{{{ft}}}"
        out.append((fv, ft))
    return out


def main() -> None:
    a = parse(__doc__)
    e26 = load(a.inputs, "ood/e26_rescored.json")["e26"]
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Case & Method & ISE $\downarrow$ & $|\Delta\tau|$ $\downarrow$ & MI err $\downarrow$ & NLL $\downarrow$ \\",
             r"\midrule"]
    for c in e26["cases"] + [{"summary": dict(e26["means"], name="Mean")}]:
        s = c["summary"]
        if s["name"] == "Mean":
            lines.append(r"\midrule")
        cells = row(s)
        lines.append(f"{label(s['name'])} & VDC (Ours) & " + " & ".join(v for v, _ in cells) + r" \\")
        lines.append(r" & pyvine-TLL & " + " & ".join(t for _, t in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(a.out, "tab_ood_families.tex", "\n".join(lines) + "\n", "ood.py")


if __name__ == "__main__":
    main()
