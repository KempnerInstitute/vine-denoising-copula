#!/usr/bin/env python3
"""Retained training-corruption variants (tab_corruption_variants and tab_corruption_edges).

Input: corruption/corruption_variants_results.json, the output of ../runners/corruption_variants.py (protocol
fixed before the run): the deployed uniform-mix model and the retained direct, Gaussian and multinomial
checkpoints scored on identical inputs. tab_corruption_variants gives the Table 1 suite (mean held-out KL and
ISE over 45 fits), the out-of-distribution suite (27 fits) and each checkpoint's own D-vine on the MAF data
(held-out NLL, bits/dim); tab_corruption_edges gives the common real-data edge panel (held-out edge NLL summed
over all trees, bits/dim; reference TLL edges on the same inputs) and the number of edges, out of all four
datasets, on which each alternative has a higher held-out edge log-density than the deployed model.
"""
from __future__ import annotations

from common import load, parse, write

ROWS = [("uniform", "Uniform-mix (deployed)"), ("direct", "Direct"), ("gaussian", "Gaussian"),
        ("multinomial", "Multinomial"), ("multinomial_190k", "Multinomial")]
DATASETS = [("power", "POWER"), ("gas", "GAS"), ("hepmass", "HEPMASS"), ("miniboone", "MINIBOONE")]


def bold_min(cells):
    """Bold the lowest value of each column at the displayed precision."""
    best = min(float(c) for c in cells)
    return [rf"\textbf{{{c}}}" if float(c) == best else c for c in cells]


def columns(values, spec):
    return bold_min([format(v, spec) for v in values])


def main() -> None:
    a = parse(__doc__)
    r = load(a.inputs, "corruption/corruption_variants_results.json")
    ck, s = r["manifest"]["checkpoints"], r["suites"]
    names = [n for n, _ in ROWS]
    t1, ood = s["E1a_table1"]["means"], s["E1b_ood"]["means"]
    cols = [columns([t1[n]["kl"] for n in names], ".4f"), columns([t1[n]["ise"] for n in names], ".2e"),
            columns([ood[n]["kl"] for n in names], ".4f"), columns([ood[n]["ise"] for n in names], ".4f")]
    cols += [columns([s["E3_vine"]["nll_bits_per_dim"][d][n] for n in names], ".4f") for d, _ in DATASETS]
    body = ("\\begin{tabular}{llcccccccc}\n\\toprule\n"
            " & & \\multicolumn{2}{c}{Table~1 suite} & \\multicolumn{2}{c}{Out-of-distribution} & "
            "\\multicolumn{4}{c}{Own D-vine, held-out NLL (bits/dim)} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-10}\n"
            "Variant & Step & KL & ISE & KL & ISE & " + " & ".join(lab for _, lab in DATASETS) + " \\\\\n\\midrule\n")
    for i, (n, lab) in enumerate(ROWS):
        body += f"{lab} & {ck[n]['step'] // 1000}k & " + " & ".join(c[i] for c in cols) + " \\\\\n"
    write(a.out, "tab_corruption_variants.tex", body + "\\bottomrule\n\\end{tabular}\n", "corruption.py")

    panel = s["E2_edge_panel"]
    srcs = names + ["tll_reference"]
    ecols = [columns([panel["means_bits_per_dim"][d][x]["total"] for x in srcs], ".4f") for d, _ in DATASETS]
    n_edges = sum(sum(v[1] for v in panel["vs_uniform"][d]["direct"]["edges_alt_better"].values()) for d, _ in DATASETS)
    body = ("\\begin{tabular}{llccccc}\n\\toprule\n"
            "Edge estimator & Step & " + " & ".join(lab for _, lab in DATASETS) + f" & Edges better (of {n_edges}) \\\\\n\\midrule\n")
    for i, x in enumerate(srcs):
        lab = dict(ROWS).get(x, "pyvine-TLL (reference)")
        step = f"{ck[x]['step'] // 1000}k" if x in ck else "--"
        better = "--" if x in ("uniform", "tll_reference") else str(sum(
            sum(v[0] for v in panel["vs_uniform"][d][x]["edges_alt_better"].values()) for d, _ in DATASETS))
        body += f"{lab} & {step} & " + " & ".join(c[i] for c in ecols) + f" & {better} \\\\\n"
    write(a.out, "tab_corruption_edges.tex", body + "\\bottomrule\n\\end{tabular}\n", "corruption.py")


if __name__ == "__main__":
    main()
