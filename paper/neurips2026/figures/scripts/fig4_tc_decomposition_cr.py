#!/usr/bin/env python3
"""Camera-ready Fig. 4: TC decomposition (vector PDF at printed size).

Inputs folder:
  (a) paper_outputs/tc_decomposition_ground_truth_compare.json      edge MI vs analytic truth, d=8
  (b) tc_knockout/tc_decomposition_uci_maf_{power,gas,hepmass}.json tree-level shares of |TC|, MAF data
  (c) tc_knockout/e17_tree_knockout_tc_maf.json                     tree knockout on the MAF data
All three panels are drawn by the panel functions of figA_tc_story_cr.py. Drawn at 5.5 in with the shared
vdc_figstyle fonts and method encodings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figA_tc_story_cr as tcs  # noqa: E402  (importing it restyles matplotlib)
import vdc_figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fs.add_io_args(ap, "fig4_tc_decomposition")
    ap.add_argument("--knockout-json", type=Path, help="override tc_knockout/e17_tree_knockout_tc_maf.json")
    a = ap.parse_args()
    plt.rcdefaults()  # the panel module's helper import applied its own style; start from defaults
    fs.apply()

    def load(rel, override=None):
        return json.loads(fs.input_file(a.inputs, rel, override).read_text())

    gt = load("paper_outputs/tc_decomposition_ground_truth_compare.json")
    decomp = {k: load(f"tc_knockout/tc_decomposition_uci_maf_{k}.json") for k in tcs.DATASETS}
    knockout = load("tc_knockout/e17_tree_knockout_tc_maf.json", a.knockout_json)

    fig = plt.figure(figsize=(fs.TEXTWIDTH, 1.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.0], wspace=0.42, left=0.085, right=0.99,
                          top=0.865, bottom=0.2)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
    info_a = tcs.panel_edge_mi(ax_a, gt)
    info_b = tcs.panel_tree_shares(ax_b, decomp)
    info_c = tcs.panel_knockout(ax_c, knockout)
    fs.panel_labels(fig, [ax_a, ax_b, ax_c], "abc", y=0.995)
    print(f"(a) edges={info_a['edges']} MAE={info_a['mae']:.5f} (json {info_a['mae_json']:.5f})")
    for key, v in info_b.items():
        print(f"(b) {key}", v)
    print(f"(c) n={info_c['n']} pearson r={info_c['r']:.4f}")
    fs.save(fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
