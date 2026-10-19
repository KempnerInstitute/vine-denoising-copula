#!/usr/bin/env python3
"""Appendix figure: TC decomposition, scaling, knockout and self-consistency (camera-ready).

Redraws the earlier fig_tc_story_composite at the full NeurIPS text width in the shared camera-ready
style (vdc_figstyle). It reads only the saved results that the earlier version of the figure used
(2026-10-13 render; the inputs manifest records their source hashes), from the inputs folder:
paper_outputs/tc_decomposition_ground_truth_compare.json, tc_scaling/tc_benchmark.json,
tc_knockout/tc_decomposition_uci_maf_{power,gas,hepmass}.json, tc_knockout/e17_tree_knockout_tc_maf.json and
paper_outputs/mi_self_consistency_scaleup.json. Tree shares are computed with the earlier script's own
helpers, so every plotted value is unchanged.

  (a) edge MI of the d=8 Gaussian-factor ground-truth vine, VDC estimate vs analytic value
  (b) TC of a Gaussian AR(1) process vs dimension (true, VDC, KSG; mean over three seeds)
  (c) tree-level shares of |TC| for the MAF POWER, GAS and HEPMASS fits
  (d) tree knockout: removed share vs observed VDC TC drop
  (e) MI self-consistency on the 480-trial protocol (DPI violations, additivity error)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_tc_story_composite import _aggregate_tree_shares, _tree_abs_shares  # noqa: E402
import vdc_figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

# Datasets are told apart by fill lightness together with hatch in (c) and marker shape in (d);
# method colours are kept for methods.
DATASETS = {
    "power": dict(label="POWER", color="0.30", hatch="", marker="o"),
    "gas": dict(label="GAS", color="0.62", hatch="////", marker="^"),
    "hepmass": dict(label="HEPMASS", color="0.90", hatch="....", marker="s"),
}
TREE_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
DPI_TOL = 1e-3  # nats; the protocol's default tolerance


def plain_log_ticks(ax, axis, ticks, labels=None):
    getattr(ax, f"set_{axis}scale")("log")
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    getattr(ax, f"set_{axis}ticklabels")(labels or [f"{t:g}" for t in ticks])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fs.add_io_args(ap, "figA_tc_story")
    ap.add_argument("--gt-json", type=Path, help="override paper_outputs/tc_decomposition_ground_truth_compare.json")
    ap.add_argument("--tc-json", type=Path, help="override tc_scaling/tc_benchmark.json")
    ap.add_argument("--decomp-dir", type=Path, help="override folder with tc_decomposition_uci_maf_{power,gas,hepmass}.json")
    ap.add_argument("--knockout-json", type=Path, help="override tc_knockout/e17_tree_knockout_tc_maf.json")
    ap.add_argument("--selfcons-json", type=Path, help="override paper_outputs/mi_self_consistency_scaleup.json")
    a = ap.parse_args()

    def load(rel, override=None):
        return json.loads(fs.input_file(a.inputs, rel, override).read_text())

    gt = load("paper_outputs/tc_decomposition_ground_truth_compare.json", a.gt_json)
    tc = load("tc_scaling/tc_benchmark.json", a.tc_json)
    decomp = {k: load(f"tc_knockout/tc_decomposition_uci_maf_{k}.json",
                      a.decomp_dir / f"tc_decomposition_uci_maf_{k}.json" if a.decomp_dir else None) for k in DATASETS}
    knockout = load("tc_knockout/e17_tree_knockout_tc_maf.json", a.knockout_json)
    sc = {str(r["estimator"]).upper(): r for r in load("paper_outputs/mi_self_consistency_scaleup.json", a.selfcons_json)}

    plt.rcdefaults()  # importing the earlier script applied its own style; start from defaults
    fs.apply()
    fig = plt.figure(figsize=(fs.TEXTWIDTH, 3.30))
    outer = fig.add_gridspec(2, 3, left=0.085, right=0.985, top=0.925, bottom=0.115,
                             wspace=0.58, hspace=0.66, height_ratios=[1.0, 0.82])
    ax_a, ax_b, ax_c = (fig.add_subplot(outer[0, i]) for i in range(3))
    ax_d = fig.add_subplot(outer[1, 0])
    sub = outer[1, 1:].subgridspec(1, 2, wspace=0.62)
    ax_e1, ax_e2 = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])

    # (a) edge MI against the analytic ground truth
    merged = gt["merged_edges"]
    mi_true = np.array([r["mi_true"] for r in merged], float)
    mi_est = np.array([r["mi_est"] for r in merged], float)
    tree = np.array([r["tree"] for r in merged], int)
    trees = sorted(set(tree.tolist()))
    tree_cols = plt.get_cmap("Oranges")(np.linspace(0.95, 0.38, len(trees)))
    ax_a.plot([1e-6, 1.0], [1e-6, 1.0], linestyle="--", color="0.6", linewidth=0.7, zorder=1)
    for i, t in enumerate(trees):
        m = tree == t
        ax_a.scatter(mi_true[m], mi_est[m], s=13, marker=TREE_MARKERS[i], color=tree_cols[i],
                     edgecolors="0.1", linewidths=0.35, label=f"T{t}", zorder=3)
    dec = [1e-6, 1e-4, 1e-2, 1.0]
    labs = ["$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "$1$"]
    plain_log_ticks(ax_a, "x", dec, labs)
    plain_log_ticks(ax_a, "y", dec, labs)
    ax_a.set_xlim(1e-6, 1.0)
    ax_a.set_ylim(1e-6, 1.0)
    ax_a.set_xlabel("True edge MI (nats)")
    ax_a.set_ylabel("VDC edge MI (nats)")
    ax_a.set_title("Edge MI vs. ground truth")
    mae = float(np.mean(np.abs(mi_est - mi_true)))
    ax_a.text(0.04, 0.97, f"{len(merged)} edges\nMAE {mae:.3f} nats", transform=ax_a.transAxes,
              ha="left", va="top", fontsize=fs.FS_ANNOT)
    ax_a.legend(loc="lower right", ncol=2, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.1,
                labelspacing=0.2)

    # (b) TC scaling with dimension
    dims = np.asarray(tc["dimensions"], float)
    tc_true = np.asarray(tc["tc_true"], float)
    tc_vdc = np.asarray(tc["tc_dcd_mean"], float)
    tc_ksg = np.asarray(tc["tc_ksg_mean"], float)
    ax_b.plot(dims, tc_true, color="0.75", linewidth=3.4, solid_capstyle="round", label="True TC", zorder=1)
    ax_b.plot(dims, tc_vdc, **fs.line_kw("vdc", zorder=3, label="VDC (tree 1)"))
    ax_b.plot(dims, tc_ksg, **fs.line_kw("ksg", zorder=2))
    plain_log_ticks(ax_b, "x", [2, 5, 10, 20, 50])
    plain_log_ticks(ax_b, "y", [0.3, 1, 3, 10])
    ax_b.set_xlim(1.7, 105)
    ax_b.set_ylim(0.22, 28)
    xa = 66.0
    ax_b.annotate("", xy=(xa, tc_true[-1]), xytext=(xa, tc_ksg[-1]),
                  arrowprops=dict(arrowstyle="<->", color="0.25", linewidth=0.6, shrinkA=0, shrinkB=0))
    ax_b.text(xa * 1.07, np.sqrt(tc_true[-1] * tc_ksg[-1]), f"{tc_true[-1] / tc_ksg[-1]:.1f}$\\times$",
              ha="left", va="center", fontsize=fs.FS_ANNOT, color="0.15")
    ax_b.set_xlabel("Dimension $d$")
    ax_b.set_ylabel("TC (nats)")
    ax_b.set_title("TC vs. dimension, AR(1)")
    ax_b.legend(loc="upper left")

    # (c) tree-level shares of |TC|
    labels_c = None
    xs = np.arange(5)
    w = 0.27
    for off, key in zip((-w, 0.0, w), DATASETS):
        shares = _aggregate_tree_shares(_tree_abs_shares(decomp[key]), max_tree=4)
        labels_c = list(shares)
        st = DATASETS[key]
        ax_c.bar(xs + off, [shares[k] for k in labels_c], width=w, color=st["color"], hatch=st["hatch"],
                 edgecolor="0.1", linewidth=0.5, label=st["label"], zorder=2)
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels(labels_c)
    ax_c.tick_params(axis="x", length=0)
    ax_c.set_ylim(0, 85)
    ax_c.set_yticks([0, 20, 40, 60, 80])
    ax_c.set_xlabel("Vine tree")
    ax_c.set_ylabel("Share of |TC| (%)")
    ax_c.set_title("Tree shares, MAF data")
    ax_c.legend(loc="upper right", handlelength=1.3)

    # (d) tree knockout
    pts = []
    for ds in knockout["datasets"]:
        for ab in ds["ablations"]:
            if ab["label"] == "Full":
                continue
            pts.append((str(ds["dataset"]).lower(), ab["label"], float(ab["removed_abs_share_pct"]),
                        100.0 * float(ab["rel_drop_vdc"])))
    share = np.array([p[2] for p in pts])
    drop = np.array([p[3] for p in pts])
    r = float(np.corrcoef(share, drop)[0, 1])
    ax_d.plot([-12, 95], [-12, 95], linestyle="--", color="0.6", linewidth=0.7, zorder=1)
    for key, st in DATASETS.items():
        m = np.array([p[0] == key for p in pts])
        ax_d.scatter(share[m], drop[m], s=17, marker=st["marker"], color=st["color"], edgecolors="0.1",
                     linewidths=0.45, label=st["label"], zorder=3)
    ax_d.set_xlim(-9, 88)
    ax_d.set_ylim(-9, 88)
    ax_d.set_xticks([0, 20, 40, 60, 80])
    ax_d.set_yticks([0, 20, 40, 60, 80])
    t1 = np.array(["T1" in p[1] for p in pts])
    ax_d.annotate("T1 removed", xy=(share[t1].mean(), drop[t1].min() - 3.0), xytext=(88, 45),
                  ha="right", va="center", fontsize=fs.FS_ANNOT,
                  arrowprops=dict(arrowstyle="-", color="0.4", linewidth=0.5, shrinkA=2, shrinkB=0))
    ax_d.text(share[~t1].max() + 6.0, 6.0, "T2, T3 or T4+\nremoved", ha="left", va="center",
              fontsize=fs.FS_ANNOT)
    ax_d.text(0.05, 0.58, f"$r={r:.3f}$", transform=ax_d.transAxes, ha="left", va="top",
              fontsize=fs.FS_ANNOT)
    ax_d.set_xlabel("Removed share of |TC| (%)")
    ax_d.set_ylabel("Observed TC drop (%)")
    ax_d.set_title("Tree knockout")
    ax_d.legend(loc="upper left", handletextpad=0.2, borderaxespad=0.1)

    # (e) self-consistency on the 480-trial protocol
    methods = ["vdc", "ksg"]
    ypos = [1.0, 0.0]
    viol = {}
    for k in methods:
        recs = sc[k.upper()]["dpi"]["records"]
        v = np.array([x["mi_noisy"] > x["mi_base"] + DPI_TOL for x in recs])
        assert abs(v.mean() - sc[k.upper()]["dpi"]["summary"]["violation_rate"]) < 1e-12
        viol[k] = (int(v.sum()), len(v), int(v.reshape(-1, 3).any(axis=1).sum()))
    add = {k: float(sc[k.upper()]["additivity"]["summary"]["mean_abs_error"]) for k in methods}
    for k, y in zip(methods, ypos):
        n_v, n_t, _ = viol[k]
        pct = 100.0 * n_v / n_t
        ax_e1.barh(y, pct, height=0.56, **fs.bar_kw(k), zorder=2)
        ax_e1.text(pct + 0.25, y, f"{pct:.1f}%", ha="left", va="center", fontsize=fs.FS_ANNOT)
        ax_e2.barh(y, add[k], height=0.56, **fs.bar_kw(k), zorder=2)
        ax_e2.text(add[k] + 0.0015, y, f"{add[k]:.3f}", ha="left", va="center", fontsize=fs.FS_ANNOT)
    # Fisher exact test over the 160 base datasets (no base dataset violates at more than one level)
    from scipy.stats import fisher_exact
    nb = viol["vdc"][1] // 3
    p_base = float(fisher_exact([[viol["vdc"][2], nb - viol["vdc"][2]], [viol["ksg"][2], nb - viol["ksg"][2]]])[1])
    ax_e1.text(0.99, 0.98, f"Fisher exact $p={p_base:.3f}$", transform=ax_e1.transAxes, ha="right", va="top",
               fontsize=fs.FS_ANNOT, color="0.15")
    for ax in (ax_e1, ax_e2):
        ax.set_yticks(ypos)
        ax.set_yticklabels([fs.STYLE[k].label for k in methods])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, 1.6)
    ax_e1.set_xlim(0, 7.2)
    ax_e1.set_xticks([0, 2, 4, 6])
    ax_e1.set_xlabel(f"DPI violations (% of {viol['vdc'][1]} trials)")
    ax_e1.set_title("DPI violations")
    ax_e2.set_xlim(0, 0.058)
    ax_e2.set_xticks([0, 0.02, 0.04])
    ax_e2.set_xticklabels(["0", "0.02", "0.04"])
    ax_e2.set_xlabel("Mean absolute error (nats)")
    ax_e2.set_title("Additivity error")

    fs.panel_labels(fig, [ax_a, ax_b, ax_c], "abc")
    fs.panel_labels(fig, [ax_d, ax_e1], "de")

    print(f"(a) edges={len(merged)} MAE={mae:.5f} (json {gt['meta']['edge_mae']:.5f})")
    print(f"(b) d={dims.tolist()} true/KSG at d=50: {tc_true[-1] / tc_ksg[-1]:.3f}")
    for key in DATASETS:
        print(f"(c) {key}", {k: round(v, 3) for k, v in _aggregate_tree_shares(_tree_abs_shares(decomp[key]), 4).items()})
    print(f"(d) n={len(pts)} pearson r={r:.4f}")
    print(f"(e) DPI (violations, trials, base datasets) {viol}; Fisher p over base datasets={p_base:.4f}; additivity {add}")
    fs.save(fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
