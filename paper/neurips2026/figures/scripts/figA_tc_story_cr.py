#!/usr/bin/env python3
"""Appendix figure: MI and TC as the dimension grows (camera-ready), and the shared information panels.

Reads only saved results (the inputs manifest records their source hashes), from the inputs folder:
  (a) paper_outputs/e6_theory_synthetic_results.json   pairwise MI error of a Gaussian AR(1) process vs d
  (b) tc_scaling/tc_benchmark.json                     TC of a Gaussian AR(1) process vs d (true, VDC with a
      D-vine truncated after its first tree, full chain-rule KSG; mean over three seeds)

The panel functions below also draw Fig. 3(c) (fig3_information_cr.py: MI self-consistency on the 480-trial
protocol, paper_outputs/mi_self_consistency_scaleup.json) and Fig. 4 (fig4_tc_decomposition_cr.py: edge MI
against the analytic ground truth, tree-level shares of |TC| for the MAF POWER, GAS and HEPMASS fits and the
tree knockout). Tree shares are computed with the earlier script's own helpers, so every plotted value is
unchanged.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

# Datasets are told apart by fill lightness together with hatch (bars) and marker shape (points);
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


def panel_edge_mi(ax, gt):
    """Edge MI of the d=8 Gaussian-factor ground-truth vine: VDC estimate vs analytic value."""
    merged = gt["merged_edges"]
    mi_true = np.array([r["mi_true"] for r in merged], float)
    mi_est = np.array([r["mi_est"] for r in merged], float)
    tree = np.array([r["tree"] for r in merged], int)
    trees = sorted(set(tree.tolist()))
    tree_cols = plt.get_cmap("Oranges")(np.linspace(0.95, 0.38, len(trees)))
    ax.plot([1e-6, 1.0], [1e-6, 1.0], linestyle="--", color="0.6", linewidth=0.7, zorder=1)
    for i, t in enumerate(trees):
        m = tree == t
        ax.scatter(mi_true[m], mi_est[m], s=14, marker=TREE_MARKERS[i], color=tree_cols[i],
                   edgecolors="0.1", linewidths=0.35, label=f"T{t}", zorder=3)
    dec = [1e-6, 1e-4, 1e-2, 1.0]
    labs = ["$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "$1$"]
    plain_log_ticks(ax, "x", dec, labs)
    plain_log_ticks(ax, "y", dec, labs)
    ax.set_xlim(1e-6, 1.0)
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlabel("True edge MI (nats)")
    ax.set_ylabel("VDC edge MI (nats)")
    ax.set_title("Edge MI vs. truth")
    mae = float(np.mean(np.abs(mi_est - mi_true)))
    ax.text(0.04, 0.97, f"{len(merged)} edges\nMAE {mae:.3f} nats", transform=ax.transAxes,
            ha="left", va="top", fontsize=fs.FS_ANNOT)
    ax.legend(loc="lower right", ncol=2, handletextpad=0.05, columnspacing=0.35, borderaxespad=0.05,
              labelspacing=0.1, handlelength=1.0)
    return dict(edges=len(merged), mae=mae, mae_json=float(gt["meta"]["edge_mae"]))


def panel_mi_dim(ax, e6):
    rows = [r for r in e6["mi_summary_records"] if r.get("scenario") == "gaussian_ar1"]
    keys = [("dcd", "vdc"), ("ksg", "ksg"), ("gaussian", "gauss"), ("infonce", "infonce"), ("mine", "mine")]
    dodge = np.linspace(-0.13, 0.13, len(keys))
    handles = []
    for (m, k), off in zip(keys, dodge):
        pts = sorted((int(r["d"]), float(r["mi_abs_err_mean"]), float(r["mi_abs_err_std"]))
                     for r in rows if r.get("method") == m and int(r.get("n_success", 0)) > 0
                     and np.isfinite(float(r["mi_abs_err_mean"])))
        if not pts:
            continue
        d = np.array([p[0] for p in pts], float)
        mu = np.array([p[1] for p in pts])
        sd = np.array([p[2] if np.isfinite(p[2]) else 0.0 for p in pts])
        lo = np.minimum(sd, mu * 0.75)  # keep whiskers positive on the log axis (truncated, see caption)
        x = d * np.exp(off)
        lab = "VDC" if k == "vdc" else fs.STYLE[k].label.replace(" copula", "")
        kw = fs.line_kw(k, label=lab, linewidth=1.3 if k == "vdc" else 0.9)
        eb = ax.errorbar(x, mu, yerr=[lo, sd], elinewidth=0.55, capsize=1.2, **kw)
        for art in eb[2]:
            art.set_alpha(0.55)
        handles.append(Line2D([], [], **kw))  # legend entry without the error-bar glyph
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([5, 10, 20, 50])
    ax.set_xticklabels(["5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(7e-5, 0.12)  # the band below 6e-4 holds the legend, under every (truncated) whisker
    ax.set_xlim(4.0, 62.0)
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("MI absolute error (nats)")
    ax.set_title("MI error vs. dimension")
    ax.legend(handles=handles, loc="lower center", ncol=3, columnspacing=0.6, handlelength=1.5,
              handletextpad=0.3, borderaxespad=0.1, labelspacing=0.15)


def panel_tc_dimension(ax, tc):
    """TC of a Gaussian AR(1) process vs dimension: truth, VDC (tree 1) and full chain-rule KSG."""
    dims = np.asarray(tc["dimensions"], float)
    tc_true = np.asarray(tc["tc_true"], float)
    tc_vdc = np.asarray(tc["tc_dcd_mean"], float)
    tc_ksg = np.asarray(tc["tc_ksg_mean"], float)
    ax.plot(dims, tc_true, color="0.75", linewidth=3.4, solid_capstyle="round", label="True TC", zorder=1)
    ax.plot(dims, tc_vdc, **fs.line_kw("vdc", zorder=3, label="VDC (tree 1)"))
    ax.plot(dims, tc_ksg, **fs.line_kw("ksg", zorder=2, label="KSG (full chain rule)"))
    plain_log_ticks(ax, "x", [2, 5, 10, 20, 50])
    plain_log_ticks(ax, "y", [0.3, 1, 3, 10])
    ax.set_xlim(1.7, 105)
    ax.set_ylim(0.22, 28)
    xa = 66.0
    ax.annotate("", xy=(xa, tc_true[-1]), xytext=(xa, tc_ksg[-1]),
                arrowprops=dict(arrowstyle="<->", color="0.25", linewidth=0.6, shrinkA=0, shrinkB=0))
    ax.text(xa * 1.07, np.sqrt(tc_true[-1] * tc_ksg[-1]), f"{tc_true[-1] / tc_ksg[-1]:.1f}$\\times$",
            ha="left", va="center", fontsize=fs.FS_ANNOT, color="0.15")
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("TC (nats)")
    ax.set_title("TC vs. dimension, AR(1)")
    ax.legend(loc="upper left", borderaxespad=0.1)
    return dict(dims=dims.tolist(), true_over_ksg_d50=float(tc_true[-1] / tc_ksg[-1]))


def panel_tree_shares(ax, decomp):
    """Tree-level shares of |TC| on the MAF POWER, GAS and HEPMASS fits (trees 5+ pooled)."""
    labels = None
    xs = np.arange(5)
    w = 0.27
    out = {}
    for off, key in zip((-w, 0.0, w), DATASETS):
        shares = _aggregate_tree_shares(_tree_abs_shares(decomp[key]), max_tree=4)
        out[key] = {k: round(v, 3) for k, v in shares.items()}
        labels = list(shares)
        st = DATASETS[key]
        ax.bar(xs + off, [shares[k] for k in labels], width=w, color=st["color"], hatch=st["hatch"],
               edgecolor="0.1", linewidth=0.5, label=st["label"], zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 85)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlabel("Vine tree")
    ax.set_ylabel("Share of |TC| (%)")
    ax.set_title("Tree shares, MAF data")
    ax.legend(loc="upper right", handlelength=1.3)
    return out


def panel_knockout(ax, knockout):
    """Tree knockout on the MAF data: fitted share of |TC| removed vs observed drop in VDC TC."""
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
    ax.plot([-12, 95], [-12, 95], linestyle="--", color="0.6", linewidth=0.7, zorder=1)
    for key, st in DATASETS.items():
        m = np.array([p[0] == key for p in pts])
        ax.scatter(share[m], drop[m], s=17, marker=st["marker"], color=st["color"], edgecolors="0.1",
                   linewidths=0.45, label=st["label"], zorder=3)
    ax.set_xlim(-9, 88)
    ax.set_ylim(-9, 88)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_yticks([0, 20, 40, 60, 80])
    t1 = np.array(["T1" in p[1] for p in pts])
    ax.annotate("T1\nremoved", xy=(share[t1].mean(), drop[t1].min() - 3.0), xytext=(80, 47),
                ha="center", va="center", fontsize=fs.FS_ANNOT,
                arrowprops=dict(arrowstyle="-", color="0.4", linewidth=0.5, shrinkA=1, shrinkB=0))
    ax.text(share[~t1].max() + 6.0, 4.0, "T2, T3 or\nT4+ removed", ha="left", va="center", fontsize=fs.FS_ANNOT)
    ax.text(0.05, 0.60, f"$r={r:.3f}$", transform=ax.transAxes, ha="left", va="top", fontsize=fs.FS_ANNOT)
    ax.set_xlabel("Removed share of |TC| (%)")
    ax.set_ylabel("Observed TC drop (%)")
    ax.set_title("Tree knockout")
    ax.legend(loc="upper left", handletextpad=0.15, borderaxespad=0.1, labelspacing=0.15)
    return dict(n=len(pts), r=r)


def selfcons_stats(sc):
    """DPI violations (count, trials, base datasets), Fisher p over base datasets, additivity error."""
    from scipy.stats import fisher_exact
    viol = {}
    for k in ("vdc", "ksg"):
        recs = sc[k.upper()]["dpi"]["records"]
        v = np.array([x["mi_noisy"] > x["mi_base"] + DPI_TOL for x in recs])
        assert abs(v.mean() - sc[k.upper()]["dpi"]["summary"]["violation_rate"]) < 1e-12
        viol[k] = (int(v.sum()), len(v), int(v.reshape(-1, 3).any(axis=1).sum()))
    add = {k: float(sc[k.upper()]["additivity"]["summary"]["mean_abs_error"]) for k in ("vdc", "ksg")}
    nb = viol["vdc"][1] // 3  # no base dataset violates at more than one noise level
    p_base = float(fisher_exact([[viol["vdc"][2], nb - viol["vdc"][2]], [viol["ksg"][2], nb - viol["ksg"][2]]])[1])
    return viol, add, p_base


def panel_selfcons(ax_dpi, ax_add, sc, title_ax=None):
    """MI self-consistency on the 480-trial protocol: DPI violation rate (top) and additivity error (bottom)."""
    viol, add, p_base = selfcons_stats(sc)
    methods, ypos = ["vdc", "ksg"], [1.0, 0.0]
    for k, y in zip(methods, ypos):
        n_v, n_t, _ = viol[k]
        pct = 100.0 * n_v / n_t
        ax_dpi.barh(y, pct, height=0.62, **fs.bar_kw(k), zorder=2)
        ax_dpi.text(pct + 0.2, y, f"{pct:.1f}%", ha="left", va="center", fontsize=fs.FS_ANNOT)
        ax_add.barh(y, add[k], height=0.62, **fs.bar_kw(k), zorder=2)
        ax_add.text(add[k] + 0.0015, y, f"{add[k]:.3f}", ha="left", va="center", fontsize=fs.FS_ANNOT)
    for ax in (ax_dpi, ax_add):
        ax.set_yticks(ypos)
        ax.set_yticklabels([fs.STYLE[k].label for k in methods])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, 1.6)
    ax_dpi.set_xlim(0, 8.6)
    ax_dpi.set_xticks([0, 2, 4, 6, 8])
    ax_dpi.text(0.98, ypos[0], f"$p={p_base:.3f}$", transform=ax_dpi.get_yaxis_transform(), ha="right",
                va="center", fontsize=fs.FS_ANNOT, color="0.15")
    ax_dpi.set_xlabel("DPI violations (%)", labelpad=1.0)
    ax_add.set_xlim(0, 0.058)
    ax_add.set_xticks([0, 0.02, 0.04])
    ax_add.set_xticklabels(["0", "0.02", "0.04"])
    ax_add.set_xlabel("Additivity error (nats)", labelpad=1.0)
    (title_ax or ax_dpi).set_title("Self-consistency")
    return dict(viol=viol, add=add, p_base=p_base)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fs.add_io_args(ap, "figA_tc_story")
    ap.add_argument("--tc-json", type=Path, help="override tc_scaling/tc_benchmark.json")
    a = ap.parse_args()

    def load(rel, override=None):
        return json.loads(fs.input_file(a.inputs, rel, override).read_text())

    e6 = load("paper_outputs/e6_theory_synthetic_results.json")
    tc = load("tc_scaling/tc_benchmark.json", a.tc_json)

    plt.rcdefaults()  # importing the earlier script applied its own style; start from defaults
    fs.apply()
    fig = plt.figure(figsize=(fs.TEXTWIDTH, 1.70))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], left=0.085, right=0.99, top=0.865, bottom=0.195,
                          wspace=0.3)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    panel_mi_dim(ax_a, e6)
    info_b = panel_tc_dimension(ax_b, tc)
    fs.panel_labels(fig, [ax_a, ax_b], "ab", y=0.995)
    print(f"(b) d={info_b['dims']} true/KSG at d=50: {info_b['true_over_ksg_d50']:.3f}")
    fs.save(fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
