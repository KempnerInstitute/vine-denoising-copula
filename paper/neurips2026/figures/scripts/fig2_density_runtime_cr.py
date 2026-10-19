#!/usr/bin/env python3
"""Camera-ready Fig. 2: density quality and fitting cost (vector PDF at printed size).

Same data and panels as fig4_density_runtime_composite_v2.py (whose loaders are reused); inputs folder:
  (a) tables/tab_bivariate.tex                      bivariate ISE vs per-pair latency
  (b) paper_outputs/e22_highdim_scaling.json         Clayton-vine held-out NLL vs d
  (c) fast_fit_scaling/timing_summary.json           fixed-thread fit time vs d
  (d) paper_outputs/e23_sp100_density.json, e28_sp100_chrono.json   S&P100 held-out NLL
Drawn at 5.5 in (NeurIPS \\textwidth) with the shared vdc_figstyle fonts and method encodings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig4_density_runtime_composite_v2 as old  # noqa: E402  (data loaders)
import vdc_figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BIVAR_KEY = {"Histogram": "hist", "KDE": "kde", "pyvine-param": "param", "pyvine-TLL": "tll",
             "VDC (one-shot)": "vdc"}
NLL_KEYS = [("VDC", "vdc"), ("pyvine_tll", "tll"), ("flow_realnvp", "flow"), ("gaussian_copula", "gauss")]
SHORT = {"vdc": "VDC", "tll": "TLL", "flow": "RealNVP", "gauss": "Gaussian"}  # TLL = pyvine-TLL


def panel_bivariate(ax, rows):
    # label placement (x multiplier, y multiplier, ha) chosen for the corrected Table 1 values
    place = {"hist": (1.55, 1.0, "left"), "kde": (1.0, 0.44, "center"), "param": (0.68, 1.0, "right"),
             "tll": (1.0, 2.35, "center"), "vdc": (1.0, 0.46, "center")}
    for r in rows:
        k = BIVAR_KEY.get(r["method"])
        if k is None:
            continue
        s = fs.STYLE[k]
        ax.scatter(r["time_ms"], r["ise"], s=34 if k == "vdc" else 22, marker=s.marker, color=s.color,
                   edgecolor="0.1", linewidth=0.5, zorder=3)
        xm, ym, ha = place[k]
        ax.text(r["time_ms"] * xm, r["ise"] * ym, "VDC" if k == "vdc" else s.label.replace("Histogram", "Hist."),
                fontsize=fs.FS_ANNOT, ha=ha, va="center", color="0.1",
                fontweight="bold" if k == "vdc" else "normal", zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.2, 1500)
    ax.set_ylim(1.2e-3, 8.0)
    ax.set_xlabel("Runtime per edge (ms)")
    ax.set_ylabel("ISE")
    ax.set_title("Edge accuracy vs. latency")


def panel_scaling(ax, e22):
    dims = [100, 200, 500]
    arr = old._by_d_method(e22, [m for m, _ in NLL_KEYS], dims)
    for j, (_m, k) in enumerate(NLL_KEYS):
        y = arr[:, j]
        ok = ~np.isnan(y)
        ax.plot(np.array(dims)[ok], y[ok], **fs.line_kw(k, label=SHORT[k]))
    ax.set_xticks(dims)
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Held-out NLL (bits/dim)")
    ax.set_title("Clayton-vine likelihood")
    ax.set_ylim(-0.66, -0.30)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.355), ncol=2, columnspacing=0.6, handlelength=1.8)


def panel_time(ax, summary):
    assert summary["status"] == "complete_and_exact"
    rows = {row["d"]: row["methods"] for row in summary["summary"]}
    dims = [100, 200, 500]

    def get(m):
        return (np.array([rows[d][m]["mean_seconds"] for d in dims]),
                np.array([rows[d][m]["sample_sd_seconds"] for d in dims]))

    t_vdc, e_vdc = get("vdc_fast")
    t16, e16 = get("tll_16")
    t8, e8 = get("tll_8")
    ax.errorbar(dims, t8, yerr=e8, **fs.line_kw("tll", color="#5B5B5B", marker="s", linestyle=":",
                                                label="TLL, 8 threads"))
    ax.errorbar(dims, t16, yerr=e16, **fs.line_kw("tll", label="TLL, 16 threads"))
    ax.errorbar(dims, t_vdc, yerr=e_vdc, **fs.line_kw("vdc", label="VDC (GPU)"))
    # speedup over 16-thread TLL, printed under each VDC point
    for d, a, b in zip(dims, t_vdc, t16):
        off, ha = ((6, -2), "left") if d < 500 else ((0, -7), "center")
        ax.annotate(f"{b / a:.1f}×", xy=(d, a), xytext=off, textcoords="offset points",
                    ha=ha, va="top", fontsize=fs.FS_ANNOT, color="0.1")
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Fit time (s)")
    ax.set_title("Vine fitting cost")
    ax.set_ylim(9, 3.0e4)
    ax.legend(loc="upper left", handlelength=1.8)


def panel_sp100(ax, rnd, chrono):
    groups = [("Random", rnd), ("Chronological", chrono)]
    w = 0.19
    for j, (m, k) in enumerate(NLL_KEYS):
        xs, ys, lo, hi = [], [], [], []
        for g, (_name, look) in enumerate(groups):
            rec = look.get(m)
            if rec is None:
                continue
            xs.append(g + (j - 1.5) * w)
            ys.append(rec["nll"])
            lo.append(max(rec["nll"] - rec["lo"], 0))
            hi.append(max(rec["hi"] - rec["nll"], 0))
        ax.bar(xs, ys, width=w * 0.88, zorder=3,
               **fs.bar_kw(k, label=SHORT[k]))
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="0.2", elinewidth=0.6, capsize=1.4, zorder=4)
    ax.axhline(0, color="0.2", linewidth=0.7, zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random", "Chrono."])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.52, 1.52)
    ax.set_ylim(-1.0, 0.0)
    ax.set_ylabel("Held-out NLL (bits/dim)")
    ax.set_title("S&P100 ($d{=}100$)")
    ax.legend(loc="lower right", ncol=2, columnspacing=0.35, handlelength=1.0, handletextpad=0.3, handleheight=0.85, borderaxespad=0.1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fs.add_io_args(ap, "fig2_density_runtime")
    a = ap.parse_args()
    fs.apply()
    for rel in ("tables/tab_bivariate.tex", "paper_outputs/e22_highdim_scaling.json",
                "paper_outputs/e23_sp100_density.json", "paper_outputs/e28_sp100_chrono.json"):
        fs.input_file(a.inputs, rel)
    old.PAPER_ROOT = a.inputs  # the shared loaders resolve "drafts/<rel>" against PAPER_ROOT
    rows = old._parse_bivariate_table("drafts/tables/tab_bivariate.tex")
    e22 = old._load_records("drafts/paper_outputs/e22_highdim_scaling.json")
    summary = json.loads(fs.input_file(a.inputs, "fast_fit_scaling/timing_summary.json").read_text())
    rnd = old._load_sp100_nll("drafts/paper_outputs/e23_sp100_density.json")
    chrono = old._load_sp100_nll("drafts/paper_outputs/e28_sp100_chrono.json")

    fig = plt.figure(figsize=(fs.TEXTWIDTH, 1.95))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.97, 1.0, 1.0, 1.15], wspace=0.6, left=0.07, right=0.997, top=0.86, bottom=0.2)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    panel_bivariate(axes[0], rows)
    panel_scaling(axes[1], e22)
    panel_time(axes[2], summary)
    panel_sp100(axes[3], rnd, chrono)
    fs.panel_labels(fig, axes, "abcd", y=0.995)
    fs.save(fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
