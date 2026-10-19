#!/usr/bin/env python3
"""Camera-ready Fig. 3: information-estimation results (vector PDF at printed size).

Same data as the earlier composite (fig5_information_composite.py and fig_error_vs_dimension.py); inputs folder:
  (a) tables/tab_mi.tex                                   bivariate MI MAE, eight cases (no Joe)
  (b) paper_outputs/e6_theory_synthetic_results.json      MI abs. error vs d, Gaussian AR(1)
  (c) paper_outputs/tc_benchmark_neural_full.json         median TC runtime vs d
Drawn at 5.5 in with the shared vdc_figstyle fonts and method encodings.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vdc_figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

MI_KEYS = {"VDC (Ours)": "vdc", "MINE": "mine", "NWJ": "nwj", "KSG": "ksg", "InfoNCE": "infonce",
           "Gaussian": "gauss", "MINDE": "minde"}


def parse_mi_table(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if "&" not in line or "Method" in line:
            continue
        cells = [re.sub(r"\\textbf\{([^}]*)\}", r"\1", c).strip() for c in line.split("&")]
        name = cells[0]
        if name not in MI_KEYS:
            continue
        m = re.match(r"([0-9.]+)\s*(?:\$\\pm\$\s*([0-9.]+))?", cells[1])
        rows.append({"key": MI_KEYS[name], "mae": float(m.group(1)),
                     "sd": float(m.group(2)) if m.group(2) else None})
    return rows


def panel_mi_bars(ax, rows):
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        s = fs.STYLE[r["key"]]
        ax.barh(yi, r["mae"], height=0.68, **fs.bar_kw(r["key"]), zorder=3)
        if r["sd"]:
            ax.errorbar(r["mae"], yi, xerr=r["sd"], fmt="none", ecolor="0.15", elinewidth=0.6, capsize=1.4, zorder=4)
        x_text = r["mae"] + (r["sd"] or 0)
        ax.text(x_text * 1.12, yi, f"{r['mae']:.3f}", va="center", ha="left", fontsize=fs.FS_ANNOT, color="0.1")
    ax.set_yticks(y)
    ax.set_yticklabels(["VDC" if r["key"] == "vdc" else fs.STYLE[r["key"]].label.replace(" copula", "")
                        for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    ax.set_xlim(5e-3, 1.2)
    ax.set_xlabel("MI mean absolute error (nats)")
    ax.set_title("Bivariate MI error")


def panel_mi_dim(ax, e6):
    rows = [r for r in e6["mi_summary_records"] if r.get("scenario") == "gaussian_ar1"]
    keys = [("dcd", "vdc"), ("ksg", "ksg"), ("gaussian", "gauss"), ("infonce", "infonce"), ("mine", "mine")]
    dodge = np.linspace(-0.13, 0.13, len(keys))
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
        eb = ax.errorbar(x, mu, yerr=[lo, sd], elinewidth=0.55, capsize=1.2,
                         **fs.line_kw(k, label=lab, linewidth=1.3 if k == "vdc" else 0.9))
        for art in eb[2]:
            art.set_alpha(0.55)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([5, 10, 20, 50])
    ax.set_xticklabels(["5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(5e-4, 0.9)
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("MI absolute error (nats)")
    ax.set_title("MI error vs. dimension")
    ax.legend(loc="upper center", ncol=3, columnspacing=0.5, handlelength=1.6)


def panel_tc_time(ax, tc):
    raw = tc["raw_records"]
    dims = sorted({int(r["d"]) for r in raw})
    for key, k in (("time_dcd_s", "vdc"), ("time_ksg_s", "ksg"), ("time_infonce_s", "infonce"), ("time_mine_s", "mine"),
                   ("time_nwj_s", "nwj")):
        xs, ys = [], []
        for d in dims:
            vals = [float(r[key]) for r in raw if int(r["d"]) == d and r.get(key) is not None
                    and np.isfinite(float(r[key])) and float(r[key]) > 0]
            if vals:
                xs.append(d)
                ys.append(float(np.median(vals)))
        if xs:
            ax.plot(xs, ys, **fs.line_kw(k, label="VDC" if k == "vdc" else fs.STYLE[k].label,
                                         linewidth=1.3 if k == "vdc" else 0.9))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.minorticks_off()
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("TC runtime (s)")
    ax.set_title("Total-correlation runtime")
    ax.legend(loc="upper left", handlelength=1.6)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fs.add_io_args(ap, "fig3_information")
    a = ap.parse_args()
    fs.apply()
    rows = parse_mi_table(fs.input_file(a.inputs, "tables/tab_mi.tex"))
    e6 = json.loads(fs.input_file(a.inputs, "paper_outputs/e6_theory_synthetic_results.json").read_text())
    tc = json.loads(fs.input_file(a.inputs, "paper_outputs/tc_benchmark_neural_full.json").read_text())
    fig = plt.figure(figsize=(fs.TEXTWIDTH, 1.60))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.0], wspace=0.5, left=0.105, right=0.995,
                          top=0.85, bottom=0.21)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_mi_bars(axes[0], rows)
    panel_mi_dim(axes[1], e6)
    panel_tc_time(axes[2], tc)
    fs.panel_labels(fig, axes, "abc", y=0.995)
    fs.save(fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
