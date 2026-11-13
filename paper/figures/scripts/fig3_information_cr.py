#!/usr/bin/env python3
"""Camera-ready Fig. 3: information-estimation results (vector PDF at printed size).

Inputs folder:
  (a) tables/tab_mi.tex                                   bivariate MI MAE, eight cases (no Joe)
  (b) tc_scaling/tc_benchmark.json                        median TC runtime vs d over seeds 7, 17, 42
                                                          (VDC: D-vine tree 1; KSG: full chain rule)
  (c) paper_outputs/mi_self_consistency_scaleup.json      DPI violations and additivity error, 480 trials
Panel (c) is drawn by the panel function of figA_tc_story_cr.py, which also holds the MI-versus-dimension
panel (now in the appendix) and the panels of Fig. 4. Drawn at 5.5 in with the shared vdc_figstyle fonts and
method encodings.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figA_tc_story_cr as tcs  # noqa: E402  (panel (c); importing it restyles matplotlib)
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
    ax.set_xlim(5e-3, 2.5)
    ax.set_xlabel("MI mean absolute error (nats)")
    ax.set_title("Bivariate MI error")


def panel_tc_time(ax, tc):
    raw = [r for recs in tc["per_seed_raw_records"].values() for r in recs]   # three trials per seed and d
    dims = sorted({int(r["d"]) for r in raw})
    # legend rows narrow from top to bottom, so the legend clears the rising baseline curves
    for key, k in (("time_dcd_s", "vdc"), ("time_infonce_s", "infonce"), ("time_ksg_s", "ksg"), ("time_mine_s", "mine"),
                   ("time_nwj_s", "nwj")):
        xs, ys = [], []
        for d in dims:
            vals = [float(r[key]) for r in raw if int(r["d"]) == d and r.get(key) is not None
                    and np.isfinite(float(r[key])) and float(r[key]) > 0]
            if vals:
                xs.append(d)
                ys.append(float(np.median(vals)))
        if xs:
            # the AR(1) TC timing fits a D-vine truncated after its first tree (d - 1 edges)
            ax.plot(xs, ys, **fs.line_kw(k, label={"vdc": "VDC (tree 1)", "ksg": "KSG (full chain rule)"}.get(k, fs.STYLE[k].label),
                                         linewidth=1.3 if k == "vdc" else 0.9))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.minorticks_off()
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("TC runtime (s)")
    ax.set_ylim(7e-3, 2e4)  # headroom above the data holds the legend
    ax.set_title("TC runtime")
    ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.1, labelspacing=0.15)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fs.add_io_args(ap, "fig3_information")
    a = ap.parse_args()
    plt.rcdefaults()  # the panel module's helper import applied its own style; start from defaults
    fs.apply()

    def load(rel):
        return json.loads(fs.input_file(a.inputs, rel).read_text())

    rows = parse_mi_table(fs.input_file(a.inputs, "tables/tab_mi.tex"))
    tc = load("tc_scaling/tc_benchmark.json")
    sc = {str(r["estimator"]).upper(): r for r in load("paper_outputs/mi_self_consistency_scaleup.json")}

    fig = plt.figure(figsize=(fs.TEXTWIDTH, 1.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.15, 0.95], wspace=0.5, left=0.105, right=0.995,
                          top=0.865, bottom=0.2)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    sub = gs[0, 2].subgridspec(2, 1, hspace=1.25)
    ax_c1, ax_c2 = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[1, 0])
    panel_mi_bars(ax_a, rows)
    panel_tc_time(ax_b, tc)
    info_c = tcs.panel_selfcons(ax_c1, ax_c2, sc)
    fs.panel_labels(fig, [ax_a, ax_b, ax_c1], "abc", y=0.995)
    print(f"(c) DPI (violations, trials, base datasets) {info_c['viol']}; Fisher p={info_c['p_base']:.4f}; "
          f"additivity {info_c['add']}")
    fs.save(fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
