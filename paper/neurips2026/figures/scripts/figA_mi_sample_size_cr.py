#!/usr/bin/env python3
"""Camera-ready appendix figure: fixed-checkpoint MI sample-size sensitivity (E12, Clayton chain, d=50).

Replaces figA2_mi_sensitivity_composite.py. Reads only the merged E12 artifact (inputs folder)
paper_outputs/e12_block_mi_sample_size_clayton_chain_results_with_neural_seed42.json (VDC, KSG and Gaussian from
..._results_seed42.json, MINE and InfoNCE from ..._neural_results_seed42.json). For every estimator it plots the
mean over the three trials at each sample size of
  (a) the relative MI error 100 |I_hat - I| / I against the artifact's true block MI (0.640 nats), and
  (b) the wall-clock runtime.
Shaded bands show the range (minimum to maximum) over the three trials, which stays positive on the log axes, so
nothing is clipped or floored. ``--band sd`` draws mean +- 1 s.d. (ddof=1) instead; where mean - s.d. <= 0 the band
is cut at the lower axis limit and the affected points are printed. Drawn at 5.5 in with the shared vdc_figstyle
fonts and method encodings (same keys and order as Fig. 3(b)).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vdc_figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullFormatter  # noqa: E402

DATA = "paper_outputs/e12_block_mi_sample_size_clayton_chain_results_with_neural_seed42.json"
KEYS = [("dcd", "vdc"), ("ksg", "ksg"), ("gaussian", "gauss"), ("infonce", "infonce"), ("mine", "mine")]
YLIM = {"range": {"err": (0.1, 1000.0), "time": (1e-4, 4e3)},
        "sd": {"err": (1.0, 1000.0), "time": (1e-4, 4e3)}}


def collect(obj):
    """Per method: sample sizes, per-trial relative errors (%) and per-trial runtimes (s)."""
    true_mi = float(obj["true_mi"])
    series = {}
    for row in obj["results"]:
        assert abs(float(row["true_mi"]) - true_mi) < 1e-12, "rows use different true MI values"
        for m, s in row["methods"].items():
            est = np.asarray(s["estimates"], float)
            series.setdefault(m, []).append((int(row["n_samples"]), 100.0 * np.abs(est - true_mi) / true_mi,
                                             np.asarray(s["times_s"], float)))
    return true_mi, {m: sorted(rows, key=lambda r: r[0]) for m, rows in series.items()}


def label(k):
    return "VDC" if k == "vdc" else fs.STYLE[k].label.replace(" copula", "")


def panel(ax, series, which, band, ylim):
    idx = 1 if which == "err" else 2
    for m, k in KEYS:
        rows = series[m]
        n = np.array([r[0] for r in rows], float)
        vals = [r[idx] for r in rows]
        mu = np.array([v.mean() for v in vals])
        if band == "range":
            lo, hi = np.array([v.min() for v in vals]), np.array([v.max() for v in vals])
        else:
            sd = np.array([v.std(ddof=1) for v in vals])
            lo, hi = mu - sd, mu + sd
            for ni, l in zip(n[lo <= ylim[0]], lo[lo <= ylim[0]]):
                print(f"  {which}: {label(k)} band cut at the axis at n={int(ni)} (mean - s.d. = {l:.3g})")
            lo = np.where(lo > ylim[0], lo, ylim[0] * 0.5)
        assert np.all(lo > 0) and np.all(hi < ylim[1]) and (band == "sd" or np.all(lo > ylim[0])), (k, which)
        s = fs.STYLE[k]
        ax.fill_between(n, lo, hi, color=s.color, alpha=0.17, linewidth=0, zorder=1)
        ax.plot(n, mu, zorder=4 if k == "vdc" else 3,
                **fs.line_kw(k, label=label(k), linewidth=1.3 if k == "vdc" else 0.9))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(75, 1.35e5)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(FixedLocator([1e2, 1e3, 1e4, 1e5]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(FixedLocator([3e2, 3e3, 3e4]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Sample size $n$")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fs.add_io_args(ap, "figA_mi_sample_size")
    ap.add_argument("--data", type=Path, help="use this E12 artifact instead of the saved one")
    ap.add_argument("--band", choices=["range", "sd"], default="range")
    a = ap.parse_args()
    fs.apply()
    obj = json.loads(fs.input_file(a.inputs, DATA, a.data).read_text())
    true_mi, series = collect(obj)
    assert {len(r[1]) for rows in series.values() for r in rows} == {3}, "expected three trials per point"

    fig = plt.figure(figsize=(fs.TEXTWIDTH, 1.9))
    gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.085, right=0.99, top=0.805, bottom=0.185)
    ax_e, ax_t = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ylim = YLIM[a.band]
    panel(ax_e, series, "err", a.band, ylim["err"])
    panel(ax_t, series, "time", a.band, ylim["time"])

    ax_e.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100, 1000][0 if a.band == "range" else 1:]))
    ax_e.set_yticklabels(["0.1", "1", "10", "100", "1000"][0 if a.band == "range" else 1:])
    ax_e.yaxis.set_minor_formatter(NullFormatter())
    ax_e.set_ylabel("Relative MI error (%)")
    ax_e.set_title("MI error vs. sample size")
    ax_e.text(0.985, 0.97, f"Clayton chain, $d{{=}}50$\ntrue block MI ${true_mi:.3f}$ nats", transform=ax_e.transAxes,
              ha="right", va="top", fontsize=fs.FS_ANNOT, color="0.3", linespacing=1.15)

    ax_t.yaxis.set_major_locator(FixedLocator([1e-4, 1e-2, 1e0, 1e2]))
    ax_t.yaxis.set_major_formatter(LogFormatterMathtext())
    ax_t.yaxis.set_minor_locator(FixedLocator([1e-3, 1e-1, 1e1, 1e3]))
    ax_t.yaxis.set_minor_formatter(NullFormatter())
    ax_t.set_ylabel("Runtime (s)")
    ax_t.set_title("Runtime vs. sample size")

    handles, labels = ax_e.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(labels),
               handlelength=2.6, columnspacing=1.6)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    y_title = max(ax.title.get_window_extent(r).transformed(fig.transFigure.inverted()).y1 for ax in (ax_e, ax_t))
    fs.panel_labels(fig, [ax_e, ax_t], "ab", y=y_title + 0.004)
    fs.save(fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
