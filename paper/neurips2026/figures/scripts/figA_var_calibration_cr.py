#!/usr/bin/env python3
"""Camera-ready appendix figure: VaR calibration on S&P100 daily returns (vector PDF at printed size).

Data: var_backtest/e3_var_results.json in the inputs folder, the merged camera-ready backtest (four replay chunks,
1442 days, 289 refits; its source_sha256 in the inputs manifest, af53cfc6..., is the merged_sha256 of the
backtest receipt). Same numbers as the earlier fig_var_calibration
(scripts/paper_artifacts.py:generate_fig_var_calibration), redrawn with the shared vdc_figstyle encodings:
  (a) 1% VaR, (b) 5% VaR; observed violation rate per method with 95% Wilson score intervals,
  the nominal rate (dashed), and the central 95% range of the observed rate under correct
  calibration, Binomial(n, alpha) / n (shaded). Methods sit on a categorical axis, so no point
  is offset along a rate axis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vdc_figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


# (results-file method name, style); historical simulation appears only here, so it gets its own shape.
METHODS = [("VDC (Ours)", fs.STYLE["vdc"]),
           ("Gaussian", fs.STYLE["gauss"]),
           ("Historical", fs.MethodStyle("Historical simulation", "#333333", "*", "-", ""))]
TICKS = {"VDC (Ours)": "VDC", "Gaussian": "Gaussian copula", "Historical": "Historical simulation"}


def wilson(k: int, n: int, z: float = 1.959963984540054):
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = (z / den) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(centre - half, 0.0), min(centre + half, 1.0)


def check_against_paper(rows):
    """Numbers quoted in Appendix app:var and Section 5 must come from this file."""
    vdc1, vdc5, g1 = rows["VDC (Ours)", 0.01], rows["VDC (Ours)", 0.05], rows["Gaussian", 0.01]
    assert all(r["n"] == 1442 for r in rows.values())
    assert vdc1["violations"] == 27 and round(100 * 27 / 1442, 1) == 1.9 and round(vdc1["kupiec_p"], 3) == 0.003
    assert round(100 * g1["violations"] / g1["n"], 1) == 1.9
    assert round(100 * vdc5["violations"] / vdc5["n"], 1) == 5.4
    assert round(vdc5["kupiec_p"], 2) == 0.48 and round(vdc5["christoffersen_p"], 3) == 0.003


def panel(ax, rows, alpha, labels: bool):
    """One forest plot: a row per method with the observed rate, its 95% Wilson interval and the count."""
    n = 1442
    lo_k, hi_k = binom.ppf(0.025, n, alpha), binom.ppf(0.975, n, alpha)
    ys = np.arange(len(METHODS))[::-1]  # first method on top
    ax.axvspan(100 * lo_k / n, 100 * hi_k / n, color="0.88", linewidth=0, zorder=0)
    ax.axvline(100 * alpha, color="0.1", linestyle="--", linewidth=0.8, zorder=1)
    for yi, (name, st) in zip(ys, METHODS):
        r = rows[name, alpha]
        k = r["violations"]
        rate = 100 * k / n
        lo, hi = (100 * v for v in wilson(k, n))
        # inside the shaded range <=> not rejected at 5% by the Kupiec test, for all six points here
        assert (lo_k <= k <= hi_k) == (r["kupiec_p"] >= 0.05), (name, alpha)
        ax.errorbar(rate, yi, xerr=[[rate - lo], [hi - rate]], fmt="none", ecolor=st.color, elinewidth=0.9,
                    capsize=2.0, capthick=0.9, zorder=3)
        ax.plot(rate, yi, linestyle="none", marker=st.marker, color=st.color, markeredgecolor="white",
                markeredgewidth=0.4, markersize=6.4 if st.marker == "*" else 4.8, zorder=4)
        ax.text(rate, yi + 0.2, f"{k}", ha="center", va="bottom", fontsize=fs.FS_ANNOT, color="0.15")
    ax.set_yticks(ys)
    ax.set_yticklabels([TICKS[name] for name, _ in METHODS] if labels else [])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.55, len(METHODS) - 0.25)
    ax.set_xlabel("Violation rate (%)")
    ax.set_title(f"{100 * alpha:.0f}% VaR")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fs.add_io_args(ap, "figA_var_calibration")
    ap.add_argument("--data", type=Path, help="use this e3_var_results.json instead of the saved one")
    a = ap.parse_args()
    data = json.loads(fs.input_file(a.inputs, "var_backtest/e3_var_results.json", a.data).read_bytes())
    assert data["coverage_verified_days"] == 1442 and data["n_refits"] == 289
    rows = {(r["method"], r["alpha"]): r for r in data["rows"]}
    check_against_paper(rows)

    fs.apply()
    fig = plt.figure(figsize=(4.4, 1.55))
    gs = fig.add_gridspec(1, 2, wspace=0.12, left=0.235, right=0.975, top=0.85, bottom=0.355)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    panel(ax1, rows, 0.01, labels=True)
    panel(ax2, rows, 0.05, labels=False)
    ax1.set_xlim(0.4, 3.0)
    ax1.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax2.set_xlim(3.2, 7.0)
    ax2.set_xticks([4, 5, 6, 7])
    handles = [Patch(facecolor="0.88", edgecolor="none", label="95% range if calibrated"),
               Line2D([], [], color="0.1", linestyle="--", linewidth=0.8, label="nominal rate")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.605, 0.0), ncol=2, handlelength=1.8,
               columnspacing=1.6)
    fs.panel_labels(fig, [ax1, ax2], "ab", y=0.995)
    fs.save(fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
