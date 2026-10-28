#!/usr/bin/env python3
"""Camera-ready appendix figure: non-simplified conditional pair-copula stress tests.

Redraws the figure of e18_nonsimplified_stress.py from its saved output only, paper_outputs/
e18_nonsimplified_stress.json in the inputs folder (one seed; n_train=2000, n_test=10000); nothing is refitted.
  (a) excess test NLL over the true conditional oracle (nats per test pair) of the five simplified
      pooled estimators, one sub-axes per case;
  (b) empirical Kendall tau of (U2, U3) within each U1 regime and pooled, Gaussian sign flip;
  (c) empirical tail concentrations lambda_L(q), lambda_U(q) per regime, Clayton/Gumbel tail switch,
      at q = 0.1 (the default of e18_nonsimplified_stress._tail_concentration, which the run used).
Estimators keep their vdc_figstyle colour and hatch and are named on the axis; panels (b)-(c) show the
data, not methods, and use greys, hatches and direct labels. Plotted values are checked against
tables/tab_nonsimplified_stress.tex.
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

CASES = [("gaussian_sign_flip", "Sign flip"), ("clayton_gumbel_tail_switch", "Tail switch")]
# simplified pooled estimators, top to bottom (the ranking is the same in both cases)
POOLED = [("VDC", "vdc"), ("pyvine-param.", "param"), ("pyvine-TLL", "tll"), ("Gaussian", "gauss"),
          ("Histogram", "hist")]


def num(v, digits=3):
    return f"{v:.{digits}f}".replace("-", "−")


def table_cells(path):
    """(test NLL, excess NLL) strings per (case, estimator) in the LaTeX table."""
    out, case = {}, None
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("%"):
            continue
        if "Gaussian sign flip" in line:
            case = "gaussian_sign_flip"
            continue
        if "tail switch" in line:
            case = "clayton_gumbel_tail_switch"
            continue
        cells = [c.strip() for c in line.strip().rstrip("\\").split("&")]
        if case and len(cells) == 5:
            out[(case, cells[0])] = (cells[2], cells[3])
    return out


def check(data, table):
    rows = {(r["case_key"], r["method"]): r for r in data["rows"]}
    assert len(rows) == len(table) == 14, (len(rows), len(table))
    for key, (nll, excess) in table.items():
        r = rows[key]
        assert (f"{r['test_nll']:.3f}", f"{r['excess_nll']:.3f}") == (nll, excess), (key, nll, excess)
    for case, _ in CASES:
        pooled = {m: r["excess_nll"] for (c, m), r in rows.items() if c == case and r["uses_u1"] == "no"}
        assert set(pooled) == {m for m, _ in POOLED}
        assert sorted(pooled, key=pooled.get) == [m for m, _ in POOLED], (case, pooled)  # VDC best
    assert rows[("clayton_gumbel_tail_switch", "VDC")]["excess_nll"] < \
        rows[("clayton_gumbel_tail_switch", "Regime Gaussian")]["excess_nll"]
    assert not any(k in r for r in data["rows"] for k in ("excess_nll_seeds", "excess_nll_sem", "excess_nll_std"))
    g, t = data["diagnostics"]["gaussian_sign_flip"], data["diagnostics"]["clayton_gumbel_tail_switch"]
    assert g["tau_low_empirical"] < -0.5 < 0.5 < g["tau_high_empirical"] and abs(g["tau_pooled_empirical"]) < 0.02
    assert abs(t["tau_low_empirical"] - 0.6) < 0.01 and abs(t["tau_high_empirical"] - 0.6) < 0.01
    assert t["tail_low_lower"] > t["tail_low_upper"] and t["tail_high_upper"] > t["tail_high_lower"]
    return rows


def panel_excess(axs, rows):
    y = np.arange(len(POOLED))[::-1]
    for ax, (case, title) in zip(axs, CASES):
        for yi, (name, key) in zip(y, POOLED):
            v = rows[(case, name)]["excess_nll"]
            ax.barh(yi, v, height=0.68, zorder=3, **fs.bar_kw(key))
            ax.text(v + 0.014, yi, num(v), va="center", ha="left", fontsize=fs.FS_ANNOT, color="0.1")
        ax.set_xlim(0, 0.57)
        ax.set_xticks([0, 0.2, 0.4])
        ax.set_xticklabels(["0", "0.2", "0.4"])
        ax.set_ylim(-0.6, len(POOLED) - 0.4)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([name for name, _ in POOLED])
    axs[1].tick_params(axis="y", labelleft=False)


def panel_tau(ax, g):
    vals = [g["tau_low_empirical"], g["tau_pooled_empirical"], g["tau_high_empirical"]]
    ax.axhline(0, color="0.1", linewidth=0.6, zorder=2)
    ax.bar([0, 2], [vals[0], vals[2]], width=0.62, color="0.5", edgecolor="0.15", linewidth=0.5, zorder=3)
    ax.plot([1], [vals[1]], linestyle="none", marker="D", markersize=3.6, color="white",
            markeredgecolor="0.1", markeredgewidth=0.7, zorder=4)
    for x, v in ((0, vals[0]), (2, vals[2])):
        ax.text(x, v + (0.07 if v >= 0 else -0.07), num(v), ha="center", va="bottom" if v >= 0 else "top",
                fontsize=fs.FS_ANNOT, color="0.1")
    ax.text(1, vals[1] + 0.09, f"pooled\n{num(vals[1])}", ha="center", va="bottom", fontsize=fs.FS_ANNOT,
            color="0.1", linespacing=1.1)
    ax.set_xlim(-0.55, 2.55)
    ax.set_xticks([0, 2])
    ax.set_xticklabels(["$U_1<0.5$", "$U_1\\geq0.5$"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(-0.82, 0.82)
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_yticklabels(["\u22120.5", "0", "0.5"])
    ax.set_ylabel("Kendall $\\tau$ of $(U_2,U_3)$")
    ax.set_title("Sign flip")


def panel_tails(ax, t):
    x = np.arange(2.0)
    off, w = 0.2, 0.36
    lam = {"L": [t["tail_low_lower"], t["tail_high_lower"]], "U": [t["tail_low_upper"], t["tail_high_upper"]]}
    ax.bar(x - off, lam["L"], width=w, color="0.35", edgecolor="0.15", linewidth=0.5, zorder=3)
    ax.bar(x + off, lam["U"], width=w, color="white", edgecolor="0.15", hatch="xxxx", linewidth=0.5, zorder=3)
    for side, dx in (("L", -off), ("U", off)):
        for xi, v in zip(x + dx, lam[side]):
            ax.text(xi, v + 0.025, f"$\\lambda_{side}$", ha="center", va="bottom", fontsize=fs.FS_ANNOT, color="0.1")
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks(x)
    ax.set_xticklabels(["$U_1<0.5$\nClayton", "$U_1\\geq0.5$\nGumbel"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 0.92)
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75"])
    ax.set_ylabel("Tail concentration")
    ax.set_title("Tail switch")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fs.add_io_args(ap, "figA_nonsimplified_stress")
    ap.add_argument("--data", type=Path, help="use this e18 output instead of the saved one")
    ap.add_argument("--table", type=Path, help="check against this table instead of the saved one")
    a = ap.parse_args()
    data = json.loads(fs.input_file(a.inputs, "paper_outputs/e18_nonsimplified_stress.json", a.data).read_text())
    rows = check(data, table_cells(fs.input_file(a.inputs, "tables/tab_nonsimplified_stress.tex", a.table)))
    fs.apply()

    fig = plt.figure(figsize=(fs.TEXTWIDTH, 1.8))
    outer = fig.add_gridspec(1, 3, width_ratios=[2.15, 1.0, 1.0], wspace=0.42, left=0.128, right=0.995,
                             top=0.87, bottom=0.2)
    inner = outer[0, 0].subgridspec(1, 2, wspace=0.09)
    ax_a = [fig.add_subplot(inner[0, 0])]
    ax_a.append(fig.add_subplot(inner[0, 1], sharey=ax_a[0]))
    ax_b, ax_c = fig.add_subplot(outer[0, 1]), fig.add_subplot(outer[0, 2])
    panel_excess(ax_a, rows)
    panel_tau(ax_b, data["diagnostics"]["gaussian_sign_flip"])
    panel_tails(ax_c, data["diagnostics"]["clayton_gumbel_tail_switch"])
    # one x label centred under both sub-axes of (a)
    p0, p1 = ax_a[0].get_position(), ax_a[1].get_position()
    ax_a[0].set_xlabel("Excess test NLL over the oracle (nats)")
    ax_a[0].xaxis.set_label_coords(((p0.x0 + p1.x1) / 2 - p0.x0) / p0.width, -0.15)
    fs.panel_labels(fig, [ax_a[0], ax_b, ax_c], "abc", y=0.995)
    fs.save(fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
