#!/usr/bin/env python3
"""Figure 4 v2: density and vine-fitting summary, four panels.

(a) Bivariate accuracy-latency tradeoff on the held-out copula suite (preserved
    from the original Figure 4).
(b) Full-vine NLL vs d on synthetic non-Gaussian Clayton vines.
(c) Vine-only fit time on the same synthetic high-d vines.
(d) Real S&P100 held-out NLL under random and chronological split protocols.

Replaces drafts/figures/fig4_density_runtime_composite.{pdf,png} with the
four-panel version used in the main text.

Sources:
  drafts/tables/tab_bivariate.tex  -- bivariate (a)
  drafts/paper_outputs/e22_highdim_scaling.json  -- uniform Clayton (b)
  drafts/analysis/fast_fit_scaling_20261018/timing_summary.json -- audited timing (c)
  drafts/paper_outputs/e23_sp100_density.json    -- S&P100 random split (d)
  drafts/paper_outputs/e28_sp100_chrono.json     -- S&P100 chronological split (d)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import COLORS, apply_style, save_figure

# CVD-safe color overrides local to this figure (the shared paper_style
# palette is untouched): pyvine-TLL moves from green to blue, and KDE gets
# its own brown so purple stays RealNVP-only.
PAPER_ROOT = Path(__file__).resolve().parents[1]

TLL_BLUE = "#0173b2"
KDE_BROWN = "#8c6d31"


def _load_records(p: str) -> List[Dict]:
    return json.loads((PAPER_ROOT / Path(p).relative_to("drafts")).read_text())["records"]


def _parse_bivariate_table(p: str):
    text = (PAPER_ROOT / Path(p).relative_to("drafts")).read_text()
    rows = []
    for line in text.splitlines():
        if "\\\\" not in line or line.strip().startswith("%") or "midrule" in line or "toprule" in line or "bottomrule" in line:
            continue
        if "Method" in line:
            continue
        cells = [c.strip().replace("\\textbf{", "").replace("}", "") for c in line.split("&")]
        if len(cells) < 5:
            continue
        method = cells[0]
        try:
            ise = float(cells[1])
            time_ms = float(re.sub(r"[^0-9.eE+-]", "", cells[-1]))  # last column (camera-ready table adds KL)
        except ValueError:
            continue
        rows.append({"method": method, "ise": ise, "time_ms": time_ms})
    return rows


METHOD_SPEC = [
    ("VDC", "VDC (Ours)", COLORS["vdc"], "o", "-"),
    ("pyvine_param", "pyvine-param", COLORS["pyvine_par"], "s", "-"),
    ("pyvine_tll", "pyvine-TLL", TLL_BLUE, "^", "--"),
    ("flow_realnvp", "Flow (RealNVP)", COLORS["flow"], "D", "-"),
    ("gaussian_copula", "Gaussian copula", COLORS["gaussian"], "v", "-"),
]

# (id, label, color, marker, linestyle): pyvine-TLL is dashed so the
# VDC/TLL contrast does not rely on hue alone.
NLL_METHOD_SPEC = [
    ("VDC", "VDC", COLORS["vdc"], "o", "-"),
    ("pyvine_tll", "pyvine-TLL", TLL_BLUE, "^", "--"),
    ("flow_realnvp", "RealNVP", COLORS["flow"], "D", "-"),
    ("gaussian_copula", "Gaussian", COLORS["gaussian"], "v", "-"),
]


def _by_d_method(records, methods, d_values):
    out = np.full((len(d_values), len(methods)), np.nan)
    for r in records:
        if r.get("status", "ok") != "ok":
            continue
        d = int(r.get("d") or -1)
        m = r.get("method")
        nll = r.get("nll_bits_per_dim")
        if d in d_values and m in methods and isinstance(nll, (int, float)) and not np.isnan(nll):
            out[d_values.index(d), methods.index(m)] = float(nll)
    return out


def _panel_bivariate(ax, rows):
    bivar_color = {
        "Histogram": COLORS["histogram"],
        "KDE": KDE_BROWN,
        "pyvine-param": COLORS["pyvine_par"],
        "pyvine-TLL": TLL_BLUE,
        "VDC (one-shot)": COLORS["vdc"],
    }
    label_offsets = {
        # Positions for the camera-ready values (correct-sample Table 1): every method is
        # identified by a text label next to its marker, not by colour alone.
        "Histogram": (1.45, 1.00, "left"),
        "KDE": (1.22, 1.70, "left"),
        "pyvine-param": (0.92, 1.80, "right"),
        "pyvine-TLL": (1.00, 1.70, "center"),
        "VDC (one-shot)": (1.00, 0.55, "center"),
    }
    label_lookup = {
        "Histogram": "Hist",
        "KDE": "KDE",
        "pyvine-param": "pyvine-param",
        "pyvine-TLL": "pyvine-TLL",
        "VDC (one-shot)": "VDC",
    }
    for r in rows:
        c = bivar_color.get(r["method"], "0.5")
        is_vdc = "VDC" in r["method"]
        ax.scatter(r["time_ms"], r["ise"], s=70 if is_vdc else 42,
                   color=c, edgecolor="0.15", linewidth=0.7, zorder=3,
                   label=r["method"])
        xmult, ymult, ha = label_offsets.get(r["method"], (1.08, 1.0, "left"))
        ax.text(r["time_ms"] * xmult, r["ise"] * ymult,
                label_lookup.get(r["method"], r["method"]),
                fontsize=6.2, color="0.18", ha=ha, va="center", zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Runtime per pair (ms)")
    ax.set_ylabel("ISE")
    ax.set_title("Edge accuracy–latency")
    ax.set_xlim(1.1, 850)
    ax.set_ylim(1.5e-3, 6.0)


def _panel_scaling(ax, e22_records):
    """Uniform Clayton NLL vs d, with global baselines only as quality references."""
    methods = [m[0] for m in NLL_METHOD_SPEC]
    dims_clayton = [100, 200, 500]
    arr_c = _by_d_method(e22_records, methods, dims_clayton)

    for j, (m_id, m_label, color, marker, ls) in enumerate(NLL_METHOD_SPEC):
        y = arr_c[:, j]
        if np.all(np.isnan(y)):
            continue
        x = [d for k, d in enumerate(dims_clayton) if not np.isnan(y[k])]
        yv = [y[k] for k in range(len(dims_clayton)) if not np.isnan(y[k])]
        ax.plot(x, yv, marker=marker, linestyle=ls, linewidth=1.3, markersize=4.8,
                color=color, label=m_label)

        # Annotate explicit timeout for any method with a missing endpoint at d=500.
        if m_id == "pyvine_tll" and np.isnan(y[-1]) and len(yv) > 0:
            last_x, last_y = x[-1], yv[-1]
            tail_x = dims_clayton[-1]
            ax.scatter([tail_x], [last_y], marker="x", s=42,
                       color=color, linewidths=1.2, zorder=5)
            ax.annotate("timeout", xy=(tail_x, last_y),
                        xytext=(tail_x, last_y + 0.045),
                        ha="center", va="bottom",
                        fontsize=5.6, color=color, style="italic")

    ax.set_xlabel(r"Dimension $d$")
    ax.set_ylabel("Held-out NLL (bits/dim)")
    ax.set_title("Clayton-vine NLL")
    ax.set_xticks([100, 200, 500])
    ax.set_xticklabels(["100", "200", "500"])
    ax.axhline(0, color="0.85", linewidth=0.5, zorder=-1)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left",
              handlelength=1.2, labelspacing=0.25, borderpad=0.1)


def _fit_time_lookup(records):
    lookup = {}
    for r in records:
        if r.get("status", "ok") != "ok":
            continue
        d = int(r.get("d") or -1)
        method = r.get("method")
        fit_time = r.get("fit_s", r.get("fit_time_s"))
        fit_std = r.get("fit_s_std")
        if isinstance(fit_time, (int, float)) and fit_time > 0:
            std = float(fit_std) if isinstance(fit_std, (int, float)) and np.isfinite(fit_std) else 0.0
            lookup[(d, method)] = (float(fit_time), std)
    return lookup


def _panel_vine_time(ax, e22_records, timing_summary):
    """Compare fit time only among vine-copula methods on a shared dimension grid."""
    assert timing_summary['status'] == 'complete_and_exact'
    assert [row['d'] for row in timing_summary['summary']] == [100, 200, 500]
    e22 = {(row['d'], method): (value['mean_seconds'], value['sample_sd_seconds'])
           for row in timing_summary['summary'] for method, value in row['methods'].items()}
    dims_clayton = [100, 200, 500]

    def times(lookup, dims, method):
        vals, stds = [], []
        for d in dims:
            val = lookup.get((d, method))
            vals.append(np.nan if val is None else val[0])
            stds.append(0.0 if val is None else val[1])
        return np.asarray(vals, dtype=float), np.asarray(stds, dtype=float)

    y_vdc, e_vdc = times(e22, dims_clayton, "vdc_fast")
    y_tll, e_tll = times(e22, dims_clayton, "tll_16")

    y_tll8, e_tll8 = times(e22, dims_clayton, "tll_8")

    # Seed-std errorbars are tiny at this scale (the caption notes that the
    # std is smaller than the markers); capsize keeps them visible.
    ax.errorbar(dims_clayton, y_vdc, yerr=e_vdc, color=COLORS["vdc"], marker="o",
                linewidth=1.4, markersize=4.8, capsize=3, elinewidth=0.8,
                label="VDC (H100)")
    ax.errorbar(dims_clayton, y_tll, yerr=e_tll, color=TLL_BLUE, marker="^",
                linestyle="--", linewidth=1.4, markersize=4.8, capsize=3,
                elinewidth=0.8, label="TLL (16 CPU)")
    ax.errorbar(dims_clayton, y_tll8, yerr=e_tll8, color="#555555", marker="s",
                linestyle=":", linewidth=1.4, markersize=4.3, capsize=3,
                elinewidth=0.8, label="TLL (8 CPU)")

    for x, vdc_t, tll_t in zip(dims_clayton, y_vdc, y_tll):
        if np.isfinite(vdc_t) and np.isfinite(tll_t):
            x_text = x + 8 if x == dims_clayton[0] else x
            ha = "left" if x == dims_clayton[0] else "center"
            ax.text(x_text, tll_t * 1.12, f"{tll_t / vdc_t:.1f}x",
                    fontsize=6.0, color=TLL_BLUE,
                    ha=ha, va="bottom")

    # If pyvine-TLL is missing at the largest d (timeout), draw an explicit
    # hollow X at the per-fit budget so the gap is not silently absent.
    pyvine_budget_s = 2400.0
    for k, d in enumerate(dims_clayton):
        if np.isfinite(y_vdc[k]) and not np.isfinite(y_tll[k]):
            ax.scatter([d], [pyvine_budget_s], marker="x", s=46,
                       color=TLL_BLUE, linewidths=1.3, zorder=5)
            ax.annotate(f"timeout (>{pyvine_budget_s:.0f} s)",
                        xy=(d, pyvine_budget_s),
                        xytext=(d, pyvine_budget_s * 1.18),
                        ha="center", va="bottom",
                        fontsize=5.8, color=TLL_BLUE, style="italic")

    ax.set_yscale("log")
    ax.set_xlabel(r"Dimension $d$")
    ax.set_ylabel("Fit time (s)")
    ax.set_title("Vine fitting cost")
    ax.set_xticks([100, 200, 500])
    ax.set_xticklabels(["100", "200", "500"])
    ax.set_ylim(20, 7500)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left",
              handlelength=1.3, labelspacing=0.25, borderpad=0.1)


def _load_sp100_nll(p: str) -> Dict[str, Dict[str, float]]:
    """Return {method: {nll, lo, hi}} from an S&P100 density JSON."""
    out: Dict[str, Dict[str, float]] = {}
    for r in _load_records(p):
        if r.get("status", "ok") != "ok":
            continue
        nll = r.get("nll_bits_per_dim")
        ci = r.get("ci_self") or {}
        if not isinstance(nll, (int, float)):
            continue
        out[r["method"]] = {
            "nll": float(nll),
            "lo": float(ci.get("lo", nll)),
            "hi": float(ci.get("hi", nll)),
        }
    return out


def _panel_sp100(ax, random_nll, chrono_nll):
    """Grouped bars: held-out NLL on real S&P100 under two split protocols.

    Bars hang downward from zero (all NLLs are negative; lower is better),
    with 95% bootstrap CI whiskers from ci_self.
    """
    groups = [("Random", random_nll), ("Chronological", chrono_nll)]
    group_x = np.arange(len(groups), dtype=float)
    n_m = len(NLL_METHOD_SPEC)
    width = 0.19

    for j, (m_id, m_label, color, _marker, _ls) in enumerate(NLL_METHOD_SPEC):
        xs, vals, err_lo, err_hi = [], [], [], []
        for k, (_gname, lookup) in enumerate(groups):
            rec = lookup.get(m_id)
            if rec is None:
                continue
            xs.append(group_x[k] + (j - (n_m - 1) / 2) * width)
            vals.append(rec["nll"])
            err_lo.append(max(rec["nll"] - rec["lo"], 0.0))
            err_hi.append(max(rec["hi"] - rec["nll"], 0.0))
        ax.bar(xs, vals, width=width * 0.9, color=color,
               edgecolor="0.15", linewidth=0.5, hatch=["///", "...", "xx", "---"][j], label=m_label, zorder=3)
        ax.errorbar(xs, vals, yerr=[err_lo, err_hi], fmt="none",
                    ecolor="0.25", elinewidth=0.7, capsize=1.6,
                    capthick=0.7, zorder=4)

    # Emphasized zero baseline: bars hang below it, so lower = better.
    ax.axhline(0, color="0.25", linewidth=0.8, zorder=2)
    ax.annotate("", xy=(0.5, -0.30), xytext=(0.5, -0.13),
                arrowprops=dict(arrowstyle="-|>", color="0.35",
                                linewidth=0.9, shrinkA=0, shrinkB=0))
    ax.text(0.5, -0.10, "better", fontsize=6.2, color="0.35",
            ha="center", va="bottom", style="italic")

    ax.set_xticks(group_x)
    ax.set_xticklabels([g for g, _ in groups])
    ax.set_xlim(-0.55, group_x[-1] + 0.55)
    ax.set_ylim(-0.78, 0.02)
    ax.set_ylabel("Held-out NLL (bits/dim)")
    ax.set_title("S&P100 splits")
    ax.legend(fontsize=5.8, frameon=False, loc="lower right",
              handlelength=1.0, labelspacing=0.25, borderpad=0.1,
              handletextpad=0.4)


def main():
    global PAPER_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, default=PAPER_ROOT)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--timing-summary", type=Path,
        default=PAPER_ROOT / "analysis/fast_fit_scaling_20261018/timing_summary.json")
    args = parser.parse_args()
    PAPER_ROOT = args.paper_root
    apply_style(font_size=8, column_width=3.25, text_width=6.75)

    bivar_rows = _parse_bivariate_table("drafts/tables/tab_bivariate.tex")
    e22 = _load_records("drafts/paper_outputs/e22_highdim_scaling.json")
    sp100_random = _load_sp100_nll("drafts/paper_outputs/e23_sp100_density.json")
    sp100_chrono = _load_sp100_nll("drafts/paper_outputs/e28_sp100_chrono.json")

    # 1x4 row: saves ~1.5in of page height versus the 2x2 grid. Panel titles
    # are shortened to fit the ~1.7in panel width; annotations sized to match.
    fig = plt.figure(figsize=(7.35, 2.60))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.06],
                          wspace=0.42,
                          left=0.065, right=0.99, top=0.885, bottom=0.185)
    axes = [fig.add_subplot(gs[0, j]) for j in range(4)]

    _panel_bivariate(axes[0], bivar_rows)
    _panel_scaling(axes[1], e22)
    _panel_vine_time(axes[2], e22, json.loads(args.timing_summary.read_text()))
    _panel_sp100(axes[3], sp100_random, sp100_chrono)

    # Keep panel letters separate from titles.  All four share one baseline
    # and sit just left of their y-axis labels.
    for ax, label in zip(axes, ["a", "b", "c", "d"]):
        ax.text(
            -0.28,
            1.12,
            f"({label})",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="right",
            clip_on=False,
        )

    save_figure(fig, "fig4_density_runtime_composite", figures_dir=args.figures_dir)


if __name__ == "__main__":
    main()
