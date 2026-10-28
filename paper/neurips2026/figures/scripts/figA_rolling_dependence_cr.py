#!/usr/bin/env python3
"""Appendix figure: rolling S&P100 dependence dashboard (camera-ready style).

Same data and quantities as scripts/fig_rolling_dependence_dashboard.py run with --tree-mode fraction
--exclude-dual-share (the earlier figures/fig_rolling_dependence_dashboard_full_frac_cross.pdf): E16 refits of
the deployed VDC checkpoint on 289 rolling 252-day windows of S&P100 daily log-returns, refit every 5 trading
days. Inputs folder: paper_outputs/e16_rolling_dependence_compact.json (the fields of
e16_rolling_dependence_full.json this figure reads, written by extract_rolling_dependence_inputs.py) and
sp100/sp100_returns_meta.json (symbols, first and last dates, row count). Each window's vine is evaluated on its own
pseudo-observations: (a) mean log-copula density, (b) share of it in each tree band, (c) mean tree-1 edge
contributions of the five cross-name pairs that most often rank among the recorded top-5 tree-1 edges
(GOOG-GOOGL, a same-issuer dual share, is excluded). Only the top-5 tree-1 edges are stored per window, so
series in (c) have gaps where a pair fell outside the top five.

x-axis: the date of the last return in each window. The returns matrix stores no per-row dates; its metadata
gives the first and last dates and the row count. With --xaxis date (default) the script maps rows to
consecutive NYSE trading days and checks that the NYSE calendar has exactly as many trading days in that
range as there are rows. Given the returns matrix (--returns, not distributed), it also checks that known
extreme market days (2020-03-12, 2020-03-16, 2020-03-24, 2022-11-10, 2025-04-09) are among the 16 largest
cross-sectional mean moves; the dates, and so the figure, are the same with or without this check. It stops
if a check fails; --xaxis window plots against the window index instead.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vdc_figstyle as fs  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DUAL_SHARE = {frozenset({"GOOG", "GOOGL"})}
BANDS = [(0, 3), (3, 10), (10, None)]  # tree bands, as in the earlier figure
ANCHORS = ["2020-03-12", "2020-03-16", "2020-03-24", "2022-11-10", "2025-04-09"]

# (c): one entry per series slot, in persistence order. Colours pass the dataviz palette validator
# (all-pairs CVD dE >= 9.6); the two light ones are carried by markers, line styles and direct labels.
SERIES_STYLE = [
    ("#0072B2", "-", "o"),
    ("#D55E00", (0, (4, 1.5)), "s"),
    ("#882255", (0, (4, 1.2, 1, 1.2)), "^"),
    ("#56B4E9", (0, (7, 2)), "D"),
    ("#CC79A7", (0, (1, 1.1)), "v"),
]
BAND_STYLE = [("#9ECAE1", "////"), ("#C6DBEF", "\\\\\\\\"), ("#EAF2FA", "xxx")]
# Pairs whose series ends before the last window: (anchor, dx, dy, ha) with the anchor "end" (last window) or
# "max" (highest window), offsets in points; chosen so the labels sit in free space.
END_LABEL = {"BAC\u2013JPM": ("max", 6, 0, "left"), "XOM\u2013CVX": ("end", -7, -9, "right")}
BOX = dict(boxstyle="round,pad=0.18,rounding_size=0.12", facecolor="white", edgecolor="none", alpha=0.92)
HALO = [pe.withStroke(linewidth=1.8, foreground="white")]


def nyse_trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """NYSE trading days in [start, end] for 2019-2026 (full-day closures only)."""
    def nth(y, m, wd, k):
        d = dt.date(y, m, 1)
        d += dt.timedelta((wd - d.weekday()) % 7)
        return d + dt.timedelta(7 * (k - 1))

    def last(y, m, wd):
        d = (dt.date(y, m + 1, 1) if m < 12 else dt.date(y + 1, 1, 1)) - dt.timedelta(1)
        return d - dt.timedelta((d.weekday() - wd) % 7)

    def easter(y):
        a, b, c = y % 19, y // 100, y % 100
        h = (19 * a + b - b // 4 - (b - (b + 8) // 25 + 1) // 3 + 15) % 30
        l = (32 + 2 * (b % 4) + 2 * (c // 4) - h - c % 4) % 7
        m = (a + 11 * h + 22 * l) // 451
        return dt.date(y, (h + l - 7 * m + 114) // 31, (h + l - 7 * m + 114) % 31 + 1)

    def observed(d, new_year=False):
        if d.weekday() == 5:
            return None if new_year else d - dt.timedelta(1)  # Saturday New Year's Day is not observed
        return d + dt.timedelta(1) if d.weekday() == 6 else d

    hol = {dt.date(2025, 1, 9)}  # national day of mourning for President Carter
    for y in range(start.year, end.year + 1):
        assert 2019 <= y <= 2026, "holiday rules are only written out for 2019-2026"
        hol |= {observed(dt.date(y, 1, 1), True), nth(y, 1, 0, 3), nth(y, 2, 0, 3), easter(y) - dt.timedelta(2),
                last(y, 5, 0), observed(dt.date(y, 7, 4)), nth(y, 9, 0, 1), nth(y, 11, 3, 4), observed(dt.date(y, 12, 25))}
        if y >= 2022:
            hol.add(observed(dt.date(y, 6, 19)))  # Juneteenth, observed by NYSE from 2022
    n = (end - start).days + 1
    return [d for d in (start + dt.timedelta(i) for i in range(n)) if d.weekday() < 5 and d not in hol]


def row_dates(meta: dict, returns_path: Path | None = None) -> np.ndarray:
    days = nyse_trading_days(dt.date.fromisoformat(meta["start_date"]), dt.date.fromisoformat(meta["end_date"]))
    n_rows = int(meta["shape"][0])
    if len(days) != n_rows:
        raise SystemExit(f"{len(days)} NYSE trading days but {n_rows} rows; use --xaxis window")
    if returns_path is None:
        print(f"date check: {n_rows} rows = NYSE trading days {days[0]}..{days[-1]} (anchor check skipped: no --returns)")
        return np.array(days, dtype="datetime64[D]")
    r = np.load(returns_path)
    if r.shape[0] != n_rows:
        raise SystemExit(f"returns have {r.shape[0]} rows, metadata says {n_rows}")
    top = {days[i].isoformat() for i in np.argsort(-np.abs(r.mean(axis=1)))[:16]}
    missing = [a for a in ANCHORS if a not in top]
    if missing:
        raise SystemExit(f"anchor days {missing} are not among the largest mean moves; use --xaxis window")
    print(f"date check: {n_rows} rows = NYSE trading days {days[0]}..{days[-1]}; anchors {ANCHORS} matched")
    return np.array(days, dtype="datetime64[D]")


def load(json_path: Path, meta: dict):
    obj = json.loads(json_path.read_text())
    win = obj["windows"]
    symbols = [str(s).replace(".us", "").upper() for s in meta["symbols"]]
    total = np.array([float(w["mean_total_log_copula"]) for w in win])
    # (trees, windows), C-ordered so band sums round exactly as in the earlier figure script
    trees = np.ascontiguousarray(np.array([[float(v) for v in w["per_tree_mean"]] for w in win]).T)
    shares = [100.0 * trees[lo:hi].sum(axis=0) / np.maximum(np.abs(total), 1e-12) for lo, hi in BANDS]
    band_names = [f"Trees {lo + 1}–{hi if hi else trees.shape[0]}" for lo, hi in BANDS]
    series: dict[str, np.ndarray] = {}
    for k, w in enumerate(win):
        for rec in w["top_tree1_edges"]:
            i, j = sorted((int(rec["edge"]["i"]), int(rec["edge"]["j"])))
            if frozenset({symbols[i], symbols[j]}) in DUAL_SHARE:
                continue
            series.setdefault(f"{symbols[i]}–{symbols[j]}", np.full(len(win), np.nan))[k] = float(rec["mean_log_contribution"])
    ranked = sorted(series, key=lambda s: (int(np.isfinite(series[s]).sum()), float(np.nanmean(series[s]))), reverse=True)
    return obj, total, shares, band_names, [(s, series[s]) for s in ranked[:5]]


def sparse_markers(y: np.ndarray, every: int, phase: int) -> np.ndarray:
    """Marker positions: every ``every`` windows (offset by ``phase``), every isolated point and the last point."""
    ok = np.isfinite(y)
    idx = np.arange(len(y))
    keep = ok & ((idx - phase) % every == 0)
    prev = np.r_[False, ok[:-1]]
    nxt = np.r_[ok[1:], False]
    keep |= ok & ~prev & ~nxt
    keep[np.flatnonzero(ok)[-1]] = True
    return np.flatnonzero(keep)


def spread(ys: list[float], gap: float) -> list[float]:
    """Shift label heights apart by at least ``gap`` while keeping their order."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    shift = (np.mean(ys) - np.mean(out))
    return list(out + shift)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fs.add_io_args(ap, "figA_rolling_dependence")
    ap.add_argument("--json", type=Path, help="use this E16 output (full or compact) instead of the saved extract")
    ap.add_argument("--meta", type=Path, help="use this sp100_returns_meta.json instead of the saved one")
    ap.add_argument("--returns", type=Path, help="sp100_returns.npy, for the optional anchor-day check")
    ap.add_argument("--xaxis", choices=["date", "window"], default="date")
    a = ap.parse_args()
    fs.apply()
    meta = json.loads(fs.input_file(a.inputs, "sp100/sp100_returns_meta.json", a.meta).read_text())
    obj, total, shares, band_names, top = load(fs.input_file(a.inputs, "paper_outputs/e16_rolling_dependence_compact.json",
                                                             a.json), meta)
    n = len(total)
    if a.xaxis == "date":
        dates = row_dates(meta, a.returns)
        day = np.array([w["day_index"] for w in obj["windows"]])
        x = dates[day + int(obj["window"]) - 1]  # last return in each window
    else:
        x = np.arange(n)

    W, H = fs.TEXTWIDTH, 2.5
    fig = plt.figure(figsize=(W, H))

    def axes_in(x0, y0, w, h, **kw):
        return fig.add_axes([x0 / W, y0 / H, w / W, h / H], **kw)

    ax_a = axes_in(0.50, 1.47, 1.90, 0.80)
    ax_b = axes_in(0.50, 0.40, 1.90, 0.80, sharex=ax_a)
    ax_c = axes_in(2.96, 0.40, 1.88, 1.87, sharex=ax_a)

    # (a) total
    ax_a.grid(axis="y", color="0.88", linewidth=0.35, linestyle="-")
    ax_a.set_axisbelow(True)
    ax_a.plot(x, total, color="0.12", linewidth=1.0)
    ax_a.set_ylabel("Mean log-copula\ndensity (nats)")
    ax_a.set_title("Total dependence per window")
    ax_a.tick_params(labelbottom=False)
    ax_a.set_ylim(40, 92)
    ax_a.set_yticks([50, 70, 90])

    # (b) tree-band shares: light sequential fills, dark hatches, white separators, no grid
    cum = np.zeros(n)
    for (fill, hatch), s in zip(BAND_STYLE, shares):
        ax_b.fill_between(x, cum, cum + s, facecolor=fill, hatch=hatch, edgecolor="0.3", linewidth=0.0)
        cum = cum + s
        ax_b.plot(x, cum, color="white", linewidth=0.8)
    ax_b.set_ylim(0, 100)
    ax_b.set_yticks([0, 50, 100])
    ax_b.set_ylabel("Share of\ntotal (%)")
    ax_b.set_title("Share of each tree band")
    hi = np.cumsum(shares, axis=0)
    mid = hi - np.asarray(shares) / 2
    ax_b.text(x[n // 2], mid[0, n // 2], band_names[0], ha="center", va="center", fontsize=fs.FS_ANNOT, color="0.1", bbox=BOX)
    # the thin middle band is labelled in the band above, with a leader to where it is widest under a label-wide
    # stretch (~71 windows); the top band's own label goes where that band is widest, away from this one
    span = 71

    def widest(share, away_from=None):
        m = np.lib.stride_tricks.sliding_window_view(share, span).min(axis=1)  # m[i]: windows i..i+span-1
        centre = np.arange(len(m)) + span // 2
        m[(centre < 45) | (centre > n - 46)] = -np.inf  # keep labels off the spines
        if away_from is not None:
            m[np.abs(centre - away_from) < 85] = -np.inf  # no horizontal overlap with the other label
        return int(centre[np.argmax(m)])

    c = widest(shares[1])
    c3 = widest(shares[2], away_from=c)
    ax_b.text(x[c3], mid[2, c3], band_names[2], ha="center", va="center", fontsize=fs.FS_ANNOT, color="0.1", bbox=BOX)
    ax_b.annotate(band_names[1], xy=(x[c], mid[1, c]), xytext=(x[c], mid[2, c]), ha="center", va="center",
                  fontsize=fs.FS_ANNOT, color="0.1", bbox=BOX,
                  arrowprops=dict(arrowstyle="-", color="0.1", linewidth=0.6, shrinkA=0, shrinkB=0))
    ax_b.plot([x[c]], [mid[1, c]], marker="o", markersize=1.8, color="0.1")

    # (c) persistent tree-1 cross-name pairs
    ax_c.grid(axis="y", color="0.88", linewidth=0.35, linestyle="-")
    ax_c.set_axisbelow(True)
    ends = []
    for k, ((label, y), (color, ls, marker)) in enumerate(zip(top, SERIES_STYLE)):
        ax_c.plot(x, y, color=color, linestyle=ls, linewidth=1.05)
        mk = sparse_markers(y, every=20, phase=4 * k)
        ax_c.plot(x[mk], y[mk], linestyle="none", marker=marker, color=color, markersize=3.2,
                  markeredgecolor="white", markeredgewidth=0.4)
        last = int(np.flatnonzero(np.isfinite(y))[-1])
        ends.append((label, last, float(y[last]), color, marker))
    ax_c.set_ylim(0.42, 1.25)
    ax_c.set_ylabel("Mean tree-1 edge\ncontribution (nats)")
    ax_c.set_title("Most persistent top-5 tree-1 pairs")
    # direct labels: series that reach the last window are labelled in the right margin, the others at their end
    at_edge = [e for e in ends if e[1] == n - 1]
    ys = spread([e[2] for e in at_edge], gap=0.055)
    for (label, last, yv, color, marker), yl in zip(at_edge, ys):
        ax_c.annotate(label, xy=(x[last], yv), xytext=(1.03, yl), textcoords=("axes fraction", "data"),
                      ha="left", va="center", fontsize=fs.FS_ANNOT, color="0.1",
                      arrowprops=dict(arrowstyle="-", color=color, linewidth=0.6, shrinkA=0.5, shrinkB=2.0))
    for label, last, yv, color, marker in ends:
        if last == n - 1:
            continue
        anchor, dx, dy, ha = END_LABEL.get(label, ("end", 4, 0, "left"))
        y = top[[t[0] for t in top].index(label)][1]
        k = int(np.nanargmax(y)) if anchor == "max" else last
        ax_c.annotate(label, xy=(x[k], y[k]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                      fontsize=fs.FS_ANNOT, color="0.1", path_effects=HALO,
                      arrowprops=dict(arrowstyle="-", color=color, linewidth=0.6, shrinkA=0.5, shrinkB=2.0))

    for ax in (ax_b, ax_c):
        if a.xaxis == "date":
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
            ax.set_xlabel("Window end date")
        else:
            ax.set_xlabel("Rolling window index (5-day refits)")
    if a.xaxis == "date":
        pad = np.timedelta64(20, "D")
        ax_a.set_xlim(x[0] - pad, x[-1] + pad)
    fs.panel_labels(fig, [ax_a, ax_c], "ac")
    fs.panel_labels(fig, [ax_b], "b")
    fs.save(fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
