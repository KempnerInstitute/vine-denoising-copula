#!/usr/bin/env python3
"""Compose TC story figure:
1) Ground-truth edge MI recovery (truth vs estimate),
2) Synthetic TC scaling from the benchmark JSON,
3) Real-data decomposition contrast (Power vs one or two other datasets),
4) Knockout consistency: fitted tree share vs TC drop after tree ablation.

Output:
  drafts/figures/fig_tc_story_composite.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Apply shared paper style
try:
    from paper_style import apply_style, COLORS
    apply_style()
except ImportError:
    pass


def _drafts_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSON: {p}")
    obj = json.loads(p.read_text())
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected dict JSON at {p}, got {type(obj)}")
    return obj


def _tree_abs_shares(dec: Dict[str, Any]) -> Dict[int, float]:
    # Prefer precomputed summary.
    fig = dec.get("figure", {})
    rows = fig.get("tree_summary", []) if isinstance(fig, dict) else []
    out: Dict[int, float] = {}
    if isinstance(rows, list) and rows:
        for r in rows:
            try:
                out[int(r["tree"])] = float(r["share_pct_abs"])
            except Exception:
                pass
        if out:
            return out

    # Fallback from edge records.
    edges = dec.get("edge_records", [])
    if not isinstance(edges, list) or not edges:
        return out
    by_tree: Dict[int, List[float]] = {}
    for e in edges:
        t = int(e["tree"])
        by_tree.setdefault(t, []).append(float(e["mi"]))
    denom = float(sum(abs(float(e["mi"])) for e in edges))
    if denom <= 1e-12:
        return {t: 0.0 for t in by_tree}
    for t, vals in by_tree.items():
        out[t] = 100.0 * float(np.sum(np.abs(vals))) / denom
    return out


def _aggregate_tree_shares(shares: Dict[int, float], max_tree: int = 4) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t in range(1, max_tree + 1):
        out[f"T{t}"] = float(shares.get(t, 0.0))
    higher = float(sum(v for k, v in shares.items() if int(k) > max_tree))
    out[f"T{max_tree+1}+"] = higher
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build composite TC story figure.")
    p.add_argument(
        "--gt-json",
        type=Path,
        default=_drafts_dir() / "paper_outputs" / "tc_decomposition_ground_truth_compare.json",
    )
    p.add_argument(
        "--power-json",
        type=Path,
        default=_drafts_dir() / "paper_outputs" / "tc_decomposition_uci_power.json",
    )
    p.add_argument("--second-dataset", type=str, default="gas")
    p.add_argument(
        "--second-json",
        type=Path,
        default=None,
        help="If omitted, use drafts/paper_outputs/tc_decomposition_uci_<second-dataset>.json",
    )
    p.add_argument("--third-dataset", type=str, default=None)
    p.add_argument(
        "--third-json",
        type=Path,
        default=None,
        help="If omitted, use drafts/paper_outputs/tc_decomposition_uci_<third-dataset>.json",
    )
    p.add_argument(
        "--tc-json",
        type=Path,
        default=_repo_root() / "results" / "tc_benchmark.json",
    )
    p.add_argument(
        "--knockout-json",
        type=Path,
        default=_drafts_dir() / "paper_outputs" / "e17_tree_knockout_tc_full.json",
    )
    p.add_argument(
        "--selfcons-json",
        type=Path,
        default=_drafts_dir() / "paper_outputs" / "mi_self_consistency_scaleup.json",
        help="Scaled 480-trial MI self-consistency results (list of per-estimator dicts).",
    )
    p.add_argument("--out-prefix", type=str, default="fig_tc_story_composite")
    p.add_argument("--figures-dir", type=Path)
    args = p.parse_args()

    gt = _load_json(args.gt_json)
    pwr = _load_json(args.power_json)
    second_json = (
        args.second_json
        if args.second_json is not None
        else (_drafts_dir() / "paper_outputs" / f"tc_decomposition_uci_{str(args.second_dataset).lower()}.json")
    )
    sec = _load_json(second_json)
    third = None
    if args.third_dataset is not None or args.third_json is not None:
        third_json = (
            args.third_json
            if args.third_json is not None
            else (_drafts_dir() / "paper_outputs" / f"tc_decomposition_uci_{str(args.third_dataset).lower()}.json")
        )
        third = _load_json(third_json)
    tc = _load_json(args.tc_json)
    knockout = _load_json(args.knockout_json)

    # Scaled 480-trial self-consistency protocol (panel e). The file is a list
    # of per-estimator dicts (KSG, VDC), unlike the dict JSONs above.
    sc_path = Path(args.selfcons_json)
    if not sc_path.exists():
        raise FileNotFoundError(f"Missing JSON: {sc_path}")
    sc_raw = json.loads(sc_path.read_text())
    if not isinstance(sc_raw, list):
        raise RuntimeError(f"Expected list JSON at {sc_path}, got {type(sc_raw)}")
    sc_by_est = {str(r.get("estimator", "")).upper(): r for r in sc_raw}
    for est in ("VDC", "KSG"):
        if est not in sc_by_est:
            raise RuntimeError(f"Missing estimator '{est}' in {sc_path}")

    merged = gt.get("merged_edges", [])
    if not isinstance(merged, list) or not merged:
        raise RuntimeError("Ground-truth JSON has no merged_edges.")

    # Panel A: truth vs estimate scatter.
    x_true = np.array([float(r["mi_true"]) for r in merged], dtype=np.float64)
    y_est = np.array([float(r["mi_est"]) for r in merged], dtype=np.float64)
    t_edge = np.array([int(r["tree"]) for r in merged], dtype=np.int64)
    trees = sorted(set(int(t) for t in t_edge.tolist()))

    # Panel C: grouped bar tree abs-shares for two or three datasets.
    pwr_shares = _aggregate_tree_shares(_tree_abs_shares(pwr), max_tree=4)
    sec_shares = _aggregate_tree_shares(_tree_abs_shares(sec), max_tree=4)
    tree_labels = list(pwr_shares.keys())
    pwr_vals = np.array([pwr_shares[k] for k in tree_labels], dtype=np.float64)
    sec_vals = np.array([sec_shares.get(k, 0.0) for k in tree_labels], dtype=np.float64)
    third_vals = None
    if third is not None:
        thr_shares = _aggregate_tree_shares(_tree_abs_shares(third), max_tree=4)
        third_vals = np.array([thr_shares.get(k, 0.0) for k in tree_labels], dtype=np.float64)

    tc_power = float(pwr.get("meta", {}).get("tc_edge_sum", float("nan")))
    ds2_name = str(sec.get("meta", {}).get("dataset", str(args.second_dataset).lower()))
    tc_second = float(sec.get("meta", {}).get("tc_edge_sum", float("nan")))
    ds3_name = None
    tc_third = None
    if third is not None:
        ds3_name = str(third.get("meta", {}).get("dataset", str(args.third_dataset).lower()))
        tc_third = float(third.get("meta", {}).get("tc_edge_sum", float("nan")))

    # Panel B: TC scaling from existing benchmark.
    dims = np.array(tc.get("dimensions", []), dtype=np.float64)
    tc_true = np.array(tc.get("tc_true", []), dtype=np.float64)
    tc_dcd = np.array(tc.get("tc_dcd_mean", []), dtype=np.float64)
    tc_dcd_std = np.array(tc.get("tc_dcd_std", []), dtype=np.float64)
    tc_ksg = np.array(tc.get("tc_ksg_mean", []), dtype=np.float64)
    tc_ksg_std = np.array(tc.get("tc_ksg_std", []), dtype=np.float64)

    # Panel D: fitted tree share vs VDC TC drop under tree knockout.
    ko_rows: List[Dict[str, Any]] = []
    ko_datasets = knockout.get("datasets", [])
    if isinstance(ko_datasets, list):
        for ds in ko_datasets:
            ds_name = str(ds.get("dataset", "")).lower()
            for ab in ds.get("ablations", []):
                if str(ab.get("label", "")) == "Full":
                    continue
                ko_rows.append(
                    {
                        "dataset": ds_name,
                        "label": str(ab.get("label", "")),
                        "share_pct": float(ab.get("removed_abs_share_pct", float("nan"))),
                        "drop_pct": 100.0 * float(ab.get("rel_drop_vdc", float("nan"))),
                    }
                )

    if ko_rows:
        _x = np.array([r["share_pct"] for r in ko_rows]); _y = np.array([r["drop_pct"] for r in ko_rows])
        print(f"[knockout] n_points={len(ko_rows)} pearson_r={np.corrcoef(_x, _y)[0, 1]:.3f}")
    # Use paper-style colors consistently
    VDC_RED = COLORS["vdc"]
    KSG_BLUE = COLORS["ksg"]
    # Distinct ordinal colors for tree levels (no recycling across T1..T7).
    TREE_COLORS = plt.cm.viridis(np.linspace(0.0, 0.9, max(len(trees), 2)))
    # Okabe-Ito dataset triad; blue stays reserved for KSG, red for VDC.
    DATASET_COLORS = {"power": "#009E73", "gas": "#E69F00", "hepmass": "#CC79A7", "credit": "#CC79A7"}
    # Redundant (colour-independent) encodings: hatch per dataset in (c), marker per tree in (a).
    DATASET_HATCH = {"power": "///", "gas": "...", "hepmass": "xx", "credit": "xx"}
    TREE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

    # Column-native layout for the submitted two-column paper.  Panels (a-d)
    # form a 2x2 block and panel (e) spans the bottom row; this preserves
    # readable type when the asset is included at \columnwidth.
    apply_style(font_size=7)
    fig = plt.figure(figsize=(3.35, 4.2))
    gs = fig.add_gridspec(
        3, 2,
        left=0.17, right=0.985, bottom=0.075, top=0.935,
        wspace=0.82, hspace=0.68,
        height_ratios=[1.0, 1.0, 0.88],
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])
    axes = [ax1, ax2, ax3, ax4, ax5]

    # (a) Ground-truth edge MI scatter — log scale
    for i, t in enumerate(trees):
        m = t_edge == t
        ax1.scatter(
            x_true[m],
            y_est[m],
            s=12,
            alpha=0.85,
            color=TREE_COLORS[i],
            marker=TREE_MARKERS[i % len(TREE_MARKERS)],
            label=f"T{t}",
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )
    # Use log scale to reveal low-MI edges
    pos_true = x_true[x_true > 0]
    pos_est = y_est[y_est > 0]
    lo = float(min(np.min(pos_true), np.min(pos_est))) * 0.5
    hi = float(max(np.max(x_true), np.max(y_est))) * 1.5
    ax1.plot([lo, hi], [lo, hi], "--", color="0.65", linewidth=0.8, zorder=1)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_xlabel("True MI (nats)")
    ax1.set_ylabel("Estimated MI (nats)", fontsize=7)
    # Compact two-column legend in the empty lower-right triangle so it does
    # not overlap the y-axis label or the data along the diagonal.
    ax1.legend(fontsize=4.8, ncol=2, loc="lower right",
               framealpha=0.9, handletextpad=0.15, columnspacing=0.4,
               borderpad=0.25, labelspacing=0.25)

    # (b) TC scaling with dimension. True TC is drawn first as a thick light
    # solid reference so it stays visible beneath the overlapping VDC curve.
    ax2.plot(dims, tc_true, color="#999999", linestyle="-", linewidth=3.0,
             label="True TC", zorder=1, solid_capstyle="round")
    ax2.plot(
        dims, tc_dcd, marker="o", linewidth=1.4, color=VDC_RED, label="VDC",
        markersize=3, zorder=3, alpha=0.9,
    )
    if tc_dcd_std.size == tc_dcd.size:
        ax2.fill_between(dims, tc_dcd - tc_dcd_std, tc_dcd + tc_dcd_std, color=VDC_RED, alpha=0.12)
    ax2.plot(
        dims, tc_ksg, marker="s", linewidth=1.2, color=KSG_BLUE, label="KSG",
        markersize=3, zorder=2, alpha=0.9,
    )
    if tc_ksg_std.size == tc_ksg.size:
        ax2.fill_between(dims, tc_ksg - tc_ksg_std, tc_ksg + tc_ksg_std, color=KSG_BLUE, alpha=0.10)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Dimension $d$")
    ax2.set_ylabel("TC (nats)")
    ax2.legend(fontsize=5.2, loc="lower right", framealpha=0.9,
               handletextpad=0.3, borderpad=0.3)

    # (c) Tree-level TC decomposition — grouped bars
    x = np.arange(len(tree_labels))
    if third is None:
        w = 0.36
        offsets = [-w / 2, +w / 2]
        series = [
            (pwr_vals, "power", "POWER"),
            (sec_vals, ds2_name.lower(), ds2_name.upper()),
        ]
    else:
        w = 0.24
        offsets = [-w, 0.0, +w]
        series = [
            (pwr_vals, "power", "POWER"),
            (sec_vals, ds2_name.lower(), ds2_name.upper()),
            (third_vals, ds3_name.lower(), ds3_name.upper()),
        ]
    for off, (vals, key, label) in zip(offsets, series):
        ax3.bar(
            x + off,
            vals,
            width=w,
            color=DATASET_COLORS.get(key, COLORS["neutral"]),
            alpha=0.85,
            edgecolor="0.25",
            linewidth=0.3,
            hatch=DATASET_HATCH.get(key, ""),
            label=label,
        )
    ax3.set_xticks(x)
    ax3.set_xticklabels(tree_labels)
    ax3.set_xlabel("Tree level")
    ax3.set_ylim(0.0, 100.0)
    ax3.set_ylabel("Share of |TC| (%)")
    ax3.legend(fontsize=4.5, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 1.02), framealpha=0.0,
               handletextpad=0.18, columnspacing=0.35, borderpad=0.0)

    # (d) Knockout consistency scatter — distinct shape per dataset for legibility
    DATASET_MARKERS = {"power": "o", "gas": "^", "hepmass": "s", "credit": "s"}
    if ko_rows:
        vals = [float(r["share_pct"]) for r in ko_rows] + [float(r["drop_pct"]) for r in ko_rows]
        lo = min(vals)
        hi = max(vals)
        pad = 0.05 * max(1e-6, hi - lo)
        ax4.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="0.65",
                 linewidth=0.8, zorder=1, label="y = x")
        for ds_name in ["power", "gas", "hepmass", "credit"]:
            subset = [r for r in ko_rows if r["dataset"] == ds_name]
            if not subset:
                continue
            ax4.scatter(
                [float(r["share_pct"]) for r in subset],
                [float(r["drop_pct"]) for r in subset],
                s=18,
                alpha=0.92,
                color=DATASET_COLORS.get(ds_name, COLORS["neutral"]),
                marker=DATASET_MARKERS.get(ds_name, "o"),
                edgecolors="white",
                linewidths=0.4,
                label=ds_name.upper(),
                zorder=3,
            )
            # Annotate only well-separated points (x-share > 15%); labels for
            # the cluster of small shares near the origin would collide.
            for r in subset:
                if float(r["share_pct"]) <= 15.0:
                    continue
                ax4.annotate(
                    r["label"].replace("Remove ", ""),
                    (float(r["share_pct"]), float(r["drop_pct"])),
                    xytext=(3, 2),
                    textcoords="offset points",
                    fontsize=4.8,
                    color=DATASET_COLORS.get(ds_name, COLORS["neutral"]),
                    clip_on=False,
                )
        ax4.set_xlim(lo - pad, hi + pad)
        ax4.set_ylim(lo - pad, hi + pad)
    ax4.set_xlabel("Fitted share (%)")
    ax4.set_ylabel("TC drop (%)")
    ax4.legend(fontsize=5.0, loc="lower right", framealpha=0.9,
               handletextpad=0.3, borderpad=0.3)

    # (e) MI self-consistency: two compact bar plots. This keeps the two units
    # separate while reading more like a figure panel than a pasted table.
    # Data: scaled 480-trial protocol (mi_self_consistency_scaleup.json).
    sc_methods = ["VDC", "KSG"]
    sc_dpi = [
        100.0 * float(sc_by_est[m]["dpi"]["summary"]["violation_rate"])
        for m in sc_methods
    ]  # percent
    sc_add = [
        float(sc_by_est[m]["additivity"]["summary"]["mean_abs_error"])
        for m in sc_methods
    ]  # nats
    sc_colors = [VDC_RED, KSG_BLUE]
    ax5.axis("off")
    ax5.set_xlim(0.0, 1.0)
    ax5.set_ylim(0.0, 1.0)

    dpi_ax = ax5.inset_axes([0.08, 0.23, 0.38, 0.68])
    add_ax = ax5.inset_axes([0.59, 0.23, 0.38, 0.68])

    def _metric_bars(sub_ax, vals, xlim, label, fmt):
        y = np.arange(len(sc_methods))
        sub_ax.barh(y, vals, color=sc_colors, height=0.62, edgecolor="none", alpha=0.92)
        sub_ax.set_yticks(y)
        sub_ax.set_yticklabels(sc_methods, fontsize=6.0)
        sub_ax.invert_yaxis()
        sub_ax.set_xlim(0.0, xlim)
        sub_ax.set_title(label, fontsize=6.0, loc="left", pad=1.0)
        sub_ax.tick_params(axis="x", labelsize=5.2, pad=1)
        sub_ax.tick_params(axis="y", length=0, pad=1)
        sub_ax.grid(axis="x", alpha=0.20, linewidth=0.35)
        for yi, val, color in zip(y, vals, sc_colors):
            xpos = min(xlim * 0.96, val + xlim * 0.05)
            sub_ax.text(xpos, yi, fmt.format(val), va="center", ha="left",
                        fontsize=5.5, color=color, fontweight="bold")

    _metric_bars(dpi_ax, sc_dpi, 6.0, "DPI violations (%)", "{:.1f}")
    _metric_bars(add_ax, sc_add, 0.050, "Additivity error (nats)", "{:.3f}")

    # Trial count and significance of the DPI-violation contrast (10/480 vs
    # 25/480, Fisher exact test), rendered inside the panel below the bars.
    ax5.text(
        0.52,
        0.01,
        "160 base datasets; Fisher $p=0.011$",
        transform=ax5.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="0.15",
    )

    gt_meta = gt.get("meta", {})
    # Use figure coordinates so panel numbers share an exact vertical guide
    # within each column.  They sit left of the y-axis labels and slightly
    # above the plotting area, clear of titles, legends, and data.
    label_x = [0.012, 0.505, 0.012, 0.505, 0.012]
    for label, ax, xpos in zip(["a", "b", "c", "d", "e"], axes, label_x):
        pos = ax.get_position()
        fig.text(
            xpos,
            pos.y1 + 0.012,
            f"({label})",
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    from paper_style import save_figure
    save_figure(fig, args.out_prefix, figures_dir=args.figures_dir)
    print(
        "GT metrics:",
        f"TC_true={float(gt_meta.get('tc_true', float('nan'))):.4f}",
        f"TC_est={float(gt_meta.get('tc_est_edge', float('nan'))):.4f}",
        f"edge_MAE={float(gt_meta.get('edge_mae', float('nan'))):.4f}",
    )
    msg = [
        "Dataset TC:",
        f"Power={tc_power:.4f}",
        f"{ds2_name}={tc_second:.4f}",
    ]
    if ds3_name is not None and tc_third is not None:
        msg.append(f"{ds3_name}={tc_third:.4f}")
    msg.append("(signed, unclamped)")
    print(*msg)
    if ko_rows:
        xs = np.asarray([float(r["share_pct"]) for r in ko_rows], dtype=np.float64)
        ys = np.asarray([float(r["drop_pct"]) for r in ko_rows], dtype=np.float64)
        corr = float(np.corrcoef(xs, ys)[0, 1]) if xs.size >= 2 else float("nan")
        print("Knockout correlation (share vs VDC TC drop):", f"{corr:.3f}")


if __name__ == "__main__":
    main()
