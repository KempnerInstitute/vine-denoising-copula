#!/usr/bin/env python3
"""Appendix figure: training curve and IPFP marginal profile of the deployed checkpoint.

These two panels were Fig. 1(c)-(d) in earlier versions. Data: fig1_overview/fig1_overview_data.npz in the
inputs folder (training-loss history of the deployed run, downsampled every 200 steps;
raw and projected outputs for the Clayton theta=3, n=1500, seed-123 sample).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vdc_figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402



def smooth(y, k=25):
    k = max(1, int(k))
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fs.add_io_args(ap, "figA_training_diagnostics")
    ap.add_argument("--data", type=Path, help="use this fig1_overview_data.npz instead of the saved one")
    a = ap.parse_args()
    fs.apply()
    D = np.load(fs.input_file(a.inputs, "fig1_overview/fig1_overview_data.npz", a.data))
    fig = plt.figure(figsize=(fs.TEXTWIDTH, 1.72))
    gs = fig.add_gridspec(1, 3, wspace=0.5, left=0.075, right=0.985, top=0.83, bottom=0.225,
                          width_ratios=[1.0, 1.0, 1.08])
    ax_l, ax_r, ax_p = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a), (b): one quantity per axis (no shared second y-scale), same training-step axis
    step = D["loss_step"] / 1000.0
    excess = smooth(D["loss_value"] - D["loss_min"][0]) + 1e-3
    rms = smooth(D["loss_row_marg_rms"])
    for ax, y, color, title, ylabel, yt in [
            (ax_l, excess, "#D55E00", "Training loss", "Loss minus its minimum", [0.3, 1, 3]),
            (ax_r, rms, "#0072B2", "Pre-projection marginals", "RMS deviation from uniform", [0.006, 0.01, 0.02, 0.03])]:
        ax.plot(step, y, color=color, linestyle="-", linewidth=0.9)
        ax.set_yscale("log")
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{t:g}" for t in yt])
        ax.minorticks_off()
        ax.set_xlim(-5, 245)
        ax.set_xticks([0, 100, 200])
        ax.set_xlabel("Training step (thousands)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axvline(190, color="0.35", linewidth=0.6, linestyle=":")
    ax_l.text(186, 0.97, "deployed\n(190k)", transform=ax_l.get_xaxis_transform(), ha="right", va="top",
              fontsize=fs.FS_ANNOT, color="0.25")

    # (c): row-marginal profile of one raw output before and after the deployed projection
    m = D["pre_raw"].shape[0]
    u = (np.arange(m) + 0.5) / m
    ax_p.plot(u, D["pre_raw"].mean(axis=1), color="0.35", linestyle="-", linewidth=1.0, label="raw network output")
    ax_p.plot(u, D["pre_proj"].mean(axis=1), color="#D55E00", linestyle="--", linewidth=1.1,
              label="after IPFP ($K{=}30$)")
    ax_p.axhline(1.0, color="0.1", linestyle=":", linewidth=0.8, label="uniform target")
    ax_p.set_yscale("log")
    ax_p.set_yticks([0.5, 1, 2, 5, 10])
    ax_p.set_yticklabels(["0.5", "1", "2", "5", "10"])
    ax_p.minorticks_off()
    ax_p.set_xlabel("$u$ (grid row)")
    ax_p.set_ylabel("Row marginal")
    ax_p.set_title("Marginals of one output")
    ax_p.legend(loc="upper right", handlelength=1.8)
    fs.panel_labels(fig, [ax_l, ax_r, ax_p], "abc", y=0.995)
    fs.save(fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
