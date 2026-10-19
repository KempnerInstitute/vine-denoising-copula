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
    fig = plt.figure(figsize=(fs.TEXTWIDTH, 1.75))
    gs = fig.add_gridspec(1, 2, wspace=0.42, left=0.085, right=0.92, top=0.86, bottom=0.22)
    ax1, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    step = D["loss_step"] / 1000.0
    excess = smooth(D["loss_value"] - D["loss_min"][0]) + 1e-3
    rms = smooth(D["loss_row_marg_rms"])
    ax1.plot(step, excess, color="#D55E00", linestyle="-", linewidth=0.9, label="training loss minus its minimum")
    ax1.set_yscale("log")
    ax1.set_xlabel("Training step (thousands)")
    ax1.set_ylabel("Loss minus minimum")
    ax1.axvline(190, color="0.4", linewidth=0.6, linestyle=":")
    ax1.text(186, 0.97, "deployed\n(step 190k)", transform=ax1.get_xaxis_transform(), ha="right", va="top",
             fontsize=fs.FS_ANNOT, color="0.25")
    ax1.set_yticks([0.3, 1, 3])
    ax1.set_yticklabels(["0.3", "1", "3"])
    ax1.minorticks_off()
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(step, rms, color="#0072B2", linestyle="--", linewidth=0.9, label="row-marginal RMS before projection")
    ax2.set_yscale("log")
    ax2.set_ylabel("Pre-projection marginal RMS")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", bbox_to_anchor=(0.99, 0.55))
    ax1.set_title("Training of the deployed estimator")

    m = D["pre_raw"].shape[0]
    u = (np.arange(m) + 0.5) / m
    ax3.plot(u, D["pre_raw"].mean(axis=1), color="0.35", linestyle="-", linewidth=1.0, label="raw network output")
    ax3.plot(u, D["pre_proj"].mean(axis=1), color="#D55E00", linestyle="--", linewidth=1.1, label="after IPFP ($K{=}30$)")
    ax3.axhline(1.0, color="0.1", linestyle=":", linewidth=0.8, label="uniform target")
    ax3.set_yscale("log")
    ax3.set_xlabel("$u$ (row of the grid)")
    ax3.set_ylabel("Row marginal")
    ax3.set_title("Projection onto uniform marginals")
    ax3.legend(loc="upper center")
    fs.panel_labels(fig, [ax1, ax3], "ab", y=0.995)
    fs.save(fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
