#!/usr/bin/env python3
"""Plot a density grid saved by `vdc estimate-pair`. No illustrative data is generated."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=Path, required=True, help="Saved (m, m) density .npy file")
    parser.add_argument("--output", type=Path, default=Path("results/density.png"))
    args = parser.parse_args()
    density = np.load(args.density, allow_pickle=False)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        parser.error("density must be a square two-dimensional array")
    if not np.isfinite(density).all() or np.any(density < 0):
        parser.error("density must be finite and nonnegative")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(density.T, origin="lower", extent=[0, 1, 0, 1], cmap="cividis")
    ax.set(xlabel="u", ylabel="v", title="Estimated copula density")
    fig.colorbar(im, ax=ax, label="Density")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
