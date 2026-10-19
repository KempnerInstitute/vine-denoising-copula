#!/usr/bin/env python3
"""Camera-ready Fig. 1: VDC overview, drawn as a vector schematic at printed size.

(a) Pretrain once: zoo samples -> noisy histogram -> U-Net f_theta -> projected density.
(b) Fit a vine tree by tree: every edge of tree k is histogrammed, the batch goes through the
    frozen f_theta in one forward pass, IPFP projects each grid, and the cached h-functions
    transform the pseudo-observations that feed tree k+1.
(c) Vine factorization and what the fitted vine provides.
(d) Example on the X-shaped copula (Complex-X) of the training zoo: true, pyvine-TLL, VDC.
All images are real outputs read from fig1_overview/fig1_overview_data.npz in the inputs folder
(written by fig1_overview_data.py with the frozen checkpoint and the pinned library).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vdc_figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

W, H = fs.TEXTWIDTH, 2.60
INK = "#2B2B2B"
ARROW = "#3C4A5C"
CMAP = "viridis"
FILL_A, EDGE_A = "#EEF3F9", "#7D93AE"
FILL_B, EDGE_B = "#FBF3EA", "#C49A6C"
FILL_C, EDGE_C = "#F2F2F2", "#9A9A9A"


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")

    def box(self, x, y, w, h, fc, ec, title=None, letter=None):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                         fc=fc, ec=ec, lw=0.7, zorder=0))
        if title:
            self.ax.text(x + 0.27, y + h - 0.06, title, ha="left", va="top", fontsize=fs.FS_TITLE,
                         fontweight="bold", color=INK)
        if letter:
            self.ax.text(x + 0.05, y + h - 0.05, f"({letter})", ha="left", va="top", fontsize=fs.FS_PANEL,
                         fontweight="bold", color=INK)

    def img(self, x, y, w, h, data, vmin=None, vmax=None, z=2, frame=INK, cmap=CMAP):
        a = self.fig.add_axes([x / W, y / H, w / W, h / H])
        a.imshow(data.T, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=vmin, vmax=vmax,
                 interpolation="nearest", aspect="auto")
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(0.5)
            s.set_color(frame)
        a.set_zorder(z)
        return a

    def scatter(self, x, y, w, h, pts, s=0.6, z=2, color="#33475B"):
        a = self.fig.add_axes([x / W, y / H, w / W, h / H])
        a.scatter(pts[:, 0], pts[:, 1], s=s, color=color, linewidths=0, rasterized=True)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.set_xticks([])
        a.set_yticks([])
        a.set_facecolor("white")
        for sp in a.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color(INK)
        a.set_zorder(z)
        return a

    def arrow(self, x1, y1, x2, y2, rad=0.0, lw=0.8, color=ARROW, style="-|>", ls="-"):
        self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=6.5,
                                          connectionstyle=f"arc3,rad={rad}", lw=lw, color=color,
                                          linestyle=ls, zorder=1, shrinkA=0, shrinkB=0))

    def text(self, x, y, s, size=None, ha="center", va="top", **kw):
        self.ax.text(x, y, s, ha=ha, va=va, fontsize=size or fs.FS_ANNOT, color=kw.pop("color", INK), **kw)

    def unet(self, xc, yc, w, h, label, frozen=False):
        heights = [1.0, 0.72, 0.46, 0.72, 1.0]
        bw = w / 7.2
        xs = [xc - w / 2 + i * (w - bw) / 4 for i in range(5)]
        col = "#D55E00" if not frozen else "#8C8C8C"
        for x0, hh in zip(xs, heights):
            self.ax.add_patch(Rectangle((x0, yc - hh * h / 2), bw, hh * h, fc=col, ec="none", alpha=0.85, zorder=2))
        for i, j in ((0, 4), (1, 3)):
            self.ax.add_patch(FancyArrowPatch((xs[i] + bw / 2, yc + heights[i] * h / 2 + 0.01),
                                              (xs[j] + bw / 2, yc + heights[j] * h / 2 + 0.01),
                                              connectionstyle="arc3,rad=-0.45", arrowstyle="-", lw=0.5,
                                              color=col, zorder=2))
        self.text(xc, yc - h / 2 - 0.035, label)


def logd(d, lo=-2.0):
    return np.log10(np.clip(d, 10 ** lo, None))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fs.add_io_args(ap, "fig1_overview")
    ap.add_argument("--data", type=Path, help="use this fig1_overview_data.npz instead of the saved one")
    a = ap.parse_args()
    fs.apply()
    D = np.load(fs.input_file(a.inputs, "fig1_overview/fig1_overview_data.npz", a.data))
    c = Canvas()

    # ---------------- (a) pretrain once ----------------
    ya, ha_ = 1.35, 1.21
    c.box(0.04, ya, 1.98, ha_, FILL_A, EDGE_A, "Pretrain once", "a")
    c.text(1.24, ya + ha_ - 0.075, "on simulated copulas", ha="left", style="italic", size=6.0)
    y0, s = ya + 0.40, 0.34          # icon bottom, icon size
    ylab = y0 + s + 0.09             # labels sit above the icons
    zoo = [D[f"zoo_{k}"] for k in range(4)]
    q = s / 2 - 0.01
    for k, (dx, dy) in enumerate([(0, q + 0.02), (q + 0.02, q + 0.02), (0, 0), (q + 0.02, 0)]):
        c.scatter(0.12 + dx, y0 + dy, q, q, zoo[k], s=0.35)
    c.text(0.12 + s / 2, ylab, "copula zoo", va="bottom")
    c.arrow(0.50, y0 + s / 2, 0.61, y0 + s / 2)
    c.img(0.63, y0, s, s, logd(D["pre_hist_mixed"]), vmin=-1.0, vmax=1.4)
    c.text(0.63 + s / 2, ylab, "noisy\nhistogram", va="bottom")
    c.arrow(1.00, y0 + s / 2, 1.11, y0 + s / 2)
    c.unet(1.31, y0 + s / 2, 0.36, 0.30, "")
    c.text(1.31, ylab, r"U-Net $f_\theta$", va="bottom")
    c.arrow(1.51, y0 + s / 2, 1.60, y0 + s / 2)
    c.img(1.62, y0, s, s, logd(D["pre_proj"]), vmin=-1.0, vmax=1.4)
    c.text(1.62 + s / 2, ylab, "projected\ndensity $\\hat D$", va="bottom")
    c.text(1.03, ya + 0.10, r"loss: $\hat D$ vs. the true copula density", size=6.0, va="bottom", style="italic")

    # ---------------- (b) fit a vine, tree by tree ----------------
    xb, wb = 2.10, 3.36
    c.box(xb, ya, wb, ha_, FILL_B, EDGE_B, "Fit any vine, tree by tree", "b")
    s2, off = 0.30, 0.035

    def stack(x, kind):
        for j in (2, 1, 0):  # back cards up and to the right
            xx, yy = x + j * off, y0 + j * off
            if kind == "scatter":
                c.scatter(xx, yy, s2, s2, D[f"dep{j}_sample"], s=0.25, z=3 + (2 - j))
            else:
                c.img(xx, yy, s2, s2, logd(D[f"dep{j}_{kind}"]), vmin=-1.0, vmax=1.4, z=3 + (2 - j))
        return x + s2 + 2 * off

    xs = xb + 0.13
    xe = stack(xs, "scatter")
    mid = y0 + s2 / 2 + off
    c.text(xs + (s2 + 2 * off) / 2, ylab, "tree-$k$ edge\npseudo-obs.", va="bottom")
    c.arrow(xe + 0.04, mid, xe + 0.15, mid)
    xh = xe + 0.17
    xe2 = stack(xh, "hist")
    c.text(xh + (s2 + 2 * off) / 2, ylab, "histogram\nbatch", va="bottom")
    c.arrow(xe2 + 0.04, mid, xe2 + 0.15, mid)
    xu = xe2 + 0.37
    c.unet(xu, mid, 0.40, 0.32, "", frozen=True)
    c.text(xu, ylab, "frozen $f_\\theta$:\none forward pass", va="bottom")
    c.arrow(xu + 0.22, mid, xu + 0.33, mid)
    xp = xu + 0.35
    xe3 = stack(xp, "proj")
    c.text(xp + (s2 + 2 * off) / 2, ylab, "projected edge\ndensities $\\hat c_e$", va="bottom")
    c.arrow(xe3 + 0.04, mid, xe3 + 0.15, mid)
    xt = xe3 + 0.17
    c.ax.add_patch(FancyBboxPatch((xt, y0 + 0.02), 0.42, s2 + 2 * off - 0.04, boxstyle="round,pad=0,rounding_size=0.04",
                                  fc="white", ec=INK, lw=0.5, zorder=2))
    c.text(xt + 0.21, y0 + 0.31, r"$h_e,\ h_e^{-1}$", size=fs.FS_LABEL)
    c.text(xt + 0.21, y0 + 0.16, "cached", size=6.0)
    c.text(xt + 0.21, ylab, "conditional\ntables", va="bottom")
    # recursion loop below the icons: h-transforms feed the next tree
    yl = y0 - 0.17
    c.ax.plot([xt + 0.21, xt + 0.21, xs + (s2 + 2 * off) / 2], [y0 + 0.02, yl, yl], color=ARROW, lw=0.8, zorder=1)
    c.arrow(xs + (s2 + 2 * off) / 2, yl, xs + (s2 + 2 * off) / 2, y0 - 0.01)
    c.text((xs + xt) / 2 + 0.33, yl - 0.035, r"$h$-transforms give the pseudo-observations of tree $k{+}1$",
           size=6.0, style="italic")

    # ---------------- (c) vine and outputs ----------------
    yc0, hc = 0.03, 1.27
    c.box(0.04, yc0, 2.40, hc, FILL_C, EDGE_C, "One estimator for every edge", "c")
    rows = [("$T_1$", ["1", "2", "3", "4"], 0.93), ("$T_2$", ["12", "23", "34"], 0.66), ("$T_3$", ["13|2", "24|3"], 0.39)]
    for lab, nodes, yy in rows:
        y = yc0 + yy
        c.text(0.17, y, lab, size=fs.FS_LABEL, va="center")
        n = len(nodes)
        step = 0.30
        xs_n = [0.80 - step * (n - 1) / 2 + step * i for i in range(n)]
        for i in range(n - 1):
            c.ax.plot([xs_n[i], xs_n[i + 1]], [y, y], color=INK, lw=0.6, zorder=1)
            c.text((xs_n[i] + xs_n[i + 1]) / 2, y + 0.06, r"$f_\theta$", size=5.6, va="bottom", color="#7A7A7A")
        for x0, t in zip(xs_n, nodes):
            wbox = 0.10 + 0.034 * len(t)
            c.ax.add_patch(FancyBboxPatch((x0 - wbox / 2, y - 0.055), wbox, 0.11,
                                          boxstyle="round,pad=0,rounding_size=0.03", fc="white", ec=INK, lw=0.5, zorder=2))
            c.text(x0, y, t, size=6.0, va="center", zorder=3)
    c.text(0.80, yc0 + 0.15, r"$c(\mathbf{u})=\prod_{e}\,\hat c_e(u_{j|D_e},\,u_{k|D_e})$",
           size=fs.FS_LABEL, va="center")
    outs = [("Likelihood", "product of edge densities"), ("Sampling", r"inverse $h$-functions"),
            ("Information", "MI per edge, TC per tree")]
    for i, (t1, t2) in enumerate(outs):
        y = yc0 + 0.93 - 0.29 * i
        c.ax.add_patch(FancyBboxPatch((1.46, y - 0.12), 0.94, 0.235, boxstyle="round,pad=0,rounding_size=0.03",
                                      fc="white", ec=EDGE_C, lw=0.5, zorder=2))
        c.text(1.51, y + 0.075, t1, ha="left", size=6.3, fontweight="bold", zorder=3)
        c.text(1.51, y - 0.025, t2, ha="left", size=6.0, zorder=3)
    c.arrow(1.29, yc0 + 0.64, 1.42, yc0 + 0.64)

    # ---------------- (d) example ----------------
    xd, wd = 2.50, 2.96
    c.box(xd, yc0, wd, hc, FILL_C, EDGE_C, "Example: X-shaped copula", "d")
    s3 = 0.70
    iy3 = yc0 + 0.30
    c.scatter(xd + 0.10, iy3, s3, s3, D["cx_sample"], s=0.25)
    c.text(xd + 0.10 + s3 / 2, iy3 - 0.03, "sample ($n{=}2000$)")
    ise_tll, ise_vdc = D["cx_ise"]
    imgs = [("true density", D["cx_true"], None), ("pyvine-TLL", D["cx_tll"], ise_tll), ("VDC", D["cx_vdc"], ise_vdc)]
    for i, (t, dd, ise) in enumerate(imgs):
        x0 = xd + 0.88 + i * 0.64
        im = c.img(x0, iy3, 0.58, s3, np.log10(np.clip(dd, 1e-4, None)), vmin=-4.0, vmax=1.3)
        lab = t if ise is None else f"{t}\nISE {ise:.1e}".replace("e-0", "e-").replace("e+0", "e+")
        c.text(x0 + 0.29, iy3 - 0.03, lab)
    xcb = xd + 0.88 + 3 * 0.64 - 0.03
    cax = c.fig.add_axes([xcb / W, iy3 / H, 0.05 / W, s3 / H])
    cb = c.fig.colorbar(im.images[0], cax=cax)
    cb.set_ticks([-4, -2, 0])
    cb.ax.tick_params(labelsize=5.6, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    c.text(xcb + 0.025, iy3 + s3 + 0.05, r"$\log_{10}$", size=5.6, va="bottom", ha="center")
    fs.save(c.fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
