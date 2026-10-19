#!/usr/bin/env python3
"""Camera-ready Fig. 1: VDC overview, drawn as a vector schematic at printed size.

(a) Pretrain once: zoo samples -> noisy histogram -> U-Net f_theta -> IPFP -> projected density,
    trained against the true copula density.
(b) Fit a D-vine tree by tree: the edges of tree k are histogrammed and passed through the frozen
    f_theta in batches (one network evaluation per batch), IPFP projects each grid, and the cached
    h-functions transform the pseudo-observations that feed tree k+1.
(c) Vine factorization: one estimator for every edge, and what the fitted vine provides.
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

W, H = fs.TEXTWIDTH, 2.70
INK = "#262626"
MUTED = "#6E6E6E"
ARROW = "#4A4A4A"
PANEL_FC, PANEL_EC = "#F6F6F6", "#CCCCCC"
TRAINED = fs.STYLE["vdc"].color   # the network being trained: method colour
FROZEN = "#8E8E8E"                # the same network, frozen
CMAP = "viridis"
DOTS = "#333333"

M, GAP = 0.03, 0.06               # outer margin and gap between panels (in)
SPLIT = 2.22                      # width of the left column; the right column takes the rest
ROW2 = (M, 1.28)                  # (bottom, height) of the lower row
ROW1 = (M + 1.28 + GAP, H - 2 * M - 1.28 - GAP)


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")

    def panel(self, x, y, w, h, letter, title):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.045",
                                         fc=PANEL_FC, ec=PANEL_EC, lw=0.5, zorder=0))
        self.ax.text(x + 0.07, y + h - 0.055, f"({letter})", ha="left", va="top", fontsize=fs.FS_PANEL,
                     fontweight="bold", color=INK)
        self.ax.text(x + 0.33, y + h - 0.062, title, ha="left", va="top", fontsize=fs.FS_TITLE,
                     fontweight="bold", color=INK)

    def _frame(self, a, z):
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():  # all four sides (the shared style hides top and right spines)
            s.set_visible(True)
            s.set_linewidth(0.5)
            s.set_color(INK)
        a.set_zorder(z)
        return a

    def img(self, x, y, s, data, vmin, vmax, z=2):
        a = self.fig.add_axes([x / W, y / H, s / W, s / H])
        a.imshow(data.T, origin="lower", extent=(0, 1, 0, 1), cmap=CMAP, vmin=vmin, vmax=vmax,
                 interpolation="nearest", aspect="auto")
        return self._frame(a, z)

    def scatter(self, x, y, s, pts, size=0.5, z=2):
        a = self.fig.add_axes([x / W, y / H, s / W, s / H])
        a.scatter(pts[:, 0], pts[:, 1], s=size, color=DOTS, linewidths=0, rasterized=True)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.set_facecolor("white")
        return self._frame(a, z)

    def arrow(self, x1, y1, x2, y2, lw=0.9):
        self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7.5, lw=lw,
                                          color=ARROW, zorder=1, shrinkA=0, shrinkB=0))

    def text(self, x, y, s, size=None, ha="center", va="top", color=INK, **kw):
        self.ax.text(x, y, s, ha=ha, va=va, fontsize=size or fs.FS_ANNOT, color=color, **kw)

    def unet(self, xc, yc, w, h, color):
        heights = [1.0, 0.70, 0.44, 0.70, 1.0]
        bw = w / 6.5
        xs = [xc - w / 2 + i * (w - bw) / 4 for i in range(5)]
        for x0, hh in zip(xs, heights):
            self.ax.add_patch(Rectangle((x0, yc - hh * h / 2), bw, hh * h, fc=color, ec="none", zorder=2))
        for i, j in ((0, 4), (1, 3)):  # skip connections
            self.ax.add_patch(FancyArrowPatch((xs[i] + bw / 2, yc + heights[i] * h / 2 + 0.012),
                                              (xs[j] + bw / 2, yc + heights[j] * h / 2 + 0.012),
                                              connectionstyle="arc3,rad=-0.45", arrowstyle="-", lw=0.6,
                                              color=color, zorder=2))

    def htable(self, x, y, w, h):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.035",
                                         fc="white", ec=INK, lw=0.5, zorder=2))
        self.text(x + w / 2, y + 0.66 * h, r"$h_e,\ h_e^{-1}$", size=fs.FS_LABEL, va="center", zorder=3)
        self.text(x + w / 2, y + 0.30 * h, "cached", va="center", color=MUTED, zorder=3)


def logd(d, lo=-2.0):
    return np.log10(np.clip(d, 10 ** lo, None))


def place(x0, x1, widths, gaps):
    """Left edges of elements laid out from x0 with the given relative gaps, spread to end at x1."""
    free = (x1 - x0) - sum(widths)
    gaps = [g * free / sum(gaps) for g in gaps]
    xs, x = [], x0
    for i, w in enumerate(widths):
        xs.append(x)
        x += w + (gaps[i] if i < len(gaps) else 0.0)
    return xs, gaps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fs.add_io_args(ap, "fig1_overview")
    ap.add_argument("--data", type=Path, help="use this fig1_overview_data.npz instead of the saved one")
    a = ap.parse_args()
    fs.apply()
    D = np.load(fs.input_file(a.inputs, "fig1_overview/fig1_overview_data.npz", a.data))
    c = Canvas()
    lw_col, rw_col = SPLIT, W - 2 * M - GAP - SPLIT
    xl, xr = M, M + SPLIT + GAP
    s, off = 0.40, 0.028             # icon size and card offset of a stack
    vlo, vhi = -1.0, 1.4             # log10 density range of the small icons

    # ---------------- row 1: geometry shared by (a) and (b) ----------------
    y1, h1 = ROW1
    yi = y1 + 0.33                   # bottom of the front icons
    ymid = yi + s / 2                # arrow height
    ylab = yi + s + 2 * off + 0.05   # baseline of the labels above the icons (same in both panels)
    ynote = yi - 0.11                # loop line / note below the icons

    # (a) pretrain once
    c.panel(xl, y1, lw_col, h1, "a", "Pretrain once")
    uw = 0.36
    xs, gaps = place(xl + 0.08, xl + lw_col - 0.08, [s, s, uw, s], [1.0, 1.0, 1.0])
    q = s / 2 - 0.012
    for k, (dx, dy) in enumerate([(0, q + 0.024), (q + 0.024, q + 0.024), (0, 0), (q + 0.024, 0)]):
        c.scatter(xs[0] + dx, yi + dy, q, D[f"zoo_{k}"], size=0.3)
    c.text(xs[0] + s / 2, ylab, "simulated\ncopulas", va="bottom")
    c.img(xs[1], yi, s, logd(D["pre_hist_mixed"]), vlo, vhi)
    c.text(xs[1] + s / 2, ylab, "noisy\nhistogram", va="bottom")
    c.unet(xs[2] + uw / 2, ymid, uw, 0.32, TRAINED)
    c.text(xs[2] + uw / 2, ylab, "U-Net $f_\\theta$\n(trained)", va="bottom")
    c.img(xs[3], yi, s, logd(D["pre_proj"]), vlo, vhi)
    c.text(xs[3] + s / 2, ylab, "projected\ndensity", va="bottom")
    for i in range(3):
        c.arrow(xs[i] + [s, s, uw][i] + 0.025, ymid, xs[i + 1] - 0.025, ymid)
    c.text(xl + lw_col / 2, ynote + 0.04, "loss: IPFP-projected output vs. true copula density", va="top", style="italic")

    # (b) fit a D-vine, tree by tree, with the frozen network
    c.panel(xr, y1, rw_col, h1, "b", "Fit a D-vine, tree by tree")
    sw = s + 2 * off
    tw = 0.42
    xs, gaps = place(xr + 0.08, xr + rw_col - 0.08, [sw, sw, uw, sw, tw], [0.9, 1.5, 1.5, 0.9])

    def stack(x, kind):
        for j in (2, 1, 0):  # back cards up and to the right
            xx, yy = x + j * off, yi + j * off
            if kind == "scatter":  # back cards stay blank so their dots do not clutter the stack
                c.scatter(xx, yy, s, D[f"dep{j}_sample"] if j == 0 else np.empty((0, 2)), size=0.22, z=3 + (2 - j))
            else:
                c.img(xx, yy, s, logd(D[f"dep{j}_{kind}"]), vlo, vhi, z=3 + (2 - j))

    ym = ymid + off
    stack(xs[0], "scatter")
    c.text(xs[0] + sw / 2, ylab, "tree-$k$ edges\n(pseudo-obs.)", va="bottom")
    stack(xs[1], "hist")
    c.text(xs[1] + sw / 2, ylab, "edge\nhistograms", va="bottom")
    c.unet(xs[2] + uw / 2, ym, uw, 0.32, FROZEN)
    c.text(xs[2] + uw / 2, ylab, "frozen $f_\\theta$:\none pass per batch", va="bottom")
    stack(xs[3], "proj")
    c.text(xs[3] + sw / 2, ylab, "edge\ndensities", va="bottom")
    c.htable(xs[4], yi + 0.02, tw, s + 2 * off - 0.04)
    c.text(xs[4] + tw / 2, ylab, "conditional\nCDFs", va="bottom")
    ends = [xs[0] + sw, xs[1] + sw, xs[2] + uw, xs[3] + sw]
    for i in range(4):
        c.arrow(ends[i] + 0.025, ym, xs[i + 1] - 0.025, ym)
    c.text(xs[3] - gaps[2] / 2, ym + 0.03, "IPFP", va="bottom")
    # recursion: the h-transforms give the pseudo-observations of tree k+1
    xa, xz = xs[4] + tw / 2, xs[0] + sw / 2
    c.ax.plot([xa, xa, xz], [yi + 0.02, ynote, ynote], color=ARROW, lw=0.9, zorder=1, solid_capstyle="butt")
    c.arrow(xz, ynote, xz, yi - 0.012)
    c.text((xa + xz) / 2, ynote - 0.035, "$h$-transforms give the pseudo-observations of tree $k{+}1$",
           va="top", style="italic")

    # ---------------- row 2 ----------------
    y2, h2 = ROW2
    # (c) one estimator for every edge
    c.panel(xl, y2, lw_col, h2, "c", "One estimator for every edge")
    xc0, step = xl + 0.64, 0.28      # centre of the vine diagram and node spacing in tree 1
    rows = [("$T_1$", ["1", "2", "3", "4"]), ("$T_2$", ["12", "23", "34"]), ("$T_3$", ["13|2", "24|3"])]
    yrow = [y2 + 0.93, y2 + 0.65, y2 + 0.37]
    for (lab, nodes), y in zip(rows, yrow):
        n = len(nodes)
        xn = [xc0 + step * (i - (n - 1) / 2) for i in range(n)]
        c.text(xl + 0.07, y, lab, size=fs.FS_LABEL, va="center")
        for i in range(n - 1):
            c.ax.plot([xn[i], xn[i + 1]], [y, y], color=INK, lw=0.6, zorder=1)
            c.text((xn[i] + xn[i + 1]) / 2, y + 0.06, r"$f_\theta$", va="bottom", color=MUTED)
        for x0, t in zip(xn, nodes):
            wb = 0.06 + 0.040 * len(t)
            c.ax.add_patch(FancyBboxPatch((x0 - wb / 2, y - 0.065), wb, 0.13, boxstyle="round,pad=0,rounding_size=0.03",
                                          fc="white", ec=INK, lw=0.5, zorder=2))
            c.text(x0, y, t, va="center", zorder=3)
    c.text(xl + lw_col / 2, y2 + 0.115, r"$c(\mathbf{u})=\prod_{e}\,\hat c_e(u_{j|D_e},\,u_{k|D_e})$", size=fs.FS_LABEL,
           va="center")
    xo = xl + 1.20                   # output chips, one per tree row: what the fitted vine provides
    wo, ho = xl + lw_col - 0.07 - xo, 0.25
    outs = [("Likelihood", r"$\log c=\Sigma_e \log \hat c_e$"), ("Sampling", "inverse $h$, tree by tree"),
            ("Information", "edge MI and tree TC")]
    for y, (t1, t2) in zip(yrow, outs):
        c.ax.add_patch(FancyBboxPatch((xo, y - ho / 2), wo, ho, boxstyle="round,pad=0,rounding_size=0.03",
                                      fc="white", ec=PANEL_EC, lw=0.6, zorder=2))
        c.text(xo + wo / 2, y + 0.056, t1, va="center", fontweight="bold", zorder=3)
        c.text(xo + wo / 2, y - 0.052, t2, va="center", zorder=3)
    xa0, xf = xc0 + 1.5 * step + 0.07, xo - 0.035    # bracket from the vine to the three uses
    c.ax.plot([xa0, xf], [yrow[1], yrow[1]], color=ARROW, lw=0.8, zorder=1, solid_capstyle="butt")
    c.ax.plot([xf, xf], [yrow[2], yrow[0]], color=ARROW, lw=0.8, zorder=1, solid_capstyle="butt")
    for y in yrow:
        c.ax.plot([xf, xo], [y, y], color=ARROW, lw=0.8, zorder=1, solid_capstyle="butt")

    # (d) example: X-shaped copula
    c.panel(xr, y2, rw_col, h2, "d", "Example: X-shaped copula")
    s3, g3 = 0.665, 0.045
    iy = y2 + 0.32
    x0 = xr + 0.09
    c.scatter(x0, iy, s3, D["cx_sample"], size=0.2)
    c.text(x0 + s3 / 2, iy - 0.04, "sample ($n{=}2000$)")
    ise_tll, ise_vdc = D["cx_ise"]
    imgs = [("true density", D["cx_true"], None), ("pyvine-TLL", D["cx_tll"], ise_tll), ("VDC", D["cx_vdc"], ise_vdc)]
    for i, (t, dd, ise) in enumerate(imgs):
        xi = x0 + (i + 1) * (s3 + g3)
        im = c.img(xi, iy, s3, np.log10(np.clip(dd, 1e-4, None)), -4.0, 1.3)
        lab = t if ise is None else f"{t}\nISE {ise:.1e}".replace("e-0", "e-").replace("e+0", "e+")
        c.text(xi + s3 / 2, iy - 0.04, lab)
    xcb = x0 + 4 * (s3 + g3) - g3 + 0.05
    cax = c.fig.add_axes([xcb / W, iy / H, 0.05 / W, s3 / H])
    cb = c.fig.colorbar(im.images[0], cax=cax)
    cb.set_ticks([-4, -2, 0])
    cb.ax.tick_params(labelsize=fs.FS_ANNOT, length=1.8, width=0.5, pad=1.2)
    cb.outline.set_linewidth(0.5)
    c.text(xcb + 0.025, iy + s3 + 0.04, r"$\log_{10}$", va="bottom")
    fs.save(c.fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
