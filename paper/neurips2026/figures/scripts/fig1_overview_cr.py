#!/usr/bin/env python3
"""Camera-ready Fig. 1: VDC overview, drawn as a vector schematic at printed size.

(a) Pretrain once on simulated copulas: a deck of noisy histograms H~ -> trained U-Net f_theta -> IPFP
    -> output D^, compared with the true density c by the loss L, which updates f_theta. The pictured H~ is
    one training input (uniform mixing of weight 0.3) and D^ the deployed network's output for exactly that
    input at its noise level.
(b) Fit a D-vine tree by tree: the pairs of tree k are histogrammed and passed through the frozen
    f_theta in batches (one network pass per batch), IPFP projects each grid, and the cached h-functions
    transform the pairs that feed tree k+1.
(c) One estimator for every edge of the vine factorization, and what the fitted vine provides.
(d) Example on the X-shaped copula (Complex-X) of the training zoo: true, pyvine-TLL, VDC.
All images are real outputs read from fig1_overview/fig1_overview_data.npz in the inputs folder
(written by fig1_overview_data.py with the frozen checkpoint and the pinned library).

The figure is 5.5 in wide; each row's panel widths and the figure height follow from the measured
content, with equal outer margins and gutters.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vdc_figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, PowerNorm  # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

W, H_MAX = fs.TEXTWIDTH, 2.75
M, GAP, PM = 0.03, 0.06, 0.07                 # outer margin, gutter between panels, inner panel margin (in)

INK, MUTED, ARROW, FRAME = "#262626", "#6E6E6E", "#4A4A4A", "#8A8A8A"
PANEL_FC, PANEL_EC = "#F6F6F6", "#CCCCCC"
TRAINED, FROZEN, LOCK = fs.STYLE["vdc"].color, "#9A9A9A", "#6E6E6E"
TRAINED_SH = (TRAINED, "#EE9A66", "#A04600", "#7A3500")   # front, top and side faces and edges of the network boxes
FROZEN_SH = (FROZEN, "#C4C4C4", "#7A7A7A", "#5A5A5A")
CMAP = LinearSegmentedColormap.from_list(
    "white_navy", ["#FFFFFF", "#DEEBF7", "#9ECAE1", "#4292C6", "#08519C", "#08306B"])
DOTS = "#3A3A3A"
LW_FRAME, LW_ARROW = 0.5, 0.8
S, OFF = 0.39, 0.020                          # card size and deck offset in (a) and (b)
UW = 0.38                                     # network glyph width
UA = 0.185                                    # depth of the glyph below its centre line
BW, BH = 0.23, 0.15                           # IPFP box
HW = 0.37                                     # cached h-table box
TITLE_BAND = 0.165                            # panel top to the bottom of its title


class Canvas:
    def __init__(self, h):
        self.h = h
        self.fig = plt.figure(figsize=(W, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")
        self.r = self.fig.canvas.get_renderer()

    def measure(self, s, size=None, **kw):
        t = self.ax.text(0, 0, s, fontsize=size or fs.FS_ANNOT, **kw)
        bb = t.get_window_extent(self.r)
        t.remove()
        return bb.width / self.fig.dpi, bb.height / self.fig.dpi

    def text(self, x, y, s, size=None, ha="center", va="top", color=INK, **kw):
        return self.ax.text(x, y, s, ha=ha, va=va, fontsize=size or fs.FS_ANNOT, color=color, **kw)

    def panel(self, x, y, w, h, letter, title, subtitle=None):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.045",
                                         fc=PANEL_FC, ec=PANEL_EC, lw=LW_FRAME, zorder=0))
        yt = y + h - 0.055
        self.text(x + 0.07, yt, f"({letter})", size=fs.FS_PANEL, ha="left", fontweight="bold")
        self.text(x + 0.33, yt - 0.007, title, size=fs.FS_TITLE, ha="left", fontweight="bold")
        if subtitle:
            tw, _ = self.measure(title, fs.FS_TITLE, fontweight="bold")
            self.text(x + 0.33 + tw + 0.07, yt - 0.012, subtitle, ha="left", style="italic", color=MUTED)

    def axes(self, x, y, w, h):
        return self.fig.add_axes([x / W, y / self.h, w / W, h / self.h])

    def _frame(self, a, z):
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():  # all four sides (the shared style hides top and right spines)
            sp.set_visible(True)
            sp.set_linewidth(LW_FRAME)
            sp.set_color(FRAME)
        a.set_zorder(z)
        return a

    def card(self, x, y, s, z=2, data=None, norm=None, pts=None, dot=0.5):
        a = self.axes(x, y, s, s)
        a.set_facecolor("white")
        if data is not None:
            a.imshow(data.T, origin="lower", extent=(0, 1, 0, 1), cmap=CMAP, norm=norm,
                     interpolation="nearest", aspect="auto")
        if pts is not None:
            a.scatter(pts[:, 0], pts[:, 1], s=dot, color=DOTS, linewidths=0, rasterized=True)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        return self._frame(a, z)

    def deck(self, x, y, s, **front):
        """Two blank cards behind the front card (a batch of many examples)."""
        for j in (2, 1):
            self.card(x + j * OFF, y + j * OFF, s, z=3 + (2 - j))
        return self.card(x, y, s, z=5, **front)

    def arrow(self, x1, y1, x2, y2, style="-|>", scale=7):
        self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=scale, lw=LW_ARROW,
                                          color=ARROW, zorder=1, shrinkA=0, shrinkB=0))

    def path(self, xs, ys):
        self.ax.plot(xs, ys, color=ARROW, lw=LW_ARROW, zorder=1, solid_capstyle="butt", solid_joinstyle="miter")

    def unet(self, xc, yc, shades):
        """U-Net glyph in the usual feature-map style: boxes whose height (spatial size) shrinks and whose
        thickness (channels) grows with depth step down from the input level to the bottleneck and back up.
        A thin trunk joins the levels and dashed skip connections cross the two upper levels; the top
        level spans the centre line, where the pipeline arrows enter and leave."""
        front, top, side, edge = shades
        levels = [(0.185, 0.020), (0.105, 0.030), (0.070, 0.046)]      # (spatial size, thickness), in
        bottoms = [-0.072, -0.140, -UA]                                  # front-face bottoms below yc
        boxes = []
        for dx, lev in zip((-0.190, -0.106, -0.036, 0.043, 0.114), (0, 1, 2, 1, 0)):   # e0 e1 bottleneck d1 d0
            s, t = levels[lev]
            boxes.append((xc + dx, yc + bottoms[lev], t, s, 0.28 * s))
        self.ax.plot([x + t / 2 for x, _, t, _, _ in boxes], [y + s / 2 for _, y, _, s, _ in boxes],
                     color=edge, lw=0.6, zorder=1.5, solid_joinstyle="round")
        kw = dict(ec=edge, lw=0.3, joinstyle="round", zorder=2)
        for x, y, t, s, d in boxes:     # top face, side face (depth up and to the right), front face
            self.ax.add_patch(Polygon([(x, y + s), (x + d, y + s + d), (x + t + d, y + s + d), (x + t, y + s)], fc=top, **kw))
            self.ax.add_patch(Polygon([(x + t, y), (x + t + d, y + d), (x + t + d, y + s + d), (x + t, y + s)], fc=side, **kw))
            self.ax.add_patch(Rectangle((x, y), t, s, fc=front, **kw))
        for (x0, y0, t0, s0, d0), (x1, *_) in ((boxes[0], boxes[4]), (boxes[1], boxes[3])):
            yy = y0 + 0.62 * s0         # above the deeper levels; the tail starts behind the side face
            self.ax.add_patch(FancyArrowPatch((x0 + t0 + 0.62 * d0 + 0.006, yy), (x1 - 0.006, yy), arrowstyle="-|>",
                                              mutation_scale=3.8, lw=0.6, color=ARROW, linestyle=(0, (2.2, 1.2)),
                                              shrinkA=0, shrinkB=0, zorder=1.5))

    def tag(self, xc, y0):
        """Small rounded f_theta tag (the same estimator on every edge), outlined like the frozen network."""
        tw, th = 0.12, 0.10
        self.ax.add_patch(FancyBboxPatch((xc - tw / 2, y0), tw, th, boxstyle="round,pad=0,rounding_size=0.02",
                                         fc="white", ec=FROZEN, lw=0.6, zorder=2))
        self.text(xc, y0 + th / 2, r"$f_\theta$", va="center", color=ARROW, zorder=3)

    def padlock(self, x, y):
        """Small padlock with its body's lower-left corner at (x, y)."""
        bw, bh = 0.052, 0.040
        self.ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.007",
                                         fc=LOCK, ec="none", zorder=3))
        self.ax.add_patch(Arc((x + bw / 2, y + bh), 0.032, 0.050, theta1=0, theta2=180, lw=0.9, color=LOCK, zorder=3))

    def box(self, xc, yc, w, h, lines):
        self.ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                         fc="white", ec=FRAME, lw=0.6, zorder=2))
        for dy, s, kw in lines:
            self.text(xc, yc + dy, s, va="center", zorder=3, **kw)


def row_plan(c, items, min_gap=0.09, label_pad=0.05):
    """Minimal width of a row of icons and its gaps: every gap leaves room for an arrow and keeps the
    neighbouring one-line labels apart, and the outer labels stay inside the row.
    items: (width, label or label width, minimum gap after)."""
    lw = [l if isinstance(l, float) else (c.measure(l)[0] if l else 0.0) for _, l, _ in items]
    lead = max(0.0, (lw[0] - items[0][0]) / 2)
    tail = max(0.0, (lw[-1] - items[-1][0]) / 2)
    gaps = []
    for i in range(len(items) - 1):
        clear = (lw[i] + lw[i + 1]) / 2 + label_pad - (items[i][0] + items[i + 1][0]) / 2 if lw[i] and lw[i + 1] else 0.0
        gaps.append(max(min_gap, clear, items[i][2]))
    return lead + sum(w for w, _, _ in items) + sum(gaps) + tail, lead, gaps


def place(x0, width, items, plan):
    """Left edges of the row elements, spreading the spare width equally over the gaps."""
    need, lead, gaps = plan
    extra = (width - need) / len(gaps)
    xs, x = [], x0 + lead
    for i, (w, _, _) in enumerate(items):
        xs.append(x)
        x += w + (gaps[i] + extra if i < len(gaps) else 0.0)
    return xs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fs.add_io_args(ap, "fig1_overview")
    ap.add_argument("--data", type=Path, help="use this fig1_overview_data.npz instead of the saved one")
    a = ap.parse_args()
    fs.apply()
    D = np.load(fs.input_file(a.inputs, "fig1_overview/fig1_overview_data.npz", a.data))

    def pnorm(*arrs):  # light-to-dark sequential scale with a square-root stretch, capped at the 99th percentile
        return PowerNorm(gamma=0.5, vmin=0.0, vmax=max(float(np.percentile(x, 99)) for x in arrs))

    # ---------------- geometry (measured on a scratch canvas) ----------------
    m = Canvas(H_MAX)
    dw = S + 2 * OFF                   # deck width
    lock_w = m.measure(r"frozen $f_\theta$")[0] + 0.09
    items_a = [(dw, r"histogram $\tilde H$", 0), (UW, r"U-Net $f_\theta$", 0), (BW, None, 0),
               (S, r"output $\hat D$", 0.15), (S, "true density $c$", 0)]   # room for the loss connector
    items_b = [(dw, "tree-$k$ pairs", 0), (dw, "histograms", 0), (UW, lock_w, 0), (BW, None, 0),
               (dw, r"edge densities $\hat c_e$", 0), (HW, None, 0)]
    plan_a, plan_b = row_plan(m, items_a), row_plan(m, items_b)
    spare1 = W - 2 * M - GAP - 4 * PM - plan_a[0] - plan_b[0]
    assert spare1 >= 0, f"row 1 does not fit: {spare1:.3f} in short"
    wa, wb = plan_a[0] + 2 * PM + spare1 / 2, plan_b[0] + 2 * PM + spare1 / 2
    # (c): vine diagram and chips; (d) takes the rest of the row
    nw, nh, step = 0.20, 0.12, 0.27    # one node size for every tree; D-vine pyramid
    outs = [("Likelihood", r"$c(\mathbf{u})=\prod_e \hat c_e(u_{j|D_e},\,u_{k|D_e})$"),
            ("Sampling", "inverse $h$, tree by tree"), ("Information", "MI per edge, TC per tree")]
    wo = max(m.measure(t)[0] for _, t in outs) + 0.10
    wc = 0.19 + 3 * step + nw + 0.10 + wo + PM
    wd = W - 2 * M - GAP - wc
    s3, g3 = (wd - 2 * PM - 3 * 0.05) / 4, 0.05
    # heights: row 1 from the icon stack, row 2 from the larger of (c) and (d)
    h1 = 0.325 + dw + 0.045 + 0.075 + 0.035 + TITLE_BAND
    hc = 0.30
    h2 = max(TITLE_BAND + 0.04 + 3 * hc + 2 * 0.03 + 0.045,          # (c)
             TITLE_BAND + 0.105 + s3 + 0.04 + 0.235 + 0.045)          # (d): colour bar in the header
    H = 2 * M + GAP + h1 + h2
    assert H <= H_MAX + 1e-9, H
    plt.close(m.fig)
    c = Canvas(H)

    # ---------------- row 1 ----------------
    y1 = M + h2 + GAP
    yi = y1 + 0.325                    # bottom of the front cards
    yc = yi + S / 2                    # icon centre line (arrows)
    ylab = yi + dw + 0.045             # common baseline of the one-line labels
    yloop = yi - 0.155                 # loop lines below the row
    xa0, xb0 = M, M + wa + GAP

    # (a) pretrain once
    c.panel(xa0, y1, wa, h1, "a", "Pretrain once", "on simulated copulas")
    xa = place(xa0 + PM, wa - 2 * PM, items_a, plan_a)
    c.deck(xa[0], yi, S, data=D["pre_hist_mixed"], norm=pnorm(D["pre_hist_mixed"]))
    c.unet(xa[1] + UW / 2, yc, TRAINED_SH)
    c.box(xa[2] + BW / 2, yc, BW, BH, [(0, "IPFP", {})])
    dn = pnorm(D["pre_mixed_proj"], D["pre_true"])
    c.card(xa[3], yi, S, data=D["pre_mixed_proj"], norm=dn)   # the forward pass of the pictured input
    c.card(xa[4], yi, S, data=D["pre_true"], norm=dn)
    for x, (w, lab, _) in zip(xa, items_a):
        if lab:
            c.text(x + w / 2, ylab, lab, va="baseline")
    ends = [xa[0] + dw, xa[1] + UW, xa[2] + BW]
    for i in range(3):
        c.arrow(ends[i] + 0.015, yc, xa[i + 1] - 0.015, yc)
    c.arrow(xa[3] + S + 0.02, yc, xa[4] - 0.02, yc, style="<|-|>", scale=5)
    xm = (xa[3] + S + xa[4]) / 2
    _, lh = c.measure(r"loss $\mathcal{L}$")
    c.text(xm, yi - 0.018, r"loss $\mathcal{L}$")      # same band as "one pass per batch" in (b)
    xu = xa[1] + UW / 2                                 # the loss updates the network
    c.path([xm, xm, xu], [yi - 0.028 - lh, yloop, yloop])
    c.arrow(xu, yloop, xu, yc - UA - 0.012)
    c.text((xm + xu) / 2, yloop - 0.025, r"gradient steps update $f_\theta$", style="italic")

    # (b) fit a D-vine, tree by tree, with the frozen network
    c.panel(xb0, y1, wb, h1, "b", "Fit a D-vine, tree by tree")
    xb = place(xb0 + PM, wb - 2 * PM, items_b, plan_b)
    c.deck(xb[0], yi, S, pts=D["dep0_sample"][:200], dot=0.32)
    c.deck(xb[1], yi, S, data=D["dep0_hist"], norm=pnorm(D["dep0_hist"]))
    c.unet(xb[2] + UW / 2, yc, FROZEN_SH)
    c.box(xb[3] + BW / 2, yc, BW, BH, [(0, "IPFP", {})])
    c.deck(xb[4], yi, S, data=D["dep0_proj"], norm=pnorm(D["dep0_proj"]))
    c.box(xb[5] + HW / 2, yc, HW, S, [(0.06, r"$h_e,\ h_e^{-1}$", {"size": fs.FS_LABEL}), (-0.075, "cached", {"color": MUTED})])
    for x, (w, lab, _) in zip(xb, items_b):
        if isinstance(lab, str):
            c.text(x + w / 2, ylab, lab, va="baseline")
    fw = c.measure(r"frozen $f_\theta$")[0]
    xuc = xb[2] + UW / 2
    c.text(xuc - 0.045, ylab, r"frozen $f_\theta$", va="baseline")
    c.padlock(xuc - 0.045 + fw / 2 + 0.03, ylab - 0.004)
    c.text(xuc, yi - 0.018, "one pass per batch", color=MUTED)
    ends = [xb[0] + dw, xb[1] + dw, xb[2] + UW, xb[3] + BW, xb[4] + dw]
    for i in range(5):
        c.arrow(ends[i] + 0.015, yc, xb[i + 1] - 0.015, yc)
    xh, xs0 = xb[5] + HW / 2, xb[0] + S / 2
    c.path([xh, xh, xs0], [yc - S / 2, yloop, yloop])
    c.arrow(xs0, yloop, xs0, yi - 0.01)
    c.text((xh + xs0) / 2, yloop - 0.025, r"$h$-transforms give the pairs of tree $k{+}1$", style="italic")

    # ---------------- row 2 ----------------
    y2 = M
    xc0p, xd0 = M, M + wc + GAP
    # (c) one estimator for every edge
    c.panel(xc0p, y2, wc, h2, "c", "One estimator for every edge")
    top = y2 + h2 - TITLE_BAND - 0.04
    yrow = [top - hc / 2 - k * (hc + 0.03) for k in range(3)]
    xv = xc0p + 0.19 + 1.5 * step + nw / 2        # centre of the vine diagram
    rows = [("$T_1$", ["1", "2", "3", "4"]), ("$T_2$", ["12", "23", "34"]), ("$T_3$", ["13|2", "24|3"])]
    for (lab, nodes), y in zip(rows, yrow):
        n = len(nodes)
        xn = [xv + step * (i - (n - 1) / 2) for i in range(n)]
        c.text(xc0p + 0.095, y, lab, size=fs.FS_LABEL, va="center")
        for i in range(n - 1):
            c.path([xn[i] + nw / 2, xn[i + 1] - nw / 2], [y, y])
            c.tag((xn[i] + xn[i + 1]) / 2, y + nh / 2 + 0.012)   # the same network on every edge
        for x0, t in zip(xn, nodes):
            c.ax.add_patch(FancyBboxPatch((x0 - nw / 2, y - nh / 2), nw, nh, boxstyle="round,pad=0,rounding_size=0.03",
                                          fc="white", ec=FRAME, lw=LW_FRAME, zorder=2))
            c.text(x0, y, t, va="center", zorder=3)
    xo = xc0p + wc - PM - wo           # output chips, one per tree row: what the fitted vine provides
    for k, (y, (t1, t2)) in enumerate(zip(yrow, outs)):
        c.box(xo + wo / 2, y, wo, hc, [(0.078 if k else 0.088, t1, {"fontweight": "bold"}),
                                       (-0.048 if k else -0.062, t2, {})])
    xbr0, xf = xv + 1.5 * step + nw / 2 + 0.03, xo - 0.035     # bracket from the vine to the three uses
    c.path([xbr0, xf], [yrow[1], yrow[1]])
    c.path([xf, xf], [yrow[2], yrow[0]])
    for y in yrow:
        c.path([xf, xo], [y, y])

    # (d) example: X-shaped copula
    c.panel(xd0, y2, wd, h2, "d", "Example: X-shaped copula")
    lnorm = LogNorm(vmin=1e-3, vmax=20.0)
    iy = y2 + 0.045 + 0.235 + 0.04
    x0 = xd0 + PM
    c.card(x0, iy, s3, pts=D["cx_sample"], dot=0.18)
    c.text(x0 + s3 / 2, iy - 0.04, "sample\n$n{=}2000$")
    ise_tll, ise_vdc = D["cx_ise"]
    imgs = [("true density", D["cx_true"], None), ("pyvine-TLL", D["cx_tll"], ise_tll), ("VDC", D["cx_vdc"], ise_vdc)]
    for i, (t, dd, ise) in enumerate(imgs):
        xi = x0 + (i + 1) * (s3 + g3)
        im = c.card(xi, iy, s3, data=np.clip(dd, 1e-3, None), norm=lnorm)
        lab = t if ise is None else f"{t}\nISE {ise:.2g}"
        c.text(xi + s3 / 2, iy - 0.04, lab)
    # horizontal log colour bar in the header, right-aligned with the images
    cbw, cbh = 0.78, 0.045
    xcb, ycb = x0 + 4 * s3 + 3 * g3 - cbw, y2 + h2 - 0.09
    cax = c.axes(xcb, ycb, cbw, cbh)
    cb = c.fig.colorbar(im.images[0], cax=cax, orientation="horizontal")
    cb.set_ticks([1e-3, 1e-1, 10])
    cb.ax.tick_params(labelsize=fs.FS_ANNOT, length=1.8, width=0.5, pad=1.0, which="major")
    cb.ax.tick_params(which="minor", length=0)
    cb.outline.set_linewidth(LW_FRAME)
    cb.outline.set_edgecolor(FRAME)
    print(f"size {W:.2f} x {H:.2f} in; panels a {wa:.2f} b {wb:.2f} c {wc:.2f} d {wd:.2f}; spare row 1 {spare1:.3f}; "
          f"card {S:.2f}, squares {s3:.3f}")
    fs.save(c.fig, a.name, a.figures_dir)


if __name__ == "__main__":
    main()
