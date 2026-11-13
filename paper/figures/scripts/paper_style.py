"""Shared matplotlib style for all VDC paper figures.

Usage:
    from paper_style import apply_style, COLORS, save_figure

    apply_style()
    fig, ax = plt.subplots(...)
    ...
    save_figure(fig, 'fig_name')  # saves .pdf and .png to drafts/figures/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ---------- Color palette ----------
# Professional palette inspired by Nature/Science style
COLORS = {
    "vdc": "#D64541",       # warm red — our method
    "pyvine_tll": "#2ECC71",  # green
    "pyvine_par": "#3498DB",  # blue
    "ksg": "#3498DB",       # blue
    "mine": "#9B59B6",      # purple
    "infonce": "#27AE60",   # green
    "nwj": "#8B6914",       # brown
    "gaussian": "#F39C12",  # amber
    "minde": "#E91E8B",     # pink
    "histogram": "#95A5A6", # gray
    "kde": "#9B59B6",       # purple
    "flow": "#8E44AD",      # dark purple
    # Vine tree colors
    "tree1": "#4A90D9",     # blue
    "tree2": "#F5A623",     # orange
    "tree3": "#7ED321",     # green
    "tree_deep": "#2C3E50", # dark
    # General
    "accent": "#D64541",
    "neutral": "#7F8C8D",
    "bg_light": "#FAFAFA",
}

# Method display names
METHOD_NAMES = {
    "vdc": "VDC (Ours)",
    "pyvine_tll": "pyvine (TLL)",
    "pyvine_par": "pyvine (param.)",
    "ksg": "KSG",
    "mine": "MINE",
    "infonce": "InfoNCE",
    "nwj": "NWJ",
    "gaussian": "Gaussian",
    "minde": "MINDE",
    "flow": "Flow (RealNVP)",
}


def apply_style(
    *,
    font_size: int = 9,
    use_tex: bool = False,
    column_width: float = 3.25,  # ICML single column in inches
    text_width: float = 6.75,    # ICML full width in inches
) -> None:
    """Apply consistent publication style to all matplotlib figures."""
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": font_size,
        "mathtext.fontset": "cm",
        # Use LaTeX rendering only if available
        "text.usetex": use_tex,
        # Axes
        "axes.labelsize": font_size + 1,
        "axes.titlesize": font_size + 1,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.prop_cycle": plt.cycler(color=[
            COLORS["vdc"], COLORS["pyvine_tll"], COLORS["ksg"],
            COLORS["mine"], COLORS["gaussian"], COLORS["flow"],
        ]),
        # Ticks
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # Legend
        "legend.fontsize": font_size - 1,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.85",
        "legend.fancybox": False,
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
        # Figure
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Grid (when used)
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "grid.color": "0.8",
    })


def fig_size(
    width: str = "column",
    aspect: float = 0.7,
    n_panels: int = 1,
) -> tuple[float, float]:
    """Return (width, height) in inches for standard figure sizes.

    Args:
        width: "column" (3.25"), "wide" (5.0"), or "full" (6.75")
        aspect: height/width ratio
        n_panels: number of side-by-side panels (scales width)
    """
    widths = {"column": 3.25, "wide": 5.0, "full": 6.75}
    w = widths.get(width, 3.25)
    return (w, w * aspect)


def save_figure(
    fig: plt.Figure,
    name: str,
    *,
    figures_dir: Optional[Path] = None,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    """Save figure to drafts/figures/ in multiple formats."""
    if figures_dir is None:
        figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        out = figures_dir / f"{name}.{fmt}"
        fig.savefig(str(out), format=fmt)
        print(f"  Saved: {out}")
    plt.close(fig)


def annotate_panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    """Add panel label (a), (b), etc."""
    ax.text(
        x, y, f"({label})",
        transform=ax.transAxes,
        fontsize=10, fontweight="bold",
        va="top", ha="right",
    )
