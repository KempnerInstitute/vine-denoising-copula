"""Camera-ready figure style for the VDC paper (NeurIPS 2026).

Figures are drawn at their printed size so that font sizes in the PDF are the sizes set here:
the NeurIPS text width is 5.5 in, and every main-text figure is included at \\textwidth without
rescaling. Text and math use STIX, a Times-style TrueType face close to the paper's Times (ptm) body font,
so figure and body typography match (the system Nimbus Roman is OpenType/CFF and embeds poorly).

Every method has one colour (Okabe-Ito based, colour-blind safe) together with its own marker,
line style and hatch, so no plot identifies a series by colour alone.

Saved inputs are read from one folder with a MANIFEST.json of sha256 values: ``inputs/`` next to the
scripts folder in the code release, otherwise ``analysis/figure_inputs/`` of the paper repository.

Usage:
    from vdc_figstyle import apply, STYLE, TEXTWIDTH, panel_labels, save
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXTWIDTH = 5.5  # inches, NeurIPS \textwidth
_HERE = Path(__file__).resolve().parent
_RELEASE_INPUTS = _HERE.parent / "inputs"
IS_RELEASE = (_RELEASE_INPUTS / "MANIFEST.json").exists()
INPUTS_DIR = _RELEASE_INPUTS if IS_RELEASE else _HERE.parent / "analysis" / "figure_inputs"
FIGURES_DIR = _HERE.parent / ("output" if IS_RELEASE else "figures")

# Font sizes (pt) at printed size; captions in the paper are 9 pt. Nothing prints below 7 pt.
FS_TICK = 7.0
FS_LABEL = 7.5
FS_TITLE = 8.0
FS_LEGEND = 7.0
FS_ANNOT = 7.0
FS_PANEL = 9.0


@dataclass(frozen=True)
class MethodStyle:
    label: str
    color: str
    marker: str
    linestyle: object
    hatch: str


# One entry per method used anywhere in the paper.
STYLE = {
    "vdc": MethodStyle("VDC", "#D55E00", "o", "-", "////"),
    "tll": MethodStyle("pyvine-TLL", "#0072B2", "^", "--", "...."),
    "param": MethodStyle("pyvine-param", "#56B4E9", "s", "-.", "\\\\\\\\"),
    "flow": MethodStyle("RealNVP", "#CC79A7", "D", ":", "xxx"),
    "gauss": MethodStyle("Gaussian copula", "#E69F00", "v", (0, (6, 2)), "---"),
    "hist": MethodStyle("Histogram", "#999999", "h", (0, (1, 1)), "++"),
    "kde": MethodStyle("KDE", "#8C6D31", "p", (0, (3, 1, 1, 1)), "oo"),
    "ksg": MethodStyle("KSG", "#009E73", "P", "--", "...."),
    "mine": MethodStyle("MINE", "#4D4D4D", "X", "-.", "\\\\\\\\"),
    "infonce": MethodStyle("InfoNCE", "#8C7A00", "<", ":", "xxx"),
    "nwj": MethodStyle("NWJ", "#7F7F7F", ">", (0, (5, 1)), "++"),
    "minde": MethodStyle("MINDE", "#882255", "d", (0, (3, 1, 1, 1)), "oo"),
}


def apply() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],  # Times-like TrueType; embeds cleanly as Type 42
        "mathtext.fontset": "stix",
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "bold",
        "axes.titlepad": 3.0,
        "axes.labelpad": 2.0,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "xtick.major.pad": 1.8,
        "ytick.major.pad": 1.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": FS_LEGEND,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.25,
        "legend.borderaxespad": 0.2,
        "legend.columnspacing": 0.8,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.8,
        "errorbar.capsize": 1.8,
        "hatch.linewidth": 0.5,
        "patch.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.35,
        "grid.color": "0.8",
    })


def line_kw(key: str, **over) -> dict:
    s = STYLE[key]
    kw = dict(color=s.color, marker=s.marker, linestyle=s.linestyle, label=s.label,
              markeredgecolor="white", markeredgewidth=0.4)
    kw.update(over)
    return kw


def bar_kw(key: str, **over) -> dict:
    s = STYLE[key]
    kw = dict(color=s.color, hatch=s.hatch, edgecolor="0.15", linewidth=0.5, label=s.label)
    kw.update(over)
    return kw


def panel_labels(fig, axes, letters, y: float | None = None, dx: float = 0.0) -> None:
    """Bold panel letters at the left edge of each axes' tight bounding box (tick and axis labels
    included), all on one baseline at height ``y`` in figure coordinates (default: just above the
    highest axes title in the row)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [ax.get_tightbbox(r).transformed(inv) for ax in axes]
    if y is None:
        y = max(b.y1 for b in boxes) + 0.005
    for ax, b, letter in zip(axes, boxes, letters):
        fig.text(b.x0 + dx, y, f"({letter})", fontsize=FS_PANEL, fontweight="bold",
                 ha="left", va="top")


def add_io_args(ap, name: str) -> None:
    """The options every figure script shares: input folder, output folder and output name."""
    ap.add_argument("--inputs", type=Path, default=INPUTS_DIR,
                    help="folder of saved inputs with MANIFEST.json (default: %(default)s)")
    ap.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    ap.add_argument("--name", default=name)


def input_file(inputs: Path, rel: str, override: Path | None = None) -> Path:
    """Path of a saved input: ``inputs/rel`` after checking its sha256 against ``inputs/MANIFEST.json``,
    or an explicitly given ``override`` path, used as is."""
    if override is not None:
        print(f"input {hashlib.sha256(Path(override).read_bytes()).hexdigest()}  {override} (not checked)")
        return Path(override)
    entry = json.loads((Path(inputs) / "MANIFEST.json").read_text())["files"].get(rel)
    if entry is None:
        raise SystemExit(f"{rel} is not listed in {inputs}/MANIFEST.json")
    p = Path(inputs) / rel
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise SystemExit(f"{p}: sha256 {digest} does not match MANIFEST.json ({entry['sha256']})")
    print(f"input {digest}  {rel}")
    return p


def save(fig, name: str, figures_dir: Path | None = None) -> None:
    d = Path(figures_dir) if figures_dir else FIGURES_DIR
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{name}.{ext}", format=ext)
        print("saved", d / f"{name}.{ext}")
    plt.close(fig)
