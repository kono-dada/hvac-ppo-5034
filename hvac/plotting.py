"""Shared figure style for the report (docs/report/figures/*.png).

Every figure script calls ``use_style()`` once and takes its colours from here,
so that the same thing (the thermostat, the oracle, the baseline, the comfort
band, ...) looks the same in every figure.

Colour conventions used throughout the report:

* thermostat = orange, oracle = black, baseline = black (the text relies on these);
* heating = red, cooling = blue, comfort band = green;
* in the ablation figures, a value below the baseline's is drawn in blues, a
  value above it in oranges/reds, darker the further it is from the baseline.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- palette
INK = "#1f2328"        # text, axes, the oracle, the baseline
MUTED = "#6e7781"      # secondary text, reference lines
FAINT = "#d0d7de"      # grid, thin separators
BG_SOFT = "#f6f8fa"    # soft panel backgrounds

THERMOSTAT = "#e8833a"
ORACLE = INK
BASELINE = INK
OFF = "#8c959f"

HEAT = "#cf3e3e"
HEAT_SOFT = "#fbe3e0"
COOL = "#2f6fb0"
COOL_SOFT = "#dde9f6"
COMFORT = "#4c9a5a"
COMFORT_SOFT = "#dcefdc"

AIR = "#e8833a"        # air temperature (same family as the heat pump's quick effect)
MASS = "#7a6249"       # wall/floor temperature
OUTDOOR = "#5b8fbf"
SUN = "#d99a1e"
INTERNAL = "#4c9a5a"
PREDICTION = "#c9b8e8"  # predicted ranges in the ablation figures

# Values below / above the baseline: index 0 is nearest the baseline.
_BELOW = ["#6baed6", "#2b7bba", "#08427e"]
_ABOVE = ["#f4a259", "#d9542b", "#8f1d14"]

FONT_CANDIDATES = ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC",
                   "Source Han Sans SC", "DejaVu Sans"]


def _available_fonts() -> list[str]:
    names = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in FONT_CANDIDATES if f in names] or ["DejaVu Sans"]


def use_style() -> None:
    """Apply the report style to matplotlib's global rcParams."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _available_fonts(),
            "axes.unicode_minus": False,
            "mathtext.fontset": "stixsans",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 8,
            "axes.labelsize": 10,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#8c959f",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": FAINT,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "xtick.color": "#57606a",
            "ytick.color": "#57606a",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.handlelength": 1.8,
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "text.color": INK,
            "figure.dpi": 100,
            "figure.facecolor": "white",
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "savefig.facecolor": "white",
        }
    )


def ablation_colors(values: list[float], baseline: float) -> list[str]:
    """Blues for values below the baseline, oranges/reds above it, the baseline itself in black."""
    below = sorted((v for v in values if v < baseline), reverse=True)  # nearest the baseline first
    above = sorted(v for v in values if v > baseline)
    out = []
    for v in values:
        if v == baseline:
            out.append(BASELINE)
        elif v < baseline:
            out.append(_BELOW[below.index(v)])
        else:
            out.append(_ABOVE[above.index(v)])
    return out


def panel_title(ax, letter: str, text: str, **kw) -> None:
    """Bold left-aligned title with a panel letter, e.g. "(a)  成本"."""
    ax.set_title(f"({letter})  {text}", **kw)


def header(fig, title: str, note: str | None = None, x: float = 0.01, top: float = 1.0) -> None:
    """Left-aligned figure title, with an optional grey note line, placed just above `top` (figure fraction)."""
    from matplotlib.transforms import offset_copy

    y = 4
    if note:
        fig.text(x, top, note, ha="left", va="bottom", fontsize=9.5, color=MUTED,
                 transform=offset_copy(fig.transFigure, fig=fig, y=y, units="points"))
        y += 17
    fig.text(x, top, title, ha="left", va="bottom", fontsize=13.5, weight="semibold", color=INK,
             transform=offset_copy(fig.transFigure, fig=fig, y=y, units="points"))


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
