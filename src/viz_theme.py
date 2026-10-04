"""
viz_theme.py
============
One shared visual system for every figure in this project, so the EDA plots, the
model-benchmark plots and the Streamlit app read as a single publication.

The colour values below are a *validated* palette: the ordinal stage ramp, the
ordinal depth ramp and the two-series comparison palette were each checked with
the data-visualisation validator for lightness banding, chroma floor, adjacent
colour-vision-deficiency separation, normal-vision separation and contrast
against the chart surface. Do not substitute colours ad hoc - re-validate first.

Encoding rules applied throughout
---------------------------------
* The five scale-up stages are an ORDERED sequence, so they take a single-hue
  ordinal ramp (light -> dark), never eight arbitrary categorical hues.
* Depth classes are likewise ordinal.
* Correlations span -1..+1, a polarity, so they take the diverging blue<->red
  pair with a neutral grey midpoint - never a rainbow.
* Shade families are nominal but too numerous for distinct hues, so family
  charts use a single hue and let position carry identity.
* Baseline-vs-AI comparisons are two true categories: slots 1 and 2.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Tokens (light surface; figures are produced for print)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots 1-3: validated for ALL pairs, so safe for scatter, bubble
# and any form where every series can sit beside every other.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

# The full eight-slot categorical order, validated on the ADJACENT pairlist
# (lines, bars, stacks). Assign in this fixed order and never cycle: a ninth
# series folds into "Other" or becomes a small multiple.
#
# Slots 3 and 4 fall below 3:1 contrast on the light surface, so any chart
# using four or more slots must ship relief - visible direct labels or an
# adjacent table view. The recipe table beside the trajectory chart serves
# that purpose in this application.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Ordinal ramp for the five scale-up stages (validated --ordinal, light).
STAGE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

# Ordinal ramp for the four depth classes (validated --ordinal, light).
DEPTH_RAMP = ["#86b6ef", "#3987e5", "#256abf", "#104281"]

# Single sequential hue for magnitude.
SEQ_HUE = "#2a78d6"

# Diverging pair with a neutral grey midpoint, for correlation matrices.
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red",
    ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f0a3a3", "#d03b3b", "#7d1f1f"],
)

STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

DEPTH_ORDER = ["Pastel/Light", "Medium", "Dark", "Extra Dark"]


def stage_colors(n: int = 5) -> list:
    """n evenly spaced steps of the ordinal stage ramp."""
    if n == 5:
        return list(STAGE_RAMP)
    idx = [round(i * (len(STAGE_RAMP) - 1) / max(n - 1, 1)) for i in range(n)]
    return [STAGE_RAMP[i] for i in idx]


def depth_colors() -> dict:
    return dict(zip(DEPTH_ORDER, DEPTH_RAMP))


def apply_theme() -> None:
    """Install the house style as matplotlib defaults."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "figure.edgecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.edgecolor": SURFACE,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 110,

        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.prop_cycle": mpl.cycler(color=SERIES),

        "grid.color": GRID,
        "grid.linewidth": 0.7,

        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,

        "lines.linewidth": 2.0,
        "lines.markersize": 8,

        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "text.color": INK,
    })


def titled(ax, title: str, subtitle: str | None = None) -> None:
    """
    Bold title with an optional recessive subtitle underneath.

    The title is padded clear of the subtitle rather than both being drawn at
    the axes top, which would overlap them.
    """
    if subtitle:
        ax.set_title(title, loc="left", color=INK, fontweight="bold",
                     fontsize=12, pad=30)
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=9, color=INK_MUTED)
    else:
        ax.set_title(title, loc="left", color=INK, fontweight="bold",
                     fontsize=12, pad=10)


def source_note(fig, text: str) -> None:
    """Recessive provenance note at the foot of a figure."""
    fig.text(0.005, -0.012, text, ha="left", va="top", fontsize=7.5,
             color=INK_MUTED)


def strip_grid(ax) -> None:
    ax.grid(False)


def save(fig, path: str) -> None:
    """Save a figure, tolerating OneDrive's transient file locks."""
    import time
    for attempt in range(6):
        try:
            fig.savefig(path)
            plt.close(fig)
            return
        except PermissionError:
            time.sleep(0.4 * (attempt + 1))
    alt = path.replace(".png", ".new.png")
    fig.savefig(alt)
    plt.close(fig)
    print(f"[viz] target locked; wrote {alt}")
