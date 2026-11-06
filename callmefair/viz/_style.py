"""Shared visual language for callmefair plots.

One validated palette and one set of chart rules for every figure:

* categorical hues are assigned in this fixed order, never cycled; scatter-like
  charts with more than three series add marker shapes as a second encoding;
* sequential magnitude uses one blue ramp; signed changes use blue <-> red with a
  neutral gray midpoint;
* "fair" / "unfair" use the reserved status colors and always carry a text label;
* one y-axis per panel (no twin axes), thin marks, recessive grid.
"""

from __future__ import annotations

# Categorical slots, light surface (validated: adjacent-pair CVD dE >= 8).
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
              "#1c5cab", "#104281", "#0d366b")
DIVERGING = ("#1c5cab", "#6da7ec", "#f0efec", "#ec8b8a", "#c03a3a")  # better ... worse

GOOD, CRITICAL = "#0ca30c", "#d03b3b"          # status: fair / unfair (+ text label)
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def color(i: int) -> str:
    """Categorical color for series index ``i`` (fixed order)."""
    return CATEGORICAL[i % len(CATEGORICAL)]


def marker(i: int) -> str:
    return MARKERS[i % len(MARKERS)]


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("callmefair_seq", SEQUENTIAL)


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("callmefair_div", DIVERGING)


def style_axes(ax) -> None:
    """Recessive chrome: light grid behind data, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def new_axes(ax=None, figsize=(6.4, 4.0)):
    """Return (fig, ax): reuse ``ax`` if given, else create a styled figure."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    else:
        fig = ax.figure
    style_axes(ax)
    return fig, ax
