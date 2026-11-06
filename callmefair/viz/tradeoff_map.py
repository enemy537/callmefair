"""Fairea trade-off map (paper Fig. 2): baseline, five regions and method points."""

from __future__ import annotations

import numpy as np
import pandas as pd

from callmefair.viz import _style

_TINT = {"win-win": "#ffffff", "inverted": "#e9e8e4", "lose-lose": "#c9c8c3",
         "good trade-off": "#dff0d4", "bad trade-off": "#f6e3c8"}
_AXIS_NAME = {"acc": "accuracy", "balanced_acc": "balanced accuracy", "spd": "|SPD|",
              "aod": "|AOD|", "eod": "|EOD|", "fair_score": r"fairness score $\mathcal{F}$",
              "fair_score_uncapped": r"uncapped fairness score $\mathcal{F}$"}


def plot_tradeoff_map(baseline, regions, *, ax=None, label: bool = True):
    """Draw the Fairea map for ``regions`` (output of :func:`callmefair.tradeoff.tradeoff_regions`).

    Axes follow the paper: x = normalised bias (0 = least biased baseline point),
    y = normalised accuracy (1 = original model). The original model F_OM is at (1, 1);
    the baseline joins it to the fully mutated model at (0, 0). Regions: win-win (above
    and left of F_OM), inverted (above, right), lose-lose (below, right), and below-left
    split by the baseline into good (above it) and poor (below it) trade-offs.

    Args:
        baseline: output of :func:`callmefair.tradeoff.create_baseline`.
        regions: output of :func:`callmefair.tradeoff.tradeoff_regions` (its ``attrs``
            select the strategy and metrics used for the baseline).
        ax: optional axes.
        label: direct-label win-win and good trade-off points.
    Returns:
        The matplotlib Figure.
    """
    from matplotlib.lines import Line2D
    from callmefair.tradeoff import PAPER_NAMES, baseline_curve, normalize

    meta = regions.attrs or {}
    acc_name, bias_name = meta.get("accuracy", "balanced_acc"), meta.get("bias", "spd")
    curve = baseline_curve(baseline, meta.get("strategy"), accuracy=acc_name, bias=bias_name)
    b_acc, b_bias, _ = normalize(curve["accuracy"], curve["bias"])
    fig, ax = _style.new_axes(ax, figsize=(7.2, 5.6))

    pts_x, pts_y = regions["bias_norm"].to_numpy(float), regions["acc_norm"].to_numpy(float)
    x0 = min(-0.05, np.nanmin(pts_x, initial=0) - 0.05)
    x1 = max(1.25, np.nanmax(pts_x, initial=1) + 0.05)
    y0 = min(-0.05, np.nanmin(pts_y, initial=0) - 0.05)
    y1 = max(1.25, np.nanmax(pts_y, initial=1) + 0.05)
    o_bias, o_acc = float(b_bias[0]), float(b_acc[0])

    # regions (paper Fig. 2)
    order = np.argsort(b_bias, kind="stable")
    xs = np.linspace(x0, o_bias, 400)
    line_y = np.interp(xs, b_bias[order], b_acc[order])
    ax.fill_between([x0, o_bias], o_acc, y1, color=_TINT["win-win"], zorder=0)
    ax.fill_between([o_bias, x1], o_acc, y1, color=_TINT["inverted"], zorder=0)
    ax.fill_between([o_bias, x1], y0, o_acc, color=_TINT["lose-lose"], zorder=0)
    ax.fill_between(xs, np.minimum(line_y, o_acc), o_acc, color=_TINT["good trade-off"], zorder=0)
    ax.fill_between(xs, y0, np.minimum(line_y, o_acc), color=_TINT["bad trade-off"], zorder=0)
    for name, (x, y) in {"win-win": (x0 + 0.03, y1 - 0.04), "inverted": (x1 - 0.03, y1 - 0.04),
                         "lose-lose": (x1 - 0.03, y0 + 0.04),
                         "good trade-off": (x0 + 0.03, o_acc - 0.05),
                         "bad trade-off": (o_bias - 0.03, y0 + 0.04)}.items():
        ha = "left" if x < o_bias else "right"
        ax.text(x, y, PAPER_NAMES[name], ha=ha, va="center", fontsize=9, style="italic",
                color=_style.INK_SECONDARY, zorder=1)

    # baseline (mutation degrees) and the original model
    ax.plot(b_bias, b_acc, color=_style.INK, lw=1.6, marker="|", ms=9, zorder=3)
    for d, x, y in zip(curve["degree"], b_bias, b_acc):
        if np.isclose(d, 1.0) or np.isclose(d, 0.5):
            ax.annotate(f"F{int(round(d * 100))}", (x, y), xytext=(6, -10),
                        textcoords="offset points", fontsize=8, color=_style.INK_SECONDARY)
    ax.plot([o_bias], [o_acc], marker="*", ms=14, color=_style.INK, ls="", zorder=5)
    ax.annotate("original model (F_OM)", (o_bias, o_acc), xytext=(8, 6),
                textcoords="offset points", fontsize=8, color=_style.INK)

    # method points and selective labels (largest trade-off areas)
    models = list(dict.fromkeys(regions["model"]))
    many = len(models) > 3
    for i, m in enumerate(models):
        sub = regions[regions["model"] == m]
        ax.plot(sub["bias_norm"], sub["acc_norm"], ls="", marker=_style.marker(i) if many else "o",
                ms=7, color=_style.color(i), mec=_style.SURFACE, mew=1.0, zorder=4)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if label:   # selective labels: win-win points and the five largest good trade-offs
        top = regions[regions["region"] == "good trade-off"].nlargest(5, "area")
        shown = pd.concat([regions[regions["region"] == "win-win"], top])
        _place_labels(ax, [(r["bias_norm"], r["acc_norm"], f'{r["BM"]} ({r["model"][:3]})')
                           for _, r in shown.iterrows()])
    ax.set_xlabel(f"normalised bias ({_AXIS_NAME.get(bias_name, bias_name)}, lower is fairer)")
    ax.set_ylabel(f"normalised {_AXIS_NAME.get(acc_name, acc_name)} (higher is better)", labelpad=8)
    ax.set_title("Fairea trade-off map", fontsize=11)
    handles = [Line2D([], [], color=_style.INK, lw=1.6, marker="|", ms=9,
                      label=f"baseline (mutate to {meta.get('strategy', '?').replace('to_', '')})")]
    handles += [Line2D([], [], ls="", marker=_style.marker(i) if many else "o", ms=7,
                       color=_style.color(i), label=str(m)) for i, m in enumerate(models)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
              fontsize=8, frameon=False)
    return fig


def _place_labels(ax, items, fontsize: int = 7) -> None:
    """Greedy label placement: first offset whose box clears earlier labels and points."""
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    taken = [(*to_px((x, y)), 6, 6) for x, y, _ in items]          # point marker boxes
    offsets = [(10, 8), (10, -14), (-10, 8), (-10, -14), (10, 22), (-10, 22),
               (10, -28), (-10, -28), (24, 0), (-24, 0)]
    for x, y, text in items:
        px, py = to_px((x, y))
        w, h = 0.55 * fontsize * len(text), 1.4 * fontsize
        for dx, dy in offsets:
            cx = px + dx + (w / 2 if dx > 0 else -w / 2)
            cy = py + dy + h / 2
            if all(abs(cx - tx) > (w + tw) / 2 or abs(cy - ty) > (h + th) / 2
                   for tx, ty, tw, th in taken):
                break
        taken.append((cx, cy, w, h))
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", fontsize=fontsize, color=_style.INK,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=_style.INK_SECONDARY))
