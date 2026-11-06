"""Fairea trade-off map (paper Fig. 2): baseline, five regions and method points, in the metrics' units."""

from __future__ import annotations

import numpy as np
import pandas as pd

from callmefair.viz import _style

_TINT = {"win-win": "#ffffff", "inverted": "#e9e8e4", "lose-lose": "#c9c8c3",
         "good trade-off": "#dff0d4", "bad trade-off": "#f6e3c8"}
_AXIS_NAME = {"acc": "accuracy", "balanced_acc": "balanced accuracy", "spd": "|SPD|",
              "aod": "|AOD|", "eod": "|EOD|", "fair_score": r"fairness score $\mathcal{F}$",
              "fair_score_uncapped": r"uncapped fairness score $\mathcal{F}$"}


def plot_tradeoff_map(baseline, regions, *, normalized: bool = False, ax=None,
                      label: bool = True):
    """Draw the Fairea map for ``regions`` (output of :func:`callmefair.tradeoff.tradeoff_regions`).

    Like the paper's Fig. 2, the map is drawn in the metrics' own units: x = bias
    (``|SPD|`` by default, lower is fairer), y = accuracy. The baseline is a line from the
    fully mutated model F100 (bias 0, the constant model's accuracy) to the original
    model F_OM. Regions around F_OM: win-win (above-left: more accurate *and* fairer),
    inverted (above-right), lose-lose (below-right); below-left is split by the baseline
    into good (above it) and poor (below it) trade-offs. Regions and areas come from
    Fairea's normalised space (``tradeoff_regions``); min-max scaling each axis does not
    change which region a point falls in, so both views agree. ``normalized=True`` draws
    that space instead (F_OM at (1, 1), F100 at (0, 0)).

    With one baseline per model (a mapping, as ``tradeoff_regions`` expects for several
    models) the figure has one panel per model, each against its own original model.

    Args:
        baseline: :func:`callmefair.tradeoff.create_baseline` output, or a mapping
            model name -> output.
        regions: output of :func:`callmefair.tradeoff.tradeoff_regions`.
        normalized: draw normalised axes instead of the metrics' own units.
        ax: optional axes (single-model maps only).
        label: direct-label win-win points and the largest good trade-offs.
    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    meta = regions.attrs or {}
    models = list(dict.fromkeys(regions["model"]))
    baselines = ({m: baseline for m in models} if isinstance(baseline, pd.DataFrame)
                 else {m: baseline[m] for m in models})
    if ax is not None and len(models) > 1:
        raise ValueError("ax is only supported for single-model maps")
    if ax is None:
        fig, axs = plt.subplots(1, max(1, len(models)), figsize=(6.4 * max(1, len(models)), 5.4),
                                facecolor=_style.SURFACE, squeeze=False)
        axs = list(axs[0])
    else:
        fig, axs = ax.figure, [ax]
    strategies = meta.get("strategies", {})
    for i, (m, a) in enumerate(zip(models, axs)):
        _style.style_axes(a)
        _draw_map(a, baselines[m], regions[regions["model"] == m].reset_index(drop=True),
                  strategy=strategies.get(m, meta.get("strategy")),
                  acc_name=meta.get("accuracy", "balanced_acc"), bias_name=meta.get("bias", "spd"),
                  normalized=normalized, label=label, slot=i)
        a.set_title(m if len(models) > 1 else "Fairea trade-off map", fontsize=11)
    if len(models) > 1:
        fig.suptitle("Fairea trade-off map (each model against its own original model)",
                     fontsize=12, color=_style.INK)
    handles = [Line2D([], [], color=_style.INK, lw=1.6, marker="|", ms=9,
                      label="baseline (mutation 0 % to 100 %)"),
               Line2D([], [], color=_style.INK, marker="*", ms=12, ls="",
                      label="original model (F_OM)")]
    handles += [Line2D([], [], ls="", marker=_style.marker(i), ms=7, color=_style.color(i),
                       label=f"{m} + mitigation") for i, m in enumerate(models)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95 if len(models) > 1 else 1))
    return fig


def _draw_map(ax, baseline, regions, *, strategy, acc_name, bias_name, normalized, label, slot):
    from callmefair.tradeoff import PAPER_NAMES, baseline_curve, normalize

    curve = baseline_curve(baseline, strategy, accuracy=acc_name, bias=bias_name)
    if normalized:
        b_acc, b_bias, _ = normalize(curve["accuracy"], curve["bias"])
        pts_x, pts_y = regions["bias_norm"].to_numpy(float), regions["acc_norm"].to_numpy(float)
    else:
        b_acc, b_bias = curve["accuracy"].to_numpy(float), curve["bias"].to_numpy(float)
        pts_x, pts_y = regions["bias"].to_numpy(float), regions["accuracy"].to_numpy(float)

    o_bias, o_acc = float(b_bias[0]), float(b_acc[0])
    all_x, all_y = np.concatenate([b_bias, pts_x]), np.concatenate([b_acc, pts_y])
    sx = np.nanmax(all_x) - min(0.0, np.nanmin(all_x)) or 1.0
    sy = np.nanmax(all_y) - np.nanmin(all_y) or 1.0
    x0 = min(0.0, np.nanmin(all_x)) - 0.04 * sx
    x1 = max(np.nanmax(all_x), o_bias + 0.2 * sx) + 0.05 * sx
    y0 = np.nanmin(all_y) - 0.08 * sy
    y1 = max(np.nanmax(all_y), o_acc + 0.15 * sy) + 0.06 * sy

    # regions (paper Fig. 2)
    order = np.argsort(b_bias, kind="stable")
    xs = np.linspace(x0, o_bias, 400)
    line_y = np.minimum(np.interp(xs, b_bias[order], b_acc[order]), o_acc)
    ax.fill_between([x0, o_bias], o_acc, y1, color=_TINT["win-win"], zorder=0)
    ax.fill_between([o_bias, x1], o_acc, y1, color=_TINT["inverted"], zorder=0)
    ax.fill_between([o_bias, x1], y0, o_acc, color=_TINT["lose-lose"], zorder=0)
    ax.fill_between(xs, line_y, o_acc, color=_TINT["good trade-off"], zorder=0)
    ax.fill_between(xs, y0, line_y, color=_TINT["bad trade-off"], zorder=0)
    dx, dy = 0.02 * (x1 - x0), 0.035 * (y1 - y0)
    for name, (x, y, ha) in {"win-win": (x0 + dx, y1 - dy, "left"),
                             "inverted": (x1 - dx, y1 - dy, "right"),
                             "lose-lose": (x1 - dx, y0 + dy, "right"),
                             "good trade-off": (x0 + dx, o_acc - dy, "left"),
                             "bad trade-off": (o_bias - dx, y0 + dy, "right")}.items():
        ax.text(x, y, PAPER_NAMES[name], ha=ha, va="center", fontsize=9, style="italic",
                color=_style.INK_SECONDARY, zorder=1)

    # baseline (mutation degrees) and the original model
    ax.plot(b_bias, b_acc, color=_style.INK, lw=1.6, marker="|", ms=9, zorder=3)
    for d, x, y in zip(curve["degree"], b_bias, b_acc):
        if np.isclose(d, 1.0) or np.isclose(d, 0.5):
            ax.annotate(f"F{int(round(d * 100))}", (x, y), xytext=(6, -10),
                        textcoords="offset points", fontsize=8, color=_style.INK_SECONDARY)
    ax.plot([o_bias], [o_acc], marker="*", ms=14, color=_style.INK, ls="", zorder=5)
    ax.annotate("F_OM", (o_bias, o_acc), xytext=(8, 6), textcoords="offset points",
                fontsize=8, color=_style.INK)

    # method points and selective labels
    ax.plot(pts_x, pts_y, ls="", marker=_style.marker(slot), ms=7, color=_style.color(slot),
            mec=_style.SURFACE, mew=1.0, zorder=4)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if label:
        shown = list(regions.index[regions["region"] == "win-win"])
        good = regions[regions["region"] == "good trade-off"]
        shown += list(good.nlargest(5, "area").index)
        _place_labels(ax, [(pts_x[i], pts_y[i], str(regions.at[i, "BM"])) for i in shown])
    prefix = "normalised " if normalized else ""
    ax.set_xlabel(f"{prefix}bias ({_AXIS_NAME.get(bias_name, bias_name)}, lower is fairer)")
    ax.set_ylabel(f"{prefix}{_AXIS_NAME.get(acc_name, acc_name)} (higher is better)", labelpad=8)


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
