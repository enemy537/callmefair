"""Trade-off map: mitigation points against the degradation baseline."""

from __future__ import annotations

from . import _style

# Light tints per region (never the only cue: each region is also named in the plot).
_TINT = {"win-win": "#d5eee8", "good trade-off": "#e6f2d2", "bad trade-off": "#fbefd3",
         "lose-lose": "#f8dcdc", "inverted": "#ebe5f4"}


def plot_tradeoff_map(baseline_df, regions_df, *, strategy: str = "to_1", ax=None,
                      label: bool = True):
    """Plot the normalised trade-off map and return its Figure.

    x = normalised bias (0 = least biased), y = normalised accuracy loss (0 = most
    accurate), so the ideal corner is the bottom-left origin. The baseline (degrading the
    model) is the line from the unmutated model O, with ticks at 0/50/100 % mutated.
    ``regions_df`` is the output of :func:`callmefair.tradeoff.tradeoff_regions`; points are
    colored (and shaped) per model; good trade-offs show their area over the baseline.
    ``label`` names the points that beat or tie O or the baseline (win-win, good).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon as MplPolygon
    from shapely.geometry import LineString
    from callmefair.tradeoff import (PAPER_NAMES, area_polygon, baseline_curve, normalize)

    fig, ax = _style.new_axes(ax, figsize=(7.4, 5.2))
    curve = baseline_curve(baseline_df, strategy)
    keys = [f"{m} | {b}" for m, b in zip(regions_df["model"], regions_df["BM"])]
    methods = {k: (a, s) for k, a, s in
               zip(keys, regions_df["balanced_acc"], regions_df["fair_score"])}
    loss_b, bias_b, norm = normalize(curve.balanced_acc, curve.fair_score, methods)
    line = LineString(zip(bias_b, loss_b))
    x_o, y_o = line.coords[0]

    hi_x = max([1.0, *(v[1] for v in norm.values())]) + 0.05
    hi_y = max([1.0, *(v[0] for v in norm.values())]) + 0.05
    lo = -0.04

    # --- region tints: exactly the areas classify_region assigns ---
    region_text = []
    geoms = region_geometries(line, (lo, lo, hi_x, hi_y))
    from shapely.ops import polylabel
    for name, g in geoms.items():
        for part in getattr(g, "geoms", [g]):
            if part.is_empty or part.area <= 0:
                continue
            ax.add_patch(MplPolygon(list(part.exterior.coords), closed=True, fc=_TINT[name],
                                    ec="none", zorder=0))
        big = max(getattr(g, "geoms", [g]), key=lambda q: q.area)
        c = polylabel(big, tolerance=0.01)
        region_text.append((c.x, c.y, PAPER_NAMES[name]))
        ax.text(c.x, c.y, PAPER_NAMES[name], ha="center", va="center", fontsize=9,
                style="italic", color=_style.INK_SECONDARY, zorder=1)

    # --- baseline with degree ticks, and O ---
    to_label = []
    ax.plot(bias_b, loss_b, "-", color=_style.INK, lw=1.8, zorder=2)
    for d, x, y in zip(curve.degree, bias_b, loss_b):
        ax.plot(x, y, "|" if d not in (0, 0.5, 1) else "o", color=_style.INK, ms=6,
                mfc=_style.SURFACE, zorder=3)
        if d in (0.5, 1):
            to_label.append((f"{d:.0%} mutated", x, y))
    ax.plot(x_o, y_o, "*", color=_style.INK, ms=13, mfc=_style.INK, zorder=4)
    to_label.append(("O (unmutated)", x_o, y_o))

    # --- method points ---
    models = list(dict.fromkeys(regions_df["model"]))
    shaped = len(models) > 3
    for (key, (loss, bias)), model, bm, region in zip(
            norm.items(), regions_df["model"], regions_df["BM"], regions_df["region"]):
        i = models.index(model)
        if region == "good trade-off":
            poly = area_polygon(line, (loss, bias))
            if poly is not None and not poly.is_empty:
                ax.fill(*poly.exterior.xy, fc=_style.color(i), alpha=0.18, ec="none", zorder=1.5)
        ax.scatter(bias, loss, s=46, color=_style.color(i), ec=_style.SURFACE, lw=0.8,
                   marker=_style.marker(i) if shaped else "o", zorder=5)
        if label and region in ("win-win", "good trade-off"):
            to_label.append((bm, bias, loss))
    _place_labels(ax, to_label, [(v[1], v[0]) for v in norm.values()]
                  + list(zip(bias_b, loss_b)), region_text)

    ax.set(xlim=(lo, hi_x), ylim=(lo, hi_y),
           xlabel="Normalized bias (fair_score, 0 = least biased)",
           ylabel="Normalized accuracy loss (0 = most accurate)",
           title="Trade-off map: lower-left is better (origin = ideal)")
    ax.grid(False)

    handles = [Line2D([], [], color=_style.INK, lw=1.8, label="Baseline (degrade the model)")]
    handles += [Line2D([], [], ls="", marker=_style.marker(i) if shaped else "o",
                       color=_style.color(i), label=m) for i, m in enumerate(models)]
    if len(handles) >= 2:
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.14), ncol=min(len(handles), 4))
    return fig


def region_geometries(line, bounds, levels: int = 400) -> dict:
    """Shapely region polygons (bias on x, loss on y) matching ``classify_region``.

    ``line`` is the normalised baseline (first vertex = O); ``bounds`` = (x0, y0, x1, y1).
    Good = points of the less-bias / more-loss quadrant whose rightward horizontal ray meets
    the curve, i.e. x <= the curve's largest x at that height (right envelope, so a
    non-monotone curve is handled like the ray test); Poor = the rest of that quadrant.
    """
    from shapely import get_coordinates
    from shapely.geometry import LineString, box
    from shapely.ops import unary_union
    x0, y0, x1, y1 = bounds
    xo, yo = line.coords[0]
    quad = box(x0, yo, xo, y1)
    ys = sorted({*(c[1] for c in line.coords),
                 *(yo + (y1 - yo) * k / levels for k in range(levels + 1))})
    strips = []
    for a, b in zip(ys[:-1], ys[1:]):
        hit = line.intersection(LineString([(x0 - 1, (a + b) / 2), (x1 + 1, (a + b) / 2)]))
        if hit.is_empty:
            continue
        xmax = get_coordinates(hit)[:, 0].max()
        strips.append(box(x0, a, xmax, b))
    good = unary_union(strips).intersection(quad) if strips else quad.difference(quad)
    return {"win-win": box(x0, y0, xo, yo), "inverted": box(xo, y0, x1, yo),
            "lose-lose": box(xo, yo, x1, y1), "good trade-off": good,
            "bad trade-off": quad.difference(good)}


def _place_labels(ax, items, anchors, texts=()):
    """Greedy label placement: first of 8 offsets whose box avoids earlier labels and points."""
    fig = ax.figure
    fig.canvas.draw()                                   # fixes the data -> pixel transform
    scale = fig.dpi / 72
    to_px = ax.transData.transform
    taken = [(*to_px(a), *to_px(a)) for a in anchors]   # point boxes (padded below)
    taken = [(x - 7 * scale, y - 7 * scale, x + 7 * scale, y + 7 * scale) for x, y, _, _ in taken]
    for x, y, t in texts:                                # region names are obstacles too
        px, py = to_px((x, y))
        w = (len(t) * 5 + 4) * scale / 2
        taken.append((px - w, py - 8 * scale, px + w, py + 8 * scale))
    for text, x, y in items:
        px, py = to_px((x, y))
        w, h = (len(text) * 4.6 + 4) * scale, 11 * scale
        best = None
        for dx, dy, ha in ((10, 6, "left"), (10, -16, "left"), (-10, 6, "right"),
                           (-10, -16, "right"), (14, 18, "left"), (-14, 18, "right"),
                           (14, -28, "left"), (-14, -28, "right")):
            x0 = px + dx * scale - (w if ha == "right" else 0)
            y0 = py + dy * scale
            box = (x0, y0, x0 + w, y0 + h)
            hit = any(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3]
                      for t in taken)
            if best is None:
                best = (dx, dy, ha, box)
            if not hit:
                best = (dx, dy, ha, box)
                break
        dx, dy, ha, box = best
        taken.append(box)
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=8, color=_style.INK, zorder=6,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": _style.INK_SECONDARY,
                                "shrinkA": 0, "shrinkB": 4})
