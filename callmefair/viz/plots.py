"""Plots for bias-mitigation results.

``results`` arguments accept the list of dicts returned by
``BMGridSearch.run_single_sensitive`` or an equivalent DataFrame. Every function
returns the matplotlib Figure and neither shows nor saves it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from callmefair.util.fair_util import METRIC_OPTIMA, METRIC_RANGES, REPORT_KEY_TO_METRIC
from callmefair.viz import _style

# How a result column improves: lower is better, closer to 0 / 1 is better, higher is better.
_LOWER = {'fair_score', 'theil_idx'}
_TOWARD_ZERO = {'eq_opp_diff', 'avg_odd_diff', 'spd'}
_TOWARD_ONE = {'disparate_impact'}
_LABELS = {
    'fair_score': 'fairness score (0 = fair)', 'balanced_acc': 'balanced accuracy',
    'acc': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'F1', 'mcc': 'MCC',
    'eq_opp_diff': 'equal opportunity diff (0 = fair)', 'avg_odd_diff': 'average odds diff (0 = fair)',
    'spd': 'statistical parity diff (0 = fair)', 'disparate_impact': 'disparate impact (1 = fair)',
    'theil_idx': 'Theil index (0 = fair)',
}


# ------------------------------------------------------------------ helpers
def _frame(results) -> pd.DataFrame:
    return results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(list(results))


def _slots(values) -> dict:
    """Entity -> slot index by first appearance (fixes color/marker per entity)."""
    return {v: i for i, v in enumerate(dict.fromkeys(values))}


def _short(bm: str) -> str:
    """'preReweighing posEqqOds' -> 'Reweighing+EqqOds'."""
    parts = []
    for p in str(bm).split():
        for prefix in ('pre', 'pos', 'in'):
            if p.startswith(prefix) and len(p) > len(prefix) and p[len(prefix)].isupper():
                p = p[len(prefix):]
                break
        parts.append(p)
    return '+'.join(parts)


def _label(col: str) -> str:
    return _LABELS.get(col, col)


def _direction(col: str) -> str:
    if col in _LOWER:
        return 'lower is better'
    if col in _TOWARD_ZERO:
        return 'closer to 0 is better'
    if col in _TOWARD_ONE:
        return 'closer to 1 is better'
    return 'higher is better'


def _goodness(col: str, base, new):
    """Positive when `new` is better than `base` for metric `col`."""
    if col in _LOWER:
        return base - new
    if col in _TOWARD_ZERO:
        return abs(base) - abs(new)
    if col in _TOWARD_ONE:
        return abs(base - 1) - abs(new - 1)
    return new - base


def _axes(axes, n, figsize, stacked=False, **kw):
    """(fig, list of n styled axes), reusing `axes` when given."""
    import matplotlib.pyplot as plt
    if axes is None:
        fig, axes = plt.subplots(*((n, 1) if stacked else (1, n)), figsize=figsize,
                                 facecolor=_style.SURFACE, **kw)
    axes = list(np.atleast_1d(axes).ravel())
    if len(axes) < n:
        raise ValueError(f'expected {n} axes, got {len(axes)}')
    for a in axes:
        _style.style_axes(a)
    return axes[0].figure, axes


def _fit(fig, created: bool) -> None:
    if created:
        fig.tight_layout()


# ------------------------------------------------------------------ 1. Pareto front
def pareto_front(results, *, x: str = 'fair_score', y: str = 'balanced_acc') -> pd.DataFrame:
    """Rows not dominated by any other (lower ``x`` and higher ``y`` are better).

    Args:
        results: list of dicts or DataFrame.
        x: column to minimise.
        y: column to maximise.
    Returns:
        The non-dominated rows, sorted by ``x``.
    """
    df = _frame(results)
    xv, yv = df[x].to_numpy(float), df[y].to_numpy(float)
    keep = [not np.any((xv <= xv[i]) & (yv >= yv[i]) & ((xv < xv[i]) | (yv > yv[i])))
            for i in range(len(df))]
    return df[keep].sort_values([x, y], ascending=[True, False])


# ------------------------------------------------------------------ 2. trade-off scatter
def plot_tradeoff(results, *, x: str = 'fair_score', y: str = 'balanced_acc',
                  hue: str = 'model', label: str = 'BM', pareto: bool = True, ax=None):
    """Scatter of every model + mitigation; arrows run from each baseline to its mitigations.

    Args:
        results: list of dicts or DataFrame (baseline row first per model).
        x, y: columns for the axes (x: lower is better, y: higher is better).
        hue: column that selects color/marker (one entity per value).
        label: column with the mitigation name (rows named 'baseline' are hollow).
        pareto: draw the Pareto front as a step line with direct labels.
        ax: optional axes to draw on.
    Returns:
        The matplotlib Figure.
    """
    from matplotlib.lines import Line2D
    df = _frame(results)
    created = ax is None
    fig, ax = _style.new_axes(ax, figsize=(7.4, 5.0))
    slots = _slots(df[hue]) if hue in df else {None: 0}
    many = len(slots) > 3
    handles = []
    for name, i in slots.items():
        sub = df[df[hue] == name] if hue in df else df
        c, m = _style.color(i), (_style.marker(i) if many else 'o')
        base = sub[sub[label] == 'baseline']
        mit = sub[sub[label] != 'baseline']
        ax.scatter(mit[x], mit[y], s=45, color=c, marker=m, linewidths=0.6,
                   edgecolors=_style.SURFACE, zorder=3)
        for _, b in base.iterrows():
            ax.scatter([b[x]], [b[y]], s=130, facecolors='none', edgecolors=c, marker=m,
                       linewidths=1.6, zorder=4)
            for _, r in mit.iterrows():
                ax.annotate('', xy=(r[x], r[y]), xytext=(b[x], b[y]), zorder=2,
                            arrowprops=dict(arrowstyle='->', color=c, lw=0.6, alpha=0.4,
                                            shrinkA=7, shrinkB=4))
        handles.append(Line2D([], [], color=c, marker=m, ls='', markersize=7, label=str(name)))
    handles.append(Line2D([], [], color=_style.INK_SECONDARY, marker='o', mfc='none', ls='',
                          markersize=10, label='baseline'))
    if pareto:
        front = pareto_front(df, x=x, y=y)
        ax.step(front[x], front[y], where='post', color=_style.INK_SECONDARY, lw=1.0, ls='--',
                zorder=1)
        handles.append(Line2D([], [], color=_style.INK_SECONDARY, lw=1.0, ls='--',
                              label='Pareto front'))
    ax.margins(0.12)
    ax.set_xlabel('fairness score (0 = fair)  →  more biased' if x == 'fair_score' else _label(x))
    ax.set_ylabel(_label(y))
    ax.set_title('Fairness vs. accuracy trade-off')
    if x == 'fair_score' and y == 'balanced_acc':
        ax.annotate('better', xy=(0.02, 0.98), xycoords='axes fraction', xytext=(0.09, 0.945),
                    textcoords='axes fraction', fontsize=8, color=_style.INK_SECONDARY,
                    ha='left', va='center',
                    arrowprops=dict(arrowstyle='->', color=_style.INK_SECONDARY, lw=0.8))
    if len(handles) >= 2:
        ax.legend(handles=handles, fontsize=8, frameon=False, loc='lower right')
    _fit(fig, created)
    if pareto:
        _label_points(ax, df[[x, y]].to_numpy(float), front, x, y, label)
    return fig


def _label_points(ax, pts, front, x, y, label) -> None:
    """Direct-label `front` rows, trying offsets that avoid other labels and points."""
    fig = ax.figure
    fig.canvas.draw()                                # final layout before measuring pixels
    scale = fig.dpi / 72
    pts = ax.transData.transform(pts)
    placed = []
    for _, r in front.iterrows():
        text = _short(r[label])
        w, h = 5.2 * len(text) * scale, 11 * scale
        px, py = ax.transData.transform((r[x], r[y]))
        best = None
        for dx, dy in [(8, 6), (8, -16), (-8 - w, 6), (-8 - w, -16), (8, 22), (8, -32),
                       (-8 - w, 22), (-8 - w, -32), (-w / 2, 16), (-w / 2, -28),
                       (14, 40), (-14 - w, 40), (14, -50), (-14 - w, -50),
                       (20, -80), (-20 - w, -80), (-w / 2, -100), (40, -120), (-40 - w, -120)]:
            box = (px + dx, py + dy, px + dx + w, py + dy + h)
            cost = 10 * sum(q[0] < box[2] and box[0] < q[2] and q[1] < box[3] and box[1] < q[3]
                            for q in placed)
            cost += sum(box[0] - 12 < qx < box[2] + 12 and box[1] - 12 < qy < box[3] + 12
                        for qx, qy in pts if (qx, qy) != (px, py))
            if best is None or cost < best[0]:
                best = (cost, dx, dy, box)
            if cost == 0:
                break
        _, dx, dy, box = best
        placed.append(box)
        ax.annotate(text, (r[x], r[y]), textcoords='offset pixels', xytext=(dx, dy),
                    fontsize=8, color=_style.INK_SECONDARY, zorder=5,
                    arrowprops=dict(arrowstyle='-', lw=0.5, color=_style.INK_SECONDARY, alpha=0.6,
                                    shrinkA=0, shrinkB=4) if abs(dy) > 30 else None)


# ------------------------------------------------------------------ 3. heatmap
def plot_metric_heatmap(results, metric: str = 'fair_score', *, index: str = 'BM',
                        columns: str = 'model', ax=None, annotate: bool = True):
    """Heatmap of one metric per mitigation (rows) and model (columns).

    Args:
        results: list of dicts or DataFrame.
        metric: result column to show.
        index, columns: columns for rows / columns (rows keep first-appearance
            order with 'baseline' first).
        ax: optional axes.
        annotate: write the value in each cell.
    Returns:
        The matplotlib Figure.
    """
    df = _frame(results)
    created = ax is None
    table = df.pivot_table(index=index, columns=columns, values=metric, aggfunc='first',
                           sort=False)
    rows = list(dict.fromkeys(df[index]))
    rows = [r for r in rows if r == 'baseline'] + [r for r in rows if r != 'baseline']
    table = table.loc[rows, list(dict.fromkeys(df[columns]))]
    n_r, n_c = table.shape
    fig, ax = _style.new_axes(ax, figsize=(max(4.5, 1.6 * n_c + 3.2), max(3.0, 0.42 * n_r + 1.6)))
    ax.grid(False)
    vals = table.to_numpy(float)
    im = ax.imshow(vals, cmap=_style.sequential_cmap(), aspect='auto')
    ax.set_xticks(range(n_c), [str(c) for c in table.columns])
    ax.set_yticks(range(n_r), [_short(r) if index == 'BM' else str(r) for r in table.index])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    if annotate:
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        for i in range(n_r):
            for j in range(n_c):
                if np.isfinite(vals[i, j]):
                    dark = hi > lo and (vals[i, j] - lo) / (hi - lo) > 0.55
                    ax.text(j, i, f'{vals[i, j]:.3f}', ha='center', va='center', fontsize=8,
                            color='white' if dark else _style.INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(f'{_label(metric)}  ({_direction(metric)})', fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=_style.INK_SECONDARY)
    ax.set_title(f'{_label(metric)} by mitigation and model')
    _fit(fig, created)
    return fig


# ------------------------------------------------------------------ 4. mitigation effect
def plot_mitigation_effect(results, metrics=('fair_score', 'balanced_acc'), *,
                           model: str | None = None, axes=None):
    """Per metric, bars of (mitigation - baseline of the same model) for each mitigation.

    Bars are colored blue when the change is better for that metric and red when
    worse (whatever the metric's direction); one panel per metric.

    Args:
        results: list of dicts or DataFrame.
        metrics: result columns, one panel each.
        model: restrict to one model (default: all, grouped by model).
        axes: optional sequence of ``len(metrics)`` axes.
    Returns:
        The matplotlib Figure.
    """
    df = _frame(results)
    if model is not None:
        df = df[df['model'] == model]
    models = list(dict.fromkeys(df['model']))
    entries = []                                  # (header | None, model, BM)
    for m in models:
        sub = df[df['model'] == m]
        if len(models) > 1:
            entries.append((m, m, None))
        entries += [(None, m, b) for b in sub['BM'] if b != 'baseline']
    n = len(metrics)
    created = axes is None
    fig, axs = _axes(axes, n, (3.6 * n + 2.2, max(2.8, 0.34 * len(entries) + 1.6)),
                     sharey=True)
    cmap = _style.diverging_cmap()
    ypos = np.arange(len(entries))
    for ax, metric in zip(axs, metrics):
        delta, good = [], []
        for header, m, b in entries:
            if header:
                delta.append(np.nan); good.append(np.nan)
                continue
            sub = df[df['model'] == m].set_index('BM')[metric]
            base = sub.loc['baseline']
            new = sub.loc[b]
            delta.append(new - base); good.append(_goodness(metric, base, new))
        good = np.array(good)
        lim = np.nanmax(np.abs(good)) if np.isfinite(good).any() else 1.0
        lim = lim or 1.0
        colors = [cmap(0.5 - 0.5 * g / lim) if np.isfinite(g) else 'none' for g in good]
        ax.barh(ypos, np.nan_to_num(delta), color=colors, height=0.62,
                edgecolor=_style.SURFACE, linewidth=1.5, zorder=3)
        ax.axvline(0, color=_style.INK, lw=0.8, zorder=4)
        ax.set_yticks(ypos, [e[0] or _short(e[2]) for e in entries])
        for tick, e in zip(ax.get_yticklabels(), entries):
            if e[0]:
                tick.set_fontweight('bold')
        ax.grid(False, axis='y')
        ax.set_xlabel(f'change in {_label(metric)}\nvs. baseline  (blue = better, red = worse; '
                      f'{_direction(metric)})', fontsize=8)
        ax.set_title(_label(metric).split(' (')[0], fontsize=10)
    axs[0].set_ylim(len(entries) - 0.5, -0.5)
    if n > 1:
        fig.suptitle('Effect of each mitigation relative to its model baseline', fontsize=11,
                     color=_style.INK)
    _fit(fig, created)
    return fig


# ------------------------------------------------------------------ 5. fair-range small multiples
def plot_fair_ranges(results, *, label: str = 'BM', hue: str = 'model', axes=None):
    """Small multiples (EOD, AOD, SPD, DI, TI) showing each row against its acceptable range.

    Filled dots lie inside the acceptable range, hollow dots outside; values
    beyond the plotted window sit on the edge with an arrow.

    Args:
        results: list of dicts or DataFrame.
        label: column used as the row label.
        hue: column that selects the dot color.
        axes: optional sequence of 5 axes.
    Returns:
        The matplotlib Figure.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    df = _frame(results).reset_index(drop=True)
    created = axes is None
    rows = _slots(df[label])                       # one row per label
    fig, axs = _axes(axes, 5, (15.5, max(3.2, 0.42 * len(rows) + 2.0)), sharey=True)
    slots = _slots(df[hue]) if hue in df else {None: 0}
    cols = [_style.color(slots[v]) for v in (df[hue] if hue in df else [None] * len(df))]
    # dodge the hue vertically inside each row
    ypos = np.array([rows[b] + (slots[h] - (len(slots) - 1) / 2) * 0.28
                     for b, h in zip(df[label], df[hue] if hue in df else [None] * len(df))])
    for ax, (key, name) in zip(axs, REPORT_KEY_TO_METRIC.items()):
        lo, hi = METRIC_RANGES[name]
        half, centre = (hi - lo) / 2, (hi + lo) / 2
        v = df[key].to_numpy(float)
        finite = v[np.isfinite(v)]
        vmin = min(lo - 0.3 * half, finite.min()) if len(finite) else lo
        vmax = max(hi + 0.3 * half, finite.max()) if len(finite) else hi
        left, right = max(vmin, centre - 4 * half), min(vmax, centre + 4 * half)
        pad = 0.08 * (right - left)
        left, right = left - pad, right + pad
        ax.axvspan(lo, hi, color=_style.GOOD, alpha=0.13, lw=0, zorder=1)
        ax.axvline(METRIC_OPTIMA[name], color=_style.INK_SECONDARY, lw=1.0, ls='--', zorder=2)
        for i in range(len(df)):
            x = v[i]
            if np.isnan(x):
                continue
            inside = lo <= x <= hi
            edge = None
            if x > right - pad / 2:
                x, edge = right - pad / 2, '>'
            elif x < left + pad / 2:
                x, edge = left + pad / 2, '<'
            ax.plot([x], [ypos[i]], marker=edge or 'o', ms=7, color=cols[i], zorder=3,
                    mfc=cols[i] if inside else 'none', mew=1.3, ls='')
        ax.set_xlim(left, right)
        ax.set_title(f'{name}', fontsize=10)
        ax.set_xlabel(_label(key).split(' (')[0], fontsize=8)
        ax.grid(False, axis='y')
    axs[0].set_yticks(range(len(rows)), [_short(b) if label == 'BM' else str(b) for b in rows])
    axs[0].set_ylim(len(rows) - 0.5, -0.5)
    handles = [Line2D([], [], color=_style.color(i), marker='o', ls='', ms=7, label=str(n))
               for n, i in slots.items() if n is not None]
    handles += [Line2D([], [], color=_style.INK_SECONDARY, marker='o', ls='', ms=7,
                       label='inside range (filled)'),
                Line2D([], [], color=_style.INK_SECONDARY, marker='o', mfc='none', ls='', ms=7,
                       label='outside range (hollow)'),
                Patch(facecolor=_style.GOOD, alpha=0.25, label='acceptable range'),
                Line2D([], [], color=_style.INK_SECONDARY, ls='--', lw=1, label='optimum')]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle('Fairness metrics against their acceptable ranges', fontsize=11, color=_style.INK)
    if created:
        fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    return fig


# ------------------------------------------------------------------ 6. bias search
def plot_bias_search(table, *, score: str = 'Normalized Fairness score', ax=None):
    """Horizontal bars of the per-attribute fairness score, most biased first.

    Args:
        table: ``BiasSearch.evaluate_average`` table (header row first) or DataFrame.
        score: column to plot.
        ax: optional axes.
    Returns:
        The matplotlib Figure.
    """
    if isinstance(table, pd.DataFrame):
        df = table.copy()
    else:
        df = pd.DataFrame(table[1:], columns=table[0])
    created = ax is None
    df = df.sort_values(score, ascending=False).reset_index(drop=True)
    fig, ax = _style.new_axes(ax, figsize=(7.0, max(2.4, 0.42 * len(df) + 1.4)))
    y = np.arange(len(df))
    ax.barh(y, df[score], color=_style.color(0), height=0.6, zorder=3)
    top = float(df[score].max()) or 1.0
    ax.set_xlim(0, top * 1.45)
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.text(r[score] + top * 0.015, yi,
                f"{r[score]:.3f}   n = {r['Privileged N']} priv. / {r['Unprivileged N']} unpriv.",
                va='center', fontsize=8, color=_style.INK_SECONDARY)
    ax.set_yticks(y, df['Attribute'])
    ax.invert_yaxis()
    ax.grid(False, axis='y')
    ax.set_xlabel(f'{score.lower()} (0 = fair)  →  more biased')
    ax.set_title('Bias by sensitive attribute')
    _fit(fig, created)
    return fig


# ------------------------------------------------------------------ 7. group rates
def plot_group_rates(bmM, *, ax=None):
    """Selection rate, TPR and FPR for privileged vs. unprivileged groups (test split).

    Args:
        bmM: a ``BMMetrics`` (uses its ``cmetrics``).
        ax: optional axes.
    Returns:
        The matplotlib Figure.
    """
    cm = bmM.cmetrics
    created = ax is None
    names = ['selection rate', 'true positive rate', 'false positive rate']
    fns = [cm.selection_rate, cm.true_positive_rate, cm.false_positive_rate]
    priv = [f(privileged=True) for f in fns]
    unpriv = [f(privileged=False) for f in fns]
    fig, ax = _style.new_axes(ax, figsize=(6.6, 4.2))
    x, w = np.arange(3), 0.36
    ax.bar(x - w / 2, priv, w, color=_style.color(0), label='privileged', zorder=3,
           edgecolor=_style.SURFACE, linewidth=1.5)
    ax.bar(x + w / 2, unpriv, w, color=_style.color(1), label='unprivileged', zorder=3,
           edgecolor=_style.SURFACE, linewidth=1.5)
    top = max(priv + unpriv) or 1.0
    notes = ['SPD', 'EOD', 'ΔFPR']
    for xi, p, u, note in zip(x, priv, unpriv, notes):
        ax.text(xi, max(p, u) + top * 0.04, f'{note} = {u - p:+.3f}', ha='center', fontsize=8,
                color=_style.INK_SECONDARY)
    ax.set_ylim(0, top * 1.25)
    ax.set_xticks(x, names)
    ax.grid(False, axis='x')
    ax.set_ylabel('rate (test split)')
    ax.set_title('Group rates behind the fairness metrics')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    _fit(fig, created)
    return fig


# ------------------------------------------------------------------ 8. threshold sweep
def plot_threshold_sweep(bmM, *, metric: str = 'spd', axes=None):
    """Balanced accuracy and one fairness metric against the decision threshold (validation).

    Args:
        bmM: a ``BMMetrics``.
        metric: 'spd', 'disparate_impact' or 'eq_opp_diff'.
        axes: optional sequence of 2 axes (stacked, sharing x).
    Returns:
        The matplotlib Figure.
    """
    curve = bmM.threshold_curve()
    name = REPORT_KEY_TO_METRIC[metric]
    lo, hi = METRIC_RANGES[name]
    created = axes is None
    fig, (a1, a2) = _axes(axes, 2, (6.6, 5.4), stacked=True, sharex=created)
    if not created:
        a2.sharex(a1)
    t = curve['threshold']
    a1.plot(t, curve['balanced_acc'], color=_style.color(0), lw=1.6)
    a1.set_ylabel('balanced accuracy\n(higher is better)')
    a1.set_title('Effect of the decision threshold (validation split)')
    y = curve[metric].replace([np.inf, -np.inf], np.nan)
    a2.axhspan(lo, hi, color=_style.GOOD, alpha=0.13, lw=0, label='acceptable range')
    a2.axhline(METRIC_OPTIMA[name], color=_style.INK_SECONDARY, lw=1.0, ls='--', label='optimum')
    a2.plot(t, y, color=_style.color(1), lw=1.6, label=name)
    if y.notna().any():
        span = max(y.max(), hi) - min(y.min(), lo)
        a2.set_ylim(min(y.min(), lo) - 0.08 * span, max(y.max(), hi) + 0.08 * span)
    a2.set_ylabel(_label(metric))
    a2.set_xlabel('decision threshold')
    a2.legend(frameon=False, fontsize=8, loc='best')
    for a in (a1, a2):
        a.axvline(bmM.best_class_thresh, color=_style.INK, lw=1.0, ls=':')
    a1.annotate('chosen on validation', (bmM.best_class_thresh, a1.get_ylim()[0]),
                xytext=(5, 6), textcoords='offset points', fontsize=8, color=_style.INK)
    _fit(fig, created)
    return fig
