"""Trade-off map (FAIREA-style): is a mitigation better than just degrading the model?

Baseline: take the unmitigated model and overwrite a growing fraction ("degree") of its
predictions with one constant label; accuracy and bias both fall, which is the price of
fairness paid with no mitigation at all. Both are rescaled to [0, 1] with one shared range
per axis (x = bias, 0 = least biased; y = accuracy loss, 0 = most accurate). Relative to
the unmutated model O and the baseline curve, each method lands in a region: win-win,
good trade-off, bad trade-off, lose-lose or inverted. Good trade-offs also get an area
(the bargain over the baseline). Bias is callmefair's ``fair_score`` (0 = fair).
One call for notebooks: :func:`tradeoff_regions`; plot with ``callmefair.viz.plot_tradeoff_map``.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd

REGIONS = ("win-win", "good trade-off", "bad trade-off", "lose-lose", "inverted")
PAPER_NAMES = {"win-win": "Win-Win", "good trade-off": "Good", "bad trade-off": "Poor",
               "lose-lose": "Lose-Lose", "inverted": "Inverted"}
DEFAULT_DEGREES = tuple(np.round(np.linspace(0, 1, 11), 1))
DEFAULT_ODDS = {"to_0": (1.0, 0.0), "to_1": (0.0, 1.0)}   # P(overwrite with 0), P(with 1)
_EPS = 1e-9


# ---------------------------------------------------------------------- baseline

def create_baseline(make_model: Callable[[], object], df: pd.DataFrame, label: str,
                    attribute: str, privileged_groups: list[dict],
                    unprivileged_groups: list[dict], data_splits: int = 3,
                    repetitions: int = 3, degrees=DEFAULT_DEGREES,
                    odds: Mapping[str, tuple[float, float]] = DEFAULT_ODDS,
                    seed: int = 0) -> pd.DataFrame:
    """Accuracy/bias of a model whose predictions are randomly overwritten.

    For each of ``data_splits`` splits (seed ``seed + s``) a fresh ``make_model()`` is
    trained and its test predictions are computed exactly like ``BMGridSearch`` does
    (validation-tuned threshold). Then a fraction ``degree`` of the test predictions is
    overwritten with labels drawn from ``odds[strategy]`` = (P(0), P(1)), ``repetitions``
    times. Returns one row per (degree, strategy), averaged over splits and repetitions,
    with columns ``degree, strategy, balanced_acc, fair_score``.
    """
    from callmefair import BMInterface
    from callmefair.mitigation.fair_grid import get_model_proba
    from callmefair.util.fair_util import BMMetrics

    rows = []
    for split in range(data_splits):
        rng = np.random.default_rng(seed + split)
        bm_i = BMInterface(df, label=label, protected=[attribute], random_state=seed + split)
        val_p, test_p = get_model_proba(make_model(), bm_i)
        metrics = BMMetrics(bm_i, np.array([0, 1]), val_p, test_p,
                            privileged_groups, unprivileged_groups)
        predicted = metrics.cmetrics.classified_dataset
        base_labels = predicted.labels.copy()
        for strategy, p in odds.items():
            for degree in degrees:
                for _ in range(repetitions):
                    mutated = predicted.copy(deepcopy=True)
                    mutated.labels = _mutate(base_labels, degree, p, rng)
                    metrics.set_pos_pred(mutated)
                    rows.append({"degree": degree, "strategy": strategy,
                                 "balanced_acc": metrics.get_report()["balanced_acc"],
                                 "fair_score": metrics.get_score()["overall_score"]})
    return (pd.DataFrame(rows).groupby(["strategy", "degree"], as_index=False, sort=False)
            [["balanced_acc", "fair_score"]].mean())


def _mutate(labels: np.ndarray, degree: float, p: tuple[float, float],
            rng: np.random.Generator) -> np.ndarray:
    """Overwrite ``int(n * degree)`` random entries with draws from {0, 1} ~ ``p`` (vectorised)."""
    out = labels.copy()
    n = int(len(out) * degree)
    idx = rng.choice(len(out), size=n, replace=False)
    out[idx] = rng.choice([0, 1], size=(n,) + out.shape[1:], p=p)
    return out


def baseline_curve(baseline: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Rows of one mutation strategy, ordered by degree (first row = unmutated model)."""
    return baseline[baseline.strategy == strategy].sort_values("degree").reset_index(drop=True)


# ---------------------------------------------------------------- normalisation

def normalize(base_acc, base_bias, methods: Mapping[str, tuple[float, float]] | None = None):
    """Rescale baseline and methods to [0, 1] with one shared range per axis.

    ``methods`` maps name -> (balanced_acc, bias). Returns ``(loss, bias, methods_norm)``:
    loss = (max_acc - acc) / range (0 = most accurate), bias = (bias - min) / range
    (0 = least biased); ``methods_norm`` maps name -> (loss, bias). The original
    ``fairea.py`` scaled the baseline by its own range and the methods by the joint
    range, which put them on different scales whenever a method fell outside the baseline.
    """
    methods = dict(methods or {})
    acc = np.concatenate([np.asarray(base_acc, float), [a for a, _ in methods.values()]])
    bias = np.concatenate([np.asarray(base_bias, float), [b for _, b in methods.values()]])

    def loss_of(a):
        span = acc.max() - acc.min()
        return (acc.max() - np.asarray(a, float)) / span if span > 0 else np.zeros_like(a, float)

    def bias_of(b):
        span = bias.max() - bias.min()
        return (np.asarray(b, float) - bias.min()) / span if span > 0 else np.zeros_like(b, float)

    return (loss_of(base_acc), bias_of(base_bias),
            {k: (float(loss_of(a)), float(bias_of(b))) for k, (a, b) in methods.items()})


# --------------------------------------------------------------- classification

def classify_region(base: "LineString", methods_norm: Mapping[str, tuple[float, float]]) -> dict:
    """Region of each normalised method ``(loss, bias)`` relative to the baseline.

    ``base`` is the normalised baseline as a LineString of ``(bias, loss)`` points whose
    first vertex is the unmutated model O. With point P = (bias, loss):

    * ``win-win``       - no more biased and no less accurate than O;
    * ``lose-lose``     - more biased and less accurate than O;
    * ``inverted``      - more biased but more accurate than O;
    * otherwise P trades accuracy for bias: ``good trade-off`` if the baseline reaches
      P's accuracy only with at least as much bias (a horizontal ray to the right of P
      meets the curve), else ``bad trade-off``.
    """
    from shapely.geometry import LineString
    x_o, y_o = base.coords[0]
    regions = {}
    for name, (loss, bias) in methods_norm.items():
        if bias <= x_o + _EPS and loss <= y_o + _EPS:
            regions[name] = "win-win"
        elif bias > x_o and loss > y_o:
            regions[name] = "lose-lose"
        elif bias > x_o:
            regions[name] = "inverted"
        elif base.intersects(LineString([(bias, loss), (2, loss)])):
            regions[name] = "good trade-off"
        else:
            regions[name] = "bad trade-off"
    return regions


def _nearest(geom, p: "Point"):
    """Point of ``geom`` (an intersection result) closest to ``p``; None if empty."""
    from shapely import get_coordinates
    from shapely.geometry import Point
    if geom.is_empty:
        return None
    return min((Point(c) for c in get_coordinates(geom)), key=p.distance)


def area_polygon(base: "LineString", method: tuple[float, float]) -> "Polygon | None":
    """Polygon whose area is :func:`compute_area` (None unless ``good trade-off``).

    The region is bounded by the horizontal and vertical lines from the point
    ``(bias, loss)`` to the baseline and by the baseline between the two hits. If the
    vertical line misses the curve (the point is less biased than the whole baseline),
    the curve's last vertex closes the region.
    """
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import substring
    if classify_region(base, {"m": method})["m"] != "good trade-off":
        return None
    loss, bias = method
    p = Point(bias, loss)
    right = _nearest(base.intersection(LineString([(bias, loss), (2, loss)])), p)
    up = _nearest(base.intersection(LineString([(bias, loss), (bias, 2)])), p) \
        or Point(base.coords[-1])
    t_right, t_up = base.project(right), base.project(up)
    arc = list(substring(base, min(t_right, t_up), max(t_right, t_up)).coords)
    if t_right > t_up:
        arc.reverse()
    return Polygon([(bias, loss), *arc, (bias, loss)]).buffer(0)


def compute_area(base: "LineString", method: tuple[float, float]) -> float:
    """Area between a method and the baseline (FAIREA's mitigation-region area).

    0.0 unless the point is a ``good trade-off`` (see :func:`classify_region`);
    the region is described in :func:`area_polygon`.
    """
    poly = area_polygon(base, method)
    return 0.0 if poly is None else float(poly.area)


# ------------------------------------------------------------------ convenience

def tradeoff_regions(baseline_df: pd.DataFrame, results, *, strategy: str = "to_1") -> pd.DataFrame:
    """Region and area of every mitigation row against the baseline.

    ``results`` are ``BMGridSearch`` rows (list of dicts or DataFrame with ``model``, ``BM``,
    ``balanced_acc``, ``fair_score``); the unmitigated row (``BM == "baseline"``) is dropped.
    Returns one row per mitigation: ``model, BM, balanced_acc, fair_score`` (raw), ``loss, bias``
    (normalised, shared range with the baseline), ``region``, ``paper_region``, ``area``.
    """
    from shapely.geometry import LineString
    res = pd.DataFrame(results)
    res = res[res["BM"] != "baseline"]
    res = (res.groupby(["model", "BM"], sort=False)[["balanced_acc", "fair_score"]]
           .mean().reset_index())                      # repeated seeds are averaged
    curve = baseline_curve(baseline_df, strategy)
    methods = {f"{r.model} | {r.BM}": (r.balanced_acc, r.fair_score) for r in res.itertuples()}
    loss_b, bias_b, norm = normalize(curve.balanced_acc, curve.fair_score, methods)
    line = LineString(zip(bias_b, loss_b))
    regions = classify_region(line, norm)
    out = res.copy()
    keys = list(methods)
    out["loss"] = [norm[k][0] for k in keys]
    out["bias"] = [norm[k][1] for k in keys]
    out["region"] = [regions[k] for k in keys]
    out["paper_region"] = out["region"].map(PAPER_NAMES)
    out["area"] = [compute_area(line, norm[k]) for k in keys]
    return out
