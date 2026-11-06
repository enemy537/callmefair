"""Fairea trade-off map (Hort et al., ESEC/FSE 2021), as used by FairHealthGrid.

1. **Baseline.** Train the original model, then *mutate* a fraction ``degree`` of its
   test predictions (0 %, 10 %, ..., 100 %) to one fixed label. Each degree gives a naive
   "mitigation" that buys fairness by giving up accuracy. Fairea keeps the strategy
   (label) with the highest accuracy at 100 % mutation (usually the majority label).
2. **Normalisation.** Accuracy and bias are min-max scaled *by the baseline's range*,
   so the original model F_OM sits at (1, 1) and the fully mutated model at (0, 0):
   with a linear bias metric (``|SPD|``, ``|AOD|``) the baseline is close to the diagonal.
   Mitigation results use the same scaling and may fall outside [0, 1].
3. **Regions** (paper, Fig. 2), relative to F_OM and the baseline:
   win-win (more accurate, less biased), inverted (more accurate, more biased),
   lose-lose (less accurate, more biased), and for less accurate but less biased
   methods: good trade-off above the baseline, poor ("bad") trade-off below it.
4. **Area** (paper, Fig. 3) quantifies a good trade-off: the region enclosed by the
   method point, its horizontal and vertical projections onto the baseline, and the
   baseline between them.

The paper's region definitions are implemented directly. They coincide with the
reference implementation's ray tests (github.com/maxhort/Fairea) inside the baseline's
range; outside it the reference labels points with less bias than the fully mutated
model "lose-lose", which contradicts the definitions, so those are classified by the
definitions instead.
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

# Metric names -> grid-search column; bias metrics are taken in absolute value (paper 3.1).
ACCURACY_METRICS = {"acc": "acc", "balanced_acc": "balanced_acc"}
BIAS_METRICS = {"spd": "spd", "aod": "avg_odd_diff", "eod": "eq_opp_diff",
                "fair_score": "fair_score"}


# ---------------------------------------------------------------- 1. baseline

def create_baseline(make_model: Callable[[], object], df: pd.DataFrame, label: str,
                    attribute: str, privileged_groups: list[dict],
                    unprivileged_groups: list[dict], *, data_splits: int = 10,
                    repetitions: int = 10, degrees=DEFAULT_DEGREES,
                    odds: Mapping[str, tuple[float, float]] = DEFAULT_ODDS,
                    seed: int = 0, bmi_kwargs: Mapping | None = None) -> pd.DataFrame:
    """Accuracy and bias of the original model with its predictions mutated.

    For each of ``data_splits`` splits (seed ``seed + s``) a fresh ``make_model()`` is
    trained and its test predictions are made exactly as ``BMGridSearch`` does
    (validation-tuned threshold). A fraction ``degree`` of them is then overwritten with
    labels drawn from ``odds[strategy]``, ``repetitions`` times (the paper uses 50 x 50).

    Returns one row per (strategy, degree), averaged over splits and repetitions, with
    columns ``strategy, degree`` and every metric in ``ACCURACY_METRICS`` and
    ``BIAS_METRICS`` (bias metrics in absolute value).
    """
    from callmefair import BMInterface
    from callmefair.mitigation.fair_grid import get_model_proba
    from callmefair.util.fair_util import BMMetrics

    rows = []
    for split in range(data_splits):
        rng = np.random.default_rng(seed + split)
        bm_i = BMInterface(df, label=label, protected=[attribute], random_state=seed + split,
                           **dict(bmi_kwargs or {}))
        val_p, test_p = get_model_proba(make_model(), bm_i)
        metrics = BMMetrics(bm_i, np.array([0, 1]), val_p, test_p,
                            privileged_groups, unprivileged_groups)
        predicted = metrics.cmetrics.classified_dataset
        original = predicted.labels.copy()
        for strategy, p in odds.items():
            for degree in degrees:
                for _ in range(repetitions):
                    mutated = predicted.copy(deepcopy=True)
                    mutated.labels = _mutate(original, degree, p, rng)
                    metrics.set_pos_pred(mutated)
                    rows.append({"strategy": strategy, "degree": degree,
                                 **_metric_values({**metrics.get_report(),
                                                   "fair_score": metrics.get_score()["overall_score"]})})
    metric_cols = list(ACCURACY_METRICS) + list(BIAS_METRICS)
    return (pd.DataFrame(rows).groupby(["strategy", "degree"], as_index=False, sort=False)
            [metric_cols].mean())


def _metric_values(report: Mapping) -> dict:
    """Accuracy metrics as-is, bias metrics in absolute value, keyed by short name."""
    out = {k: float(report[col]) for k, col in ACCURACY_METRICS.items()}
    out.update({k: abs(float(report[col])) for k, col in BIAS_METRICS.items()})
    return out


def _mutate(labels: np.ndarray, degree: float, p: tuple[float, float],
            rng: np.random.Generator) -> np.ndarray:
    """Overwrite ``int(n * degree)`` random entries with labels drawn from {0, 1} ~ ``p``."""
    out = labels.copy()
    n = int(len(out) * degree)
    idx = rng.choice(len(out), size=n, replace=False)
    out[idx] = rng.choice([0, 1], size=(n,) + out.shape[1:], p=p)
    return out


def select_strategy(baseline: pd.DataFrame, *, accuracy: str = "balanced_acc") -> str:
    """Fairea's rule: the strategy with the highest accuracy at 100 % mutation.

    Ties (e.g. balanced accuracy is 0.5 for any constant prediction) go to the strategy
    with the higher mean accuracy over all degrees, i.e. the stricter baseline.
    """
    full = baseline[np.isclose(baseline["degree"], baseline["degree"].max())]
    best = full[np.isclose(full[accuracy], full[accuracy].max())]["strategy"]
    if len(best) == 1:
        return str(best.iloc[0])
    means = baseline[baseline["strategy"].isin(best)].groupby("strategy")[accuracy].mean()
    return str(means.idxmax())


def baseline_curve(baseline: pd.DataFrame, strategy: str | None = None, *,
                   accuracy: str = "balanced_acc", bias: str = "spd") -> pd.DataFrame:
    """Baseline points ``degree, accuracy, bias`` for one strategy, ordered by degree.

    Warns when Fairea's requirement is violated: the original model must be more
    accurate than the fully mutated (constant) model, otherwise "sacrificing accuracy
    for fairness" is meaningless (paper, Sec. 4.1 and 6.1). With imbalanced labels this
    happens for plain accuracy; use ``accuracy='balanced_acc'``.
    """
    import warnings
    strategy = strategy or select_strategy(baseline, accuracy=accuracy)
    curve = baseline[baseline["strategy"] == strategy].sort_values("degree")
    out = pd.DataFrame({"degree": curve["degree"].to_numpy(),
                        "accuracy": curve[accuracy].to_numpy(float),
                        "bias": curve[bias].to_numpy(float)})
    if out["accuracy"].iloc[0] <= out["accuracy"].iloc[-1]:
        warnings.warn(f"Fairea baseline requirement not met: the original model's {accuracy} "
                      f"({out['accuracy'].iloc[0]:.3f}) is not above the fully mutated model's "
                      f"({out['accuracy'].iloc[-1]:.3f}). Regions are not meaningful; for "
                      "imbalanced labels use accuracy='balanced_acc'.", stacklevel=2)
    return out


# ---------------------------------------------------------------- 2. normalisation

def normalize(base_acc, base_bias, methods: Mapping[str, tuple[float, float]] | None = None):
    """Min-max scale by the baseline's range (reference implementation).

    ``methods`` maps name -> (accuracy, bias). Returns ``(acc_norm, bias_norm,
    methods_norm)`` where ``methods_norm`` maps name -> (acc_norm, bias_norm).
    """
    base_acc, base_bias = np.asarray(base_acc, float), np.asarray(base_bias, float)
    a_min, a_rng = base_acc.min(), np.ptp(base_acc) or 1.0
    b_min, b_rng = base_bias.min(), np.ptp(base_bias) or 1.0
    methods_norm = {k: ((a - a_min) / a_rng, (b - b_min) / b_rng)
                    for k, (a, b) in dict(methods or {}).items()}
    return (base_acc - a_min) / a_rng, (base_bias - b_min) / b_rng, methods_norm


# ---------------------------------------------------------------- 3. regions

def _baseline_fn(base_bias, base_acc) -> Callable[[np.ndarray], np.ndarray]:
    """Baseline accuracy as a piecewise-linear function of bias (flat beyond its ends)."""
    order = np.argsort(base_bias, kind="stable")
    xs, ys = np.asarray(base_bias, float)[order], np.asarray(base_acc, float)[order]
    return lambda x: np.interp(x, xs, ys)


def _origin(base_bias, base_acc) -> tuple[float, float]:
    """F_OM: the unmutated model = first baseline point (degree 0)."""
    return float(base_bias[0]), float(base_acc[0])


def classify_region(base_bias, base_acc, methods_norm: Mapping[str, tuple[float, float]]) -> dict:
    """Region of each normalised method ``(acc, bias)``; baseline ordered by degree."""
    o_bias, o_acc = _origin(base_bias, base_acc)
    baseline_acc = _baseline_fn(base_bias, base_acc)
    regions = {}
    for k, (acc, bias) in methods_norm.items():
        if acc >= o_acc:
            regions[k] = "win-win" if bias <= o_bias else "inverted"
        elif bias >= o_bias:                    # less accurate, no fairness gain
            regions[k] = "lose-lose"
        else:
            regions[k] = "good trade-off" if acc > baseline_acc(bias) else "bad trade-off"
    return regions


# ---------------------------------------------------------------- 4. area

def area_polygon(base_bias, base_acc, method: tuple[float, float]) -> np.ndarray | None:
    """Vertices (bias, acc) of the paper's Fig. 3 area, or None unless a good trade-off.

    BM -> BM' (horizontal projection onto the baseline) -> along the baseline ->
    BM'' (vertical projection onto the baseline) -> BM.
    """
    acc, bias = method
    if classify_region(base_bias, base_acc, {"m": method})["m"] != "good trade-off":
        return None
    f = _baseline_fn(base_bias, base_acc)
    o_bias, _ = _origin(base_bias, base_acc)
    xs = np.linspace(bias, o_bias, 2001)
    above = f(xs) >= acc
    x_h = xs[np.argmax(above)] if above.any() else o_bias        # first crossing to the right
    inner = np.asarray(base_bias, float)
    inner = np.sort(inner[(inner > bias) & (inner < x_h)])[::-1]
    path = [(bias, acc), (x_h, min(acc, float(f(x_h))))]
    path += [(x, float(f(x))) for x in inner]
    path += [(bias, float(f(bias)))]
    return np.array(path)


def compute_area(base_bias, base_acc, method: tuple[float, float]) -> float:
    """Area between a good trade-off method and the baseline (0 for other regions)."""
    poly = area_polygon(base_bias, base_acc, method)
    if poly is None:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)  # shoelace


# ---------------------------------------------------------------- one call for users

def tradeoff_regions(baseline: pd.DataFrame, results, *, strategy: str | None = None,
                     accuracy: str = "balanced_acc", bias: str = "spd") -> pd.DataFrame:
    """Place grid-search results on the Fairea map.

    Args:
        baseline: output of :func:`create_baseline`.
        results: ``BMGridSearch.run_single_sensitive`` rows (list of dicts or DataFrame);
            ``BM == 'baseline'`` rows are dropped and repeated (model, BM) rows averaged.
        strategy: mutation strategy; default = Fairea's rule (:func:`select_strategy`).
        accuracy: ``'balanced_acc'`` (default; required for imbalanced labels) or
            ``'acc'`` (the Fairea paper's choice for its balanced benchmark datasets).
        bias: ``'spd'`` (default), ``'aod'``, ``'eod'`` or ``'fair_score'`` (composite 𝓕;
            non-linear and capped at 1, so its baseline is not a straight line).

    Returns:
        DataFrame with ``model, BM, accuracy, bias, acc_norm, bias_norm, region,
        paper_region, area``; ``.attrs`` records strategy/accuracy/bias for plotting.
    """
    strategy = strategy or select_strategy(baseline, accuracy=accuracy)
    curve = baseline_curve(baseline, strategy, accuracy=accuracy, bias=bias)
    rows = pd.DataFrame(results)
    rows = rows[rows["BM"] != "baseline"].copy()
    rows["accuracy"] = rows[ACCURACY_METRICS[accuracy]].astype(float)
    rows["bias"] = rows[BIAS_METRICS[bias]].astype(float).abs()
    rows = rows.groupby(["model", "BM"], as_index=False, sort=False)[["accuracy", "bias"]].mean()

    base_acc, base_bias, norm = normalize(curve["accuracy"], curve["bias"],
                                          {i: (a, b) for i, a, b in
                                           zip(rows.index, rows["accuracy"], rows["bias"])})
    regions = classify_region(base_bias, base_acc, norm)
    rows["acc_norm"] = [norm[i][0] for i in rows.index]
    rows["bias_norm"] = [norm[i][1] for i in rows.index]
    rows["region"] = [regions[i] for i in rows.index]
    rows["paper_region"] = rows["region"].map(PAPER_NAMES)
    rows["area"] = [compute_area(base_bias, base_acc, norm[i]) for i in rows.index]
    rows.attrs = {"strategy": strategy, "accuracy": accuracy, "bias": bias}
    return rows.reset_index(drop=True)
