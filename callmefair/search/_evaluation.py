"""One bias-search iteration: seeded split, train, predict, score.

``run_iteration`` is a pure, picklable function of its arguments, so it can run
in the caller, in a thread or in a worker process and only returns a small
dict of scores (never a trained model).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from callmefair.search.model_factory import fit_model
from callmefair.util.fair_util import calculate_fairness_score

FAVORABLE, UNFAVORABLE = 1, 0
THRESHOLDS = np.linspace(0.01, 0.99, 100)


def _split_once(df, stratify, seed):
    df_train, df_rest = train_test_split(
        df, test_size=0.3, stratify=None if stratify is None else df[stratify], random_state=seed)
    df_test, df_val = train_test_split(
        df_rest, test_size=0.5,
        stratify=None if stratify is None else df_rest[stratify], random_state=seed)
    return df_train, df_val, df_test


def split_dataset(df: pd.DataFrame, label: str, attribute: str, seed: int,
                  apply_nearmiss: bool = False):
    """Split into (train, val, test) = 70/15/15, optionally NearMiss-resampling train.

    Stratifies on [attribute, label]; if impossible (a group too small), falls
    back to label only, then to a plain random split.
    """
    strategies = ([attribute, label], [label], None)
    for stratify in strategies:
        try:
            df_train, df_val, df_test = _split_once(df, stratify, seed)
            break
        except ValueError as exc:
            if stratify is None:
                raise
            warnings.warn(f"Stratifying on {stratify} failed ({exc}); trying a coarser split.")

    if apply_nearmiss:
        features = df_train.drop(columns=[label])
        x_res, y_res = NearMiss().fit_resample(features, df_train[label])
        df_train = pd.DataFrame(x_res, columns=features.columns)
        df_train[label] = y_res
    return df_train, df_val, df_test


def _to_bld(df, label, attribute):
    return BinaryLabelDataset(df=df, label_names=[label],
                              protected_attribute_names=[attribute],
                              favorable_label=FAVORABLE, unfavorable_label=UNFAVORABLE)


def best_threshold(val_bld: BinaryLabelDataset, scores: np.ndarray) -> float:
    """Threshold in THRESHOLDS maximising weighted balanced accuracy (first max wins)."""
    positive = val_bld.labels.ravel() == FAVORABLE
    weights = val_bld.instance_weights
    predicted = scores[None, :] > THRESHOLDS[:, None]            # (n_thresholds, n_samples)
    tpr = (predicted & positive) @ weights / weights[positive].sum()
    tnr = (~predicted & ~positive) @ weights / weights[~positive].sum()
    return float(THRESHOLDS[np.nanargmax(0.5 * (tpr + tnr))])


def fairness_at_threshold(test_bld: BinaryLabelDataset, scores: np.ndarray,
                          threshold: float, attribute: str) -> dict:
    pred = test_bld.copy(deepcopy=True)
    pred.scores = scores.reshape(-1, 1)
    favorable = pred.scores > threshold
    pred.labels[favorable] = FAVORABLE
    pred.labels[~favorable] = UNFAVORABLE

    metric = ClassificationMetric(test_bld, pred,
                                  unprivileged_groups=[{attribute: 0}],
                                  privileged_groups=[{attribute: 1}])
    try:
        avg_odds = metric.average_odds_difference()
    except (ValueError, ZeroDivisionError, RuntimeError) as exc:
        avg_odds = 1.0  # worst case
        warnings.warn(f"average_odds_difference failed at best threshold; using 1.0 ({exc})")

    result = calculate_fairness_score(
        metric.equal_opportunity_difference(), avg_odds,
        metric.statistical_parity_difference(), metric.disparate_impact(),
        metric.theil_index())
    result["balanced_accuracy"] = float(
        0.5 * (metric.true_positive_rate() + metric.true_negative_rate()))
    return result


def run_iteration(df: pd.DataFrame, label: str, attribute: str, apply_nearmiss: bool,
                  model_name: str, seed: int, n_jobs: int = 1) -> dict:
    """Split with ``seed``, fit ``model_name``, return the fairness score dict."""
    df_train, df_val, df_test = split_dataset(df, label, attribute, seed, apply_nearmiss)
    train, val, test = (_to_bld(d, label, attribute) for d in (df_train, df_val, df_test))

    scaler = StandardScaler().fit(train.features)
    predict = fit_model(model_name, scaler.transform(train.features), train.labels.ravel(),
                        train.instance_weights, n_jobs)
    val_scores = np.asarray(predict(scaler.transform(val.features)))
    test_scores = np.asarray(predict(scaler.transform(test.features)))

    return fairness_at_threshold(test, test_scores, best_threshold(val, val_scores), attribute)
