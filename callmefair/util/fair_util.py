"""
Core utilities for bias mitigation.

- `calculate_fairness_score`: aggregate EOD/AOD/SPD/DI/TI into one 0-1 score.
- `BMInterface`: stratified train/val/test splits exposed as aif360
  ``BinaryLabelDataset``\ s and as scaled (X, y) arrays.
- `BMMetrics`: classification + fairness metrics of a model's predictions on
  the test split, using a threshold tuned on the validation split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Value used for a metric that is missing/invalid; pushes the score towards "unfair".
_UNFAIR_DEFAULTS = {'EOD': 1.0, 'AOD': 1.0, 'SPD': 1.0, 'DI': 0.1, 'TI': 1.0}
# Public: acceptable range and optimum of each metric (used by plots too).
METRIC_OPTIMA = {'EOD': 0.0, 'AOD': 0.0, 'SPD': 0.0, 'DI': 1.0, 'TI': 0.0}
METRIC_RANGES = {'EOD': (-0.1, 0.1), 'AOD': (-0.1, 0.1), 'SPD': (-0.1, 0.1),
           'DI': (0.8, 1.2), 'TI': (0.0, 0.25)}
# Keys of `BMMetrics.get_report()` -> short metric names above.
REPORT_KEY_TO_METRIC = {'eq_opp_diff': 'EOD', 'avg_odd_diff': 'AOD', 'spd': 'SPD',
                        'disparate_impact': 'DI', 'theil_idx': 'TI'}
# Deviation from the optimum is divided by these (the half-width of each range).
_DEVIATION_SCALE = {'EOD': 0.1, 'AOD': 0.1, 'SPD': 0.1, 'DI': 0.2, 'TI': 0.25}
# Plausible domain of each metric; values outside are treated as invalid.
_PLAUSIBLE = {'EOD': (-1.0, 1.0), 'AOD': (-1.0, 1.0), 'SPD': (-1.0, 1.0),
              'DI': (1e-6, 5.0), 'TI': (0.0, 1.0)}

_OUT_OF_RANGE_PENALTY = 0.2
_DEVIATION_WEIGHT = 0.16


def _sanitize_metric(name: str, value):
    """Return `value` as a finite, plausible float, else the unfair default for `name`.

    Containers (list, tuple, ndarray, Series) are unwrapped to their first element.
    None, bool, blank strings, non-numeric, non-finite and implausible values map
    to the default instead of raising.
    """
    default = _UNFAIR_DEFAULTS[name]
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return _sanitize_metric(name, value[0]) if len(value) else default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    low, high = _PLAUSIBLE[name]
    if not np.isfinite(v) or not low <= v <= high:
        return default
    return v


def calculate_fairness_score(EOD: float, AOD: float, SPD: float, DI: float, TI: float) -> dict:
    """
    Aggregate five fairness metrics into a single score (0 = perfectly fair).

    Each metric adds a 0.2 penalty if outside its acceptable range plus
    0.16 * (\|value - optimum\| / range half-width). The sum is normalised by the
    maximum (5 * 0.36) and clipped to [0, 1].

    Acceptable range / optimum: EOD, AOD, SPD in (-0.1, 0.1) / 0;
    DI in (0.8, 1.2) / 1; TI in (0, 0.25) / 0.

    Invalid inputs (None, NaN/inf, non-numeric, implausible such as \|EOD\| > 1,
    DI outside [1e-6, 5], TI outside [0, 1]) never raise: they are replaced by
    "unfair" defaults (EOD/AOD/SPD/TI = 1.0, DI = 0.1).

    Returns:
        dict with 'raw_score', 'overall_score' (rounded to 3 decimals),
        'metric_evaluations' (bool per metric), 'deviations' (rounded) and
        'is_fair' (all metrics in range).
    """
    values = {name: _sanitize_metric(name, v)
              for name, v in (('EOD', EOD), ('AOD', AOD), ('SPD', SPD), ('DI', DI), ('TI', TI))}

    evaluations = {n: METRIC_RANGES[n][0] <= v <= METRIC_RANGES[n][1] for n, v in values.items()}
    deviations = {n: abs(v - METRIC_OPTIMA[n]) / _DEVIATION_SCALE[n] for n, v in values.items()}

    raw_score = 0.0
    for name, in_range in evaluations.items():
        if not in_range:
            raw_score += _OUT_OF_RANGE_PENALTY
        raw_score += deviations[name] * _DEVIATION_WEIGHT
    max_score = len(evaluations) * (_OUT_OF_RANGE_PENALTY + _DEVIATION_WEIGHT)

    return {
        'raw_score': raw_score,
        'overall_score': round(min(1.0, max(0.0, raw_score / max_score)), 3),
        'metric_evaluations': evaluations,
        'deviations': {k: round(v, 3) for k, v in deviations.items()},
        'is_fair': all(evaluations.values()),
    }


@dataclass
class BMnames:
    """Dataset column configuration for bias mitigation."""
    label_names: str
    protected_att: list
    favorable_label: float = 1.0
    unfavorable_label: float = 0.0


def _stratified_split(df: pd.DataFrame, label: str, seed: int, shuffle: bool,
                      test_size: float, val_size: float) -> tuple:
    """Split into (train, val, test), stratified by `label`.

    `test_size` and `val_size` are fractions of the full `df`.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=shuffle, stratify=df[label])
    train, val = train_test_split(
        train_val, test_size=val_size / (1.0 - test_size), random_state=seed,
        shuffle=shuffle, stratify=train_val[label])
    return train, val, test


def _has_all_classes(label: str, *splits: pd.DataFrame) -> bool:
    """True if every split contains the same two label values."""
    class_sets = [set(np.unique(s[label])) for s in splits]
    return len(class_sets[0]) == 2 and all(c == class_sets[0] for c in class_sets)


class BMInterface:
    """
    Splits a DataFrame (stratified train/val/test) and serves the splits as
    aif360 ``BinaryLabelDataset``\ s and as standardised (X, y) arrays.

    The (X, y) arrays are always standardised with a scaler fitted on the train
    split. The BLDs are standardised only after `set_transform()`.

    Splitting: seeds `random_state`, `random_state + 1`, ... (15 tries) are tried
    until train, val and test all contain both classes. If none does, the
    minority class is oversampled to at least 3 rows and the search is repeated;
    otherwise the last attempt is used.
    """

    _MAX_SPLIT_TRIES = 15

    def __init__(self, df: pd.DataFrame, label: str, protected: list,
                 random_state: int = 42, shuffle: bool = True,
                 test_size: float = 0.1, val_size: float = 0.1):
        if df is None:
            raise ValueError("`df` must be provided.")
        if label is None or protected is None:
            raise ValueError("`label` and `protected` must be provided.")
        if len(pd.unique(df[label])) < 2:
            raise ValueError("BMInterface requires at least two classes in the label column.")

        self.data_sets = list(self._split(df, label, random_state, shuffle, test_size, val_size))
        self.BM_attr = BMnames(label_names=label, protected_att=protected)
        self.transform = False
        self._generate_sets()
        self._scale_split()

    def _split(self, df, label, random_state, shuffle, test_size, val_size) -> tuple:
        def search(frame):
            split = None
            for i in range(self._MAX_SPLIT_TRIES):
                split = _stratified_split(frame, label, random_state + i, shuffle,
                                          test_size, val_size)
                if _has_all_classes(label, *split):
                    return split, True
            return split, False

        split, ok = search(df)
        if not ok:
            split, _ = search(self._oversample_minority(df, label, random_state))
        return split

    @staticmethod
    def _oversample_minority(df: pd.DataFrame, label: str, random_state: int) -> pd.DataFrame:
        """Duplicate minority-class rows (with replacement) up to 3 rows in total."""
        counts = df[label].value_counts()
        minority = counts.idxmin()
        need = max(0, 3 - int(counts[minority]))
        if need == 0:
            return df
        extra = df[df[label] == minority].sample(n=need, replace=True, random_state=random_state)
        return pd.concat([df, extra], ignore_index=True)

    def _generate_sets(self) -> None:
        """(Re)build the BinaryLabelDatasets from the raw splits."""
        self.biLData = [
            BinaryLabelDataset(
                df=data,
                label_names=[self.BM_attr.label_names],
                protected_attribute_names=self.BM_attr.protected_att,
                favorable_label=self.BM_attr.favorable_label,
                unfavorable_label=self.BM_attr.unfavorable_label)
            for data in self.data_sets
        ]

    def _scale_transform(self) -> None:
        """Standardise BLD features in place, fitting on the train BLD."""
        scaler = StandardScaler()
        self.biLData[0].features = scaler.fit_transform(self.biLData[0].features)
        self.biLData[1].features = scaler.transform(self.biLData[1].features)
        self.biLData[2].features = scaler.transform(self.biLData[2].features)

    def _scale_split(self) -> None:
        """Rebuild the (X, y) tuples from the current BLDs."""
        scaler = StandardScaler().fit(self.biLData[0].features)
        self.trainXY, self.valXY, self.testXY = [
            (scaler.transform(bld.features), bld.labels.ravel()) for bld in self.biLData]

    def set_transform(self) -> None:
        """Standardise the BLD features (idempotent: a second call is a no-op)."""
        if self.transform:
            return
        self.transform = True
        self._scale_transform()

    def get_protected_att(self) -> list:
        return self.BM_attr.protected_att

    def restore_BLD(self) -> None:
        """Discard any mitigation: rebuild the BLDs (scaled if in transform mode)."""
        self._generate_sets()
        if self.transform:
            self._scale_transform()
        self._scale_split()

    def get_train_BLD(self) -> BinaryLabelDataset:
        """Deep copy of the train BLD."""
        return self.biLData[0].copy(deepcopy=True)

    def get_val_BLD(self) -> BinaryLabelDataset:
        """Deep copy of the validation BLD."""
        return self.biLData[1].copy(deepcopy=True)

    def get_test_BLD(self) -> BinaryLabelDataset:
        """Deep copy of the test BLD (ground truth)."""
        return self.biLData[2].copy(deepcopy=True)

    def get_train_xy(self) -> tuple:
        return self.trainXY

    def get_val_xy(self) -> tuple:
        return self.valXY

    def get_test_xy(self) -> tuple:
        return self.testXY

    def pre_BM_set(self, new_train_BLD: BinaryLabelDataset) -> None:
        """Install a pre-processed train BLD (e.g. reweighed) and refresh scaling/(X, y)."""
        self.biLData[0] = new_train_BLD
        if self.transform:
            self._scale_transform()
        self._scale_split()

    def pos_bm_set(self, new_test_BLD: BinaryLabelDataset) -> None:
        """Install the features/protected attributes/weights of a post-processed test BLD.

        The ground-truth test labels are kept: a post-processed BLD carries
        *predicted* labels, and storing them would make `get_test_BLD()` (the
        reference for every metric) compare predictions with themselves.
        `new_test_BLD` is expected to derive from `get_test_BLD()` (so it is
        already scaled in transform mode); it is therefore not re-scaled.
        """
        truth = self.biLData[2]
        installed = new_test_BLD.copy(deepcopy=True)
        installed.labels = truth.labels.copy()
        installed.scores = truth.scores.copy()
        self.biLData[2] = installed


def find_best_threshold(scores: np.ndarray, truth: BinaryLabelDataset,
                        thresholds: np.ndarray, inclusive: bool = False) -> float:
    """Return the threshold with the highest balanced accuracy (first on ties).

    Balanced accuracy is 0.5 * (TPR + TNR), computed exactly as aif360's
    `ClassificationMetric` does: weighted by `truth.instance_weights`, relative
    to `truth`'s favorable/unfavorable labels. A sample is predicted favorable
    if `score > t` (or `score >= t` when `inclusive`).
    """
    y = truth.labels.ravel()
    w = truth.instance_weights
    pos, neg = y == truth.favorable_label, y == truth.unfavorable_label
    n_pos = np.sum(w[pos], dtype=np.float64)
    n_neg = np.sum(w[neg], dtype=np.float64)
    scores = np.asarray(scores).ravel()

    balanced_acc = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        pred_pos = scores >= t if inclusive else scores > t
        tp = np.sum(w[pos & pred_pos], dtype=np.float64)
        tn = np.sum(w[neg & ~pred_pos], dtype=np.float64)
        balanced_acc[i] = 0.5 * (tp / n_pos + tn / n_neg)
    return thresholds[int(np.argmax(balanced_acc))]


def _safe_ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, or 0.0 when the denominator is zero."""
    return float(numerator / denominator) if denominator != 0 else 0.0


class BMMetrics:
    """
    Classification and fairness metrics of predictions on the test split.

    The decision threshold is the one maximising balanced accuracy on the
    validation split (`best_class_thresh`, searched over
    `np.linspace(0.01, 0.99, 100)`); metrics are then computed once on the test
    split with that threshold.

    Predictions are either probability matrices of shape (n, n_classes) whose
    columns follow `class_array` (positive iff score > threshold), or aif360
    ``BinaryLabelDataset``\ s (in-processing mode; their `.scores` are used and the
    positive rule is score >= threshold). The caller's objects are not mutated.
    """

    N_THRESHOLDS = 100

    def __init__(self, bmI: BMInterface, class_array: np.ndarray,
                 pred_val: np.ndarray | BinaryLabelDataset,
                 pred_test: np.ndarray | BinaryLabelDataset,
                 privileged_group: list[dict], unprivileged_group: list[dict]):
        self.bmI = bmI
        self.pos_idx = np.where(class_array == bmI.BM_attr.favorable_label)[0][0]
        self.in_mode = isinstance(pred_val, BinaryLabelDataset)
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group
        self.class_threshold = np.linspace(0.01, 0.99, self.N_THRESHOLDS)
        self.set_new_pred(pred_val, pred_test)

    def _positive_scores(self, pred) -> np.ndarray:
        """Column of positive-class scores as an (n, 1) array."""
        if self.in_mode:
            return pred.scores
        return pred[:, self.pos_idx].reshape(-1, 1)

    def _classification_metric(self, predicted: BinaryLabelDataset) -> ClassificationMetric:
        return ClassificationMetric(self.bmI.get_test_BLD(), predicted,
                                    unprivileged_groups=self.unprivileged_group,
                                    privileged_groups=self.privileged_group)

    def set_new_pred(self, pred_val: np.ndarray | BinaryLabelDataset,
                     pred_test: np.ndarray | BinaryLabelDataset) -> None:
        """Replace the predictions, re-tune the threshold on val and rebuild test metrics."""
        self.pred_val = pred_val
        self.pred_test = pred_test

        # NOTE: inconsistency kept from the original: `>` for arrays, `>=` for BLDs.
        self.best_class_thresh = find_best_threshold(
            self._positive_scores(pred_val), self.bmI.get_val_BLD(),
            self.class_threshold, inclusive=self.in_mode)

        scores = self._positive_scores(pred_test)
        if self.in_mode:
            predicted = pred_test.copy(deepcopy=True)
            # NOTE: like the original, writes int 0/1 labels whatever the label values.
            predicted.labels = (scores >= self.best_class_thresh).astype(int)
        else:
            predicted = self.bmI.get_test_BLD()
            predicted.scores = scores
            predicted.labels = np.where(scores > self.best_class_thresh,
                                        predicted.favorable_label,
                                        predicted.unfavorable_label)
        self.cmetrics = self._classification_metric(predicted)

    def set_pos_pred(self, test_BLD_pred: BinaryLabelDataset) -> None:
        """Compute metrics for a post-processed test prediction.

        `test_BLD_pred.labels` (set by the post-processor) are used as-is, with
        no re-thresholding: post-processors such as EqOdds, CalibratedEqOdds and
        RejectOption decide labels themselves and `scores` may be the unchanged
        probabilities. Note that MCC is computed from `scores` (aif360's
        generalised confusion matrix), so it reflects those scores, not the
        post-processed labels. The argument is not modified.
        """
        self.cmetrics = self._classification_metric(test_BLD_pred)

    def get_pred_test(self):
        """The test predictions exactly as given by the caller."""
        return self.pred_test

    def _get_classification_metrics(self) -> tuple:
        """(balanced_acc, acc, precision, recall, f1, mcc); f1/mcc are 0.0 when undefined."""
        cm = self.cmetrics
        balanced_acc = 0.5 * (cm.true_positive_rate() + cm.true_negative_rate())
        precision = cm.precision()
        recall = cm.recall()
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        gtp, gfp, gtn, gfn = cm.generalized_binary_confusion_matrix().values()
        denominator = np.sqrt((gtp + gfp) * (gtp + gfn) * (gtn + gfp) * (gtn + gfn))
        mcc = _safe_ratio(gtp * gtn - gfp * gfn, denominator)
        return balanced_acc, cm.accuracy(), precision, recall, f1, mcc

    def _get_fair_metrics(self) -> tuple:
        """(EOD, AOD, SPD, DI, Theil); AOD falls back to the worst value 1.0 if undefined."""
        cm = self.cmetrics
        eq_opp_diff = cm.equal_opportunity_difference()
        try:
            avg_odd_diff = cm.average_odds_difference()
        except (ValueError, ZeroDivisionError, RuntimeError) as e:
            avg_odd_diff = 1.0
            print(f"Warning: avg_odd_diff calculation failed, using worst-case value 1.0. Error: {e}")
        return (eq_opp_diff, avg_odd_diff, cm.statistical_parity_difference(),
                cm.disparate_impact(), cm.theil_index())

    def get_report(self) -> dict:
        """All classification and fairness metrics on the test split."""
        eq_opp_diff, avg_odd_diff, spd, disparate_impact, theil_idx = self._get_fair_metrics()
        balanced_acc, acc, precision, recall, f1, mcc = self._get_classification_metrics()
        return {'balanced_acc': balanced_acc, 'acc': acc, 'precision': precision,
                'recall': recall, 'f1': f1, 'mcc': mcc, 'eq_opp_diff': eq_opp_diff,
                'avg_odd_diff': avg_odd_diff, 'spd': spd,
                'disparate_impact': disparate_impact, 'theil_idx': theil_idx}

    def threshold_curve(self) -> pd.DataFrame:
        """Validation-split metrics at every candidate threshold.

        Columns: threshold, balanced_acc, spd, disparate_impact, eq_opp_diff, using
        the same positive rule as the threshold search (`>` for probability
        arrays, `>=` for in-processing datasets) and aif360's definitions
        (unprivileged minus privileged; DI is the ratio). Undefined values are NaN.
        """
        truth = self.bmI.get_val_BLD()
        y, w = truth.labels.ravel(), truth.instance_weights
        pos, neg = y == truth.favorable_label, y == truth.unfavorable_label
        everyone = np.ones_like(pos)
        priv = self._group_mask(truth, self.privileged_group)
        unpriv = self._group_mask(truth, self.unprivileged_group)
        scores = self._positive_scores(self.pred_val).ravel()

        def rate(mask, pred_pos, base):
            den = np.sum(w[mask & base], dtype=np.float64)
            return np.sum(w[mask & base & pred_pos], dtype=np.float64) / den if den else np.nan

        rows = []
        for t in self.class_threshold:
            pp = scores >= t if self.in_mode else scores > t
            tpr, tnr = rate(everyone, pp, pos), 1 - rate(everyone, pp, neg)
            sr_p, sr_u = rate(priv, pp, everyone), rate(unpriv, pp, everyone)
            with np.errstate(divide='ignore', invalid='ignore'):
                di = np.float64(sr_u) / np.float64(sr_p)
            rows.append((t, 0.5 * (tpr + tnr), sr_u - sr_p, di,
                         rate(unpriv, pp, pos) - rate(priv, pp, pos)))
        return pd.DataFrame(rows, columns=['threshold', 'balanced_acc', 'spd',
                                           'disparate_impact', 'eq_opp_diff'])

    @staticmethod
    def _group_mask(bld: BinaryLabelDataset, groups: list[dict]) -> np.ndarray:
        """Boolean mask of the rows matching any dict in `groups`."""
        names = list(bld.protected_attribute_names)
        mask = np.zeros(len(bld.labels), dtype=bool)
        for g in groups:
            m = np.ones(len(mask), dtype=bool)
            for k, v in g.items():
                m &= bld.protected_attributes[:, names.index(k)] == v
            mask |= m
        return mask

    def get_score(self) -> dict:
        """`calculate_fairness_score` of the current fairness metrics."""
        return calculate_fairness_score(*self._get_fair_metrics())

    def get_groups(self) -> tuple:
        """(privileged_group, unprivileged_group)."""
        return self.privileged_group, self.unprivileged_group
