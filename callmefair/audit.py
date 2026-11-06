"""Audit the predictions of any classifier for bias.

The model can be anything trained outside callmefair: you only need a table with
the true outcome, the predicted outcome and the group columns.

>>> report = audit_predictions(df, label="y_true", prediction="y_pred",
...                            protected={"age": 1, "race": ["Caucasian"]})
>>> print(report)             # human-readable verdict per attribute
>>> report.to_frame()         # one row per (model, attribute)

Metrics use the definitions of ``BMMetrics`` (aif360 ``ClassificationMetric``):
differences are *unprivileged minus privileged*, so a negative SPD means the
unprivileged group receives the favourable outcome less often.

Command line::

    python -m callmefair.audit preds.csv --label y_true --prediction y_pred --protected age=1
"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from callmefair.util.fair_util import (METRIC_RANGES, REPORT_KEY_TO_METRIC,
                                       calculate_fairness_score, _safe_ratio)

__all__ = ['BiasReport', 'audit_predictions', 'audit_file']

_METRIC_KEYS = ['balanced_acc', 'acc', 'precision', 'recall', 'f1', 'mcc',
                'eq_opp_diff', 'avg_odd_diff', 'spd', 'disparate_impact', 'theil_idx']
_KEY_OF = {v: k for k, v in REPORT_KEY_TO_METRIC.items()}
_GROUP_COLUMNS = ['group', 'n', 'base_rate', 'selection_rate', 'TPR', 'FPR', 'precision']


@dataclass(frozen=True)
class _Audit:
    """Result for one (model, attribute) pair."""
    model: str
    attribute: str
    metrics: dict          # the 11 report metrics + fair_score + fair_score_uncapped
    evaluations: dict      # 'EOD' | 'AOD' | 'SPD' | 'DI' | 'TI' -> inside acceptable range?
    groups: pd.DataFrame

    @property
    def is_fair(self) -> bool:
        return all(self.evaluations.values())


@dataclass
class BiasReport:
    """Bias metrics of one or several models, for one or several protected attributes.

    ``print(report)`` shows :meth:`summary`. Use :meth:`to_rows` to feed
    ``callmefair.viz`` (``plot_fair_ranges``, ``plot_metric_heatmap``,
    ``plot_tradeoff``) and ``plot_group_rates(report, attribute=...)``.
    """
    audits: list

    @property
    def attributes(self) -> list:
        """Audited attributes (intersections included), in order."""
        return list(dict.fromkeys(a.attribute for a in self.audits))

    @property
    def models(self) -> list:
        """Audited model names, in order."""
        return list(dict.fromkeys(a.model for a in self.audits))

    def _select(self, attribute, model) -> list:
        found = [a for a in self.audits
                 if (attribute is None or a.attribute == attribute)
                 and (model is None or a.model == model)]
        if not found:
            raise KeyError(f"no audit for attribute={attribute!r}, model={model!r}; "
                           f"attributes: {self.attributes}, models: {self.models}")
        return found

    def _one(self, attribute, model) -> _Audit:
        found = self._select(attribute, model)
        if len(found) > 1:
            raise ValueError("several audits match; pass `attribute` and/or `model` "
                             f"(attributes: {self.attributes}, models: {self.models})")
        return found[0]

    def metrics(self, attribute: str | None = None, model: str | None = None) -> dict:
        """Metrics of one audit (``attribute``/``model`` may be omitted when unambiguous)."""
        return dict(self._one(attribute, model).metrics)

    def groups(self, attribute: str | None = None, model: str | None = None) -> pd.DataFrame:
        """Per-group table: group, n, base_rate, selection_rate, TPR, FPR, precision."""
        return self._one(attribute, model).groups.copy()

    def is_fair(self, attribute: str | None = None, model: str | None = None) -> bool:
        """True if all five fairness metrics are in their acceptable range."""
        return all(a.is_fair for a in self._select(attribute, model))

    def to_rows(self) -> list:
        """Grid-style rows (``model``, ``BM`` = attribute, metrics, ``fair_score``...)."""
        return [{'model': a.model, 'BM': a.attribute, **a.metrics} for a in self.audits]

    def to_frame(self) -> pd.DataFrame:
        """One row per (model, attribute): metrics, ``fair_score`` and ``is_fair``."""
        return pd.DataFrame([{'attribute': a.attribute, 'model': a.model, **a.metrics,
                              'is_fair': a.is_fair} for a in self.audits])

    def summary(self) -> str:
        """One line per (model, attribute): verdict, offending metrics and fairness score."""
        multi = len(self.models) > 1
        lines = []
        for a in self.audits:
            bad = []
            for name, ok in a.evaluations.items():
                if ok:
                    continue
                value = a.metrics[_KEY_OF[name]]
                shown = 'undefined' if pd.isna(value) else f'{value:.2f}'
                bad.append(f'{name} {shown} outside {METRIC_RANGES[name]}')
            verdict = 'biased: ' + ', '.join(bad) if bad else 'fair: all 5 metrics within range'
            prefix = f'[{a.model}] ' if multi else ''
            lines.append(f"{prefix}{a.attribute}: {verdict}; "
                         f"fairness score {a.metrics['fair_score']:.2f} (0 = fair)")
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ------------------------------------------------------------------ validation
def _columns(df: pd.DataFrame, names: list) -> None:
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(f"column(s) {missing} not found; available: {list(df.columns)}")


def _binary(df: pd.DataFrame, columns: list, favorable) -> dict:
    """Column -> 1.0 (favourable) / 0.0 array; checks NaNs and that values are binary."""
    for c in columns:
        if df[c].isna().any():
            raise ValueError(f"column {c!r} has {int(df[c].isna().sum())} missing values; "
                             "drop or impute those rows first")
    values = pd.unique(pd.concat([df[c] for c in columns], ignore_index=True))
    if len(values) > 2:
        raise ValueError(f"{columns} must be binary, found {len(values)} distinct values: "
                         f"{list(values[:6])}. Threshold scores/probabilities first.")
    if favorable not in values:
        raise ValueError(f"favorable_label={favorable!r} does not appear in {columns}; "
                         f"found values {list(values)}")
    return {c: (df[c] == favorable).to_numpy(dtype=float) for c in columns}


def _privileged_mask(series: pd.Series, spec) -> np.ndarray:
    if series.isna().any():
        raise ValueError(f"protected column {series.name!r} has missing values; "
                         "drop or impute those rows first")
    if callable(spec):
        mask = np.asarray(spec(series), dtype=bool)
        if mask.shape != (len(series),):
            raise ValueError(f"the callable for {series.name!r} must return one boolean per row")
        return mask
    if isinstance(spec, (list, tuple, set, frozenset, np.ndarray)):
        return series.isin(list(spec)).to_numpy()
    return (series == spec).to_numpy()


def _check_groups(name: str, spec, mask: np.ndarray, y: np.ndarray, w) -> None:
    for group, m in (('privileged', mask), ('unprivileged', ~mask)):
        if not m.any():
            raise ValueError(
                f"{name}: the {group} group is empty (privileged spec: {spec!r}); "
                "check the privileged value(s) against the values in that column")
        if not (w[m] * y[m]).sum():
            warnings.warn(f"{name}: the {group} group has no positive (favourable) true "
                          "outcomes, so TPR-based metrics (EOD, AOD) are undefined.",
                          UserWarning, stacklevel=5)


# ------------------------------------------------------------------ metrics
def _call(fn) -> float:
    try:
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return float(fn())
    except (ValueError, ZeroDivisionError, RuntimeError):
        return float('nan')


def _group_table(y, p, w, mask) -> pd.DataFrame:
    def row(name, m):
        y_, p_, w_ = y[m], p[m], w[m]
        tp, pos, neg = (w_ * y_ * p_).sum(), (w_ * y_).sum(), (w_ * (1 - y_)).sum()
        nan_div = lambda a, b: a / b if b else float('nan')
        return {'group': name, 'n': int(m.sum()), 'base_rate': nan_div(pos, w_.sum()),
                'selection_rate': nan_div((w_ * p_).sum(), w_.sum()), 'TPR': nan_div(tp, pos),
                'FPR': nan_div((w_ * (1 - y_) * p_).sum(), neg),
                'precision': nan_div(tp, (w_ * p_).sum())}
    return pd.DataFrame([row('privileged', mask), row('unprivileged', ~mask)],
                        columns=_GROUP_COLUMNS)


def _audit_one(model, attribute, y, p, w, mask) -> _Audit:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import ClassificationMetric

    truth = BinaryLabelDataset(
        df=pd.DataFrame({'y': y, 'g': mask.astype(float), 'w': w}), label_names=['y'],
        protected_attribute_names=['g'], instance_weights_name='w',
        favorable_label=1.0, unfavorable_label=0.0)
    predicted = truth.copy(deepcopy=True)
    predicted.labels = predicted.scores = p.reshape(-1, 1)
    cm = ClassificationMetric(truth, predicted, unprivileged_groups=[{'g': 0.0}],
                              privileged_groups=[{'g': 1.0}])

    precision, recall = _call(cm.precision), _call(cm.recall)
    gtp, gfp, gtn, gfn = cm.generalized_binary_confusion_matrix().values()
    with np.errstate(all='ignore'):
        denominator = np.sqrt((gtp + gfp) * (gtp + gfn) * (gtn + gfp) * (gtn + gfn))
    metrics = {
        'balanced_acc': 0.5 * (_call(cm.true_positive_rate) + _call(cm.true_negative_rate)),
        'acc': _call(cm.accuracy), 'precision': precision, 'recall': recall,
        'f1': _safe_ratio(2 * precision * recall, precision + recall),
        'mcc': _safe_ratio(gtp * gtn - gfp * gfn, denominator),
        'eq_opp_diff': _call(cm.equal_opportunity_difference),
        'avg_odd_diff': _call(cm.average_odds_difference),
        'spd': _call(cm.statistical_parity_difference),
        'disparate_impact': _call(cm.disparate_impact), 'theil_idx': _call(cm.theil_index)}
    score = calculate_fairness_score(*(metrics[k] for k in REPORT_KEY_TO_METRIC))
    metrics['fair_score'] = score['overall_score']
    metrics['fair_score_uncapped'] = score['uncapped_score']
    return _Audit(model, attribute, metrics, score['metric_evaluations'],
                  _group_table(y, p, w, mask))


# ------------------------------------------------------------------ public API
def audit_predictions(df: pd.DataFrame, *, label: str, prediction, protected: dict,
                      favorable_label=1, model: str = 'external', sample_weight: str | None = None,
                      intersections: bool = False) -> BiasReport:
    """Measure the bias of a classifier's predictions.

    Args:
        df: one row per individual.
        label: column with the true outcome (two distinct values).
        prediction: column with the predicted label, or a list of columns to compare
            several models (model names are then the column names).
        protected: ``{attribute: privileged}``. ``privileged`` is a value, a
            list/set of values, or a callable taking the column (Series) and returning
            a boolean mask. Everyone else forms the unprivileged group.
        favorable_label: outcome considered favourable (as in aif360).
        model: model name when ``prediction`` is a single column.
        sample_weight: optional column with instance weights.
        intersections: also audit every pair of attributes; the privileged group of a
            pair is privileged in both (``CType.intersection``), named ``'a_b'``.

    Raises:
        ValueError: missing columns, NaNs, non-binary label/prediction, an empty group.
    Returns:
        A :class:`BiasReport`.
    """
    if not protected:
        raise ValueError("`protected` must map at least one attribute to its privileged value(s)")
    if intersections and len(protected) < 2:
        raise ValueError("`intersections=True` needs at least two protected attributes")
    predictions = [prediction] if isinstance(prediction, str) else list(prediction)
    names = [model] if isinstance(prediction, str) else predictions
    _columns(df, [label, *predictions, *protected, *([sample_weight] if sample_weight else [])])

    binary = _binary(df, [label, *predictions], favorable_label)
    y = binary[label]
    if sample_weight:
        w = df[sample_weight].to_numpy(dtype=float)
        if not (np.isfinite(w).all() and (w > 0).all()):
            raise ValueError(f"sample_weight column {sample_weight!r} must be finite and positive")
    else:
        w = np.ones(len(df))

    masks = {a: _privileged_mask(df[a], spec) for a, spec in protected.items()}
    if intersections:
        for a, b in itertools.combinations(protected, 2):
            masks[f'{a}_{b}'] = masks[a] & masks[b]
    audits = []
    for attribute, mask in masks.items():
        spec = protected.get(attribute, 'privileged in both')
        _check_groups(attribute, spec, mask, y, w)
        audits += [_audit_one(name, attribute, y, binary[col], w, mask)
                   for name, col in zip(names, predictions)]
    return BiasReport(audits)


def audit_file(path, **kwargs) -> BiasReport:
    """:func:`audit_predictions` on a CSV (or parquet, needs pyarrow) file read with pandas."""
    path = Path(path)
    if path.suffix.lower() == '.parquet':
        df = pd.read_parquet(path)
    elif path.suffix.lower() in ('.csv', '.txt', '.gz'):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported file type {path.suffix!r}; use .csv or .parquet")
    return audit_predictions(df, **kwargs)


# ------------------------------------------------------------------ command line
def _parse(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m callmefair.audit',
        description='Audit the predictions of a classifier (CSV/parquet) for bias.')
    parser.add_argument('file')
    parser.add_argument('--label', required=True, help='true-outcome column')
    parser.add_argument('--prediction', required=True, nargs='+', help='prediction column(s)')
    parser.add_argument('--protected', required=True, action='append', metavar='ATTR=VALUE[,VALUE]',
                        help="attribute and its privileged value(s); repeat per attribute")
    parser.add_argument('--favorable-label', default='1')
    parser.add_argument('--sample-weight')
    parser.add_argument('--intersections', action='store_true')
    args = parser.parse_args(argv)

    protected = {}
    for item in args.protected:
        attr, sep, values = item.partition('=')
        if not sep or not values:
            parser.error(f'--protected expects ATTR=VALUE, got {item!r}')
        parsed = [_parse(v) for v in values.split(',')]
        protected[attr] = parsed if len(parsed) > 1 else parsed[0]
    prediction = args.prediction[0] if len(args.prediction) == 1 else args.prediction
    try:
        report = audit_file(args.file, label=args.label, prediction=prediction,
                            protected=protected, favorable_label=_parse(args.favorable_label),
                            sample_weight=args.sample_weight, intersections=args.intersections)
    except (ValueError, OSError) as e:
        parser.error(str(e))
    print(report.summary())
    print()
    print(report.to_frame().round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
