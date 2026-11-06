"""Tests for callmefair.util.fair_util."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.linear_model import LogisticRegression

from callmefair.util.fair_util import BMInterface, BMMetrics, calculate_fairness_score

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
PRIV, UNPRIV = [{'age': 1}], [{'age': 0}]


# ----------------------------------------------------------------- fairness score
def test_perfect_fairness_is_zero():
    r = calculate_fairness_score(0.0, 0.0, 0.0, 1.0, 0.0)
    assert r['overall_score'] == 0.0 and r['raw_score'] == 0.0
    assert r['is_fair'] and all(r['metric_evaluations'].values())
    assert set(r) == {'raw_score', 'overall_score', 'uncapped_score', 'metric_evaluations',
                      'deviations', 'is_fair'}
    assert r['uncapped_score'] == 0.0


def test_uncapped_score_keeps_information_beyond_the_cap():
    a = calculate_fairness_score(-0.6, -0.6, -0.8, 0.1, 0.1)
    b = calculate_fairness_score(-0.9, -0.9, -0.95, 0.05, 0.1)
    assert a['overall_score'] == b['overall_score'] == 1.0          # both clipped
    assert 1.0 < a['uncapped_score'] < b['uncapped_score']           # still ordered


@pytest.mark.parametrize("bad", [None, float('nan'), float('inf'), "abc", " ", True, [], 5.0])
def test_invalid_eod_uses_unfair_default(bad):
    r = calculate_fairness_score(bad, 0.0, 0.0, 1.0, 0.0)
    expected = calculate_fairness_score(1.0, 0.0, 0.0, 1.0, 0.0)
    assert r == expected and not r['is_fair']


@pytest.mark.parametrize("bad", [None, 0.0, -1.0, 6.0, float('nan')])
def test_invalid_di_uses_default_point_one(bad):
    assert calculate_fairness_score(0, 0, 0, bad, 0) == calculate_fairness_score(0, 0, 0, 0.1, 0)


def test_containers_use_first_element_and_strings_are_parsed():
    assert calculate_fairness_score([0.05], np.array([0.0]), "0.0", 1.0, 0.0)['is_fair']


def test_worst_case_is_one_and_in_range_flags():
    worst = calculate_fairness_score(None, None, None, None, None)
    assert worst['overall_score'] == 1.0
    r = calculate_fairness_score(0.1, -0.1, 0.0, 0.8, 0.25)  # boundaries are acceptable
    assert r['is_fair']
    assert not calculate_fairness_score(0.11, 0, 0, 1, 0)['metric_evaluations']['EOD']


# ------------------------------------------------------------------- BMInterface
@pytest.fixture(scope="module")
def stroke():
    return pd.read_csv(DATA / "stroke_pre" / "stroke.csv")


def test_split_sizes_stratification_and_determinism(stroke):
    a = BMInterface(stroke, 'stroke', ['age', 'gender'])
    b = BMInterface(stroke, 'stroke', ['age', 'gender'])
    n = len(stroke)
    sizes = [len(d) for d in a.data_sets]
    assert sum(sizes) == n
    assert abs(sizes[1] - 0.1 * n) <= 2 and abs(sizes[2] - 0.1 * n) <= 2
    rate = stroke['stroke'].mean()
    for d in a.data_sets:
        assert abs(d['stroke'].mean() - rate) < 0.01
    for x, y in zip(a.data_sets, b.data_sets):
        pd.testing.assert_frame_equal(x, y)
    c = BMInterface(stroke, 'stroke', ['age'], random_state=7)
    assert not a.data_sets[2].index.equals(c.data_sets[2].index)


def test_accessors_and_transform(stroke):
    bmi = BMInterface(stroke, 'stroke', ['age', 'gender'])
    X, y = bmi.get_train_xy()
    assert X.shape[0] == y.shape[0] == len(bmi.data_sets[0])
    assert np.allclose(X.mean(axis=0), 0, atol=1e-8)
    raw = bmi.get_train_BLD().features
    assert not np.allclose(raw.mean(axis=0), 0)
    bmi.set_transform()
    scaled = bmi.get_train_BLD().features
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-8)
    bmi.set_transform()  # idempotent
    assert np.array_equal(scaled, bmi.get_train_BLD().features)
    bmi.get_train_BLD().features[:] = 0  # copies: no aliasing
    assert np.array_equal(scaled, bmi.get_train_BLD().features)
    bmi.restore_BLD()
    assert np.allclose(bmi.get_train_BLD().features, scaled)
    assert bmi.get_protected_att() == ['age', 'gender']


def test_single_class_raises(stroke):
    with pytest.raises(ValueError):
        BMInterface(stroke.assign(stroke=0), 'stroke', ['age'])


def test_pos_bm_set_keeps_ground_truth(stroke):
    bmi = BMInterface(stroke, 'stroke', ['age', 'gender'])
    truth = bmi.get_test_BLD().labels.copy()
    pred = bmi.get_test_BLD()
    pred.labels = 1.0 - truth
    bmi.pos_bm_set(pred)
    assert np.array_equal(bmi.get_test_BLD().labels, truth)


# ---------------------------------------------- equivalence with the ORIGINAL code
class _OriginalThresholdLogic:
    """Faithful copy of the original BMMetrics threshold/ClassificationMetric logic
    (git HEAD~ of callmefair/util/fair_util.py), driven by a BMInterface."""

    def __init__(self, bmI, class_array, pred_val, pred_test, priv, unpriv):
        self.bmI = bmI
        self.pos_idx = np.where(class_array == bmI.get_train_BLD().favorable_label)[0][0]
        self.in_mode = isinstance(pred_val, BinaryLabelDataset)
        self.pred_test, self.pred_val = pred_test, pred_val
        self.privileged_group, self.unprivileged_group = priv, unpriv
        self.balanced_acc = np.zeros(100)
        self.class_threshold = np.linspace(0.01, 0.99, 100)
        self.get_thresould()
        self.set_pred_BLD()

    def get_thresould(self):
        if not self.in_mode:
            pred_val_BLD = self.bmI.get_val_BLD()
            pred_val_BLD.scores = self.pred_val[:, self.pos_idx].reshape(-1, 1)
        for idx, class_thresh in enumerate(self.class_threshold):
            if self.in_mode:
                self.pred_val.labels = (self.pred_val.scores >= class_thresh).astype(int)
                pred_val_BLD = self.pred_val
            else:
                fav_idx = pred_val_BLD.scores > class_thresh
                pred_val_BLD.labels[fav_idx] = pred_val_BLD.favorable_label
                pred_val_BLD.labels[~fav_idx] = pred_val_BLD.unfavorable_label
            cm = ClassificationMetric(self.bmI.get_val_BLD(), pred_val_BLD,
                                      unprivileged_groups=self.unprivileged_group,
                                      privileged_groups=self.privileged_group)
            self.balanced_acc[idx] = 0.5 * (cm.true_positive_rate() + cm.true_negative_rate())
        best_idx = np.where(self.balanced_acc == np.max(self.balanced_acc))[0][0]
        self.best_class_thresh = self.class_threshold[best_idx]

    def set_pred_BLD(self):
        if not self.in_mode:
            pred_BLD = self.bmI.get_test_BLD()
            pred_BLD.scores = self.pred_test[:, self.pos_idx].reshape(-1, 1)
        for thresh in self.class_threshold:
            if self.in_mode:
                self.pred_test.labels = (self.pred_test.scores >= thresh).astype(int)
                pred_BLD = self.pred_test
            else:
                fav_idx = pred_BLD.scores > thresh
                pred_BLD.labels[fav_idx] = pred_BLD.favorable_label
                pred_BLD.labels[~fav_idx] = pred_BLD.unfavorable_label
            self.cmetrics = ClassificationMetric(self.bmI.get_test_BLD(), pred_BLD,
                                                 unprivileged_groups=self.unprivileged_group,
                                                 privileged_groups=self.privileged_group)
            if thresh == self.best_class_thresh:
                break

    def report(self):
        c = self.cmetrics
        precision, recall = c.precision(), c.recall()
        gtp, gfp, gtn, gfn = c.generalized_binary_confusion_matrix().values()
        return {
            'balanced_acc': 0.5 * (c.true_positive_rate() + c.true_negative_rate()),
            'acc': c.accuracy(), 'precision': precision, 'recall': recall,
            'f1': 2 * ((precision * recall) / (precision + recall)),
            'mcc': (gtp * gtn - gfp * gfn) / np.sqrt((gtp + gfp) * (gtp + gfn) * (gtn + gfp) * (gtn + gfn)),
            'eq_opp_diff': c.equal_opportunity_difference(),
            'avg_odd_diff': c.average_odds_difference(),
            'spd': c.statistical_parity_difference(),
            'disparate_impact': c.disparate_impact(), 'theil_idx': c.theil_index(),
        }


@pytest.fixture(scope="module")
def fitted(stroke):
    bmi = BMInterface(stroke, 'stroke', ['age', 'gender'])
    X, y = bmi.get_train_xy()
    model = LogisticRegression(max_iter=500).fit(X, y)
    pv = model.predict_proba(bmi.get_val_xy()[0])
    pt = model.predict_proba(bmi.get_test_xy()[0])
    return bmi, model.classes_, pv, pt


def _assert_same(new, old):
    assert new.best_class_thresh == old.best_class_thresh
    ref, got = old.report(), new.get_report()
    assert set(ref) == set(got)
    for k in ref:
        assert got[k] == ref[k], k
    ref_score = calculate_fairness_score(
        ref['eq_opp_diff'], ref['avg_odd_diff'], ref['spd'], ref['disparate_impact'], ref['theil_idx'])
    assert new.get_score() == ref_score


def test_equivalence_ndarray_mode(fitted):
    bmi, classes, pv, pt = fitted
    new = BMMetrics(bmi, classes, pv, pt, PRIV, UNPRIV)
    old = _OriginalThresholdLogic(bmi, classes, pv.copy(), pt.copy(), PRIV, UNPRIV)
    _assert_same(new, old)
    assert new.get_groups() == (PRIV, UNPRIV)


def _bld_pred(bmi, getter, proba):
    bld = getter()
    bld.scores = proba[:, 1].reshape(-1, 1)
    return bld


def test_equivalence_bld_mode_and_no_mutation(fitted):
    bmi, classes, pv, pt = fitted
    mk = lambda: (_bld_pred(bmi, bmi.get_val_BLD, pv), _bld_pred(bmi, bmi.get_test_BLD, pt))
    val_bld, test_bld = mk()
    before = (val_bld.labels.copy(), test_bld.labels.copy())
    new = BMMetrics(bmi, classes, val_bld, test_bld, PRIV, UNPRIV)
    assert np.array_equal(val_bld.labels, before[0]) and np.array_equal(test_bld.labels, before[1])
    assert new.get_pred_test() is test_bld
    old = _OriginalThresholdLogic(bmi, classes, *mk(), PRIV, UNPRIV)
    _assert_same(new, old)


def test_set_new_pred_matches_fresh_construction(fitted):
    bmi, classes, pv, pt = fitted
    m = BMMetrics(bmi, classes, pt, pv, PRIV, UNPRIV)
    m.set_new_pred(pv, pt)
    fresh = BMMetrics(bmi, classes, pv, pt, PRIV, UNPRIV)
    assert m.best_class_thresh == fresh.best_class_thresh and m.get_report() == fresh.get_report()


def test_set_pos_pred_respects_given_labels(fitted):
    bmi, classes, pv, pt = fitted
    m = BMMetrics(bmi, classes, pv, pt, PRIV, UNPRIV)
    post = bmi.get_test_BLD()
    post.scores = pt[:, 1].reshape(-1, 1)  # probabilities, deliberately inconsistent with labels
    post.labels = np.ones_like(post.labels)  # everything favorable
    labels_before = post.labels.copy()
    m.set_pos_pred(post)
    assert np.array_equal(post.labels, labels_before)
    rep = m.get_report()
    assert rep['recall'] == 1.0 and rep['acc'] == pytest.approx(bmi.get_test_BLD().labels.mean())


def test_f1_mcc_degenerate_are_zero_not_nan(fitted):
    bmi, classes, pv, pt = fitted
    m = BMMetrics(bmi, classes, pv, pt, PRIV, UNPRIV)
    post = bmi.get_test_BLD()
    post.labels = np.zeros_like(post.labels)  # no positive prediction: precision = recall = 0
    post.scores = np.zeros_like(post.scores)  # generalised confusion has zero row sums
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        m.set_pos_pred(post)
        rep = m._get_classification_metrics()
    assert rep[4] == 0.0 and rep[5] == 0.0
