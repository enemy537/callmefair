"""Tests for callmefair.viz (plots) and BMMetrics.threshold_curve."""
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from aif360.metrics import ClassificationMetric  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from callmefair import BMGridSearch, BMInterface, BMType  # noqa: E402
from callmefair.util import fair_util  # noqa: E402
from callmefair.viz import (pareto_front, plot_bias_search, plot_fair_ranges,  # noqa: E402
                            plot_group_rates, plot_metric_heatmap, plot_mitigation_effect,
                            plot_threshold_sweep, plot_tradeoff)

warnings.filterwarnings("ignore")
STROKE = Path(__file__).resolve().parents[1] / "examples" / "data" / "stroke_pre" / "stroke.csv"
PRIV, UNPRIV = [{'age': 1}], [{'age': 0}]


@pytest.fixture(scope="module")
def grid():
    np.random.seed(0)
    df = pd.read_csv(STROKE)
    g = BMGridSearch(BMInterface(df, "stroke", ["age"]), LogisticRegression(max_iter=500),
                     [[BMType.preReweighing], [BMType.posEqqOds]], PRIV, UNPRIV)
    return g, g.run_single_sensitive(results_dir=None)


@pytest.fixture(scope="module")
def rows(grid):
    return grid[1]


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


# ------------------------------------------------------------------ fair_util additions
def test_constants_unchanged():
    assert fair_util.METRIC_RANGES == {'EOD': (-0.1, 0.1), 'AOD': (-0.1, 0.1), 'SPD': (-0.1, 0.1),
                                       'DI': (0.8, 1.2), 'TI': (0.0, 0.25)}
    assert fair_util.METRIC_OPTIMA == {'EOD': 0.0, 'AOD': 0.0, 'SPD': 0.0, 'DI': 1.0, 'TI': 0.0}
    assert fair_util.REPORT_KEY_TO_METRIC == {'eq_opp_diff': 'EOD', 'avg_odd_diff': 'AOD',
                                              'spd': 'SPD', 'disparate_impact': 'DI',
                                              'theil_idx': 'TI'}
    assert fair_util.calculate_fairness_score(0.2, 0, 0, 1.0, 0)['overall_score'] == 0.289


@pytest.mark.parametrize("pick", [3, 40, 90])
def test_threshold_curve_matches_aif360(grid, pick):
    bmM = grid[0].bmM
    curve = bmM.threshold_curve()
    assert list(curve.columns) == ['threshold', 'balanced_acc', 'spd', 'disparate_impact',
                                   'eq_opp_diff']
    assert len(curve) == bmM.N_THRESHOLDS
    t = curve['threshold'][pick]
    truth = bmM.bmI.get_val_BLD()
    pred = truth.copy(deepcopy=True)
    scores = bmM.pred_val[:, bmM.pos_idx]
    pred.labels = np.where(scores > t, truth.favorable_label, truth.unfavorable_label).reshape(-1, 1)
    cm = ClassificationMetric(truth, pred, unprivileged_groups=UNPRIV, privileged_groups=PRIV)
    row = curve.iloc[pick]
    assert row['balanced_acc'] == pytest.approx(0.5 * (cm.true_positive_rate() + cm.true_negative_rate()))
    assert row['spd'] == pytest.approx(cm.statistical_parity_difference())
    assert row['eq_opp_diff'] == pytest.approx(cm.equal_opportunity_difference())
    assert row['disparate_impact'] == pytest.approx(cm.disparate_impact(), nan_ok=True)
    best = curve.loc[curve['balanced_acc'].idxmax(), 'threshold']
    assert best == pytest.approx(bmM.best_class_thresh)


# ------------------------------------------------------------------ pareto
def test_pareto_front_hand_made():
    df = pd.DataFrame({'fair_score': [0.1, 0.2, 0.3, 0.1, 0.5],
                       'balanced_acc': [0.60, 0.70, 0.65, 0.55, 0.9], 'BM': list('abcde')})
    assert list(pareto_front(df)['BM']) == ['a', 'b', 'e']
    assert list(pareto_front(df.to_dict('records'))['BM']) == ['a', 'b', 'e']


# ------------------------------------------------------------------ plots
def test_tradeoff(rows):
    fig = plot_tradeoff(rows)
    assert isinstance(fig, Figure) and len(fig.axes) == 1
    assert len(fig.axes[0].get_legend().get_texts()) >= 3
    assert isinstance(plot_tradeoff(pd.DataFrame(rows), pareto=False), Figure)


def test_tradeoff_given_axes(rows):
    _, ax = plt.subplots()
    assert plot_tradeoff(rows, ax=ax) is ax.figure


def test_heatmap(rows):
    fig = plot_metric_heatmap(rows)
    ax = fig.axes[0]
    assert ax.images[0].get_array().shape == (len(rows), 1)
    assert [t.get_text() for t in ax.get_yticklabels()][0] == 'baseline'
    assert len(ax.texts) == len(rows)
    assert len(plot_metric_heatmap(pd.DataFrame(rows), 'balanced_acc', annotate=False).axes[0].texts) == 0


def test_mitigation_effect(rows):
    fig = plot_mitigation_effect(rows)
    assert len(fig.axes) == 2
    assert all(len(a.patches) == len(rows) - 1 for a in fig.axes)
    assert len(plot_mitigation_effect(pd.DataFrame(rows), ('spd',), model='LogisticRegression').axes) == 1


def test_mitigation_effect_blue_is_better(rows):
    ax = plot_mitigation_effect(rows, ('fair_score',)).axes[0]
    base = rows[0]['fair_score']
    for patch, r in zip(ax.patches, rows[1:]):
        red, _, blue, _ = patch.get_facecolor()
        if r['fair_score'] < base - 0.02:
            assert blue > red
        if r['fair_score'] > base + 0.02:
            assert red > blue


def test_fair_ranges(rows):
    fig = plot_fair_ranges(rows)
    assert len(fig.axes) == 5
    assert all(1 < len(a.lines) <= len(rows) + 1 for a in fig.axes)  # dots (+ optimum)
    assert fig.legends and len(fig.legends[0].get_texts()) >= 5


def test_fair_ranges_clips_infinite():
    r = [{'model': 'm', 'BM': 'baseline', 'eq_opp_diff': 0.0, 'avg_odd_diff': 0.0, 'spd': 0.0,
          'disparate_impact': float('inf'), 'theil_idx': 0.1}]
    fig = plot_fair_ranges(r)
    assert np.isfinite(fig.axes[3].get_xlim()).all()


def test_bias_search():
    table = [['Attribute', 'Raw Fairness Score', 'Normalized Fairness score', 'Balanced Accuracy',
              'Privileged N', 'Unprivileged N'],
             ['race', 1.0, 0.2, 0.6, 100, 50], ['age', 2.0, 0.4, 0.6, 90, 40]]
    fig = plot_bias_search(table)
    ax = fig.axes[0]
    assert len(ax.patches) == 2 and ax.get_legend() is None
    assert [t.get_text() for t in ax.get_yticklabels()] == ['age', 'race']
    assert len(ax.texts) == 2
    assert isinstance(plot_bias_search(pd.DataFrame(table[1:], columns=table[0])), Figure)


def test_group_rates(grid):
    fig = plot_group_rates(grid[0].bmM)
    ax = fig.axes[0]
    assert len(ax.patches) == 6 and len(ax.get_legend().get_texts()) == 2


def test_threshold_sweep(grid):
    bmM = grid[0].bmM
    fig = plot_threshold_sweep(bmM)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_shared_x_axes().joined(*fig.axes)
    assert all(any(abs(l.get_xdata()[0] - bmM.best_class_thresh) < 1e-9 for l in a.lines) for a in fig.axes)
    assert isinstance(plot_threshold_sweep(bmM, metric='disparate_impact'), Figure)
    assert isinstance(plot_threshold_sweep(bmM, metric='eq_opp_diff'), Figure)


# ------------------------------------------------------------------ import cost
def test_import_callmefair_does_not_import_matplotlib():
    code = "import sys, callmefair; sys.exit(int('matplotlib' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
