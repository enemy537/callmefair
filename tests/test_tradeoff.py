"""Tests for callmefair.tradeoff (Fairea map) and its plot."""

import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from callmefair.tradeoff import (_mutate, area_polygon, baseline_curve, classify_region,
                                 compute_area, normalize, select_strategy, tradeoff_regions)

# A straight baseline as in the paper: F_OM at (1, 1), F100 at (0, 0) after normalisation.
DIAG = (np.linspace(1, 0, 11), np.linspace(1, 0, 11))     # (bias, acc), ordered by degree


def _baseline_df(acc0=0.73, bias0=0.88):
    """Synthetic create_baseline() output: linear decay to (bias 0, acc 0.5) for 'to_1'."""
    deg = np.round(np.linspace(0, 1, 11), 1)
    rows = []
    for strategy, end_acc in (("to_0", 0.5), ("to_1", 0.5)):
        steep = 1.0 if strategy == "to_1" else 1.2       # to_0 loses accuracy faster
        acc = acc0 - (acc0 - end_acc) * np.clip(deg * steep, 0, 1)
        for d, a, b in zip(deg, acc, bias0 * (1 - deg)):
            rows.append(dict(strategy=strategy, degree=d, acc=a, balanced_acc=a, spd=b, aod=b,
                             eod=b, fair_score=min(1.0, b)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ regions (paper Fig. 2)

@pytest.mark.parametrize("point, region", [
    ((1.1, 0.5), "win-win"),          # more accurate, less biased than F_OM
    ((1.1, 1.2), "inverted"),         # more accurate, more biased
    ((0.5, 1.2), "lose-lose"),        # less accurate, more biased
    ((0.6, 0.3), "good trade-off"),   # above the baseline
    ((0.3, 0.6), "bad trade-off"),    # below the baseline
    ((0.2, -0.1), "good trade-off"),  # less biased than F100 but more accurate than the line
    ((0.5, 1.0), "lose-lose"),        # no bias reduction (e.g. saturated score), less accurate
])
def test_regions_follow_the_paper(point, region):
    assert classify_region(*DIAG, {"m": point}) == {"m": region}


def _ray_reference(base_bias, base_acc, acc, bias):
    """Independent implementation of Fairea's ray tests (valid inside [0, 1]^2)."""
    shapely = pytest.importorskip("shapely.geometry")
    base = shapely.LineString(list(zip(base_bias, base_acc)))
    p = shapely.Point(bias, acc)
    down, right = (shapely.LineString([(p.x, p.y), (p.x, 0)]),
                   shapely.LineString([(p.x, p.y), (2, p.y)]))
    up, left = (shapely.LineString([(p.x, p.y), (p.x, 2)]),
                shapely.LineString([(p.x, p.y), (0, p.y)]))
    if base.intersects(down) and base.intersects(right):
        return "good trade-off"
    if base.intersects(down):
        return "win-win"
    if base.intersects(up):
        return "bad trade-off"
    if base.intersects(left):
        return "lose-lose"
    return "inverted"


def test_regions_match_fairea_ray_tests_inside_the_baseline_range():
    rng = np.random.default_rng(0)
    bias = np.linspace(1, 0, 11)
    acc = np.clip(bias + rng.normal(0, 0.03, 11), 0, 1)       # noisy, near-diagonal
    acc[0], acc[-1] = 1.0, 0.0
    pts = {i: (a, b) for i, (a, b) in enumerate(rng.uniform(0.01, 0.99, (400, 2)))}
    ours = classify_region(bias, acc, pts)
    for i, (a, b) in pts.items():
        assert ours[i] == _ray_reference(bias, acc, a, b), (a, b)


# ------------------------------------------------------------------ area (paper Fig. 3)

def test_area_is_the_triangle_between_point_and_straight_baseline():
    # point (bias 0.2, acc 0.6): horizontal hits the diagonal at bias 0.6, vertical at acc 0.2
    assert compute_area(*DIAG, (0.6, 0.2)) == pytest.approx(0.5 * 0.4 * 0.4, rel=1e-3)
    poly = area_polygon(*DIAG, (0.6, 0.2))
    assert poly[0].tolist() == [0.2, 0.6]                     # BM first


def test_area_is_zero_outside_good_trade_off():
    for point in [(0.3, 0.6), (1.1, 0.5), (0.5, 1.2), (1.1, 1.2)]:
        assert compute_area(*DIAG, point) == 0.0
        assert area_polygon(*DIAG, point) is None


def test_area_grows_with_distance_from_baseline():
    assert compute_area(*DIAG, (0.8, 0.2)) > compute_area(*DIAG, (0.6, 0.2)) > 0


# ------------------------------------------------------------------ normalisation & strategy

def test_normalize_uses_the_baseline_range():
    acc, bias, m = normalize([0.73, 0.5], [0.88, 0.0], {"x": (0.8, -0.1)})
    assert acc.tolist() == [1.0, 0.0] and bias.tolist() == [1.0, 0.0]
    assert m["x"][0] > 1 and m["x"][1] < 0                    # methods may leave [0, 1]


def test_strategy_rule_and_tie_break():
    base = _baseline_df()
    # both end at 0.5 balanced accuracy -> tie broken by the stricter (higher mean) curve
    assert select_strategy(base) == "to_1"
    base.loc[(base.strategy == "to_0") & (base.degree == 1.0), "acc"] = 0.95
    assert select_strategy(base, accuracy="acc") == "to_0"


def test_warns_when_fairea_requirement_is_not_met():
    base = _baseline_df()
    base.loc[base.degree == 1.0, "acc"] = 0.95               # constant model beats original
    with pytest.warns(UserWarning, match="requirement not met"):
        baseline_curve(base, "to_1", accuracy="acc")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        baseline_curve(base, "to_1", accuracy="balanced_acc")


def test_mutate_changes_exactly_the_requested_fraction():
    labels = np.zeros((100, 1))
    out = _mutate(labels, 0.3, (0.0, 1.0), np.random.default_rng(0))
    assert int(out.sum()) == 30 and labels.sum() == 0


# ------------------------------------------------------------------ one call + plot

ROWS = [dict(model="LR", BM="baseline", balanced_acc=0.73, acc=0.8, spd=-0.88, avg_odd_diff=-0.6,
             eq_opp_diff=-0.6, fair_score=1.0),
        dict(model="LR", BM="preReweighing", balanced_acc=0.65, acc=0.7, spd=-0.10,
             avg_odd_diff=-0.1, eq_opp_diff=-0.1, fair_score=0.4),
        dict(model="LR", BM="posROC", balanced_acc=0.55, acc=0.6, spd=-0.60, avg_odd_diff=-0.5,
             eq_opp_diff=-0.5, fair_score=0.9)]


def test_tradeoff_regions_columns_and_attrs():
    reg = tradeoff_regions(_baseline_df(), ROWS)
    assert list(reg["BM"]) == ["preReweighing", "posROC"]     # baseline rows dropped
    assert list(reg["region"]) == ["good trade-off", "bad trade-off"]
    assert reg.loc[0, "area"] > 0 and reg.loc[1, "area"] == 0
    assert reg.attrs["strategies"] == {"LR": "to_1"}
    assert (reg.attrs["accuracy"], reg.attrs["bias"]) == ("balanced_acc", "spd")


def test_plot_tradeoff_map_returns_figure():
    import matplotlib.pyplot as plt
    from callmefair.viz import plot_tradeoff_map
    base = _baseline_df()
    fig = plot_tradeoff_map(base, tradeoff_regions(base, ROWS))
    assert fig.axes and fig.axes[0].get_title() == "Fairea trade-off map"
    plt.close("all")


def test_regions_do_not_depend_on_normalisation():
    """Min-max scaling each axis is monotone affine, so classifying in the metrics' own units
    (as the map is drawn, paper Fig. 2) gives the same regions as Fairea's normalised space."""
    rng = np.random.default_rng(1)
    acc = 0.73 - 0.023 * np.arange(11) + rng.normal(0, 0.003, 11)    # degree 0 .. 100 %
    bias = 0.875 * (1 - np.linspace(0, 1, 11))
    pts = {i: (a, b) for i, (a, b) in enumerate(zip(rng.uniform(0.45, 0.8, 300),
                                                    rng.uniform(-0.05, 1.1, 300)))}
    acc_n, bias_n, pts_n = normalize(acc, bias, pts)
    assert classify_region(bias, acc, pts) == classify_region(bias_n, acc_n, pts_n)


def test_plot_tradeoff_map_draws_metric_units_by_default():
    import matplotlib.pyplot as plt
    from callmefair.viz import plot_tradeoff_map
    base = _baseline_df()
    reg = tradeoff_regions(base, ROWS)
    ax = plot_tradeoff_map(base, reg).axes[0]
    assert "normalised" not in ax.get_xlabel() and ax.get_ylim()[0] > 0.3    # real accuracy
    ax_n = plot_tradeoff_map(base, reg, normalized=True).axes[0]
    assert ax_n.get_xlabel().startswith("normalised")
    plt.close("all")


def test_one_baseline_per_model():
    rows = ROWS + [dict(r, model="XGB") for r in ROWS]
    with pytest.raises(ValueError, match="one baseline per model"):
        tradeoff_regions(_baseline_df(), rows)
    other = _baseline_df(acc0=0.60, bias0=0.4)          # XGB's own original model
    reg = tradeoff_regions({"LR": _baseline_df(), "XGB": other}, rows)
    lr, xgb = reg[reg.model == "LR"], reg[reg.model == "XGB"]
    assert list(lr.region) == ["good trade-off", "bad trade-off"]
    # same mitigation numbers, different reference: XGB's original is less accurate and fairer
    assert xgb.set_index("BM").at["preReweighing", "region"] == "win-win"
    assert reg.attrs["strategies"] == {"LR": "to_1", "XGB": "to_1"}


def test_multi_model_map_has_one_panel_per_model():
    import matplotlib.pyplot as plt
    from callmefair.viz import plot_tradeoff_map
    rows = ROWS + [dict(r, model="XGB") for r in ROWS]
    bases = {"LR": _baseline_df(), "XGB": _baseline_df(acc0=0.60, bias0=0.4)}
    fig = plot_tradeoff_map(bases, tradeoff_regions(bases, rows))
    assert [a.get_title() for a in fig.axes] == ["LR", "XGB"]
    plt.close("all")
