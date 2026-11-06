"""Tests for callmefair.tradeoff (baseline, normalisation, regions, areas) and the map plot."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shapely")
from shapely.geometry import LineString  # noqa: E402

from callmefair.tradeoff import (_mutate, classify_region, compute_area,  # noqa: E402
                                 normalize)

# Normalised baseline: O = (bias 1, loss 0) -> (bias 0, loss 1), the diagonal.
DIAGONAL = LineString([(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)])


def region(loss, bias, line=DIAGONAL):
    return classify_region(line, {"m": (loss, bias)})["m"]


@pytest.mark.parametrize("loss, bias, expected", [
    (0.0, 0.5, "win-win"),          # as accurate as O, less biased
    (0.2, 0.3, "good trade-off"),   # below-left of the curve
    (0.8, 0.5, "bad trade-off"),    # above-right of the curve, still less biased than O
    (0.9, 1.1, "lose-lose"),        # worse than O in both
    (0.5, 1.2, "lose-lose"),
    (0.0, 1.3, "inverted"),         # more biased, as accurate
])
def test_regions(loss, bias, expected):
    assert region(loss, bias) == expected


def test_area_triangle():
    # P=(bias .25, loss .25): right hit (.75,.25), up hit (.25,.75) -> triangle legs .5
    assert compute_area(DIAGONAL, (0.25, 0.25)) == pytest.approx(0.125)


def test_area_zero_outside_good_region():
    for loss, bias in [(0.0, 0.5), (0.8, 0.5), (0.9, 1.1), (0.0, 1.3)]:
        assert compute_area(DIAGONAL, (loss, bias)) == 0.0


def test_area_when_vertical_line_misses_curve():
    line = LineString([(1.0, 0.0), (0.5, 0.5), (0.4, 1.0)])
    area = compute_area(line, (0.3, 0.1))     # bias .3 is below the curve's minimum .4
    assert 0 < area < 1


def test_area_curved_baseline_exceeds_straight():
    bulge = LineString([(1.0, 0.0), (0.9, 0.5), (0.0, 1.0)])
    assert compute_area(bulge, (0.25, 0.25)) > 0


def test_normalize_shared_scale_and_orientation():
    loss, bias, pts = normalize([0.8, 0.7, 0.6], [0.6, 0.3, 0.0], {"m": (0.9, 0.0)})
    assert pts["m"][0] == 0.0 and loss[0] == pytest.approx(1 / 3)
    assert bias.max() == 1.0 and bias.min() == 0.0


def test_normalize_constant_axis():
    loss, bias, _ = normalize([0.7, 0.7], [0.1, 0.1])
    assert (loss == 0).all() and (bias == 0).all()


def test_mutate_is_vectorised_and_fraction_exact():
    labels = np.zeros((100, 1), dtype=int)
    out = _mutate(labels, 0.3, (0.0, 1.0), np.random.default_rng(0))
    assert out.sum() == 30 and labels.sum() == 0
    assert _mutate(labels, 0.0, (0.0, 1.0), np.random.default_rng(0)).sum() == 0


def test_create_baseline_tidy_output():
    from sklearn.linear_model import LogisticRegression
    from callmefair.tradeoff import baseline_curve, create_baseline
    rng = np.random.default_rng(1)
    n = 600
    df = pd.DataFrame({"a": rng.integers(0, 2, n), "x": rng.normal(size=n)})
    df["y"] = ((df.x + 0.5 * df.a + rng.normal(scale=0.5, size=n)) > 0.3).astype(int)
    base = create_baseline(lambda: LogisticRegression(), df, "y", "a", [{"a": 1}], [{"a": 0}],
                           data_splits=1, repetitions=1, degrees=(0, 0.5, 1))
    assert list(base.columns) == ["strategy", "degree", "balanced_acc", "fair_score"]
    assert len(base) == 6
    curve = baseline_curve(base, "to_1")
    assert curve.degree.tolist() == [0, 0.5, 1]
    assert curve.balanced_acc.iloc[-1] == pytest.approx(0.5)


def test_area_polygon_matches_area():
    from callmefair.tradeoff import area_polygon
    assert area_polygon(DIAGONAL, (0.25, 0.25)).area == pytest.approx(0.125)
    assert area_polygon(DIAGONAL, (0.0, 0.5)) is None


# --------------------------------------------------------- tradeoff_regions / map

def _fixture():
    rng = np.random.default_rng(0)
    base = pd.DataFrame({
        "strategy": "to_1", "degree": np.round(np.linspace(0, 1, 11), 1),
        "balanced_acc": np.linspace(0.8, 0.5, 11), "fair_score": np.linspace(0.6, 0.0, 11)})
    rows = [{"model": "LR", "BM": "baseline", "balanced_acc": 0.8, "fair_score": 0.6}]
    for model in ("LR", "XGB"):
        rows += [
            {"model": model, "BM": "preGood", "balanced_acc": 0.74, "fair_score": 0.15},
            {"model": model, "BM": "posBad", "balanced_acc": 0.55, "fair_score": 0.45},
            {"model": model, "BM": "preLose", "balanced_acc": 0.70, "fair_score": 0.65},
            {"model": model, "BM": "posInv", "balanced_acc": 0.82, "fair_score": 0.65},
        ]
    return base, rows


def test_tradeoff_regions_table():
    from callmefair.tradeoff import tradeoff_regions
    base, rows = _fixture()
    out = tradeoff_regions(base, rows)
    assert list(out.columns) == ["model", "BM", "balanced_acc", "fair_score", "loss", "bias",
                                 "region", "paper_region", "area"]
    assert len(out) == 8 and "baseline" not in set(out.BM)
    reg = out.set_index(["model", "BM"]).region
    assert reg["LR", "preGood"] == "good trade-off"
    assert reg["LR", "posBad"] == "bad trade-off"
    assert reg["LR", "preLose"] == "lose-lose"
    assert reg["LR", "posInv"] == "inverted"
    assert out.set_index(["model", "BM"]).paper_region["LR", "posBad"] == "Poor"
    assert (out.area[out.region != "good trade-off"] == 0).all()
    assert (out.area[out.region == "good trade-off"] > 0).all()
    assert out[["loss", "bias"]].min().min() >= 0


def test_tradeoff_regions_averages_repeats():
    from callmefair.tradeoff import tradeoff_regions
    base, rows = _fixture()
    assert len(tradeoff_regions(base, pd.DataFrame(rows + rows))) == 8


def test_plot_tradeoff_map_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from callmefair.tradeoff import tradeoff_regions
    from callmefair.viz.tradeoff_map import plot_tradeoff_map
    base, rows = _fixture()
    fig = plot_tradeoff_map(base, tradeoff_regions(base, rows))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    plt.close(fig)
    fig2, ax = plt.subplots()
    assert plot_tradeoff_map(base, tradeoff_regions(base, rows), ax=ax, label=False) is fig2
    plt.close(fig2)


def test_region_tints_match_classification():
    from callmefair.tradeoff import tradeoff_regions
    from callmefair.viz.tradeoff_map import region_geometries
    from shapely.geometry import Point
    base, rows = _fixture()
    # noisy, non-monotone baseline to exercise the envelope
    base.loc[base.degree == 0.5, "fair_score"] = 0.45
    rng = np.random.default_rng(3)
    extra = [{"model": "LR", "BM": f"m{i}", "balanced_acc": rng.uniform(0.45, 0.85),
              "fair_score": rng.uniform(-0.05, 0.7)} for i in range(60)]
    out = tradeoff_regions(base, rows + extra)
    from callmefair.tradeoff import baseline_curve, normalize
    curve = baseline_curve(base, "to_1")
    meth = {f"{r.model} | {r.BM}": (r.balanced_acc, r.fair_score) for r in out.itertuples()}
    loss_b, bias_b, _ = normalize(curve.balanced_acc, curve.fair_score, meth)
    line = LineString(zip(bias_b, loss_b))
    geoms = region_geometries(line, (-0.04, -0.04, 1.1, 1.1))
    for r in out.itertuples():
        p = Point(r.bias, r.loss)
        assert geoms[r.region].buffer(1e-6).covers(p), (r.region, r.bias, r.loss)
