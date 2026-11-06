"""Tests for callmefair.audit (bias audit of external predictions)."""
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from callmefair import BMInterface, BMMetrics, audit_file, audit_predictions  # noqa: E402
from callmefair.audit import BiasReport, main  # noqa: E402
from callmefair.viz import plot_fair_ranges, plot_group_rates, plot_metric_heatmap  # noqa: E402

STROKE = Path(__file__).resolve().parents[1] / "examples" / "data" / "stroke_pre" / "stroke.csv"


@pytest.fixture
def tiny():
    # privileged (g=1): y=[1,1,0,0] p=[1,1,1,0]   unprivileged (g=0): y=[1,1,0,0] p=[1,0,0,0]
    return pd.DataFrame({
        "g": [1] * 4 + [0] * 4, "grp": ["a"] * 4 + ["b"] * 4,
        "y": [1, 1, 0, 0] * 2, "p": [1, 1, 1, 0, 1, 0, 0, 0],
        "p2": [1, 1, 0, 0, 1, 1, 0, 0]})


def test_hand_computed_metrics(tiny):
    rep = audit_predictions(tiny, label="y", prediction="p", protected={"g": 1})
    m = rep.metrics("g")
    # selection: priv 3/4, unpriv 1/4; TPR: priv 1, unpriv 1/2; FPR: priv 1/2, unpriv 0
    assert m["spd"] == pytest.approx(-0.5)
    assert m["disparate_impact"] == pytest.approx(1 / 3)
    assert m["eq_opp_diff"] == pytest.approx(-0.5)
    assert m["avg_odd_diff"] == pytest.approx(0.5 * ((0 - 0.5) + (0.5 - 1)))
    assert m["acc"] == pytest.approx(6 / 8)
    assert not rep.is_fair("g")
    groups = rep.groups("g").set_index("group")
    assert groups.loc["privileged", "selection_rate"] == 0.75
    assert groups.loc["unprivileged", "TPR"] == 0.5
    assert "SPD -0.50 outside (-0.1, 0.1)" in rep.summary() and repr(rep) == rep.summary()


def test_fair_predictions(tiny):
    rep = audit_predictions(tiny, label="y", prediction="y", protected={"g": 1})
    assert rep.is_fair("g") and "fair" in rep.summary()
    assert rep.metrics()["fair_score"] < 0.05


def test_matches_bmmetrics_on_stroke():
    df = pd.read_csv(STROKE)
    bmi = BMInterface(df, "stroke", ["age"])
    X, y = bmi.get_train_xy()
    clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(X, y)
    pred_labels = clf.predict(bmi.get_test_xy()[0])
    classes = np.array([0.0, 1.0])
    bmm = BMMetrics(bmi, classes, clf.predict_proba(bmi.get_val_xy()[0]),
                    clf.predict_proba(bmi.get_test_xy()[0]), [{'age': 1}], [{'age': 0}])
    hard = bmi.get_test_BLD()  # hard predictions: labels and scores both equal the 0/1 labels
    hard.labels = hard.scores = pred_labels.reshape(-1, 1).astype(float)
    bmm.set_pos_pred(hard)
    expected = bmm.get_report()

    test = bmi.get_test_BLD()
    frame = pd.DataFrame({"age": test.protected_attributes.ravel(), "y": test.labels.ravel(),
                          "p": pred_labels})
    got = audit_predictions(frame, label="y", prediction="p", protected={"age": 1}).metrics("age")
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, abs=1e-12, nan_ok=True), key
    assert got["fair_score"] == bmm.get_score()["overall_score"]


def test_privileged_spec_forms(tiny):
    kw = dict(label="y", prediction="p")
    base = audit_predictions(tiny, protected={"g": 1}, **kw).metrics("g")
    for protected in ({"g": [1]}, {"g": {1}}, {"grp": "a"}, {"grp": ["a"]},
                      {"g": lambda s: s == 1}):
        attr = next(iter(protected))
        assert audit_predictions(tiny, protected=protected, **kw).metrics(attr) == base


def test_multiple_attributes_and_intersections(tiny):
    tiny["h"] = [1, 0] * 4
    rep = audit_predictions(tiny, label="y", prediction="p",
                            protected={"g": 1, "h": 1}, intersections=True)
    assert rep.attributes == ["g", "h", "g_h"]
    both = audit_predictions(tiny.assign(gh=tiny.g & tiny.h), label="y", prediction="p",
                             protected={"gh": 1}).metrics("gh")
    assert rep.metrics("g_h") == both
    assert len(rep.to_frame()) == 3 and {"attribute", "is_fair", "fair_score"} <= set(rep.to_frame())


def test_several_models_and_viz(tiny):
    rep = audit_predictions(tiny, label="y", prediction=["p", "p2"], protected={"g": 1})
    assert rep.models == ["p", "p2"]
    rows = rep.to_rows()
    assert [r["model"] for r in rows] == ["p", "p2"] and rows[0]["BM"] == "g"
    with pytest.raises(ValueError, match="several audits"):
        rep.metrics("g")
    assert rep.metrics("g", "p2")["spd"] == pytest.approx(0.0)
    assert isinstance(plot_fair_ranges(rows), Figure)
    assert isinstance(plot_metric_heatmap(rows, ), Figure)
    assert isinstance(plot_group_rates(rep, attribute="g", model="p"), Figure)
    assert "[p2] g" in rep.summary()


def test_sample_weight(tiny):
    unweighted = audit_predictions(tiny, label="y", prediction="p", protected={"g": 1})
    doubled = audit_predictions(tiny.assign(w=2.0), label="y", prediction="p",
                                protected={"g": 1}, sample_weight="w")
    assert doubled.metrics("g")["spd"] == pytest.approx(unweighted.metrics("g")["spd"])


def test_non_numeric_labels(tiny):
    text = tiny.assign(y=tiny.y.map({1: "yes", 0: "no"}), p=tiny.p.map({1: "yes", 0: "no"}))
    rep = audit_predictions(text, label="y", prediction="p", protected={"g": 1},
                            favorable_label="yes")
    assert rep.metrics("g")["spd"] == pytest.approx(-0.5)


def test_audit_file_roundtrip(tiny, tmp_path):
    path = tmp_path / "preds.csv"
    tiny.to_csv(path, index=False)
    assert (audit_file(path, label="y", prediction="p", protected={"g": 1}).metrics("g")
            == audit_predictions(tiny, label="y", prediction="p", protected={"g": 1}).metrics("g"))
    with pytest.raises(ValueError, match="unsupported file type"):
        audit_file(tmp_path / "preds.xlsx", label="y", prediction="p", protected={"g": 1})


@pytest.mark.parametrize("change, kwargs, message", [
    (lambda d: d, dict(label="nope"), "not found"),
    (lambda d: d.assign(y=d.y.astype(float).where(d.index != 0)), {}, "missing values"),
    (lambda d: d.assign(p=[0, 1, 2, 0, 1, 0, 1, 0]), {}, "must be binary"),
    (lambda d: d.assign(g=d.g.astype(float).where(d.index != 0)), {}, "missing values"),
    (lambda d: d, dict(protected={"g": 7}), "group is empty"),
    (lambda d: d, dict(favorable_label=5), "does not appear"),
    (lambda d: d, dict(protected={}), "at least one"),
    (lambda d: d, dict(intersections=True), "at least two"),
])
def test_validation_errors(tiny, change, kwargs, message):
    args = dict(label="y", prediction="p", protected={"g": 1}) | kwargs
    with pytest.raises(ValueError, match=message):
        audit_predictions(change(tiny), **args)


def test_group_without_positives_warns(tiny):
    tiny.loc[tiny.g == 0, "y"] = 0
    with pytest.warns(UserWarning, match="no positive"):
        rep = audit_predictions(tiny, label="y", prediction="p", protected={"g": 1})
    assert np.isnan(rep.metrics("g")["eq_opp_diff"]) and not rep.is_fair("g")
    assert "undefined" in rep.summary()


def test_cli(tiny, tmp_path, capsys):
    path = tmp_path / "preds.csv"
    tiny.to_csv(path, index=False)
    cmd = [sys.executable, "-m", "callmefair.audit", str(path), "--label", "y",
           "--prediction", "p", "--protected", "g=1", "--protected", "grp=a"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    assert "g: biased" in out and "grp: biased" in out and "fair_score" in out
    with pytest.raises(SystemExit) as exit_info:  # in-process: importing aif360 is slow
        main(cmd[3:-1] + ["grp=zzz"])
    assert exit_info.value.code == 2
    assert "group is empty" in capsys.readouterr().err


def test_import_callmefair_does_not_import_aif360():
    code = "import sys, callmefair; sys.exit('aif360' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_lazy_exports_are_the_audit_objects():
    import callmefair
    assert callmefair.BiasReport is BiasReport and callmefair.audit_file is audit_file
