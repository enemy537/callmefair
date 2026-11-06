"""Tests for the refactored search package (stroke dataset, small iterate)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from aif360.metrics import ClassificationMetric

sys.path.insert(0, str(Path(__file__).parent))

from callmefair.search import _evaluation, _search_base, model_factory
from callmefair.search._evaluation import best_threshold, split_dataset
from callmefair.search._search_base import BaseSearch, CType, combine_attributes
from callmefair.search.fair_search import BiasSearch
from callmefair.util import device

STROKE = Path(__file__).resolve().parents[1] / "examples/data/stroke_pre/stroke.csv"
ATTRIBUTES = ["age", "gender", "ever_married", "Residence_type"]


@pytest.fixture(scope="module")
def stroke():
    return pd.read_csv(STROKE)


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    from _reference import load_original
    try:
        return load_original(tmp_path_factory.mktemp("original"))
    except Exception as exc:  # no git history available
        pytest.skip(f"original implementation unavailable: {exc}")


# ---------------------------------------------------------------- equivalence

@pytest.mark.parametrize("model", ["lr", "xgb"])
def test_iterate_one_matches_original(stroke, original, model):
    base, _ = original
    expected = base.BaseSearch(stroke, "stroke").evaluate_attribute("gender", False, 1, model)
    actual = BaseSearch(stroke, "stroke").evaluate_attribute("gender", False, 1, model)
    assert set(actual) == set(expected) == {"gender_raw", "gender_overall", "gender_balanced_acc"}
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], rel=1e-9, abs=1e-12)


def test_best_threshold_matches_per_threshold_metrics(stroke):
    _, df_val, _ = split_dataset(stroke, "stroke", "gender", seed=1)
    val = _evaluation._to_bld(df_val, "stroke", "gender")
    scores = np.random.default_rng(0).random(len(df_val))

    balanced = []
    for threshold in _evaluation.THRESHOLDS:
        pred = val.copy(deepcopy=True)
        pred.scores = scores.reshape(-1, 1)
        pred.labels[pred.scores > threshold] = 1
        pred.labels[~(pred.scores > threshold)] = 0
        metric = ClassificationMetric(val, pred, unprivileged_groups=[{"gender": 0}],
                                      privileged_groups=[{"gender": 1}])
        balanced.append(0.5 * (metric.true_positive_rate() + metric.true_negative_rate()))
    expected = _evaluation.THRESHOLDS[int(np.where(balanced == np.max(balanced))[0][0])]
    assert best_threshold(val, scores) == expected


# --------------------------------------------------------------------- seeds

def test_iterations_use_consecutive_seeds(monkeypatch, stroke):
    seeds = []

    def fake(df, label, attribute, nearmiss, model, seed, n_jobs):
        seeds.append(seed)
        return {"raw_score": 1.0, "overall_score": 1.0, "balanced_accuracy": 1.0}

    monkeypatch.setattr(_search_base, "run_iteration", fake)
    BaseSearch(stroke, "stroke", n_threads=1).evaluate_attribute("gender", iterate=3)
    BaseSearch(stroke, "stroke", n_threads=1, random_state=7).evaluate_attribute("gender", iterate=3)
    assert seeds == [42, 43, 44, 7, 8, 9]


def test_different_seeds_give_different_splits_and_scores(stroke):
    a, _, _ = split_dataset(stroke, "stroke", "gender", seed=42)
    b, _, _ = split_dataset(stroke, "stroke", "gender", seed=43)
    assert not a.index.equals(b.index)
    s = BaseSearch(stroke, "stroke", n_threads=1)
    one = s.evaluate_attribute("gender", iterate=1, model_name="lr")
    three = s.evaluate_attribute("gender", iterate=3, model_name="lr")
    assert one["gender_raw"] != three["gender_raw"]


def test_split_dataset_falls_back_when_group_too_small(stroke):
    df = stroke.copy()
    df["tiny"] = 0
    df.loc[df.index[0], "tiny"] = 1   # a single positive in the group
    train, val, test = split_dataset(df, "stroke", "tiny", seed=0)
    assert len(train) + len(val) + len(test) == len(df)


# ------------------------------------------------------- combination scoring

def test_combination_average_does_not_divide_by_iterate(monkeypatch, stroke):
    constant = {"gender_ever_married_raw": 2.0, "gender_ever_married_overall": 0.5,
                "gender_ever_married_balanced_acc": 0.6}
    search = BiasSearch(stroke, "stroke", ATTRIBUTES, n_threads=1)
    monkeypatch.setattr(search, "evaluate_attribute", lambda *a, **k: constant)
    table, _ = search.evaluate_combination_average("gender", "ever_married", iterate=10)
    assert len(table) == 1 + len(CType)
    assert all(row[2] == 2.0 and row[3] == 0.5 for row in table[1:])


def test_evaluate_average_and_combinations_tables(stroke):
    search = BiasSearch(stroke, "stroke", ATTRIBUTES, n_threads=1)
    table, printable = search.evaluate_average(iterate=1)
    assert table[0][:2] == ["Attribute", "Raw Fairness Score"]
    assert [row[0] for row in table[1:]] == ATTRIBUTES
    assert "gender" in str(printable)
    table, _ = search.evaluate_combinations(iterate=1, model_name="lr")
    names = {row[0] for row in table[1:]}
    assert names <= {"age_gender", "age_ever_married", "age_Residence_type", "gender_ever_married",
                     "gender_Residence_type", "ever_married_Residence_type",
                     "age_gender_ever_married", "age_gender_Residence_type",
                     "age_ever_married_Residence_type", "gender_ever_married_Residence_type"}
    assert "gender_ever_married" in names


@pytest.mark.parametrize("operation, expected", [
    (CType.union, [0, 1, 1, 1]),
    (CType.intersection, [0, 0, 0, 1]),
    (CType.difference_1_minus_2, [0, 0, 1, 0]),
    (CType.difference_2_minus_1, [0, 1, 0, 0]),
    (CType.symmetric_difference, [0, 1, 1, 0]),
])
def test_combine_attributes_truth_table(operation, expected):
    df = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1], "x": [5, 6, 7, 8]})
    out = combine_attributes(df, "a", "b", operation)
    assert list(out.columns) == ["x", "a_b"]
    assert out["a_b"].tolist() == expected
    assert list(df.columns) == ["a", "b", "x"]  # input untouched


def test_combine_attributes_rejects_non_binary():
    with pytest.raises(ValueError):
        combine_attributes(pd.DataFrame({"a": [0, 2], "b": [0, 1]}), "a", "b", CType.union)


# ------------------------------------------------------------------ backends

def test_process_thread_and_sequential_backends_agree(stroke):
    results = {}
    for name, backend, workers in [("seq", "process", 1), ("proc", "process", 2),
                                   ("thread", "thread", 2)]:
        search = BaseSearch(stroke, "stroke", n_threads=workers, backend=backend)
        results[name] = search.evaluate_attribute("gender", iterate=3, model_name="lr")
        search.optimizer.shutdown()
    for name in ("proc", "thread"):
        for key, value in results["seq"].items():
            assert results[name][key] == pytest.approx(value, rel=1e-9)


def test_parallel_workers_fit_with_one_thread(monkeypatch, stroke):
    jobs = []

    def fake_fit(name, x, y, w, n_jobs=1):
        jobs.append(n_jobs)
        return lambda features: np.full(len(features), 0.5)

    monkeypatch.setattr(_evaluation, "fit_model", fake_fit)
    BaseSearch(stroke, "stroke", n_threads=2, backend="thread").evaluate_attribute(
        "gender", iterate=3, model_name="lr")
    assert jobs == [1, 1, 1]


# ------------------------------------------------------------ model registry

def test_registry_contains_all_models():
    assert set(model_factory.MODEL_BUILDERS) == {"lr", "cat", "xgb", "mlp", "lgbm",
                                                 "tabtransformer", "tabnet"}
    for name in model_factory.MODEL_BUILDERS:
        assert isinstance(model_factory.uses_gpu(name), bool)


def test_custom_model_registration(monkeypatch):
    """Regression: a model registered without a GPU entry raised KeyError in uses_gpu."""
    monkeypatch.setattr(model_factory, "MODEL_BUILDERS", dict(model_factory.MODEL_BUILDERS))
    monkeypatch.setattr(model_factory, "_USES_GPU", dict(model_factory._USES_GPU))

    @model_factory.register_model("rf_test")
    def _build(x, y, w, n_jobs):
        from sklearn.ensemble import RandomForestClassifier
        return model_factory.positive_proba(RandomForestClassifier(n_estimators=5).fit(x, y))

    @model_factory.register_model("gpu_test", uses_gpu=lambda: True)
    def _build_gpu(x, y, w, n_jobs):
        raise NotImplementedError

    assert model_factory.uses_gpu("rf_test") is False
    assert model_factory.uses_gpu("gpu_test") is True
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(60, 3)), rng.integers(0, 2, 60)
    proba = model_factory.fit_model("rf_test", x, y, np.ones(60))(x)
    assert proba.shape == (60,)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        model_factory.fit_model("nope", np.zeros((4, 2)), np.array([0, 1, 0, 1]), np.ones(4))


def test_register_model_extends_registry(monkeypatch):
    monkeypatch.setitem(model_factory.MODEL_BUILDERS, "const", lambda x, y, w, n: (lambda f: np.ones(len(f))))
    predict = model_factory.fit_model("const", np.zeros((2, 1)), np.array([0, 1]), np.ones(2))
    assert predict(np.zeros((3, 1))).tolist() == [1, 1, 1]


@pytest.mark.parametrize("model", ["lr", "xgb", "cat", "lgbm"])
def test_models_return_probabilities(stroke, model):
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(200, 3)), rng.integers(0, 2, 200)
    proba = model_factory.fit_model(model, x, y, np.ones(200), n_jobs=1)(x)
    assert proba.shape == (200,) and ((proba >= 0) & (proba <= 1)).all()


# ------------------------------------------------------------ device params

@pytest.mark.parametrize("detected, xgb_device, cat_gpu, tabnet", [
    (device.CUDA, "cuda", True, "cuda"),
    (device.MPS, "cpu", False, model_factory.MPS_TABNET_DEVICE),
    (device.CPU, "cpu", False, "cpu"),
])
def test_device_drives_model_settings(monkeypatch, detected, xgb_device, cat_gpu, tabnet):
    monkeypatch.setattr(device, "detect", lambda: detected)
    assert device.xgboost_params()["device"] == xgb_device
    assert bool(device.catboost_params()) is cat_gpu
    assert model_factory.tabnet_device() == tabnet
    assert model_factory.uses_gpu("xgb") is (detected == device.CUDA)
    assert model_factory.uses_gpu("cat") is cat_gpu
    assert model_factory.uses_gpu("lr") is False


def test_default_workers_use_performance_cores(monkeypatch, stroke):
    monkeypatch.setattr(device, "performance_cores", lambda: 6)
    assert BaseSearch(stroke, "stroke").n_threads == 6
    monkeypatch.setattr(device, "performance_cores", lambda: 32)
    assert BaseSearch(stroke, "stroke").n_threads == 8


def test_progress_false_disables_tqdm(monkeypatch, stroke):
    flags = []
    real_tqdm = _search_base.tqdm
    monkeypatch.setattr(_search_base, "tqdm",
                        lambda it, **kw: flags.append(kw.get("disable")) or real_tqdm(it, **kw))
    for progress in (False, True):
        search = BiasSearch(stroke, "stroke", ATTRIBUTES, n_threads=1, progress=progress)
        search.evaluate_attribute("gender", iterate=1)
    assert flags == [True, False]
