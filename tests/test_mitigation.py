"""Tests for callmefair.mitigation (grid search, model adapters, CSV logging).

Reference values for the unchanged code paths come from the ORIGINAL
implementation (``git archive`` of the commit that precedes the refactor),
executed in a subprocess so both ``callmefair`` packages never share a process.
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STROKE = ROOT / "examples" / "data" / "stroke_pre" / "stroke.csv"
ORIGINAL_REV = "e2490ef"  # "Original snapshot (pre-baseline)": untouched mitigation code
NUMERIC = ["balanced_acc", "acc", "precision", "recall", "f1", "mcc", "eq_opp_diff",
           "avg_odd_diff", "spd", "disparate_impact", "theil_idx", "fair_score"]

# Executed in a subprocess against either source tree. argv: model, [bm names...]
RUNNER = textwrap.dedent("""
    import inspect, json, sys, warnings
    warnings.filterwarnings('ignore')
    import numpy as np, pandas as pd
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import ClassificationMetric
    from callmefair.util.fair_util import BMInterface, BMMetrics
    from callmefair.mitigation.fair_bm import BMType
    from callmefair.mitigation.fair_grid import BMGridSearch

    np.random.seed(0)
    model_name, sets = sys.argv[1], sys.argv[2:]
    df = pd.read_csv(%(csv)r)
    bmI = BMInterface(df, 'stroke', ['age'])
    model = None
    if model_name == 'LR':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
    bm_list = [[BMType[n] for n in s.split('+')] for s in sets]
    grid = BMGridSearch(bmI, model, bm_list, [{'age': 1}], [{'age': 0}])
    kwargs = {'results_dir': %(out)r} if 'results_dir' in inspect.signature(grid.run_single_sensitive).parameters else {}
    rows = grid.run_single_sensitive(**kwargs)
    if rows is None:
        # original implementation: read back the CSV it wrote (cwd/results)
        import glob
        (path,) = glob.glob('results/experiment_*.csv')
        rows = pd.read_csv(path, index_col=0).to_dict('records')
    print('JSON' + json.dumps(rows, default=float))
""")


def run_rows(tree: Path, cwd: Path, model: str, *sets: str) -> list[dict]:
    code = RUNNER % {"csv": str(STROKE), "out": str(cwd / "results")}
    env = dict(os.environ, PYTHONPATH=str(tree), PYTHONWARNINGS="ignore")
    proc = subprocess.run([sys.executable, "-c", code, model, *sets], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("JSON"))
    return json.loads(line[4:])


@pytest.fixture(scope="module")
def original_tree(tmp_path_factory) -> Path:
    tree = tmp_path_factory.mktemp("original")
    archive = subprocess.run(["git", "archive", ORIGINAL_REV, "callmefair"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
    return tree


# --------------------------------------------------------------------------- grid search

PRE_SETS = ["preReweighing", "preLFR"]
if importlib.util.find_spec("BlackBoxAuditing"):  # optional dependency of aif360's DIR
    PRE_SETS.append("preDisparate")
POST_SETS = ["posEqqOds", "posROC", "posCalibrated"]
CHAINED = "posEqqOds+posROC"


@pytest.fixture(scope="module")
def current(tmp_path_factory):
    """One subprocess run of the refactored code covering all non-in-processing cases."""
    cwd = tmp_path_factory.mktemp("current")
    rows = run_rows(ROOT, cwd, "LR", *PRE_SETS, *POST_SETS, CHAINED)
    return rows, cwd


def test_non_post_rows_match_original(original_tree, current, tmp_path):
    rows, _ = current
    expected = run_rows(original_tree, tmp_path, "LR", *PRE_SETS)
    assert [r["BM"] for r in rows[:len(expected)]] == ["baseline", *PRE_SETS]
    for old, new in zip(expected, rows):
        assert old["model"] == new["model"] == "LogisticRegression"
        assert old["BM"] == new["BM"]
        if new["BM"] == "preLFR":   # intentional change: LFR keeps the true training labels
            continue
        for col in NUMERIC:
            expected_val = old[col]
            # Intentional change: F1/MCC are 0.0 (not nan) when their denominator is 0.
            if col in ("f1", "mcc") and np.isnan(expected_val):
                expected_val = 0.0
            assert new[col] == pytest.approx(expected_val, rel=1e-9, abs=1e-12, nan_ok=True), (new["BM"], col)


def test_post_processing_rows_differ_from_baseline(current):
    rows, _ = current
    by_name = {r["BM"]: r for r in rows}
    for name in (*POST_SETS, CHAINED.replace("+", " ")):
        assert any(by_name[name][c] != pytest.approx(rows[0][c], nan_ok=True) for c in NUMERIC), name


def test_results_csv_written_with_one_header(current):
    rows, cwd = current
    (csv_file,) = (cwd / "results").glob("experiment_*.csv")
    lines = csv_file.read_text().splitlines()
    assert lines[0].split(",") == ["model", "BM", *NUMERIC, "fair_score_uncapped"]
    assert len(lines) == len(rows) + 1


# ----------------------------------------------------------------- in-processing (stubbed)

def _stub_bmi_grid(monkeypatch, bm_list):
    """Grid in in-processing mode whose in_Meta is replaced by a recording stub."""
    from callmefair.mitigation.fair_grid import BMGridSearch
    from callmefair.util.fair_util import BMInterface

    df = pd.read_csv(STROKE)
    grid = BMGridSearch(BMInterface(df, "stroke", ["age"]), None, bm_list,
                        [{"age": 1}], [{"age": 0}])
    taus = []

    class StubMeta:
        def fit(self, bld):
            self.x = bld
            return self

        def predict(self, bld):
            out = bld.copy(deepcopy=True)
            rng = np.random.default_rng(1)
            out.scores = np.clip(0.3 + 0.2 * out.features[:, [1]] + rng.normal(0, 0.2, (len(out.labels), 1)), 0, 1)
            out.labels = (out.scores > 0.5).astype(float)
            return out

    def fake_in_meta(sensitive_attribute, tau=0):
        taus.append(tau)
        return StubMeta()

    monkeypatch.setattr(grid.bmMR, "in_Meta", fake_in_meta)
    return grid, taus


def test_in_meta_naming_and_model_reset(monkeypatch, tmp_path):
    from callmefair.mitigation.fair_bm import BMType
    grid, taus = _stub_bmi_grid(monkeypatch, [[BMType.inMeta], [BMType.preReweighing]])
    rows = grid.run_single_sensitive(results_dir=str(tmp_path))
    assert [r["BM"] for r in rows] == ["baseline", "inMeta", "preReweighing"]
    # initial (baseline) model, mitigated model, then reset to the unmitigated one
    assert taus == [0, 0.7, 0]


def test_in_processing_with_post_processing_runs(monkeypatch, tmp_path):
    from callmefair.mitigation.fair_bm import BMType
    grid, _ = _stub_bmi_grid(monkeypatch, [[BMType.inMeta, BMType.posEqqOds]])
    rows = grid.run_single_sensitive(results_dir=str(tmp_path))
    assert rows[1]["BM"] == "inMeta posEqqOds"


def test_results_dir_none_writes_nothing(monkeypatch, tmp_path):
    from callmefair.mitigation.fair_bm import BMType
    monkeypatch.chdir(tmp_path)
    grid, _ = _stub_bmi_grid(monkeypatch, [[BMType.inMeta]])
    rows = grid.run_single_sensitive(results_dir=None)
    assert len(rows) == 2 and list(tmp_path.iterdir()) == []


def test_in_processing_with_classifier_is_rejected(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from callmefair.mitigation.fair_bm import BMType
    from callmefair.mitigation.fair_grid import BMGridSearch
    from callmefair.util.fair_util import BMInterface

    bmI = BMInterface(pd.read_csv(STROKE), "stroke", ["age"])
    grid = BMGridSearch(bmI, LogisticRegression(), [[BMType.inMeta]], [{"age": 1}], [{"age": 0}])
    with pytest.raises(ValueError):
        grid.run_single_sensitive(results_dir=str(tmp_path))


@pytest.mark.slow
def test_adversarial_debiasing_in_subprocess(tmp_path):
    """TF1 graph mode is process-global (disable_eager_execution): isolate it."""
    rows = run_rows(ROOT, tmp_path, "NONE", "inAdversarial")
    assert [r["BM"] for r in rows] == ["baseline", "inAD"]


# ----------------------------------------------------------------------------- adapters

class _Bld:
    instance_weights = np.array([1.0, 2.0, 0.5, 1.5])


class _Bmi:
    x = np.arange(8, dtype=float).reshape(4, 2)
    y = np.array([0.0, 1.0, 0.0, 1.0])

    def get_train_xy(self): return self.x, self.y
    def get_val_xy(self): return self.x[:2], self.y[:2]
    def get_test_xy(self): return self.x[2:], self.y[2:]
    def get_train_BLD(self): return _Bld()


class _Proba:
    def predict_proba(self, x): return np.tile([0.4, 0.6], (len(x), 1))


class WithWeights(_Proba):
    def fit(self, x, y, sample_weight=None):
        self.weight = sample_weight
        return self


class WithoutWeights(_Proba):
    def fit(self, x, y):
        self.weight = "none"
        return self


def test_resolve_adapter_dispatch():
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from callmefair.mitigation import model_adapters as ma

    assert isinstance(ma.resolve_adapter(LGBMClassifier()), ma.LightGBMAdapter)
    for model in (LogisticRegression(), MLPClassifier(), XGBClassifier()):
        assert isinstance(ma.resolve_adapter(model), ma.SklearnAdapter)
    from catboost import CatBoostClassifier
    assert isinstance(ma.resolve_adapter(CatBoostClassifier(verbose=0)), ma.SklearnAdapter)

    TabNetClassifier = type("TabNetClassifier", (_Proba,), {})
    TabTransformer = type("TabTransformer", (), {})
    assert isinstance(ma.resolve_adapter(TabNetClassifier()), ma.TabNetAdapter)
    assert isinstance(ma.resolve_adapter(type("MyTabNet", (TabNetClassifier,), {})()), ma.TabNetAdapter)
    assert isinstance(ma.resolve_adapter(TabTransformer()), ma.TabTransformerAdapter)

    with pytest.raises(ValueError):
        ma.resolve_adapter(object())


def test_register_adapter_extends_registry():
    from callmefair.mitigation import model_adapters as ma

    class Custom:
        def fit_predict_proba(self, model, bmI): return "custom"

    Mine = type("MineEstimator", (), {})
    ma.register_adapter(Mine, Custom())
    try:
        assert ma.resolve_adapter(Mine()).fit_predict_proba(None, None) == "custom"
    finally:
        ma._REGISTRY.pop("MineEstimator")


def test_sample_weight_only_passed_when_supported():
    from callmefair.mitigation.fair_grid import get_model_proba

    with_w, without_w = WithWeights(), WithoutWeights()
    get_model_proba(with_w, _Bmi())
    get_model_proba(without_w, _Bmi())
    np.testing.assert_array_equal(with_w.weight, _Bld.instance_weights)
    assert without_w.weight == "none"


def test_reweighing_reaches_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    from callmefair.mitigation.fair_grid import get_model_proba

    seen = {}
    class Spy(RandomForestClassifier):
        def fit(self, X, y, sample_weight=None):
            seen["w"] = sample_weight
            return super().fit(X, y, sample_weight=sample_weight)

    val, test = get_model_proba(Spy(n_estimators=3), _Bmi())
    np.testing.assert_array_equal(seen["w"], _Bld.instance_weights)
    assert val.shape == (2, 2) and test.shape == (2, 2)

def run_isolated(code: str, cwd: Path) -> None:
    """Run ``code`` in a clean interpreter (LightGBM and multiprocessing clash with TF/torch in-process)."""
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONWARNINGS="ignore")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]


def test_lightgbm_adapter_uses_weights(tmp_path):
    run_isolated(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        from test_mitigation import _Bmi
        from lightgbm import LGBMClassifier
        from callmefair.mitigation.model_adapters import resolve_adapter
        model = LGBMClassifier(n_estimators=2, min_child_samples=1, verbose=-1)
        val, test = resolve_adapter(model).fit_predict_proba(model, _Bmi())
        assert val.shape == (2, 2) and test.shape == (2, 2)
    """, tmp_path)


def test_tabnet_device_default_only_when_unset(monkeypatch):
    from callmefair.mitigation import model_adapters as ma
    from callmefair.util import device

    monkeypatch.setattr(device, "tabnet_device", lambda: "cpu")

    class TabNetClassifier(_Proba):
        def __init__(self, device_name):
            self.device_name = device_name

        def fit(self, x, y, eval_set=None):
            self.eval_set = eval_set

    auto, explicit = TabNetClassifier("auto"), TabNetClassifier("cuda:1")
    ma.TabNetAdapter().fit_predict_proba(auto, _Bmi())
    ma.TabNetAdapter().fit_predict_proba(explicit, _Bmi())
    assert auto.device_name == "cpu" and auto.device.type == "cpu"
    assert explicit.device_name == "cuda:1" and not hasattr(explicit, "device")
    assert len(auto.eval_set) == 1


def test_tabnet_auto_device_never_mps(monkeypatch):
    """Regression: 'auto' TabNet was sent to MPS, which crashes on float64 features."""
    from callmefair.util import device
    monkeypatch.setattr(device, "detect", lambda: device.MPS)
    assert device.tabnet_device() == device.CPU


def test_tab_transformer_adapter_builds_target_frame_and_2d_proba():
    from callmefair.mitigation import model_adapters as ma

    class TabTransformer:
        target = "label"

        def fit(self, df):
            self.train_cols = list(df.columns)

        def predict_proba(self, df):
            assert "label" in df.columns
            return np.full(len(df), 0.7)

    model = TabTransformer()
    val, test = ma.TabTransformerAdapter().fit_predict_proba(model, _Bmi())
    assert model.train_cols[-1] == "label"
    np.testing.assert_allclose(val, np.tile([0.3, 0.7], (2, 1)))
    assert test.shape == (2, 2)


# ---------------------------------------------------------------------------------- logging

def test_csv_logger_single_header_and_no_index(tmp_path):
    from callmefair.mitigation.fair_log import csvLogger

    logger = csvLogger("exp", path=str(tmp_path))
    logger([{"model": "a", "acc": 1.0}])
    logger([{"model": "b", "acc": 2.0}, {"model": "c", "acc": 3.0}])
    lines = (tmp_path / "exp.csv").read_text().splitlines()
    assert lines[0] == "model,acc"
    assert sum(l.startswith("model") for l in lines) == 1
    assert len(lines) == 4
    assert len(pd.read_csv(tmp_path / "exp.csv")) == 3


def test_aggregate_csv_files_delete_sources_flag(tmp_path):
    for name in ("a", "b"):
        pd.DataFrame({"x": [1]}).to_csv(tmp_path / f"{name}.csv", index=False)
    (tmp_path / "agg").mkdir()
    run_isolated(f"""
        from callmefair.mitigation.fair_log import aggregate_csv_files
        aggregate_csv_files({str(tmp_path)!r}, {str(tmp_path / "agg" / "keep.csv")!r},
                            num_processes=1, delete_sources=False)
    """, tmp_path)
    assert (tmp_path / "a.csv").exists() and len(pd.read_csv(tmp_path / "agg" / "keep.csv")) == 2

    run_isolated(f"""
        from callmefair.mitigation.fair_log import aggregate_csv_files
        aggregate_csv_files({str(tmp_path)!r}, {str(tmp_path / "agg" / "out.csv")!r}, num_processes=1)
    """, tmp_path)
    assert not (tmp_path / "a.csv").exists() and not (tmp_path / "b.csv").exists()


def test_pre_dr_keeps_matplotlib_backend(tmp_path):
    """Regression: BlackBoxAuditing (used by preDisparate) forced the Agg backend,
    so notebook plots silently stopped rendering after preDisparate."""
    code = textwrap.dedent(f"""
        import matplotlib
        matplotlib.use("svg")                       # any non-Agg backend
        import pandas as pd
        from callmefair import BMInterface, BMManager
        df = pd.read_csv({str(STROKE)!r})
        bmI = BMInterface(df, "stroke", ["age"])
        BMManager(bmI, [{{"age": 1}}], [{{"age": 0}}]).pre_DR("age")
        assert matplotlib.get_backend() == "svg", matplotlib.get_backend()
    """)
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, timeout=300)
    assert proc.returncode == 0, proc.stderr.decode()[-1500:]


def test_lfr_keeps_true_training_labels():
    """Regression: aif360's LFR overwrote training labels with its own predictions, which
    collapse to one class on rare outcomes (stroke) and make every model fail to fit."""
    import numpy as np
    from callmefair import BMInterface, BMManager
    df = pd.read_csv(STROKE)
    bmI = BMInterface(df, "stroke", ["age"], random_state=0, test_size=0.2, val_size=0.2)
    before = bmI.get_train_BLD().labels.copy()
    BMManager(bmI, [{"age": 1}], [{"age": 0}]).pre_LFR()
    after = bmI.get_train_BLD()
    assert np.array_equal(after.labels, before) and len(np.unique(after.labels)) == 2
