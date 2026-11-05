"""Execution-mode dispatch: GPU-backed models run sequentially, CPU models in parallel."""

import numpy as np
import pandas as pd
import pytest

from callmefair.search import _search_base
from callmefair.search._search_base import BaseSearch
from callmefair.util import device


@pytest.fixture
def searcher():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f": rng.random(100), "sensitive": rng.integers(0, 2, 100),
                       "target": rng.integers(0, 2, 100)})
    return BaseSearch(df, "target", n_threads=4, backend="thread")


@pytest.fixture
def calls(monkeypatch, searcher):
    """Replace the iteration with a stub; record whether the optimizer was used."""
    log = {"direct": 0, "mapped": 0}
    score = {"raw_score": 1.0, "overall_score": 0.5, "balanced_accuracy": 0.7}

    def fake_iteration(*args):
        log["direct"] += 1
        return score

    def fake_map(func, args_list):
        log["mapped"] += len(list(args_list))
        return iter([score] * log["mapped"])

    monkeypatch.setattr(_search_base, "run_iteration", fake_iteration)
    monkeypatch.setattr(searcher.optimizer, "map", fake_map)
    return log


@pytest.mark.parametrize("model", ["xgb", "cat"])
def test_cuda_runs_gpu_models_sequentially(monkeypatch, searcher, calls, model):
    monkeypatch.setattr(device, "detect", lambda: device.CUDA)
    searcher.evaluate_attribute("sensitive", iterate=4, model_name=model)
    assert calls == {"direct": 4, "mapped": 0}


@pytest.mark.parametrize("model", ["lr", "xgb", "cat", "lgbm"])
@pytest.mark.parametrize("detected", [device.CPU, device.MPS])
def test_cpu_and_mps_run_tree_models_in_parallel(monkeypatch, searcher, calls, model, detected):
    monkeypatch.setattr(device, "detect", lambda: detected)
    searcher.evaluate_attribute("sensitive", iterate=4, model_name=model)
    assert calls == {"direct": 0, "mapped": 4}


def test_single_iteration_or_single_worker_is_sequential(monkeypatch, searcher, calls):
    monkeypatch.setattr(device, "detect", lambda: device.CPU)
    searcher.evaluate_attribute("sensitive", iterate=1, model_name="lr")
    searcher.n_threads = 1
    searcher.evaluate_attribute("sensitive", iterate=3, model_name="lr")
    assert calls == {"direct": 4, "mapped": 0}
