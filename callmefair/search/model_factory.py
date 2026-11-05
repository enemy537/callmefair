"""Model registry for the bias search.

Each builder fits a model on scaled features and returns a *predictor*: a
callable mapping a feature matrix to the 1-D probability of the favorable
class (label 1). To add a model, decorate a builder with ``@register_model``.

Heavy frameworks (tensorflow, torch, xgboost, ...) are imported inside the
builders so importing this module - and spawning process workers on macOS -
stays cheap. Device decisions come exclusively from ``callmefair.util.device``.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pandas as pd

from callmefair.util import device

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

Predictor = Callable[[np.ndarray], np.ndarray]
Builder = Callable[[np.ndarray, np.ndarray, np.ndarray, int], Predictor]

FAVORABLE_LABEL = 1
# TabNet device on Apple Silicon hosts, chosen by benchmark (see report).
MPS_TABNET_DEVICE = "cpu"

MODEL_BUILDERS: dict[str, Builder] = {}


def register_model(name: str) -> Callable[[Builder], Builder]:
    def decorator(builder: Builder) -> Builder:
        MODEL_BUILDERS[name] = builder
        return builder
    return decorator


def fit_model(name: str, x: np.ndarray, y: np.ndarray,
              sample_weight: np.ndarray, n_jobs: int = 1) -> Predictor:
    """Fit model ``name`` and return its positive-class probability predictor."""
    try:
        builder = MODEL_BUILDERS[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_BUILDERS)}") from None
    return builder(x, y, sample_weight, n_jobs)


def _positive_proba(model) -> Predictor:
    """Wrap an sklearn-style classifier as a positive-class predictor."""
    classes = list(model.classes_)
    if FAVORABLE_LABEL not in classes:
        raise ValueError(f"Training data has no favorable-label ({FAVORABLE_LABEL}) samples")
    pos = classes.index(FAVORABLE_LABEL)
    return lambda x: model.predict_proba(x)[:, pos]


@register_model("lr")
def _build_lr(x, y, w, n_jobs):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(solver="liblinear").fit(x, y, sample_weight=w)
    return _positive_proba(model)


@register_model("cat")
def _build_cat(x, y, w, n_jobs):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(eval_metric="Accuracy", depth=4, learning_rate=0.01,
                               iterations=10, thread_count=n_jobs, verbose=False,
                               **device.catboost_params())
    return _positive_proba(model.fit(x, y))


@register_model("xgb")
def _build_xgb(x, y, w, n_jobs):
    from xgboost import XGBClassifier
    model = XGBClassifier(max_depth=8, learning_rate=0.01, gamma=0.25, n_estimators=500,
                          subsample=0.8, colsample_bytree=0.3, n_jobs=n_jobs,
                          **device.xgboost_params())
    return _positive_proba(model.fit(x, y))


@register_model("lgbm")
def _build_lgbm(x, y, w, n_jobs):
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, objective="binary",
                           n_jobs=n_jobs, verbose=-1, **device.lightgbm_params())
    return _positive_proba(model.fit(x, y, sample_weight=w))


@register_model("mlp")
def _build_mlp(x, y, w, n_jobs):
    # Keras uses a GPU only if TensorFlow itself sees one (CUDA, or tensorflow-metal).
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential([
        layers.Input(shape=(x.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x, y, sample_weight=w, epochs=50, batch_size=1024, verbose=0)
    return lambda features: model.predict(features, verbose=0).ravel()


def tabnet_device() -> str:
    detected = device.torch_device()
    return MPS_TABNET_DEVICE if detected == device.MPS else detected


def _tensorflow_sees_gpu() -> bool:
    import tensorflow as tf
    return bool(tf.config.list_physical_devices("GPU"))


# Per-model test: will this model train on a GPU? Such models run one at a time
# (GPU memory/context contention); CPU-only models are spread over worker processes.
_USES_GPU: dict[str, Callable[[], bool]] = {
    "lr": lambda: False,
    "cat": lambda: bool(device.catboost_params()),
    "xgb": lambda: device.xgboost_params()["device"] == device.CUDA,
    "lgbm": lambda: bool(device.lightgbm_params()),
    "mlp": _tensorflow_sees_gpu,
    "tabtransformer": _tensorflow_sees_gpu,
    "tabnet": lambda: tabnet_device() != device.CPU,
}


def uses_gpu(name: str) -> bool:
    return _USES_GPU[name]()


@register_model("tabnet")
def _build_tabnet(x, y, w, n_jobs):
    torch = device.import_torch()
    from pytorch_tabnet.tab_model import TabNetClassifier
    torch.set_num_threads(n_jobs)
    model = TabNetClassifier(seed=42, verbose=0, device_name=tabnet_device())
    model.fit(x.astype(np.float32), y)  # MPS has no float64
    positive = _positive_proba(model)
    return lambda features: positive(features.astype(np.float32))


@register_model("tabtransformer")
def _build_tabtransformer(x, y, w, n_jobs):
    from callmefair.util.models import TabTransformer
    columns = [f"feature_{i}" for i in range(x.shape[1])]
    config = {"batch_size": 2048, "epochs": 20, "learning_rate": 1e-3, "verbose": 0}
    model = TabTransformer(config=config, target="target")
    model.fit(pd.DataFrame(x, columns=columns).assign(target=y))

    def predict(features: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(features, columns=columns).assign(target=0)
        return np.asarray(model.predict_proba(frame)).ravel()
    return predict
