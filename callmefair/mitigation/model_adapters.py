"""Model adapters: one small class per model family, resolved through a registry.

``get_model_proba`` used to dispatch on ``str(model).startswith(...)``. Adapters
replace that with an Open/Closed design: supporting a new library means writing
an adapter and calling :func:`register_adapter`; nothing else changes.

Resolution walks the model's MRO and matches class *names*, so subclasses are
covered and heavy libraries (lightgbm, pytorch-tabnet, torch) are never
imported just to dispatch. Anything with ``fit``/``predict_proba`` falls back
to :class:`SklearnAdapter`.

Behaviour note: instance weights (``preReweighing``) are passed to every
estimator whose ``fit`` accepts ``sample_weight`` (LogisticRegression, random
forests, XGBoost, CatBoost, LightGBM, ...). Previously only LogisticRegression
and LightGBM received them.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.utils.validation import has_fit_parameter

from callmefair.util import device


class ModelAdapter(Protocol):
    """Trains ``model`` on ``bmI``'s train split and returns P(class) matrices."""

    def fit_predict_proba(self, model, bmI) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(val_proba, test_proba)``, each of shape (n_samples, n_classes)."""
        ...


def _as_2d_proba(proba: np.ndarray) -> np.ndarray:
    """Expand a 1-D positive-class vector to [P(neg), P(pos)] columns."""
    proba = np.asarray(proba)
    return np.column_stack((1.0 - proba, proba)) if proba.ndim == 1 else proba


def _feature_frame(x: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(x, columns=[f"feature_{i}" for i in range(x.shape[1])])


class SklearnAdapter:
    """Generic estimator: ``fit(X, y[, sample_weight])`` then ``predict_proba``."""

    def fit_predict_proba(self, model, bmI):
        x_train, y_train = bmI.get_train_xy()
        x_val, _ = bmI.get_val_xy()
        x_test, _ = bmI.get_test_xy()
        fit_kwargs = {}
        if has_fit_parameter(model, "sample_weight"):
            fit_kwargs["sample_weight"] = bmI.get_train_BLD().instance_weights
        model.fit(x_train, y_train, **fit_kwargs)
        return model.predict_proba(x_val), model.predict_proba(x_test)


class LightGBMAdapter:
    """LightGBM: DataFrames avoid the feature-names warning; weights + eval_set."""

    def fit_predict_proba(self, model, bmI):
        x_train, y_train = bmI.get_train_xy()
        x_val, y_val = bmI.get_val_xy()
        x_test, _ = bmI.get_test_xy()
        train_df, val_df, test_df = map(_feature_frame, (x_train, x_val, x_test))
        model.fit(train_df, y_train,
                  sample_weight=bmI.get_train_BLD().instance_weights,
                  eval_set=[(val_df, y_val)],
                  callbacks=[])
        return model.predict_proba(val_df), model.predict_proba(test_df)


class TabNetAdapter:
    """TabNet: eval_set on validation data; device default from the device module.

    The device is only chosen when the user left TabNet's ``device_name`` at
    ``'auto'``; it follows ``device.tabnet_device()`` (CUDA if present, else CPU).
    """

    def fit_predict_proba(self, model, bmI):
        x_train, y_train = bmI.get_train_xy()
        x_val, y_val = bmI.get_val_xy()
        x_test, _ = bmI.get_test_xy()
        self._apply_default_device(model)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
        return model.predict_proba(x_val), model.predict_proba(x_test)

    @staticmethod
    def _apply_default_device(model) -> None:
        if getattr(model, "device_name", None) != "auto":
            return
        torch = device.import_torch()  # lazy; macOS-safe OpenMP load order
        name = device.tabnet_device()
        model.device_name = name
        model.device = torch.device(name)


class TabTransformerAdapter:
    """callmefair TabTransformer: DataFrame in, includes the target column.

    Relies only on ``fit(df)``, ``predict_proba(df)`` and ``target``.
    """

    def fit_predict_proba(self, model, bmI):
        x_train, y_train = bmI.get_train_xy()
        x_val, _ = bmI.get_val_xy()
        x_test, _ = bmI.get_test_xy()
        target = getattr(model, "target", "target")

        train_df = _feature_frame(x_train)
        train_df[target] = y_train
        val_df, test_df = _feature_frame(x_val), _feature_frame(x_test)
        val_df[target] = 0   # ignored at prediction; required by the interface
        test_df[target] = 0

        model.fit(train_df)
        return (_as_2d_proba(model.predict_proba(val_df)),
                _as_2d_proba(model.predict_proba(test_df)))


_FALLBACK: ModelAdapter = SklearnAdapter()
_REGISTRY: dict[str, ModelAdapter] = {
    "LGBMClassifier": LightGBMAdapter(),
    "TabNetClassifier": TabNetAdapter(),
    "TabTransformer": TabTransformerAdapter(),
}


def register_adapter(model_class: type | str, adapter: ModelAdapter) -> None:
    """Register ``adapter`` for a class (or class name); matches subclasses too."""
    name = model_class if isinstance(model_class, str) else model_class.__name__
    _REGISTRY[name] = adapter


def resolve_adapter(model) -> ModelAdapter:
    """Find the adapter for ``model``; raise ValueError if it cannot predict probabilities."""
    for cls in type(model).__mro__:
        if cls.__name__ in _REGISTRY:
            return _REGISTRY[cls.__name__]
    if hasattr(model, "fit") and hasattr(model, "predict_proba"):
        return _FALLBACK
    raise ValueError(f"Unsupported model type: {type(model).__name__}")
