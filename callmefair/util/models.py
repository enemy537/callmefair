"""TabTransformer classifier (Keras) for tabular data.

Numeric columns go through a ``Normalization`` layer; string/categorical
columns are embedded and passed through transformer blocks. Categoricals are
fed to the model as raw strings and indexed by an in-graph ``StringLookup``,
so ``fit`` and ``predict_proba`` share one input-building path.

Devices are not forced: TensorFlow uses whatever GPU it can see (CUDA on
Linux, ``tensorflow-metal`` on macOS if installed) and falls back to CPU.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers

_NUMERIC_INPUT = "numeric"
_NA_TOKEN = "<NA>"

DEFAULT_CONFIG = {
    "batch_size": 256,
    "epochs": 20,
    "learning_rate": 1e-3,
    "transformer_layers": 6,
    "transformer_heads": 12,
    "transformer_embedding_dim": 64,
    "ff_dim": 64,
    "dropout": 0.1,
    "mlp_units": [128, 64],
    "mlp_dropout": 0.2,
    "dense_activation": "gelu",
    "loss": "binary_crossentropy",  # switched to sparse_categorical_crossentropy if >2 classes
    "metrics": None,  # None -> AUC (binary) or accuracy (multiclass)
    "shuffle": True,
    "seed": 42,
    "verbose": 1,  # Keras verbosity for fit/predict
}


class TabTransformer:
    """TabTransformer classifier.

    ``fit``/``predict_proba``/``predict`` take a DataFrame (or CSV path). The
    DataFrame passed to ``fit`` must contain the ``target`` column; at predict
    time it is optional and ignored. Numeric dtypes are numeric features;
    object, category and bool columns are categorical.

    Note: attention uses ``key_dim = embedding_dim // heads`` (standard
    multi-head sizing), not ``key_dim = embedding_dim`` per head.
    """

    def __init__(self, config: Optional[Dict] = None, target: str = "target"):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.target = target
        self.classes_: Optional[np.ndarray] = None
        self.history = None
        self._cat_cols: List[str] = []
        self._num_cols: List[str] = []
        self._model: Optional[keras.Model] = None

    # ------------------------------------------------------------------ data
    @staticmethod
    def _load_data(data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, str):
            return pd.read_csv(data)
        raise ValueError("data must be a pandas DataFrame or a csv file path (string)")

    def _detect_features(self, df: pd.DataFrame) -> None:
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in data.")
        features = df.drop(columns=[self.target])
        self._cat_cols = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self._num_cols = features.select_dtypes(include=["number"]).columns.tolist()
        if not (self._cat_cols or self._num_cols):
            raise ValueError("No features detected. Check your dataframe columns.")

    def _inputs(self, df: pd.DataFrame) -> Dict[str, Union[np.ndarray, tf.Tensor]]:
        """Model inputs for ``df``; the single path used by fit and predict."""
        missing = [c for c in self._cat_cols + self._num_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns missing from provided data: {missing}")
        inputs = {}
        for c in self._cat_cols:
            col = df[c].astype(object).where(df[c].notna(), _NA_TOKEN).astype(str)
            inputs[f"cat_{c}"] = tf.constant(col.to_numpy(dtype=object)[:, None])
        if self._num_cols:
            inputs[_NUMERIC_INPUT] = df[self._num_cols].to_numpy(dtype="float32", na_value=0.0)
        return inputs

    # ----------------------------------------------------------------- model
    def _build(self, inputs: Dict[str, Union[np.ndarray, tf.Tensor]], n_classes: int) -> keras.Model:
        cfg = self.config
        dim = cfg["transformer_embedding_dim"]
        heads = cfg["transformer_heads"]
        model_inputs, tokens, num_out = [], [], None

        for c in self._cat_cols:
            name = f"cat_{c}"
            inp = layers.Input(shape=(1,), dtype="string", name=name)
            model_inputs.append(inp)
            lookup = layers.StringLookup(num_oov_indices=1)
            lookup.adapt(inputs[name])
            tokens.append(layers.Embedding(lookup.vocabulary_size(), dim)(lookup(inp)))  # (batch, 1, dim)

        if self._num_cols:
            inp = layers.Input(shape=(len(self._num_cols),), dtype="float32", name=_NUMERIC_INPUT)
            model_inputs.append(inp)
            norm = layers.Normalization()
            norm.adapt(inputs[_NUMERIC_INPUT])
            num_out = norm(inp)

        parts = []
        if tokens:
            x = layers.Concatenate(axis=1)(tokens) if len(tokens) > 1 else tokens[0]  # (batch, n_cat, dim)
            for _ in range(cfg["transformer_layers"]):
                attn = layers.MultiHeadAttention(num_heads=heads, key_dim=max(1, dim // heads))(x, x)
                x = layers.LayerNormalization()(x + layers.Dropout(cfg["dropout"])(attn))
                ff = layers.Dense(cfg["ff_dim"], activation=cfg["dense_activation"])(x)
                ff = layers.Dropout(cfg["dropout"])(layers.Dense(dim)(ff))
                x = layers.LayerNormalization()(x + ff)
            parts.append(layers.Flatten()(x))
        if num_out is not None:
            parts.append(num_out)

        y = layers.Concatenate()(parts) if len(parts) > 1 else parts[0]
        for units in cfg["mlp_units"]:
            y = layers.Dropout(cfg["mlp_dropout"])(layers.Dense(units, activation=cfg["dense_activation"])(y))
        if n_classes == 2:
            out = layers.Dense(1, activation="sigmoid", name="output")(y)
        else:
            out = layers.Dense(n_classes, activation="softmax", name="output")(y)
        return keras.Model(model_inputs, out)

    # ------------------------------------------------------------------- API
    def fit(self, data: Union[str, pd.DataFrame], **fit_kwargs):
        """Train on ``data`` (must include the target column). Returns the Keras history."""
        cfg = self.config
        df = self._load_data(data)
        self._detect_features(df)

        self.classes_, y = np.unique(df[self.target].to_numpy(), return_inverse=True)
        n_classes = max(2, len(self.classes_))
        loss = cfg["loss"]
        if n_classes > 2 and loss == "binary_crossentropy":
            loss = "sparse_categorical_crossentropy"

        metrics = cfg["metrics"] or (["AUC"] if n_classes == 2 else ["accuracy"])
        keras.utils.set_random_seed(cfg["seed"])
        inputs = self._inputs(df)
        self._model = self._build(inputs, n_classes)
        self._model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
            loss=loss,
            metrics=list(metrics),
        )
        params = {
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
            "shuffle": cfg["shuffle"],
            "verbose": cfg["verbose"],
            **fit_kwargs,
        }
        self.history = self._model.fit(inputs, y.astype("float32" if n_classes == 2 else "int32"), **params)
        return self.history

    def predict_proba(self, data: Union[str, pd.DataFrame], batch_size: Optional[int] = None) -> np.ndarray:
        """Positive-class probabilities (1-D) for binary, else an (n, n_classes) matrix."""
        if self._model is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        preds = self._model.predict(
            self._inputs(self._load_data(data)),
            batch_size=batch_size or self.config["batch_size"],
            verbose=self.config["verbose"],
        )
        return preds.ravel() if preds.shape[-1] == 1 else preds

    def predict(self, data: Union[str, pd.DataFrame], batch_size: Optional[int] = None) -> np.ndarray:
        """Predicted class labels."""
        probs = self.predict_proba(data, batch_size=batch_size)
        idx = (probs > 0.5).astype(int) if probs.ndim == 1 else probs.argmax(axis=1)
        return self.classes_[idx]
