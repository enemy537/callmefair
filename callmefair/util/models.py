# Requires: tensorflow (2.x), pandas, numpy
# Adapted from Keras example: "Structured data learning with TabTransformer". See citation in chat.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics, callbacks, optimizers
import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, List

class TabTransformer:
    """
    TabTransformer classifier wrapper.
    - Accepts a CSV path or a pandas.DataFrame as input.
    - Auto-detects numeric vs categorical features.
    - Expects config dict to control model/training hyperparameters.
    """

    def __init__(self, config: Optional[Dict] = None, target: str = "target"):
        # default config
        default = {
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
            "loss": "binary_crossentropy",   # for binary classification by default
            "metrics": ["AUC"],
            "shuffle": True,
            "validation_split": 0.2,
            "seed": 42,
            # Controls verbosity of Keras .fit / .predict (0 = silent, 1 = progress bar)
            "verbose": 1,
            # For categorical embedding cardinality limit (optional)
            "max_embedding_cardinality": None  # keep None to allow all unique categories
        }
        self.config = default
        if config:
            self.config.update(config)
        self.target = target

        # internals to be set during fit
        self._cat_cols: List[str] = []
        self._num_cols: List[str] = []
        self._cat_lookup_layers: Dict[str, layers.Layer] = {}
        self._num_inputs = None
        self._model: Optional[keras.Model] = None
        self.history = None
        self._label_mapping = None  # if we need to map labels for multiclass

    @staticmethod
    def _load_data(data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, str):
            # assume CSV path
            return pd.read_csv(data)
        else:
            raise ValueError("data must be a pandas DataFrame or a csv file path (string)")

    def _auto_detect_features(self, df: pd.DataFrame):
        # drop rows where target is missing
        df = df.copy()
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in data.")
        # simple detection: numbers -> numeric, object/category/bool -> categorical
        num = df.select_dtypes(include=["number"]).columns.tolist()
        cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # remove target from lists if present
        num = [c for c in num if c != self.target]
        cat = [c for c in cat if c != self.target]

        # If no numeric columns found, attempt to coerce some object columns that look numeric
        if len(num) == 0:
            for c in df.columns:
                if c == self.target: 
                    continue
                try:
                    _ = pd.to_numeric(df[c].dropna())
                    # if conversion ok and many unique numeric-like values, treat as numeric
                    num.append(c)
                except Exception:
                    pass
            # remove found from cat
            cat = [c for c in cat if c not in num]

        self._num_cols = num
        self._cat_cols = cat

    def _make_preprocessing_layers(self, df: pd.DataFrame):
        """
        Build keras preprocessing layers for categorical and numerical features.
        For categorical: use StringLookup -> Integer inputs -> Embedding
        For numeric: use Normalization
        """
        # categorical lookup layers (StringLookup)
        self._cat_lookup_layers = {}
        for c in self._cat_cols:
            values = df[c].astype(str).fillna("<<NA>>").values
            # create StringLookup layer and adapt
            lookup = layers.StringLookup(output_mode="int", num_oov_indices=1)
            lookup.adapt(values)
            # optional trimming by config["max_embedding_cardinality"]
            if self.config.get("max_embedding_cardinality") is not None:
                # we won't trim the lookup, but you could map high-cardinalities to special token here
                pass
            self._cat_lookup_layers[c] = lookup

        # numeric normalization
        self._num_normalizer = None
        if len(self._num_cols) > 0:
            normalizer = layers.Normalization(axis=-1)
            # fit normalizer to numeric matrix
            num_matrix = df[self._num_cols].astype(float).fillna(0.0).values
            normalizer.adapt(num_matrix)
            self._num_normalizer = normalizer

    def _build_tab_transformer(self, df: pd.DataFrame, n_classes: int):
        cfg = self.config
        # Inputs
        cat_inputs = []
        cat_embeddings = []

        # For each categorical column, feed integer token -> embedding -> (to transformer)
        for c in self._cat_cols:
            input_c = layers.Input(shape=(1,), dtype=tf.string, name=f"in_{c}")
            cat_inputs.append(input_c)
            # lookup ints
            lookup = self._cat_lookup_layers[c]
            int_tokens = lookup(input_c)  # returns int indices
            # convert to embedding
            vocab_size = int(lookup.vocabulary_size())
            embed_dim = cfg["transformer_embedding_dim"]
            emb = layers.Embedding(input_dim=vocab_size + 1, output_dim=embed_dim)(int_tokens)
            # now emb shape: (batch, 1, embed_dim) -> we can squeeze to (batch, embed_dim) and later stack
            emb = layers.Reshape((embed_dim,))(emb)
            cat_embeddings.append(emb)

        # numeric input(s) - single dense vector
        num_input = None
        num_out = None
        if len(self._num_cols) > 0:
            num_input = layers.Input(shape=(len(self._num_cols),), dtype=tf.float32, name="numeric_inputs")
            cat_inputs.append(num_input)
            num_out = self._num_normalizer(num_input)

        # stack categorical embeddings -> produce per-feature token vectors
        if len(cat_embeddings) > 0:
            # produce a "tokens" tensor shape (batch, num_cat, embed_dim)
            token_stack = layers.Stack()(cat_embeddings) if hasattr(layers, "Stack") else tf.stack(cat_embeddings, axis=1)
            # If using older TF where layers.Stack isn't available, fallback above.
            # token_stack now (batch, num_cat, embed_dim)
            x = token_stack
            # Transformer blocks (apply to tokens)
            for _ in range(cfg["transformer_layers"]):
                # Multi-head attention on tokens
                attn = layers.MultiHeadAttention(num_heads=cfg["transformer_heads"],
                                                 key_dim=cfg["transformer_embedding_dim"])(x, x)
                attn = layers.Dropout(cfg["dropout"])(attn)
                x = layers.LayerNormalization()(x + attn)
                # Feed-forward per token
                ff = layers.Dense(cfg["ff_dim"], activation=cfg["dense_activation"])(x)
                ff = layers.Dense(cfg["transformer_embedding_dim"])(ff)
                ff = layers.Dropout(cfg["dropout"])(ff)
                x = layers.LayerNormalization()(x + ff)
            # Pool tokens into single vector
            cat_context = layers.Flatten()(x)  # alternatively can use GlobalAveragePooling1D
        else:
            cat_context = None

        # combine numeric and categorical contexts
        if cat_context is not None and num_out is not None:
            combined = layers.Concatenate()([cat_context, num_out])
        elif cat_context is not None:
            combined = cat_context
        elif num_out is not None:
            combined = num_out
        else:
            raise ValueError("No features detected. Check your dataframe columns.")

        # MLP head
        y = combined
        for u in cfg["mlp_units"]:
            y = layers.Dense(u, activation=cfg["dense_activation"])(y)
            y = layers.Dropout(cfg["mlp_dropout"])(y)

        if n_classes == 2:
            output = layers.Dense(1, activation="sigmoid", name="output")(y)
        else:
            output = layers.Dense(n_classes, activation="softmax", name="output")(y)

        model = keras.Model(inputs=cat_inputs, outputs=output)
        return model

    def _prepare_dataset(self, df: pd.DataFrame, shuffle=True):
        """
        Turn dataframe into tf.data.Dataset of model inputs and labels (X as dict/list consistent with model inputs).
        For categorical inputs we feed string tensors (we used string Input layers).
        For numeric inputs we feed a single 1D float vector.
        """
        # fill NA for categorical with a string token
        df_proc = df.copy()
        y = df_proc[self.target].values
        # Label mapping if multiclass (string labels)
        if y.dtype == object or y.dtype == str:
            # map to ints for loss if needed
            labels, uniques = pd.factorize(y)
            self._label_mapping = list(uniques)
            y_out = labels
        else:
            # numeric labels
            y_out = y

        # build inputs
        inputs = {}
        for c in self._cat_cols:
            # keep as string
            inputs[f"in_{c}"] = df_proc[c].astype(str).fillna("<<NA>>").values
        if len(self._num_cols) > 0:
            inputs["numeric_inputs"] = df_proc[self._num_cols].astype(float).fillna(0.0).values

        # tf.data creation
        ds = tf.data.Dataset.from_tensor_slices((inputs, y_out))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df_proc), seed=self.config["seed"])
        ds = ds.batch(self.config["batch_size"]).prefetch(tf.data.AUTOTUNE)
        return ds

    def fit(self, data: Union[str, pd.DataFrame], **fit_kwargs):
        """
        Fit the TabTransformer model on provided data.
        Additional Keras fit kwargs can be passed (e.g. validation_data).
        """
        df = self._load_data(data)
        self._auto_detect_features(df)
        self._make_preprocessing_layers(df)

        # prepare label dimensionality
        y_vals = df[self.target]
        # detect number of classes
        if y_vals.dtype == object or y_vals.dtype == str:
            n_classes = len(pd.factorize(y_vals)[1])
        else:
            unique = np.unique(y_vals)
            if len(unique) <= 2:
                n_classes = 2
            else:
                n_classes = len(unique)

        self._model = self._build_tab_transformer(df, n_classes)

        loss = self.config["loss"]
        opt = optimizers.Adam(learning_rate=self.config["learning_rate"])
        mtrs = []
        for m in self.config["metrics"]:
            if isinstance(m, str):
                mtrs.append(m)
            else:
                mtrs.append(m)

        if n_classes == 2:
            compiled_loss = loss
        else:
            compiled_loss = loss  # user should specify categorical_crossentropy for multi-class

        self._model.compile(optimizer=opt, loss=compiled_loss, metrics=mtrs)

        df_proc = df.copy()
        # labels
        y_vals = df_proc[self.target]
        if y_vals.dtype == object or y_vals.dtype == str:
            labels, uniques = pd.factorize(y_vals)
            y_out = labels
            self._label_mapping = list(uniques)
        else:
            y_out = y_vals.values

        # Build numpy inputs; convert categorical strings to integer token ids using the lookup layer's vocabulary
        inputs = {}
        for c in self._cat_cols:
            s = df_proc[c].astype(str).fillna("<<NA>>").values
            lookup = self._cat_lookup_layers[c]
            # Attempt to get the lookup vocabulary and build a mapping dict
            try:
                vocab = lookup.get_vocabulary()  # returns list of strings
                # StringLookup uses index 0 for mask/OOV depending on config; we build mapping consistent with get_vocabulary indices
                # mapping: token string -> its index in vocabulary.
                mapping = {tok: idx for idx, tok in enumerate(vocab)}
                # Map strings to indices; unknown tokens get index = len(vocab)  (safe OOV bucket)
                mapped = [mapping.get(x, len(vocab)) for x in s]
                inputs[f"in_{c}"] = np.array(mapped, dtype=np.int32)
            except Exception:
                # Fallback: if lookup.get_vocabulary() not available, try to call lookup on a tensor and fetch numpy
                import tensorflow as tf
                mapped_t = lookup(tf.constant(s))
                inputs[f"in_{c}"] = mapped_t.numpy().astype(np.int32)

        # Numeric inputs
        if len(self._num_cols) > 0:
            inputs["numeric_inputs"] = df_proc[self._num_cols].astype(float).fillna(0.0).values

        # Now call fit with numpy inputs
        fit_params = {
            "x": inputs,
            "y": y_out,
            "batch_size": self.config["batch_size"],
            "epochs": self.config["epochs"],
            "verbose": self.config.get("verbose", 1),
        }
        fit_params.update(fit_kwargs)
        self.history = self._model.fit(**fit_params)
        
        return self.history

    def predict_proba(self, data: Union[str, pd.DataFrame], batch_size: Optional[int] = None):
        df = self._load_data(data)
        # must detect features again (or rely on previously set) - prefer previously set to preserve preprocessing
        # but ensure columns exist
        for col in (self._cat_cols + self._num_cols + [self.target]):
            if col != self.target and col not in df.columns:
                raise ValueError(f"Column {col} missing from provided data for predict.")
        # prepare inputs as dict
        inputs = {}
        for c in self._cat_cols:
            inputs[f"in_{c}"] = df[c].astype(str).fillna("<<NA>>").values
        if len(self._num_cols) > 0:
            inputs["numeric_inputs"] = df[self._num_cols].astype(float).fillna(0.0).values

        preds = self._model.predict(
            inputs,
            batch_size=batch_size,
            verbose=self.config.get("verbose", 1),
        )
        if preds.shape[-1] == 1:
            # binary: return probability of positive class
            return preds.ravel()
        else:
            # multiclass: return probability matrix
            return preds

    def predict(self, data: Union[str, pd.DataFrame], batch_size: Optional[int] = None):
        probs = self.predict_proba(data, batch_size=batch_size)
        # map to labels
        if probs.ndim == 1:
            # binary classification threshold 0.5
            return (probs > 0.5).astype(int)
        else:
            # multiclass
            idx = probs.argmax(axis=1)
            if self._label_mapping is not None and len(self._label_mapping) == probs.shape[1]:
                return np.array(self._label_mapping)[idx]
            return idx
