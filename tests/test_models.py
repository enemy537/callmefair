import numpy as np
import pandas as pd
import pytest

from callmefair.util.models import TabTransformer

CONFIG = {
    "epochs": 3, "batch_size": 64, "verbose": 0,
    "transformer_layers": 1, "transformer_heads": 2, "transformer_embedding_dim": 8,
    "ff_dim": 8, "mlp_units": [8],
}


def make_df(n=200, n_classes=2, categorical=False, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    score = df["f0"] + 0.5 * df["f1"]
    df["target"] = np.digitize(score, np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1]))
    if categorical:
        df["color"] = rng.choice(["red", "green", "blue"], size=n)
        df.loc[0, "color"] = None
    return df


def fitted(**kw):
    df = make_df(**kw)
    model = TabTransformer(CONFIG)
    model.fit(df)
    return model, df


def test_binary_numeric():
    model, df = fitted()
    proba = model.predict_proba(df)
    assert proba.shape == (len(df),)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert set(model.predict(df)) <= {0, 1}


def test_predict_without_target_column():
    model, df = fitted()
    np.testing.assert_allclose(model.predict_proba(df.drop(columns="target")), model.predict_proba(df))


def test_mixed_categorical_train_predict_consistent():
    model, df = fitted(categorical=True)
    unseen = df.copy()
    unseen["color"] = "purple"  # unseen category goes to the OOV bucket
    assert model.predict_proba(df).shape == (len(df),)
    assert model.predict_proba(unseen).shape == (len(df),)


def test_multiclass_switches_loss():
    model, df = fitted(n_classes=3, categorical=True)
    proba = model.predict_proba(df)
    assert proba.shape == (len(df), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-4)
    assert set(model.predict(df)) <= {0, 1, 2}
    assert "sparse_categorical" in str(model._model.loss)


def test_string_labels_round_trip():
    df = make_df()
    df["target"] = df["target"].map({0: "no", 1: "yes"})
    model = TabTransformer(CONFIG)
    model.fit(df)
    assert set(model.predict(df)) <= {"no", "yes"}


def test_deterministic_with_seed():
    df = make_df()
    runs = []
    for _ in range(2):
        model = TabTransformer(CONFIG)
        model.fit(df)
        runs.append(model.predict_proba(df))
    np.testing.assert_allclose(runs[0], runs[1], atol=1e-4)


def test_errors():
    with pytest.raises(ValueError):
        TabTransformer(CONFIG).fit(make_df().drop(columns="target"))
    with pytest.raises(RuntimeError):
        TabTransformer(CONFIG).predict_proba(make_df())
