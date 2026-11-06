Trade-off Map Guide
===================

The trade-off map places each bias-mitigation result (balanced accuracy, composite
fairness score) relative to a FAIREA-style baseline: the accuracy/bias curve of a base
model whose predictions are randomly overwritten at increasing degrees (0 to 1). The map
answers whether a mitigation is better than simply degrading the model.

.. note::

   This code lives in ``examples/notebooks/fhg_examples/tradeoff.py``, not in the
   installed ``callmefair`` package. Run from a clone of the repository with
   ``examples/notebooks`` on ``sys.path`` and the ``examples`` extra installed (provides
   ``shapely``); see :doc:`/user_guide/installation`. The walk-through is
   ``examples/notebooks/04_tradeoff_map.ipynb``.

Steps
-----

1. ``create_baseline(make_model, df, label, attribute, privileged_groups,
   unprivileged_groups, data_splits=3, repetitions=3)`` trains ``make_model()`` on
   ``data_splits`` random splits, overwrites a fraction ``degree`` of the test predictions,
   and returns the mean ``balanced_acc`` and ``fair_score`` per (strategy, degree).
   ``baseline_curve(baseline, "to_1")`` selects one strategy ordered by degree. The
   documented strategy is ``"to_1"`` (overwrite with label 1); overwriting with 0 makes
   disparate impact undefined.
2. ``normalize(base_acc, base_bias, methods)`` rescales baseline and methods with one
   shared range per axis and returns ``(loss, bias, methods_norm)``. Both axes are
   "lower is better": loss = accuracy lost, bias = ``fair_score``.
3. ``classify_region(line, methods_norm)`` returns one region per method, where
   ``line = LineString(zip(bias, loss))`` and its first vertex is the unmutated model O
   (paper names in brackets):

   * ``"win-win"`` [Win-Win]: no more biased and no less accurate than O;
   * ``"lose-lose"`` [Lose-Lose]: more biased and less accurate than O;
   * ``"inverted"`` [Inverted]: more biased but more accurate than O;
   * ``"good trade-off"`` [Good]: less biased but less accurate than O, and the baseline
     reaches that accuracy only with at least as much bias;
   * ``"bad trade-off"`` [Poor]: less biased but less accurate than O, and the baseline
     reaches that accuracy with less bias.

4. ``compute_area(line, point)`` returns the area between a good trade-off point and the
   baseline (0.0 for any other region).

Example
-------

Baseline for logistic regression on the stroke data (protected attribute ``age``) and three
mitigations from a grid search. Run from the repository root.

.. code-block:: python

    import sys
    import pandas as pd
    from shapely.geometry import LineString
    from sklearn.linear_model import LogisticRegression
    from callmefair import BMGridSearch, BMInterface, BMType

    sys.path.insert(0, "examples/notebooks")
    from fhg_examples.tradeoff import (baseline_curve, classify_region, compute_area,
                                       create_baseline, normalize)

    df = pd.read_csv("examples/data/stroke_pre/stroke.csv")
    priv, unpriv = [{"age": 1}], [{"age": 0}]

    # 1. Mitigation results: one (balanced_acc, fair_score) point per method
    bmI = BMInterface(df, label="stroke", protected=["age"])
    bm_list = [[BMType.preReweighing], [BMType.posROC], [BMType.preDisparate]]
    grid = BMGridSearch(bmI, LogisticRegression(max_iter=1000), bm_list, priv, unpriv)
    rows = grid.run_single_sensitive(results_dir=None)
    methods = {r["BM"]: (r["balanced_acc"], r["fair_score"]) for r in rows[1:]}

    # 2. Baseline: overwrite a growing share of predictions with 1
    base = create_baseline(lambda: LogisticRegression(max_iter=1000), df, "stroke", "age",
                           priv, unpriv, data_splits=3, repetitions=3)
    curve = baseline_curve(base, "to_1")

    # 3. Normalize on shared axes, classify, measure area
    loss, bias, pts = normalize(curve.balanced_acc, curve.fair_score, methods)
    line = LineString(zip(bias, loss))
    regions = classify_region(line, pts)
    for name, region in regions.items():
        print(f"{name:15s} {region:15s} area={compute_area(line, pts[name]):.3f}")

.. code-block:: text

    preReweighing   good trade-off  area=0.060
    posROC          good trade-off  area=0.042
    preDisparate    good trade-off  area=0.000

``preDisparate`` barely changes the model, so its point lies on the baseline start and its
area is about 0. This is a reduced run (3 splits x 3 repetitions); use more for real analyses.
Many grid results have ``fair_score`` exactly 1 (the saturated maximum); treat the regions of
such points with caution.

.. warning::

   An earlier version of this code (``examples/Fairea/fairea.py``, now removed) had three
   bugs: baseline and methods were normalized on different scales; accuracy was inverted
   while the original FAIREA region geometry was kept, so regions were flipped (a method with
   the original accuracy and much lower bias was labelled "bad trade-off", and one with much
   lower accuracy and the same bias "win-win"); and ``compute_area`` raised on most inputs.
   Trade-off maps produced with that code should be re-examined; see :doc:`/changelog`.
