Trade-off Map Guide
===================

The trade-off map implements **Fairea** (Hort et al., ESEC/FSE 2021): it asks whether a
bias-mitigation method is better than simply degrading the original model. The walk-through is
``examples/notebooks/04_tradeoff_map.ipynb``; the code is ``callmefair.tradeoff`` and
``callmefair.viz.plot_tradeoff_map``.

Method
------

1. **Baseline.** Train the original model and overwrite a fraction ``degree`` (0, 0.1, ..., 1)
   of its test predictions with one label. The strategy is the label with the highest accuracy
   at 100% mutation (ties go to the stricter curve). Results are averaged over
   ``data_splits`` x ``repetitions`` (the paper uses 50 x 50; the default is 10 x 10).
2. **Axes.** x is bias, ``|SPD|`` by default (``"aod"``, ``"eod"`` and ``"fair_score"`` are
   options); y is accuracy, ``"balanced_acc"`` by default (``"acc"`` is an option). Both are
   min-max normalised *by the baseline's range*: the original model F_OM is at (1, 1), the fully
   mutated model F100 at (0, 0), and with a linear bias metric the baseline is close to the
   diagonal. Methods use the same scaling and may fall outside [0, 1].
3. **Regions** (paper Fig. 2), relative to F_OM:

   * ``win-win`` [Win-Win]: at least as accurate and no more biased;
   * ``inverted`` [Inverted]: at least as accurate but more biased;
   * ``lose-lose`` [Lose-Lose]: less accurate and more biased;
   * ``good trade-off`` [Good]: less accurate, less biased, above the baseline;
   * ``bad trade-off`` [Poor]: less accurate, less biased, below the baseline.

4. **Area** (paper Fig. 3) scores a good trade-off: the polygon from the method BM to its
   horizontal projection BM' on the baseline, along the baseline, and back to its vertical
   projection BM''. Other regions have area 0.

Why these defaults
------------------

* **Balanced accuracy.** With imbalanced labels, mutating to the majority label *raises* plain
  accuracy (stroke, 4.9% positives: 0.726 to 0.951). Fairea requires the original model to beat
  the constant model, so ``baseline_curve`` warns when it does not. Balanced accuracy ends at
  0.5 for any constant model.
* **|SPD| rather than the composite score.** The composite ``fair_score`` is non-linear (it jumps
  by 0.2 when a metric leaves its acceptable range) and capped at 1, so its baseline bends and
  saturates. Fairea uses a single linear group metric. ``bias="fair_score"`` stays available; read
  its map with that caveat.

Usage
-----

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from callmefair.tradeoff import create_baseline, tradeoff_regions
    from callmefair.viz import plot_tradeoff_map

    base = create_baseline(lambda: LogisticRegression(max_iter=1000), df, "stroke", "age",
                           priv, unpriv, data_splits=10, repetitions=10,
                           bmi_kwargs=dict(test_size=0.2, val_size=0.2))
    regions = tradeoff_regions(base, rows)   # rows: BMGridSearch.run_single_sensitive(...)
    fig = plot_tradeoff_map(base, regions)

``create_baseline`` returns one row per (strategy, degree) with ``acc, balanced_acc, spd, aod,
eod, fair_score``. ``tradeoff_regions`` drops ``baseline`` rows, averages repeated (model, BM)
rows and returns ``model, BM, accuracy, bias, acc_norm, bias_norm, region, paper_region, area``.
Building blocks: ``select_strategy``, ``baseline_curve``, ``normalize``, ``classify_region``,
``area_polygon`` and ``compute_area``. Pass the same ``bmi_kwargs`` split settings to the
baseline and the grid.

The figure uses **illustrative numbers** (a made-up straight baseline and grid rows), not a run.

.. plot::

    import numpy as np
    import pandas as pd
    from callmefair.tradeoff import tradeoff_regions
    from callmefair.viz import plot_tradeoff_map

    degree = np.round(np.linspace(0, 1, 11), 1)
    baseline_df = pd.DataFrame({
        "strategy": "to_1", "degree": degree,
        "acc": 0.9 - 0.4 * degree, "balanced_acc": 0.75 - 0.25 * degree,
        "spd": 0.40 * (1 - degree), "aod": 0.30 * (1 - degree),
        "eod": 0.30 * (1 - degree), "fair_score": 0.60 * (1 - degree),
    })
    def row(model, bm, bacc, spd):
        return {"model": model, "BM": bm, "acc": bacc + 0.1, "balanced_acc": bacc, "spd": spd,
                "avg_odd_diff": spd, "eq_opp_diff": spd, "fair_score": spd * 1.5}
    rows = [
        row("LogisticRegression", "baseline", 0.75, 0.40),
        row("LogisticRegression", "preReweighing", 0.74, 0.22),
        row("LogisticRegression", "posROC", 0.68, 0.10),
        row("XGBClassifier", "baseline", 0.78, 0.42),
        row("XGBClassifier", "preDisparate", 0.77, 0.45),
        row("XGBClassifier", "posROC", 0.66, 0.25),
        row("XGBClassifier", "posEqqOds", 0.70, 0.05),
    ]
    regions_df = tradeoff_regions(baseline_df, rows)
    fig = plot_tradeoff_map(baseline_df, regions_df)

Use more splits and repetitions than a quick run for real analyses; points close to the baseline
are ties within noise.

.. note::

   Earlier versions drew the regions with an inverted accuracy axis and normalised over baseline
   and methods together, so regions were flipped. Maps made with that code should be re-examined;
   see :doc:`/changelog`.
