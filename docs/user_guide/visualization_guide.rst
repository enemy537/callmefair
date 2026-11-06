Visualization Guide
===================

``callmefair.viz`` turns grid-search and bias-search results into figures. Every function
returns a matplotlib ``Figure`` and writes no files; save it yourself with
``fig.savefig("plot.png")`` if you need the image. Most functions take the list of dicts
from ``BMGridSearch.run_single_sensitive`` (or an equivalent DataFrame) and an optional
``ax``/``axes``.

The examples below use a small table of **illustrative numbers** (2 models x 4
mitigation settings, real column names), not real results.

.. plot::
   :context: close-figs
   :nofigs:

   rows = [
       # model, BM, balanced_acc, fair_score, eq_opp_diff, avg_odd_diff, spd, disparate_impact, theil_idx
       ("LogisticRegression", "baseline",       0.74, 0.62, 0.30, 0.25, 0.20, 0.55, 0.06),
       ("LogisticRegression", "preReweighing",  0.73, 0.35, 0.12, 0.10, 0.08, 0.80, 0.04),
       ("LogisticRegression", "preDisparate",   0.72, 0.41, 0.15, 0.12, 0.10, 0.75, 0.05),
       ("LogisticRegression", "posROC",         0.70, 0.18, 0.04, 0.05, 0.03, 0.93, 0.03),
       ("XGBClassifier",      "baseline",       0.79, 0.70, 0.34, 0.28, 0.24, 0.48, 0.08),
       ("XGBClassifier",      "preReweighing",  0.78, 0.52, 0.20, 0.17, 0.14, 0.66, 0.06),
       ("XGBClassifier",      "preDisparate",   0.77, 0.58, 0.24, 0.20, 0.17, 0.60, 0.07),
       ("XGBClassifier",      "posROC",         0.75, 0.30, 0.09, 0.08, 0.06, 0.85, 0.04),
   ]
   cols = ["model", "BM", "balanced_acc", "fair_score", "eq_opp_diff", "avg_odd_diff",
           "spd", "disparate_impact", "theil_idx"]
   results = [dict(zip(cols, r)) for r in rows]

Trade-off scatter
-----------------

``plot_tradeoff`` answers: which mitigations cost how much accuracy for how much fairness?
Arrows run from each model's baseline (hollow) to its mitigations, and the Pareto front is
labelled. ``pareto_front(results)`` returns the non-dominated rows.

.. plot::
   :context: close-figs

   from callmefair.viz import plot_tradeoff
   fig = plot_tradeoff(results)

Metric heatmap
--------------

``plot_metric_heatmap`` answers: how does one metric vary across mitigations and models?

.. plot::
   :context: close-figs

   from callmefair.viz import plot_metric_heatmap
   fig = plot_metric_heatmap(results, "fair_score")

Mitigation effect
-----------------

``plot_mitigation_effect`` answers: what did each mitigation change relative to its own
baseline? One panel per metric; blue bars are improvements, red bars are worse.

.. plot::
   :context: close-figs

   from callmefair.viz import plot_mitigation_effect
   fig = plot_mitigation_effect(results, ["fair_score", "balanced_acc"])

Fair ranges
-----------

``plot_fair_ranges`` answers: which of EOD, AOD, SPD, DI and TI are inside their acceptable
range? Filled dots are inside, hollow dots outside.

.. plot::
   :context: close-figs

   from callmefair.viz import plot_fair_ranges
   fig = plot_fair_ranges(results)

Bias search
-----------

``plot_bias_search`` answers: which sensitive attributes are the most biased? It takes the
table from ``BiasSearch.evaluate_average`` (header row first).

.. plot::
   :context: close-figs

   from callmefair.viz import plot_bias_search
   table = [
       ["Attribute", "Raw Fairness Score", "Normalized Fairness score",
        "Balanced Accuracy", "Privileged N", "Unprivileged N"],
       ["age",            0.52, 0.52, 0.78, 3200, 1900],
       ["gender",         0.21, 0.21, 0.77, 3000, 2100],
       ["ever_married",   0.38, 0.38, 0.77, 3300, 1800],
   ]
   fig = plot_bias_search(table)

Group rates and threshold sweep
-------------------------------

``plot_group_rates`` (selection rate, TPR and FPR for privileged vs. unprivileged groups)
and ``plot_threshold_sweep`` (balanced accuracy and one fairness metric against the
decision threshold) need a fitted ``BMMetrics``, so they are not executed here. See
``examples/notebooks/01_fairness_fundamentals.ipynb`` for a complete run.

.. code-block:: python

   from callmefair.viz import plot_group_rates, plot_threshold_sweep

   fig = plot_group_rates(bmM)                       # bmM: a fitted BMMetrics
   fig = plot_threshold_sweep(bmM, metric="spd")     # or "disparate_impact", "eq_opp_diff"

For the trade-off map see :doc:`tradeoff_map_guide`.

Visual conventions
------------------

* Categorical colors (one per model) follow a fixed order, never cycled; with more than
  three series, marker shapes add a second encoding.
* Magnitudes use one blue ramp; signed changes use blue to red around a gray midpoint,
  where blue always means better.
* Green and red are reserved for fair / unfair and always carry a text label.
* One y-axis per panel (no twin axes), thin marks and a recessive grid.
* Fairness scores read 0 = fair (disparate impact: 1 = fair); lower ``fair_score`` is better.
