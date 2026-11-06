Utilities
=========

The classes in :mod:`callmefair.util.fair_util` are used by every search.
Signatures and parameters are listed in :doc:`/api/util`; this page explains
the behaviour behind them.

Fairness score
--------------

:func:`callmefair.util.fair_util.calculate_fairness_score` reduces five
metrics to one number, ``overall_score``, in [0, 1] where **0 is perfectly
fair and higher is more biased**. The grid-search CSV column ``fair_score``
is this value.

For each metric the acceptable range and optimum are:

- EOD, AOD, SPD (equal opportunity, average odds, statistical parity
  difference): range (-0.1, 0.1), optimum 0.
- DI (disparate impact): range (0.8, 1.2), optimum 1.
- TI (Theil index): range (0, 0.25), optimum 0.

Each metric contributes ``0.2`` if it is outside its range, plus
``0.16 * deviation``, where ``deviation = |value - optimum| / half-width``
(half-widths 0.1, 0.1, 0.1, 0.2, 0.25). The sum is divided by the maximum,
``5 * 0.36``, and clipped to [0, 1].

Values that are ``None``, NaN/inf, non-numeric, or implausible (absolute EOD, AOD
or SPD above 1; DI outside [1e-6, 5]; TI outside [0, 1]) never raise. They
are replaced by "unfair" defaults: EOD = AOD = SPD = TI = 1.0 and DI = 0.1.

.. code-block:: python

   from callmefair import calculate_fairness_score

   s = calculate_fairness_score(EOD=0.05, AOD=0.12, SPD=-0.02, DI=0.9, TI=float("nan"))
   print(s["overall_score"], s["is_fair"])
   print(s["metric_evaluations"])
   print(s["deviations"])
   print(s["raw_score"])

.. code-block:: text

   0.791 False
   {'EOD': True, 'AOD': False, 'SPD': True, 'DI': True, 'TI': False}
   {'EOD': 0.5, 'AOD': 1.2, 'SPD': 0.2, 'DI': 0.5, 'TI': 4.0}
   1.424

The NaN Theil index became 1.0 (deviation 4.0), which dominates the score.
The returned dict has ``raw_score``, ``overall_score`` (3 decimals),
``metric_evaluations`` (in range?), ``deviations`` and ``is_fair`` (all in
range).

Data splits: BMInterface and BMMetrics
--------------------------------------

``BMInterface(df, label, protected, random_state=42, shuffle=True,
test_size=0.1, val_size=0.1)`` makes a stratified train/validation/test split
(sizes are fractions of the full frame) and serves each split two ways:

- ``get_train_xy()``, ``get_val_xy()``, ``get_test_xy()``: ``(X, y)`` arrays,
  always standardised with a scaler fitted on the train split.
- ``get_train_BLD()``, ``get_val_BLD()``, ``get_test_BLD()``: deep copies of
  the aif360 ``BinaryLabelDataset``\ s, unscaled until ``set_transform()`` is
  called (it is idempotent).
- ``restore_BLD()`` discards any mitigation (reweighing, repair, ...) and
  rebuilds the datasets.

If a split lacks one of the two classes, the constructor retries with seeds
``random_state + 1, ...`` (15 tries), then oversamples the minority class.

``BMMetrics`` picks the decision threshold that maximises balanced accuracy on
the **validation** split (100 values from 0.01 to 0.99) and reports metrics on
the **test** split with that threshold.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from callmefair import BMInterface, BMMetrics
   from sklearn.linear_model import LogisticRegression

   df = pd.read_csv("examples/data/stroke_pre/stroke.csv")
   bmI = BMInterface(df, label="stroke", protected=["age"])

   X_train, y_train = bmI.get_train_xy()
   print(X_train.shape, bmI.get_val_xy()[0].shape, bmI.get_test_xy()[0].shape)

   model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
   metrics = BMMetrics(bmI, np.array([0, 1]),
                       model.predict_proba(bmI.get_val_xy()[0]),
                       model.predict_proba(bmI.get_test_xy()[0]),
                       privileged_group=[{"age": 1}], unprivileged_group=[{"age": 0}])
   print("threshold:", round(metrics.best_class_thresh, 2))
   print({k: round(float(v), 3) for k, v in metrics.get_report().items()})
   print(metrics.get_score()["overall_score"])

.. code-block:: text

   (4087, 17) (511, 17) (511, 17)
   threshold: 0.03
   {'balanced_acc': 0.695, 'acc': 0.708, 'precision': 0.108, 'recall': 0.68, 'f1': 0.186, 'mcc': 0.08, 'eq_opp_diff': -0.8, 'avg_odd_diff': -0.822, 'spd': -0.842, 'disparate_impact': 0.158, 'theil_idx': 0.072}
   1.0

``get_report()`` keys:

- Classification: ``balanced_acc``, ``acc``, ``precision``, ``recall``,
  ``f1``, ``mcc`` (``f1`` and ``mcc`` are 0.0 when undefined).
- Fairness: ``eq_opp_diff``, ``avg_odd_diff``, ``spd``, ``disparate_impact``,
  ``theil_idx``. If ``avg_odd_diff`` cannot be computed it is set to 1.0.

``get_score()`` returns the ``calculate_fairness_score`` dict for these
metrics. ``pred_val``/``pred_test`` may be probability arrays of shape
``(n, n_classes)`` (columns follow ``class_array``) or aif360 datasets
(in-processing mode, using their ``.scores``).
``set_pos_pred(test_BLD_pred)`` recomputes the metrics for a post-processed
prediction whose labels are used as-is. The helper
:func:`callmefair.util.fair_util.find_best_threshold` implements the
threshold search.
