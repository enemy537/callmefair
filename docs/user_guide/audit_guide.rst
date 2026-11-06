Auditing any classifier
=======================

``callmefair`` cannot train every model, and it does not need to. Give it the
**predictions** of a model trained anywhere (scikit-learn, R, a vendor API...) together
with the true outcome and the group columns, and it reports bias with the same metrics
and fairness score as the rest of the library (:doc:`/user_guide/utility_guide`).
A runnable walk-through is notebook ``06_audit_external_predictions``.

Inputs
------

One row per individual, in a DataFrame (:func:`~callmefair.audit.audit_predictions`) or a
CSV / parquet file (:func:`~callmefair.audit.audit_file`; parquet needs ``pyarrow``).

``label``
   True outcome column (two distinct values).
``prediction``
   Predicted label column (hard labels, not scores), or a list of columns to compare
   several models; model names are then the column names.
``protected``
   ``{attribute: privileged}``. Everyone not privileged is the unprivileged group.
``favorable_label``
   The outcome that is good for the individual (default ``1``).
``sample_weight``, ``model``, ``intersections``
   Optional weight column; model name for a single prediction column; also audit every
   pair of attributes (privileged = privileged in both, named ``'a_b'``).

The privileged group can be a value, a list or set of values, or a function of the
column, so non-binary and text columns work as they are::

   protected = {"age": 1}                        # value
   protected = {"race": ["Caucasian", "Asian"]}  # several values
   protected = {"age": lambda s: s >= 65}        # any boolean mask

Missing columns, NaNs, non-binary label or prediction and empty groups raise a
``ValueError`` that says what to fix. A group without positive true outcomes only warns:
TPR-based metrics (EOD, AOD) are undefined there and count as unfair.

Outputs
-------

:class:`~callmefair.audit.BiasReport`:

* ``summary()`` (also ``print(report)``): verdict per attribute and the metrics outside range.
* ``metrics(attribute)``: the 11 metrics of ``BMMetrics.get_report()`` plus ``fair_score``
  and ``fair_score_uncapped``. Differences are *unprivileged minus privileged*.
* ``groups(attribute)``: size, base rate, selection rate, TPR, FPR and precision per group.
* ``is_fair(attribute)``, ``to_frame()`` (one row per attribute and model).
* ``to_rows()``: grid-search style rows, so ``plot_fair_ranges``, ``plot_metric_heatmap``
  and ``plot_tradeoff`` work directly; ``plot_group_rates(report, attribute=...)`` too.

Example
-------

.. code-block:: python

   from callmefair import audit_file

   report = audit_file("preds.csv", label="stroke", prediction="pred_hgb",
                       protected={"age": 1, "gender": 1}, model="HistGB")
   print(report)

Output (stroke data, scikit-learn ``HistGradientBoostingClassifier``)::

   age: biased: EOD -0.30 outside (-0.1, 0.1), AOD -0.30 outside (-0.1, 0.1), SPD -0.31 outside (-0.1, 0.1), DI 0.09 outside (0.8, 1.2); fairness score 1.00 (0 = fair)
   gender: biased: EOD -0.17 outside (-0.1, 0.1); fairness score 0.44 (0 = fair)

Command line
------------

.. code-block:: bash

   python -m callmefair.audit preds.csv --label stroke --prediction pred_hgb pred_rule \
       --protected age=1 --protected race=Caucasian,Asian [--intersections]

Values are read as numbers when possible; ``A,B`` gives several privileged values. The
summary and the metrics table are printed.

Reading the result
------------------

A gap in selection rate (SPD, DI) can reflect a real difference in base rates; a gap in
TPR or FPR (EOD, AOD) means equally situated people are treated differently. Check group
sizes: metrics from few positives are noisy. The audit shows that predictions differ, not
why. To reduce the bias of a model you can retrain, see :doc:`/user_guide/grid_search_guide`.
