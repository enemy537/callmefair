Quick Start
===========

This page runs one complete bias-mitigation search in a few lines. Install
the package first (see :doc:`/user_guide/installation`).

Run a grid search
-----------------

The example uses ``examples/data/stroke_pre/stroke.csv`` (5,109 rows, label ``stroke``,
protected attribute ``age`` where 1 = 65 or older, all features numeric) and a logistic regression.
It evaluates three mitigation combinations: pre-processing only,
post-processing only, and both.

.. code-block:: python

   import pandas as pd
   from callmefair import BMInterface, BMGridSearch, BMType
   from sklearn.linear_model import LogisticRegression

   df = pd.read_csv("examples/data/stroke_pre/stroke.csv")
   bmI = BMInterface(df, label="stroke", protected=["age"])

   privileged_group = [{"age": 1}]
   unprivileged_group = [{"age": 0}]

   bm_list = [
       [BMType.preReweighing],
       [BMType.posCalibrated],
       [BMType.preReweighing, BMType.posCalibrated],
   ]
   grid = BMGridSearch(bmI, LogisticRegression(max_iter=1000), bm_list,
                       privileged_group, unprivileged_group)
   rows = grid.run_single_sensitive(results_dir=None)  # None: skip the CSV log

   results = pd.DataFrame(rows)
   print(results[["BM", "balanced_acc", "spd", "fair_score"]].round(3).to_string())

Output:

.. code-block:: text

                               BM  balanced_acc    spd  fair_score
   0                     baseline         0.695 -0.842       1.000
   1                preReweighing         0.575 -0.094       0.483
   2                posCalibrated         0.500  0.000       0.529
   3  preReweighing posCalibrated         0.500  0.000       0.529

``BMInterface`` splits the data (80/10/10 train/validation/test, stratified)
and scales it. The first row is always the unmitigated ``baseline``. The rows
are written to ``results/experiment_<timestamp>.csv`` unless you pass
``results_dir=None`` as above.

Reading the result
------------------

- ``fair_score`` is in [0, 1]; **lower is fairer** (0 means all fairness
  metrics are at their optimum). It is computed from the test split.
- ``balanced_acc`` measures predictive quality; mitigation usually costs some
  of it. Here ``preReweighing`` cuts ``fair_score`` from 1.0 to 0.48 at a cost
  of about 0.12 balanced accuracy.
- ``posCalibrated`` collapses to a constant prediction here (balanced accuracy
  0.5), because only 4.9% of the stroke rows are positive. The test split has
  511 rows, so single-run metrics are noisy.

Next steps
----------

- :doc:`/user_guide/bias_search_guide`: find which attributes are biased.
- :doc:`/user_guide/grid_search_guide`: models, combinations and result files.
- :doc:`/user_guide/tradeoff_map_guide`: plot fairness against accuracy.
- :doc:`/user_guide/hardware_guide`: GPU and parallelism settings.
