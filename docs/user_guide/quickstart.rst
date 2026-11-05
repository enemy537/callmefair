Quick Start
===========

This page runs one complete bias-mitigation search in a few lines. Install
the package first (see :doc:`/user_guide/installation`).

Run a grid search
-----------------

The example uses ``examples/data/stroke_pre/stroke.csv`` (label ``stroke``,
protected attribute ``age``, all features numeric) and a logistic regression.
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
   rows = grid.run_single_sensitive()

   results = pd.DataFrame(rows)
   print(results[["BM", "balanced_acc", "spd", "fair_score"]].round(3).to_string())

Output:

.. code-block:: text

                               BM  balanced_acc    spd  fair_score
   0                     baseline         0.678 -0.744       1.000
   1                preReweighing         0.587 -0.120       0.532
   2                posCalibrated         0.619 -0.483       1.000
   3  preReweighing posCalibrated         0.537  0.019       0.894

``BMInterface`` splits the data (80/10/10 train/validation/test, stratified)
and scales it. The first row is always the unmitigated ``baseline``. The rows
are also written to ``results/experiment_<timestamp>.csv``.

Reading the result
------------------

- ``fair_score`` is in [0, 1]; **lower is fairer** (0 means all fairness
  metrics are at their optimum). It is computed from the test split.
- ``balanced_acc`` measures predictive quality; mitigation usually costs some
  of it. Here ``preReweighing`` cuts ``fair_score`` from 1.0 to 0.53 at a cost
  of about 0.09 balanced accuracy.
- The stroke test split has only 189 rows, so single-run metrics are noisy.

Next steps
----------

- :doc:`/user_guide/bias_search_guide`: find which attributes are biased.
- :doc:`/user_guide/grid_search_guide`: models, combinations and result files.
- :doc:`/user_guide/tradeoff_map_guide`: plot fairness against accuracy.
- :doc:`/user_guide/hardware_guide`: GPU and parallelism settings.
