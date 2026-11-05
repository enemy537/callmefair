Grid Search
===========

:class:`callmefair.mitigation.fair_grid.BMGridSearch` evaluates combinations
of bias-mitigation (BM) techniques for one model on one protected attribute
and logs the metrics of each (see :doc:`/user_guide/quickstart`, :doc:`/api/mitigation`).

How a combination runs
----------------------

Each :class:`callmefair.mitigation.fair_bm.BMType` belongs to one stage:

- pre-processing: ``preReweighing``, ``preDisparate``, ``preLFR``
- in-processing: ``inAdversarial``, ``inMeta``
- post-processing: ``posCalibrated``, ``posEqqOds``, ``posROC``

A combination is applied as pre-processing (train split), then
training, then post-processing (chained), each stage in list order. The
``BM`` label joins the names (``preReweighing posROC``). Data is restored
after every combination; the first row is always ``baseline``.
``preReweighing`` weights reach any model whose ``fit`` accepts
``sample_weight``; ``preDisparate`` needs ``BlackBoxAuditing``.

Building ``bm_list`` and running
--------------------------------

``bm_list`` is a list of lists of ``BMType``; ``itertools.product`` builds pairs.

.. code-block:: python

   from itertools import product

   import pandas as pd
   from sklearn.linear_model import LogisticRegression

   from callmefair import BMGridSearch, BMInterface, BMType

   df = pd.read_csv("examples/data/stroke_pre/stroke.csv")
   bmI = BMInterface(df, label="stroke", protected=["age"])
   privileged_group, unprivileged_group = [{"age": 1}], [{"age": 0}]

   pre_bm = [BMType.preReweighing, BMType.preDisparate]
   pos_bm = [BMType.posCalibrated, BMType.posROC]

   bm_list = [[pre, pos] for pre, pos in product(pre_bm, pos_bm)]  # pre + post
   bm_list += [[b] for b in pre_bm + pos_bm]                      # each alone

   grid = BMGridSearch(bmI, LogisticRegression(max_iter=1000), bm_list,
                       privileged_group, unprivileged_group)
   rows = grid.run_single_sensitive(results_dir="results")

   res = pd.DataFrame(rows)
   print(res[["BM", "balanced_acc", "spd", "fair_score"]].round(3).to_string())
   print("fairest:", res.loc[res["fair_score"].idxmin(), "BM"])

.. code-block:: text

                               BM  balanced_acc    spd  fair_score
   0                     baseline         0.678 -0.744       1.000
   1  preReweighing posCalibrated         0.537  0.019       0.894
   2         preReweighing posROC         0.587 -0.120       0.532
   3   preDisparate posCalibrated         0.619 -0.483       1.000
   4         preDisparate posROC         0.593 -0.133       0.559
   5                preReweighing         0.587 -0.120       0.532
   6                 preDisparate         0.675 -0.738       1.000
   7                posCalibrated         0.619 -0.483       1.000
   8                       posROC         0.593 -0.133       0.559
   fairest: preReweighing posROC

``fair_score`` is lower-is-fairer (see :doc:`/user_guide/utility_guide`), so
the best combination is ``idxmin``. Ties at 1.0 are common.

Results file
------------

``run_single_sensitive(results_dir='results')`` returns the rows and writes
them to ``<results_dir>/experiment_<timestamp>.csv``. Columns: ``model``,
``BM``, ``balanced_acc``, ``acc``, ``precision``, ``recall``, ``f1``, ``mcc``,
``eq_opp_diff``, ``avg_odd_diff``, ``spd``, ``disparate_impact``,
``theil_idx``, ``fair_score``.

To merge runs use :func:`callmefair.mitigation.fair_log.aggregate_csv_files`.
It **deletes the source files by default**; pass ``delete_sources=False`` to
keep them. It uses a process pool, so call it under
``if __name__ == "__main__":``.

Supported models and custom adapters
------------------------------------

Models are dispatched through :mod:`callmefair.mitigation.model_adapters`.
Any object with ``fit`` and ``predict_proba`` works through the generic
scikit-learn adapter (this covers scikit-learn, XGBoost and CatBoost).
LightGBM, TabNet and the callmefair TabTransformer have dedicated adapters.
For anything else, register an adapter (matched by class name, subclasses
included): a class whose ``fit_predict_proba(model, bmI)`` returns
``(val_proba, test_proba)``, each ``(n_samples, n_classes)`` with the
favorable class in column 1.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from sklearn.svm import LinearSVC

   from callmefair import BMGridSearch, BMInterface, BMType
   from callmefair.mitigation.model_adapters import register_adapter


   class LinearSVCAdapter:
       """LinearSVC has no predict_proba: squash decision_function with a sigmoid."""

       def fit_predict_proba(self, model, bmI):
           x_train, y_train = bmI.get_train_xy()
           model.fit(x_train, y_train,
                     sample_weight=bmI.get_train_BLD().instance_weights)

           def proba(x):
               p = 1.0 / (1.0 + np.exp(-model.decision_function(x)))
               return np.column_stack((1.0 - p, p))

           return proba(bmI.get_val_xy()[0]), proba(bmI.get_test_xy()[0])


   register_adapter(LinearSVC, LinearSVCAdapter())

   df = pd.read_csv("examples/data/stroke_pre/stroke.csv")
   bmI = BMInterface(df, label="stroke", protected=["age"])
   grid = BMGridSearch(bmI, LinearSVC(), [[BMType.preReweighing]],
                       [{"age": 1}], [{"age": 0}])
   rows = grid.run_single_sensitive()

In-processing mode
------------------

In-processing techniques supply their own model. Pass ``model=None`` and put
``inAdversarial`` or ``inMeta`` in ``bm_list`` (the baseline is the same
learner without mitigation). Combining one with a model raises ``ValueError``.

.. code-block:: python

   import pandas as pd

   from callmefair import BMGridSearch, BMInterface, BMType

   df = pd.read_csv("examples/data/stroke_pre/stroke.csv")
   bmI = BMInterface(df, label="stroke", protected=["age"])
   grid = BMGridSearch(bmI, None, [[BMType.inAdversarial]], [{"age": 1}], [{"age": 0}])
   rows = grid.run_single_sensitive()
   print(pd.DataFrame(rows)[["model", "BM", "balanced_acc", "fair_score"]].round(3).to_string())

.. code-block:: text

                     model        BM  balanced_acc  fair_score
   0  AdversarialDebiasing  baseline         0.683         1.0
   1  AdversarialDebiasing      inAD         0.681         1.0

Caveats
-------

- Small groups make metrics noisy (the stroke test split has 189 rows).
- ``posEqqOds`` can collapse to a trivial classifier when the validation
  split is tiny (as with stroke), giving degenerate metrics.
- ``disparate_impact`` can be ``0`` or ``inf``; the score treats it as
  invalid and uses the default 0.1.
- ``inAdversarial`` uses TensorFlow 1 graph mode and disables eager execution
  process-wide; run it apart from Keras or TabTransformer work.
- ``inMeta`` with ``tau > 0`` raised ``TypeError`` inside aif360's MetaFair
  code on both example datasets in our runs; verify it on your data first.
