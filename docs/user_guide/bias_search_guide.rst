Bias Search Guide
=================

:class:`callmefair.search.fair_search.BiasSearch` measures how biased a model is with
respect to binary sensitive attributes (``1`` = privileged group, ``0`` = unprivileged),
alone or combined with set operations. For each attribute it trains a model repeatedly on
different splits and averages a composite fairness score. **0 means perfectly fair; higher
means more biased.** The "Raw Fairness Score" is unbounded (>= 0), the "Normalized" score
lies in [0, 1] (it saturates at 1.0), and tables are sorted most biased first.

Evaluating attributes
---------------------

``BiasSearch(df, label_name, attribute_names, n_threads=None, backend='process',
random_state=42)``. Features must be numeric and the label binary (1 = favorable).

.. code-block:: python

    import pandas as pd
    from callmefair.search.fair_search import BiasSearch

    if __name__ == "__main__":  # required on macOS/Windows (process pool, spawn)
        df = pd.read_csv("examples/data/stroke_pre/stroke.csv")
        search = BiasSearch(df, "stroke", ["age", "gender", "ever_married", "Residence_type"])
        table, pretty = search.evaluate_average(iterate=3, model_name="lr")
        print(pretty)

.. code-block:: text

    +----------------+--------------------+---------------------------+--------------------+--------------+----------------+
    |   Attribute    | Raw Fairness Score | Normalized Fairness score | Balanced Accuracy  | Privileged N | Unprivileged N |
    +----------------+--------------------+---------------------------+--------------------+--------------+----------------+
    |      age       | 4.611407634682066  |            1.0            | 0.775399889685604  |     290      |      1593      |
    |  ever_married  | 3.5903379558107154 |            1.0            | 0.7137618636755825 |     1298     |      585       |
    | Residence_type | 1.257473275697019  |     0.6493333333333333    | 0.7362198933627505 |     950      |      933       |
    |     gender     | 1.043218147875954  |     0.5670000000000001    | 0.6975914690200403 |     760      |      1123      |
    +----------------+--------------------+---------------------------+--------------------+--------------+----------------+

``evaluate_average`` returns ``(table, PrettyTable)``; ``table`` is a list of rows (header
first) for further processing. The lower-level
``evaluate_attribute(attribute, treat_umbalance, iterate, model_name, df_new=None)`` returns a
dict with keys ``"<attribute>_raw"``, ``"<attribute>_overall"`` and
``"<attribute>_balanced_acc"``. Pass ``treat_umbalance=True`` (sic) to undersample the
training split with NearMiss.

Combinations and set operations
-------------------------------

``evaluate_combinations(iterate=10, model_name='lgbm')`` evaluates the intersection (AND) of
every 2- and 3-attribute combination, e.g. ``age_gender``. Combinations are skipped (and
listed in a summary line) unless the new attribute has exactly the values {0, 1}, both groups
have at least 10 rows, and each of the four (group, label) cells has at least 2 rows.

:func:`callmefair.search._search_base.combine_attributes` ``(df, col1, col2, operation)``
drops both columns and adds ``f"{col1}_{col2}"``. ``evaluate_combination_average(col_1, col_2)``
evaluates all five operations of ``CType`` for one pair:

=========================  =====  =====  =====  =====
Operation                  0,0    0,1    1,0    1,1
=========================  =====  =====  =====  =====
``union``                  0      1      1      1
``intersection``           0      0      0      1
``difference_1_minus_2``   0      0      1      0
``difference_2_minus_1``   0      1      0      0
``symmetric_difference``   0      1      1      0
=========================  =====  =====  =====  =====

Column headings are the ``(col1, col2)`` values of a row; entries are the combined value.

Methodology
-----------

Each of the ``iterate`` iterations ``i``:

* splits 70/15/15 (train/validation/test), stratified on (attribute, label), falling back to
  label only, then to a plain split; the seed is ``random_state + i``;
* optionally undersamples the training split with NearMiss;
* standardizes features and trains the model;
* picks the decision threshold on the validation split by balanced accuracy;
* computes fairness metrics and balanced accuracy on the test split;
* iterations are averaged. They run in a process pool (``n_threads`` workers, default
  ``min(8, performance cores)``); GPU-backed models run sequentially.

Models
------

``lr`` (liblinear), ``xgb``, ``cat``, ``lgbm``, ``mlp``, ``tabnet``, ``tabtransformer``.
``evaluate_average`` defaults to ``lr``, ``evaluate_combinations`` to ``lgbm``. Hyperparameters
are fixed in ``callmefair/search/model_factory.py``; device selection is described in
:doc:`/user_guide/hardware_guide`.

Adding a model
--------------

A builder receives scaled ``x``, labels ``y``, ``sample_weight`` and ``n_jobs`` and returns a
function mapping features to the probability of label 1 (``positive_proba`` wraps any
sklearn-style classifier). Register it with ``register_model``; pass
``uses_gpu=lambda: True`` for GPU models so they run sequentially.

.. code-block:: python

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from callmefair.search.fair_search import BiasSearch
    from callmefair.search.model_factory import positive_proba, register_model

    @register_model("rf")
    def build_rf(x, y, sample_weight, n_jobs):
        model = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=0)
        return positive_proba(model.fit(x, y, sample_weight=sample_weight))

    if __name__ == "__main__":
        df = pd.read_csv("examples/data/stroke_pre/stroke.csv")
        search = BiasSearch(df, "stroke", ["age", "gender"])
        print(search.evaluate_average(iterate=2, model_name="rf")[1])

Output (columns as above): ``age`` raw 4.41 / normalized 1.0, ``gender`` raw 0.77 /
normalized 0.43.

Interpretation
--------------

* A small privileged group makes scores noisy: on the WiDS data the privileged ``patient_age``
  group has only 105 rows. Report the group sizes shown in the table.
* Use ``iterate >= 10`` for reported results; 2-3 iterations are for smoke tests.
* The normalized score saturates at 1.0, so rank attributes by the raw score when several
  are at 1.0.
* Scores are model-dependent; compare attributes using the same model and ``random_state``.
