Trade-off Map Guide
===================

The FairHealthGrid trade-off map places each bias-mitigation result (balanced accuracy,
composite fairness score) relative to a FAIREA-style baseline: the accuracy/fairness curve of
a base model whose predictions are randomly overwritten at increasing degrees (0 to 1). The
map answers whether a mitigation is better than simply degrading the model.

.. note::

   This code lives in ``examples/Fairea/fairea.py``, not in the installed ``callmefair``
   package. Run from a clone of the repository with ``examples/`` on ``sys.path`` and install
   the ``examples`` extra (provides ``shapely``); see :doc:`/user_guide/installation`.

Steps
-----

1. ``create_baseline(clf_name, dataset, privileged_groups, unprivileged_groups,
   data_splits, repetitions, odds, degrees)`` trains the classifier (``lr``, ``dt``,
   ``svm``, ``bayes``, ``tabular``, ``cat``) on ``data_splits`` random splits, overwrites a
   fraction ``degree`` of the test predictions (``odds={"1": [0, 1]}`` overwrites with
   label 1), and records ``[balanced_acc, spd, aod, fair_score]`` per repetition.
   ``fair_score`` is the callmefair composite score (0 = fair, higher = more biased).
2. ``normalize(base_acc, base_fair, methods)`` rescales baseline and methods to [0, 1];
   accuracy is inverted (0 = most accurate), so the origin is the ideal corner.
3. ``classify_region(baseline_line, normalized_methods)`` returns one region per method
   (paper names in brackets):

   * ``"win-win"`` [Win-Win]: improves on the baseline in both accuracy and fairness;
   * ``"good trade-off"`` [Good]: on the favorable side of the baseline curve (a fairness gain that
     costs less accuracy than degrading the model would);
   * ``"bad trade-off"`` [Poor]: on the unfavorable side of the curve;
   * ``"lose-lose"`` [Lose-Lose]: worse than the baseline in both;
   * ``"inverted"`` [Inverted]: fallback for points matching none of the above.

   The assignment is geometric: vertical and horizontal rays from the point are tested for
   intersection with the baseline polyline (see ``classify_region``).

Example
-------

Baseline on the stroke data (protected attribute ``age``), then mitigation results taken from
a grid-search CSV: one ``(balanced_acc, fair_score)`` pair per method. Use ``fair_score`` as
is, so baseline and methods share the same orientation.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset
    from shapely.geometry import LineString
    from Fairea.fairea import create_baseline, normalize, classify_region  # examples/ on sys.path

    data = pd.read_csv("examples/data/stroke_pre/stroke.csv")
    bld = BinaryLabelDataset(df=data, label_names=["stroke"], protected_attribute_names=["age"],
                             favorable_label=1, unfavorable_label=0)
    degrees = [0, 0.2, 0.4, 0.6, 0.8, 1]
    res = create_baseline("lr", bld, [{"age": 1}], [{"age": 0}], data_splits=3, repetitions=3,
                          odds={"1": [0, 1]}, degrees=degrees)
    base_acc = np.array([np.mean([r[0] for r in res["1"][d]]) for d in degrees])
    base_fair = np.array([np.mean([r[3] for r in res["1"][d]]) for d in degrees])

    grid = pd.read_csv("examples/script/results/aggregated_stroke_age.csv")
    grid = grid[grid.BM.isin(["preReweighing", "preDisparate", "inAD"])]
    methods = {f"{r.model}+{r.BM}": (r.balanced_acc, r.fair_score) for r in grid.itertuples()}

    norm_acc, norm_fair, norm_methods = normalize(base_acc, base_fair, methods)
    line = LineString(list(zip(norm_fair, norm_acc)))
    regions = classify_region(line, norm_methods)
    print(pd.Series(regions).value_counts().to_string())
    print(pd.Series(regions).loc[["lr+preReweighing", "xgb+preDisparate"]].to_string())

.. code-block:: text

    win-win           8
    good trade-off    5
    bad trade-off     1
    lr+preReweighing    bad trade-off
    xgb+preDisparate          win-win

This is a reduced run (3 splits x 3 repetitions); use the notebook defaults
(25 splits x 10 repetitions) for real analyses. Many rows in the shipped CSVs have
``fair_score`` exactly 1 (the saturated maximum); such points sit on the edge of the
normalized square, so treat their region labels with caution.

Using your own grid results
---------------------------

Run ``callmefair.mitigation.fair_grid`` results through the same steps: each row of the
output has ``balanced_acc`` and ``fair_score``; build ``methods = {name: (balanced_acc,
fair_score)}`` and pass it to ``normalize``. The full workflow, including plots, is in
``examples/notebooks/FAIREA_CFS_Example.ipynb`` and ``fairea_plot_F.ipynb``.
``examples/Fairea/fairea.py`` also has ``compute_area(baseline_line, (acc, fair))`` for the
area between a good-trade-off point and the baseline; it raised errors on several of my test
points, so wrap it in ``try``/``except``.

.. warning::

   The grid results in ``examples/script/results/aggregated_*.csv`` were produced before the
   post-processing fix (see :doc:`/changelog`); rows whose ``BM`` contains ``pos`` are not
   valid. ``FAIREA_CFS_Example.ipynb`` also plots methods with ``1 - fair_score`` while its
   baseline uses ``fair_score``; use the same orientation for both.
