callmefair
==========

FairHealthGrid (package ``callmefair``) evaluates bias mitigation strategies
for healthcare machine learning. It searches sensitive attributes and their
intersections for bias, runs a grid search over model and bias-mitigation
combinations, and scores each with a composite fairness score 𝓕 built from
EOD, AOD, SPD, DI and TI. A trade-off map then places every combination in
a region: Win-Win, Good, Poor, Inverted or Lose-Lose.

* Bias search models: ``lr``, ``xgb``, ``cat``, ``mlp``, ``lgbm``, ``tabnet``, ``tabtransformer``.
* Grid search models: any scikit-learn compatible classifier, XGBoost, CatBoost, LightGBM, TabNet, TabTransformer.
* Mitigation (``BMType``): ``preReweighing``, ``preDisparate``, ``preLFR``, ``inAdversarial``, ``inMeta``, ``posCalibrated``, ``posEqqOds``, ``posROC``.

Quick example
-------------

.. code-block:: python

   import pandas as pd
   from callmefair import BiasSearch

   df = pd.read_csv("examples/data/stroke_pre/stroke.csv")
   search = BiasSearch(df, "stroke", ["age", "gender"])
   table, pretty = search.evaluate_average(iterate=5)
   print(pretty)

Install with ``uv pip install "callmefair[dl]"``; see :doc:`user_guide/installation`.
In fairness scores, 0 is perfectly fair and higher means more biased.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   user_guide/installation
   user_guide/quickstart
   user_guide/examples

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user_guide/bias_search_guide
   user_guide/grid_search_guide
   user_guide/tradeoff_map_guide
   user_guide/visualization_guide
   user_guide/utility_guide
   user_guide/hardware_guide

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog

Citation
--------

.. code-block:: bibtex

   @inproceedings{paiva2025fairhealthgrid,
     title     = {FairHealthGrid: A Systematic Framework for Evaluating Bias Mitigation Strategies in Healthcare Machine Learning},
     author    = {Paiva, Pedro and Dehghani, Farzaneh and Anzum, Fahim and Singhal, Mansi and Metwali, Ayah and Gavrilova, Marina and Bento, Mariana},
     booktitle = {31st Americas Conference on Information Systems (AMCIS 2025)},
     publisher = {Association for Information Systems},
     year      = {2025},
     url       = {https://aisel.aisnet.org/amcis2025/intelfuture/intelfuture/46/}
   }

License
-------

MIT; see the ``LICENSE`` file in the
`repository <https://github.com/uofc-ai2lab/callmefair/>`_.
