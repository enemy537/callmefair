CallMeFair Documentation
========================

Welcome to the CallMeFair documentation! CallMeFair is a comprehensive framework for automatic bias mitigation in AI systems. This framework provides tools and techniques to identify, measure, and reduce algorithmic bias in machine learning models.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   user_guide/installation
   user_guide/quickstart
   user_guide/examples

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/overview
   user_guide/bias_mitigation_guide
   user_guide/evaluation_guide
   user_guide/grid_search_guide
   user_guide/utility_guide
   user_guide/bias_search_guide

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/mitigation
   api/util
   api/search

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/bias_mitigation
   theory/fairness_metrics
   theory/evaluation_methods

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/custom_mitigation
   advanced/performance_optimization
   advanced/deployment

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   contributing/development_guide
   contributing/code_of_conduct
   contributing/roadmap

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

What is CallMeFair?
-------------------

CallMeFair is an open-source framework designed to help researchers and practitioners implement bias mitigation techniques in machine learning systems. The framework provides:

* **Comprehensive Bias Mitigation**: Support for preprocessing, in-processing, and postprocessing techniques
* **Multiple Algorithms**: Implementation of state-of-the-art bias mitigation methods
* **Easy Integration**: Simple API for integrating bias mitigation into existing workflows
* **Evaluation Tools**: Built-in metrics and evaluation methods for fairness assessment
* **Extensible Architecture**: Framework for adding custom bias mitigation techniques

Key Features
-----------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Feature
     - Description
   * - **Preprocessing Methods**
     - Reweighing, Disparate Impact Remover, Learning Fair Representations
   * - **In-processing Methods**
     - Adversarial Debiasing, MetaFair Classifier
   * - **Postprocessing Methods**
     - Calibrated Equalized Odds, Equalized Odds, Reject Option Classification
   * - **Evaluation Metrics**
     - Statistical Parity Difference, Equalized Odds Difference, Theil Index
   * - **Search Framework**
     - Automatic discovery of optimal bias mitigation strategies
   * - **Grid Search**
     - Systematic evaluation of bias mitigation combinations

Quick Start
-----------

Here's a simple example of how to use CallMeFair:

.. code-block:: python

   from callmefair.util.fair_util import BMInterface
   from callmefair.mitigation.fair_bm import BMManager
   import pandas as pd

   # Load your data
   train_df = pd.read_csv('train.csv')
   val_df = pd.read_csv('val.csv')
   test_df = pd.read_csv('test.csv')

   # Initialize the interface
   bm_interface = BMInterface(train_df, val_df, test_df, 'label', ['gender'])

   # Define groups
   privileged_groups = [{'gender': 1}]
   unprivileged_groups = [{'gender': 0}]

   # Create bias mitigation manager
   bm_manager = BMManager(bm_interface, privileged_groups, unprivileged_groups)

   # Apply preprocessing bias mitigation
   bm_manager.pre_Reweighing()

   # Apply in-processing bias mitigation
   ad_model = bm_manager.in_AD(debias=True)

   # Apply postprocessing bias mitigation
   mitigated_predictions = bm_manager.pos_CEO(valid_pred, test_pred)

Installation
-----------

Install CallMeFair using pip:

.. code-block:: bash

   pip install callmefair

Or install from source:

.. code-block:: bash

   git clone https://github.com/your-repo/callmefair.git
   cd callmefair
   pip install -e .

Dependencies
-----------

CallMeFair requires the following dependencies:

* Python 3.8+
* pandas
* numpy
* scikit-learn
* aif360
* tensorflow (optional, for adversarial debiasing)

Citation
--------

If you use CallMeFair in your research, please cite:

.. code-block:: bibtex

   @article{callmefair2024,
     title={CallMeFair: A Comprehensive Framework for Bias Mitigation in AI Systems},
     author={Your Name and Co-authors},
     journal={arXiv preprint},
     year={2024}
   }

Support
-------

* **Documentation**: This site contains comprehensive documentation
* **GitHub Issues**: Report bugs and request features on GitHub
* **Discussions**: Join community discussions on GitHub Discussions
* **Email**: Contact the development team at support@callmefair.org

License
-------

CallMeFair is released under the MIT License. See the LICENSE file for details. 