Grid Search Guide
================

This guide provides comprehensive information about using the grid search functionality
in CallMeFair to systematically evaluate different combinations of bias mitigation techniques.

Overview
--------

The grid search functionality in CallMeFair allows you to systematically evaluate
multiple combinations of bias mitigation techniques and compare their effectiveness.
This is particularly useful for:

* **Research**: Comparing different bias mitigation approaches
* **Optimization**: Finding the best combination for your specific use case
* **Reproducibility**: Ensuring systematic evaluation of all techniques
* **Analysis**: Understanding trade-offs between different approaches

Key Features
-----------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Feature
     - Description
   * - **Automatic Model Adaptation**
     - Supports various ML frameworks (sklearn, XGBoost, TabNet)
   * - **Comprehensive Evaluation**
     - Evaluates baseline and all bias mitigation combinations
   * - **Automatic Logging**
     - Results automatically saved to CSV files
   * - **Single Sensitive Attribute**
     - Currently supports single sensitive attribute evaluation
   * - **Extensible Design**
     - Framework ready for multiple sensitive attributes

Basic Usage
-----------

Here's a simple example of how to use the grid search functionality:

.. code-block:: python

   from callmefair.mitigation.fair_grid import BMGridSearch
   from callmefair.mitigation.fair_bm import BMType
   from sklearn.ensemble import RandomForestClassifier
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

   # Define bias mitigation combinations to test
   bm_combinations = [
       [BMType.preReweighing],  # Only preprocessing
       [BMType.preDisparate],   # Only disparate impact remover
       [BMType.preReweighing, BMType.posCEO],  # Preprocessing + postprocessing
       [BMType.inAdversarial],  # Only in-processing
   ]

   # Initialize grid search
   grid_search = BMGridSearch(
       bmI=bm_interface,
       model=RandomForestClassifier(),
       bm_list=bm_combinations,
       privileged_group=privileged_groups,
       unprivileged_group=unprivileged_groups
   )

   # Run comprehensive evaluation
   grid_search.run_single_sensitive()

Defining Bias Mitigation Combinations
-----------------------------------

The grid search evaluates combinations of bias mitigation techniques. Each combination
is defined as a list of `BMType` enums:

.. code-block:: python

   from callmefair.mitigation.fair_bm import BMType

   # Single technique combinations
   single_techniques = [
       [BMType.preReweighing],
       [BMType.preDisparate],
       [BMType.preLFR],
       [BMType.inAdversarial],
       [BMType.inMeta],
       [BMType.posCEO],
       [BMType.posEO],
       [BMType.posROC]
   ]

   # Multi-technique combinations
   multi_techniques = [
       [BMType.preReweighing, BMType.posCEO],  # Preprocessing + postprocessing
       [BMType.preDisparate, BMType.posEO],    # Preprocessing + postprocessing
       [BMType.preLFR, BMType.posROC],         # Preprocessing + postprocessing
       [BMType.preReweighing, BMType.preDisparate, BMType.posCEO]  # Complex
   ]

   # In-processing combinations (exclusive)
   in_processing = [
       [BMType.inAdversarial],  # Only in-processing
       [BMType.inMeta],         # Only in-processing
   ]

Supported Model Types
--------------------

The grid search automatically adapts to different machine learning frameworks:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Framework
     - Model Examples
   * - **scikit-learn**
     - RandomForestClassifier, LogisticRegression, MLPClassifier
   * - **XGBoost**
     - XGBClassifier
   * - **TabNet**
     - TabNetClassifier
   * - **Custom Models**
     - Any model with fit() and predict_proba() methods

Example with different models:

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier
   from sklearn.linear_model import LogisticRegression
   from xgboost import XGBClassifier

   # Test with different models
   models = [
       RandomForestClassifier(n_estimators=100),
       LogisticRegression(),
       XGBClassifier()
   ]

   for model in models:
       grid_search = BMGridSearch(
           bmI=bm_interface,
           model=model,
           bm_list=bm_combinations,
           privileged_group=privileged_groups,
           unprivileged_group=unprivileged_groups
       )
       grid_search.run_single_sensitive()

Understanding Results
-------------------

The grid search automatically logs results to CSV files with the following information:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Column
     - Description
   * - **model**
     - Name of the machine learning model used
   * - **BM**
     - Bias mitigation combination applied
   * - **SPD**
     - Statistical Parity Difference
   * - **EOD**
     - Equalized Odds Difference
   * - **AOD**
     - Average Odds Difference
   * - **DI**
     - Disparate Impact
   * - **TI**
     - Theil Index
   * - **fair_score**
     - Overall fairness score (0-1, lower is better)

Example result analysis:

.. code-block:: python

   import pandas as pd

   # Load results
   results = pd.read_csv('./results/experiment_2024_01_15.csv')

   # Find best performing combination
   best_fairness = results.loc[results['fair_score'].idxmin()]
   print(f"Best fairness score: {best_fairness['fair_score']:.4f}")
   print(f"Combination: {best_fairness['BM']}")

   # Compare preprocessing vs postprocessing
   pre_only = results[results['BM'].str.contains('pre') & ~results['BM'].str.contains('pos')]
   pos_only = results[results['BM'].str.contains('pos') & ~results['BM'].str.contains('pre')]
   
   print(f"Average fairness score (preprocessing only): {pre_only['fair_score'].mean():.4f}")
   print(f"Average fairness score (postprocessing only): {pos_only['fair_score'].mean():.4f}")

Advanced Usage
-------------

Custom Model Integration
~~~~~~~~~~~~~~~~~~~~~~~

You can integrate custom models by ensuring they have the required interface:

.. code-block:: python

   class CustomModel:
       def __init__(self):
           self.model = RandomForestClassifier()
       
       def fit(self, X, y, **kwargs):
           return self.model.fit(X, y, **kwargs)
       
       def predict_proba(self, X):
           return self.model.predict_proba(X)
       
       def __str__(self):
           return "CustomModel()"

   # Use custom model in grid search
   grid_search = BMGridSearch(
       bmI=bm_interface,
       model=CustomModel(),
       bm_list=bm_combinations,
       privileged_group=privileged_groups,
       unprivileged_group=unprivileged_groups
   )

In-Processing Only Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For in-processing techniques, you can set the model to None:

.. code-block:: python

   # Evaluate only in-processing techniques
   in_processing_combinations = [
       [BMType.inAdversarial],
       [BMType.inMeta]
   ]

   grid_search = BMGridSearch(
       bmI=bm_interface,
       model=None,  # No external model needed
       bm_list=in_processing_combinations,
       privileged_group=privileged_groups,
       unprivileged_group=unprivileged_groups
   )

Result Aggregation
~~~~~~~~~~~~~~~~~

For large-scale experiments, you can aggregate results from multiple runs:

.. code-block:: python

   from callmefair.mitigation.fair_log import aggregate_csv_files

   # Aggregate results from multiple experiments
   aggregate_csv_files(
       folder_path='./results/',
       output_file='./results/aggregated_results.csv',
       num_processes=8
   )

Best Practices
-------------

1. **Systematic Evaluation**
   - Start with single techniques before combining
   - Include baseline (no bias mitigation) for comparison
   - Test both preprocessing and postprocessing approaches

2. **Model Selection**
   - Use multiple model types for robustness
   - Consider computational requirements
   - Test with your specific data characteristics

3. **Combination Design**
   - Avoid incompatible combinations (e.g., multiple in-processing)
   - Consider the order of techniques
   - Test both simple and complex combinations

4. **Result Analysis**
   - Look at both fairness and accuracy metrics
   - Consider the trade-off between fairness and performance
   - Analyze which techniques work best for your specific case

5. **Reproducibility**
   - Use fixed random seeds
   - Document all experimental parameters
   - Save raw results for later analysis

Troubleshooting
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Import Errors**
   - Ensure all required dependencies are installed
   - Check that model types are supported

**Memory Issues**
   - Use smaller datasets for initial testing
   - Reduce the number of combinations
   - Use more efficient model types

**In-Processing Errors**
   - Ensure TensorFlow is installed for adversarial debiasing
   - Check that sensitive attributes are properly defined
   - Verify that groups are correctly specified

**Result Interpretation**
   - Check that fairness metrics are in expected ranges
   - Verify that baseline performance is reasonable
   - Ensure that improvements are statistically significant

Future Enhancements
------------------

The grid search functionality is designed for future enhancements:

* **Multiple Sensitive Attributes**: Support for intersectional fairness
* **Hyperparameter Optimization**: Automatic tuning of technique parameters
* **Advanced Model Support**: Integration with more ML frameworks
* **Parallel Processing**: Distributed evaluation across multiple machines
* **Interactive Analysis**: Web-based result visualization

For more advanced usage, see the :doc:`../api/mitigation` documentation. 