Utility Functions Guide
======================

This guide provides comprehensive information about the utility functions and
interfaces in CallMeFair, with special focus on the `calculate_fairness_score`
function and the `BMInterface` class.

Overview
--------

The utility module provides core functionality for bias mitigation operations:

* **Fairness Score Calculation**: Aggregates multiple fairness metrics into a single score
* **Dataset Management**: Unified interface for managing train/validation/test splits
* **Binary Label Dataset Creation**: AIF360 compatibility for bias mitigation
* **Feature Scaling**: Consistent scaling across all datasets
* **Comprehensive Metrics**: Both classification and fairness metrics evaluation

Key Components
-------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Component
     - Description
   * - **calculate_fairness_score**
     - Aggregates 5 fairness metrics into a single normalized score
   * - **BMInterface**
     - Main interface for dataset management and bias mitigation operations
   * - **BMMetrics**
     - Comprehensive evaluation of classification and fairness metrics
   * - **BMnames**
     - Configuration data class for bias mitigation attributes

Fairness Score Calculation
-------------------------

The `calculate_fairness_score` function is a core component that aggregates
five key fairness metrics into a single normalized score representing overall
model fairness.

Supported Metrics
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Metric
     - Optimal Value
     - Acceptable Range
     - Description
   * - **EOD** (Equal Opportunity Difference)
     - 0.0
     - (-0.1, 0.1)
     - Difference in true positive rates between groups
   * - **AOD** (Average Odds Difference)
     - 0.0
     - (-0.1, 0.1)
     - Difference in average of TPR and FPR between groups
   * - **SPD** (Statistical Parity Difference)
     - 0.0
     - (-0.1, 0.1)
     - Difference in positive prediction rates between groups
   * - **DI** (Disparate Impact)
     - 1.0
     - (0.8, 1.2)
     - Ratio of positive prediction rates between groups
   * - **TI** (Theil Index)
     - 0.0
     - (0.0, 0.25)
     - Inequality in prediction distributions

Scoring Algorithm
~~~~~~~~~~~~~~~~

The function uses a weighted scoring system:

1. **Range Penalty**: Each metric contributes up to 0.2 for being outside acceptable ranges
2. **Deviation Contribution**: Each metric contributes up to 0.16 based on deviation from optimal values
3. **Normalization**: Final score is normalized to 0-1 range where 0 = perfect fairness

Example Usage:

.. code-block:: python

   from callmefair.util.fair_util import calculate_fairness_score

   # Perfect fairness example
   perfect_result = calculate_fairness_score(
       EOD=0.0, AOD=0.0, SPD=0.0, DI=1.0, TI=0.0
   )
   print(f"Perfect fairness score: {perfect_result['overall_score']}")  # 0.0
   print(f"Is fair: {perfect_result['is_fair']}")  # True

   # Moderate unfairness example
   moderate_result = calculate_fairness_score(
       EOD=0.15, AOD=0.12, SPD=0.18, DI=0.7, TI=0.3
   )
   print(f"Moderate unfairness score: {moderate_result['overall_score']}")  # ~0.6-0.8
   print(f"Is fair: {moderate_result['is_fair']}")  # False

   # Check individual metric evaluations
   for metric, is_acceptable in moderate_result['metric_evaluations'].items():
       status = "✓" if is_acceptable else "✗"
       print(f"{metric}: {status}")

Result Interpretation
~~~~~~~~~~~~~~~~~~~

The function returns a comprehensive dictionary:

.. code-block:: python

   result = calculate_fairness_score(EOD=0.05, AOD=0.03, SPD=0.08, DI=0.95, TI=0.12)

   # Access different components
   print(f"Overall score: {result['overall_score']}")  # Normalized 0-1 score
   print(f"Raw score: {result['raw_score']}")  # Unnormalized score
   print(f"Is fair: {result['is_fair']}")  # Boolean fairness assessment

   # Check individual metric evaluations
   for metric, is_acceptable in result['metric_evaluations'].items():
       print(f"{metric}: {'Acceptable' if is_acceptable else 'Unacceptable'}")

   # Check deviations from optimal values
   for metric, deviation in result['deviations'].items():
       print(f"{metric} deviation: {deviation:.3f}")

Dataset Management with BMInterface
---------------------------------

The `BMInterface` class provides a unified interface for managing datasets
and bias mitigation operations. It handles dataset splitting, feature scaling,
and provides access to data in various formats required by different bias
mitigation techniques.

Basic Usage
~~~~~~~~~~

.. code-block:: python

   from callmefair.util.fair_util import BMInterface
   import pandas as pd

   # Load your datasets
   train_df = pd.read_csv('train.csv')
   val_df = pd.read_csv('val.csv')
   test_df = pd.read_csv('test.csv')

   # Initialize the interface
   bm_interface = BMInterface(
       df_train=train_df,
       df_val=val_df,
       df_test=test_df,
       label='income',
       protected=['gender', 'race']
   )

   # Get data in different formats
   train_bld = bm_interface.get_train_BLD()  # AIF360 format
   X_train, y_train = bm_interface.get_train_xy()  # (features, labels) tuple

Data Access Methods
~~~~~~~~~~~~~~~~~~

The interface provides multiple ways to access your data:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Method
     - Description
   * - **get_train_BLD()**
     - Get training data as BinaryLabelDataset (AIF360 format)
   * - **get_val_BLD()**
     - Get validation data as BinaryLabelDataset
   * - **get_test_BLD()**
     - Get test data as BinaryLabelDataset
   * - **get_train_xy()**
     - Get training data as (features, labels) tuple
   * - **get_val_xy()**
     - Get validation data as (features, labels) tuple
   * - **get_test_xy()**
     - Get test data as (features, labels) tuple

Feature Scaling
~~~~~~~~~~~~~~

The interface supports automatic feature scaling:

.. code-block:: python

   # Enable transform mode for feature scaling
   bm_interface.set_transform()

   # Get scaled features
   X_train_scaled, y_train = bm_interface.get_train_xy()
   X_test_scaled, y_test = bm_interface.get_test_xy()

   # Restore original data
   bm_interface.restore_BLD()

Bias Mitigation Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

The interface integrates seamlessly with bias mitigation techniques:

.. code-block:: python

   from callmefair.mitigation.fair_bm import BMManager

   # Define groups
   privileged_groups = [{'gender': 1, 'race': 1}]
   unprivileged_groups = [{'gender': 0, 'race': 0}]

   # Create bias mitigation manager
   bm_manager = BMManager(bm_interface, privileged_groups, unprivileged_groups)

   # Apply preprocessing bias mitigation
   bm_manager.pre_Reweighing()

   # Get modified training data
   modified_train_bld = bm_interface.get_train_BLD()

   # Restore original data for next experiment
   bm_interface.restore_BLD()

Comprehensive Metrics Evaluation
------------------------------

The `BMMetrics` class provides comprehensive evaluation of both classification
performance and fairness metrics.

Basic Usage
~~~~~~~~~~

.. code-block:: python

   from callmefair.util.fair_util import BMMetrics
   import numpy as np

   # Create metrics evaluator
   metrics = BMMetrics(
       bmI=bm_interface,
       class_array=np.array([0, 1]),
       pred_val=val_predictions,
       pred_test=test_predictions,
       privileged_group=privileged_groups,
       unprivileged_group=unprivileged_groups
   )

   # Get comprehensive report
   report = metrics.get_report()
   print(f"Accuracy: {report['acc']:.4f}")
   print(f"Statistical Parity Difference: {report['spd']:.4f}")

   # Get fairness score
   score_dict = metrics.get_score()
   print(f"Overall fairness score: {score_dict['overall_score']:.3f}")

Supported Metrics
~~~~~~~~~~~~~~~~

Classification Metrics:
- **Accuracy**: Overall classification accuracy
- **Balanced Accuracy**: Average of true positive and true negative rates
- **Precision**: Precision score
- **Recall**: Recall score
- **F1 Score**: Harmonic mean of precision and recall
- **MCC**: Matthews Correlation Coefficient

Fairness Metrics:
- **EOD**: Equal Opportunity Difference
- **AOD**: Average Odds Difference
- **SPD**: Statistical Parity Difference
- **DI**: Disparate Impact
- **TI**: Theil Index

Threshold Optimization
~~~~~~~~~~~~~~~~~~~~~

The class automatically finds the optimal classification threshold:

.. code-block:: python

   # The class automatically finds optimal threshold
   print(f"Optimal threshold: {metrics.best_class_thresh:.3f}")

   # Update predictions and recalculate
   new_val_pred = model.predict_proba(X_val)
   new_test_pred = model.predict_proba(X_test)
   metrics.set_new_pred(new_val_pred, new_test_pred)

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

   # Use with BMInterface
   custom_model = CustomModel()
   X_train, y_train = bm_interface.get_train_xy()
   custom_model.fit(X_train, y_train)

Multiple Protected Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The interface supports multiple protected attributes:

.. code-block:: python

   # Initialize with multiple protected attributes
   bm_interface = BMInterface(
       df_train=train_df,
       df_val=val_df,
       df_test=test_df,
       label='income',
       protected=['gender', 'race', 'age_group']
   )

   # Define intersectional groups
   privileged_groups = [
       {'gender': 1, 'race': 1},
       {'gender': 1, 'race': 2}
   ]
   unprivileged_groups = [
       {'gender': 0, 'race': 0},
       {'gender': 0, 'race': 1}
   ]

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~

For large datasets, consider these optimizations:

.. code-block:: python

   # Use transform mode for consistent scaling
   bm_interface.set_transform()

   # Batch processing for large datasets
   batch_size = 1000
   for i in range(0, len(X_train), batch_size):
       batch_X = X_train[i:i+batch_size]
       batch_y = y_train[i:i+batch_size]
       # Process batch

Best Practices
-------------

1. **Data Preparation**
   - Ensure consistent data types across train/validation/test
   - Handle missing values appropriately
   - Normalize categorical variables

2. **Protected Attributes**
   - Clearly define all protected attributes
   - Ensure consistent encoding across datasets
   - Document group definitions for reproducibility

3. **Feature Scaling**
   - Use transform mode for models requiring scaled features
   - Always restore data between experiments
   - Document scaling parameters

4. **Metrics Evaluation**
   - Use both classification and fairness metrics
   - Consider the trade-off between accuracy and fairness
   - Validate results with multiple evaluation methods

5. **Reproducibility**
   - Set random seeds for consistent results
   - Document all experimental parameters
   - Save intermediate results for analysis

Troubleshooting
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Import Errors**
   - Ensure all dependencies are installed
   - Check AIF360 compatibility
   - Verify pandas and numpy versions

**Data Format Issues**
   - Ensure consistent column names across datasets
   - Check data types for protected attributes
   - Verify label encoding (0/1 for binary classification)

**Scaling Issues**
   - Use restore_BLD() after bias mitigation operations
   - Check for NaN values before scaling
   - Ensure consistent scaling across all datasets

**Metrics Calculation**
   - Verify group definitions are correct
   - Check that predictions are in the right format
   - Ensure sufficient samples in each group

For more advanced usage, see the :doc:`../api/util` documentation. 