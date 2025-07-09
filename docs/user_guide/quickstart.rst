Quick Start Guide
================

This guide will help you get started with CallMeFair, a comprehensive framework for bias mitigation in AI systems. You'll learn how to install the framework, load your data, and apply various bias mitigation techniques.

Installation
-----------

First, install CallMeFair and its dependencies:

.. code-block:: bash

   pip install callmefair

Or install from source:

.. code-block:: bash

   git clone https://github.com/your-repo/callmefair.git
   cd callmefair
   pip install -e .

Basic Usage
----------

Here's a simple example that demonstrates the core functionality:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from callmefair.util.fair_util import BMInterface
   from callmefair.mitigation.fair_bm import BMManager

   # Load your data
   # Replace with your actual data files
   train_df = pd.read_csv('train.csv')
   val_df = pd.read_csv('val.csv')
   test_df = pd.read_csv('test.csv')

   # Initialize the bias mitigation interface
   bm_interface = BMInterface(
       df_train=train_df,
       df_val=val_df,
       df_test=test_df,
       label='target',  # Your target column name
       protected=['gender']  # Your protected attributes
   )

   # Define privileged and unprivileged groups
   privileged_groups = [{'gender': 1}]  # Male group
   unprivileged_groups = [{'gender': 0}]  # Female group

   # Create bias mitigation manager
   bm_manager = BMManager(
       bmI=bm_interface,
       privileged_group=privileged_groups,
       unprivileged_group=unprivileged_groups
   )

   # Apply preprocessing bias mitigation
   bm_manager.pre_Reweighing()

   # Train a model (example with sklearn)
   from sklearn.ensemble import RandomForestClassifier
   
   X_train, y_train = bm_interface.get_train_xy()
   model = RandomForestClassifier()
   model.fit(X_train, y_train)

   # Make predictions
   X_test, y_test = bm_interface.get_test_xy()
   predictions = model.predict(X_test)

   # Apply postprocessing bias mitigation
   from aif360.datasets import BinaryLabelDataset
   
   # Convert predictions to BinaryLabelDataset format
   test_pred_dataset = bm_interface.get_test_BLD()
   test_pred_dataset.labels = predictions.reshape(-1, 1)
   
   val_pred_dataset = bm_interface.get_val_BLD()
   val_pred_dataset.labels = model.predict(bm_interface.get_val_xy()[0]).reshape(-1, 1)

   # Apply Calibrated Equalized Odds postprocessing
   mitigated_predictions = bm_manager.pos_CEO(
       valid_BLD_pred=val_pred_dataset,
       test_BLD_pred=test_pred_dataset
   )

Available Bias Mitigation Techniques
----------------------------------

CallMeFair supports three categories of bias mitigation techniques:

Preprocessing Techniques
~~~~~~~~~~~~~~~~~~~~~~~

Applied to training data before model training:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Method
     - Description
   * - **Reweighing**
     - Assigns different weights to instances based on group membership
   * - **Disparate Impact Remover (DIR)**
     - Repairs training data to remove disparate impact
   * - **Learning Fair Representations (LFR)**
     - Learns fair representations that remove sensitive information

.. code-block:: python

   # Apply preprocessing techniques
   bm_manager.pre_Reweighing()  # Reweighing
   bm_manager.pre_DR('gender')  # Disparate Impact Remover
   bm_manager.pre_LFR()         # Learning Fair Representations

In-processing Techniques
~~~~~~~~~~~~~~~~~~~~~~~

Applied during model training:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Method
     - Description
   * - **Adversarial Debiasing**
     - Trains classifier with adversarial fairness constraint
   * - **MetaFair Classifier**
     - Uses meta-learning for fair classification

.. code-block:: python

   # Apply in-processing techniques
   ad_model = bm_manager.in_AD(debias=True)  # Adversarial Debiasing
   meta_model = bm_manager.in_Meta('gender', tau=0.1)  # MetaFair

Postprocessing Techniques
~~~~~~~~~~~~~~~~~~~~~~~~

Applied to model predictions after training:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Method
     - Description
   * - **Calibrated Equalized Odds (CEO)**
     - Adjusts predictions for equalized odds with calibration
   * - **Equalized Odds (EO)**
     - Adjusts predictions for equalized odds
   * - **Reject Option Classification (ROC)**
     - Rejects uncertain predictions to improve fairness

.. code-block:: python

   # Apply postprocessing techniques
   mitigated_ceo = bm_manager.pos_CEO(valid_pred, test_pred)
   mitigated_eo = bm_manager.pos_EO(valid_pred, test_pred)
   mitigated_roc = bm_manager.pos_ROC(valid_pred, test_pred)

Evaluation and Metrics
---------------------

CallMeFair provides comprehensive evaluation tools:

.. code-block:: python

   from callmefair.util.fair_util import BMMetrics

   # Create metrics evaluator
   metrics = BMMetrics(
       bmI=bm_interface,
       class_array=np.array([0, 1]),  # Class labels
       pred_val=val_pred_dataset,
       pred_test=test_pred_dataset,
       privileged_group=privileged_groups,
       unprivileged_group=unprivileged_groups
   )

   # Get comprehensive fairness report
   report = metrics.get_report()
   print("Fairness Report:")
   print(f"Statistical Parity Difference: {report['spd']:.4f}")
   print(f"Equalized Odds Difference: {report['eq_opp_diff']:.4f}")
   print(f"Average Odds Difference: {report['avg_odd_diff']:.4f}")
   print(f"Disparate Impact: {report['disparate_impact']:.4f}")
   print(f"Theil Index: {report['theil_idx']:.4f}")

   # Get overall fairness score
   score_dict = metrics.get_score()
   print(f"Overall Fairness Score: {score_dict['overall_score']:.4f}")
   print(f"Is Fair: {score_dict['is_fair']}")

Fairness Score Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~

The `calculate_fairness_score` function aggregates multiple fairness metrics:

.. code-block:: python

   from callmefair.util.fair_util import calculate_fairness_score

   # Calculate fairness score from individual metrics
   fairness_result = calculate_fairness_score(
       EOD=0.05,    # Equal Opportunity Difference
       AOD=0.03,    # Average Odds Difference
       SPD=0.08,    # Statistical Parity Difference
       DI=0.95,     # Disparate Impact
       TI=0.12      # Theil Index
   )

   print(f"Overall fairness score: {fairness_result['overall_score']}")
   print(f"Is fair: {fairness_result['is_fair']}")

   # Check individual metric evaluations
   for metric, is_acceptable in fairness_result['metric_evaluations'].items():
       status = "✓" if is_acceptable else "✗"
       print(f"{metric}: {status}")

Advanced Usage
-------------

Bias Search and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~

CallMeFair provides comprehensive bias search functionality to identify and evaluate
bias in your datasets and models:

.. code-block:: python

   from callmefair.search.fair_search import BiasSearch
   import pandas as pd

   # Initialize bias search with multiple sensitive attributes
   searcher = BiasSearch(df, 'target', ['gender', 'race', 'age'])

   # Evaluate individual attributes
   table, printable = searcher.evaluate_average(iterate=10, model_name='lr')
   print("Individual Attribute Bias:")
   print(printable)

   # Evaluate attribute combinations (2-way and 3-way)
   table, printable = searcher.evaluate_combinations()
   print("Attribute Combination Bias:")
   print(printable)

   # Compare different set operations between attributes
   table, printable = searcher.evaluate_combination_average('gender', 'race')
   print("Set Operation Comparison:")
   print(printable)

Grid Search for Bias Mitigation Combinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CallMeFair provides a comprehensive grid search framework for systematically
evaluating different combinations of bias mitigation techniques:

.. code-block:: python

   from callmefair.mitigation.fair_grid import BMGridSearch
   from callmefair.mitigation.fair_bm import BMType
   from sklearn.ensemble import RandomForestClassifier

   # Define bias mitigation combinations to test
   bm_combinations = [
       [BMType.preReweighing],  # Only preprocessing
       [BMType.preDisparate],   # Only disparate impact remover
       [BMType.preReweighing, BMType.posCEO],  # Preprocessing + postprocessing
       [BMType.inAdversarial],  # Only in-processing
       [BMType.preLFR, BMType.posEO]  # Complex combination
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
   # Results are automatically logged to CSV files

Combining Multiple Techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can combine different bias mitigation techniques manually:

.. code-block:: python

   # Apply preprocessing
   bm_manager.pre_Reweighing()
   
   # Train model with in-processing
   ad_model = bm_manager.in_AD(debias=True)
   ad_model.fit(X_train, y_train)
   
   # Apply postprocessing
   mitigated_predictions = bm_manager.pos_CEO(valid_pred, test_pred)

Custom Evaluation
~~~~~~~~~~~~~~~~

Create custom evaluation scenarios:

.. code-block:: python

   # Evaluate before bias mitigation
   original_metrics = BMMetrics(...)
   original_report = original_metrics.get_report()
   
   # Apply bias mitigation
   bm_manager.pre_Reweighing()
   
   # Evaluate after bias mitigation
   mitigated_metrics = BMMetrics(...)
   mitigated_report = mitigated_metrics.get_report()
   
   # Compare results
   improvement = {
       'SPD': original_report['SPD'] - mitigated_report['SPD'],
       'EOD': original_report['EOD'] - mitigated_report['EOD'],
       'AOD': original_report['AOD'] - mitigated_report['AOD']
   }
   print("Improvement in fairness metrics:", improvement)

Best Practices
-------------

1. **Data Preparation**
   - Ensure your data is properly split into train/validation/test sets
   - Identify all protected attributes in your dataset
   - Handle missing values appropriately

2. **Group Definition**
   - Clearly define privileged and unprivileged groups
   - Consider intersectional fairness (multiple protected attributes)
   - Document your group definitions for reproducibility

3. **Technique Selection**
   - Start with preprocessing techniques for simplicity
   - Use in-processing for better performance when possible
   - Apply postprocessing as a final fairness adjustment

4. **Evaluation**
   - Always evaluate both accuracy and fairness metrics
   - Use multiple fairness metrics for comprehensive assessment
   - Consider the trade-off between accuracy and fairness

5. **Validation**
   - Use cross-validation for robust evaluation
   - Test on multiple datasets when possible
   - Document your experimental setup

Troubleshooting
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

**Data Format Issues**
   - Ensure your DataFrame has the correct column names
   - Verify that protected attributes are properly encoded
   - Check that target variable is binary (0/1)

**Memory Issues**
   - Use smaller batch sizes for large datasets
   - Consider data sampling for initial experimentation
   - Use efficient data structures (numpy arrays instead of lists)

**Fairness Metrics**
   - Ensure groups are properly defined
   - Check for sufficient samples in each group
   - Verify that predictions are in the correct format

Next Steps
----------

Now that you've completed the quick start guide, you can:

1. **Explore the API Reference**: Learn about all available classes and methods
2. **Read the Theory Guide**: Understand the mathematical foundations
3. **Try the Examples**: Work through comprehensive examples
4. **Contribute**: Help improve the framework

For more advanced usage, see the :doc:`../user_guide/bias_mitigation_guide` and :doc:`../user_guide/evaluation_guide`. 