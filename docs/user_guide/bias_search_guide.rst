Bias Search Guide
================

The bias search functionality in CallMeFair provides comprehensive tools for evaluating bias in machine learning models with respect to sensitive attributes. This guide covers the core search modules and their usage.

Overview
--------

The bias search framework consists of two main modules:

- ``_search_base.py``: Core functionality for bias evaluation and model training
- ``fair_search.py``: High-level interface for comprehensive bias analysis

The framework supports:

- Individual attribute bias evaluation
- Attribute combination analysis (2-way and 3-way combinations)
- Multiple set operations (union, intersection, differences)
- Various ML models (Logistic Regression, CatBoost, XGBoost, MLP)
- Parallel processing for efficient evaluation
- Pretty table output for results

Core Classes
-----------

BaseSearch
~~~~~~~~~~

The ``BaseSearch`` class provides the foundation for bias evaluation:

.. code-block:: python

    from callmefair.search._search_base import BaseSearch
    
    # Initialize with dataset and target variable
    searcher = BaseSearch(df, 'target')
    
    # Evaluate bias for a specific attribute
    results = searcher.evaluate_attribute('gender', iterate=10, model_name='lr')

Key Methods:

- ``evaluate_attribute()``: Evaluate bias for a single attribute
- ``__pre_attribute_bias()``: Prepare datasets for evaluation
- ``__predict_attribute_bias()``: Compute fairness metrics

BiasSearch
~~~~~~~~~~

The ``BiasSearch`` class extends BaseSearch with comprehensive analysis capabilities:

.. code-block:: python

    from callmefair.search.fair_search import BiasSearch
    
    # Initialize with multiple attributes
    searcher = BiasSearch(df, 'target', ['gender', 'race', 'age'])
    
    # Evaluate individual attributes
    table, printable = searcher.evaluate_average()
    
    # Evaluate attribute combinations
    table, printable = searcher.evaluate_combinations()

Key Methods:

- ``evaluate_average()``: Evaluate all individual attributes
- ``evaluate_combinations()``: Evaluate 2-way and 3-way combinations
- ``evaluate_combination_average()``: Compare different set operations

Attribute Combination Operations
-----------------------------

The framework supports various set operations for combining sensitive attributes:

Union (OR)
~~~~~~~~~~
Combines attributes using logical OR operation:

.. code-block:: python

    from callmefair.search._search_base import CType, combine_attributes
    
    # gender OR race (either attribute is 1)
    result_df = combine_attributes(df, 'gender', 'race', CType.union)

Intersection (AND)
~~~~~~~~~~~~~~~~~
Combines attributes using logical AND operation:

.. code-block:: python

    # gender AND race (both attributes are 1)
    result_df = combine_attributes(df, 'gender', 'race', CType.intersection)

Set Differences
~~~~~~~~~~~~~~
Computes set differences between attributes:

.. code-block:: python

    # gender - race (gender=1 AND race=0)
    result_df = combine_attributes(df, 'gender', 'race', CType.difference_1_minus_2)
    
    # race - gender (race=1 AND gender=0)
    result_df = combine_attributes(df, 'gender', 'race', CType.difference_2_minus_1)

Symmetric Difference (XOR)
~~~~~~~~~~~~~~~~~~~~~~~~~
Combines attributes using XOR operation:

.. code-block:: python

    # gender XOR race (exactly one attribute is 1)
    result_df = combine_attributes(df, 'gender', 'race', CType.symmetric_difference)

Supported Models
---------------

The framework supports multiple machine learning models:

Logistic Regression
~~~~~~~~~~~~~~~~~~
Fast and interpretable model for bias evaluation:

.. code-block:: python

    results = searcher.evaluate_attribute('gender', model_name='lr')

CatBoost
~~~~~~~~
Gradient boosting model with optimized parameters:

.. code-block:: python

    results = searcher.evaluate_attribute('gender', model_name='cat')

XGBoost
~~~~~~~
Advanced gradient boosting with balanced parameters:

.. code-block:: python

    results = searcher.evaluate_attribute('gender', model_name='xgb')

Multi-layer Perceptron
~~~~~~~~~~~~~~~~~~~~~
Neural network with adaptive learning:

.. code-block:: python

    results = searcher.evaluate_attribute('gender', model_name='mlp')

Usage Examples
-------------

Individual Attribute Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate bias for individual sensitive attributes:

.. code-block:: python

    from callmefair.search.fair_search import BiasSearch
    
    # Initialize searcher
    searcher = BiasSearch(df, 'target', ['gender', 'race', 'age'])
    
    # Evaluate all attributes
    table, printable = searcher.evaluate_average(iterate=10, model_name='lr')
    print(printable)

Attribute Combinations
~~~~~~~~~~~~~~~~~~~~~

Evaluate bias for attribute combinations:

.. code-block:: python

    # Evaluate 2-way and 3-way combinations
    table, printable = searcher.evaluate_combinations()
    print(printable)

Set Operation Comparison
~~~~~~~~~~~~~~~~~~~~~~~

Compare different ways of combining attributes:

.. code-block:: python

    # Compare all set operations between gender and race
    table, printable = searcher.evaluate_combination_average('gender', 'race')
    print(printable)

Advanced Usage
-------------

Custom Dataset Preparation
~~~~~~~~~~~~~~~~~~~~~~~~

Use custom datasets for evaluation:

.. code-block:: python

    # Use modified dataset
    modified_df = df.copy()
    modified_df['new_feature'] = some_transformation(modified_df)
    
    results = searcher.evaluate_attribute('gender', df_new=modified_df)

Handling Class Imbalance
~~~~~~~~~~~~~~~~~~~~~~~

Apply NearMiss undersampling for imbalanced datasets:

.. code-block:: python

    # Apply class balancing
    results = searcher.evaluate_attribute('gender', treat_umbalance=True)

Parallel Processing
~~~~~~~~~~~~~~~~~~

The framework automatically uses parallel processing for certain models:

.. code-block:: python

    # Logistic Regression and MLP use multiprocessing
    results = searcher.evaluate_attribute('gender', model_name='lr')  # Parallel
    
    # CatBoost and XGBoost use sequential processing
    results = searcher.evaluate_attribute('gender', model_name='cat')  # Sequential

Output Interpretation
-------------------

Fairness Scores
~~~~~~~~~~~~~~

The framework computes two types of fairness scores:

- **Raw Score**: Direct fairness metric value
- **Overall Score**: Normalized fairness score (0-1 scale)

Higher scores indicate better fairness (less bias).

Result Tables
~~~~~~~~~~~~

Results are presented in pretty tables with columns:

- **Attribute**: Name of the sensitive attribute or combination
- **Raw Fairness Score**: Direct fairness metric value
- **Normalized Fairness Score**: Normalized score (0-1)

Example Output:

.. code-block:: text

    +----------+---------------------+------------------------+
    | Attribute| Raw Fairness Score | Normalized Fairness   |
    |          |                    | score                 |
    +----------+---------------------+------------------------+
    | gender   | 0.85               | 0.92                  |
    | race     | 0.72               | 0.78                  |
    | age      | 0.91               | 0.95                  |
    +----------+---------------------+------------------------+

Best Practices
-------------

1. **Multiple Iterations**: Use at least 5-10 iterations for robust results
2. **Model Selection**: Start with Logistic Regression for interpretability
3. **Attribute Combinations**: Use intersection for most meaningful combinations
4. **Class Balancing**: Apply NearMiss for highly imbalanced datasets
5. **Parallel Processing**: Use 'lr' or 'mlp' models for faster processing

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**Binary Attributes Required**: All sensitive attributes must be binary (0 or 1)

.. code-block:: python

    # Convert categorical to binary
    df['gender'] = (df['gender'] == 'male').astype(int)

**Memory Issues**: Reduce iterations or use smaller datasets

.. code-block:: python

    # Use fewer iterations
    results = searcher.evaluate_attribute('gender', iterate=5)

**Slow Performance**: Use parallel models or reduce dataset size

.. code-block:: python

    # Use Logistic Regression for speed
    results = searcher.evaluate_attribute('gender', model_name='lr')

Performance Tips
~~~~~~~~~~~~~~~

1. Use Logistic Regression for quick prototyping
2. Apply class balancing only when necessary
3. Use parallel processing for large datasets
4. Consider feature scaling for better model performance
5. Cache results for repeated evaluations

Integration with Other Modules
----------------------------

The bias search functionality integrates with other CallMeFair modules:

- **Bias Mitigation**: Use search results to identify which attributes need mitigation
- **Grid Search**: Combine with bias mitigation techniques
- **Utilities**: Use fairness score calculation from fair_util

.. code-block:: python

    from callmefair.search.fair_search import BiasSearch
    from callmefair.bm import BMManager
    
    # Identify bias
    searcher = BiasSearch(df, 'target', ['gender', 'race'])
    table, printable = searcher.evaluate_average()
    
    # Apply mitigation
    bm_manager = BMManager()
    mitigated_df = bm_manager.apply_mitigation(df, 'reweighing', 'gender') 