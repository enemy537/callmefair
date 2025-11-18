"""
Bias Search Base Module

This module provides the foundational classes and utilities for bias search functionality
in the CallMeFair framework. It implements attribute-based bias evaluation, model training,
and fairness metric calculation for individual and combined sensitive attributes.

The module supports:
- Individual attribute bias evaluation
- Attribute combination operations (union, intersection, differences)
- Multiple ML model training (Logistic Regression, CatBoost, XGBoost, MLP)
- Fairness metric calculation using AIF360
- Parallel processing for efficient evaluation

Classes:
    CType: Enumeration of attribute combination operations
    BaseSearch: Base class for bias search functionality

Functions:
    combine_attributes: Combine two binary columns using set operations
    wrapper_training: Train ML models for bias evaluation
    wrapper: Multiprocessing wrapper for model training

Example:
    >>> from callmefair.search._search_base import BaseSearch
    >>> searcher = BaseSearch(df, 'target')
    >>> results = searcher.evaluate_attribute('gender', iterate=5)
"""

from enum import Enum
from callmefair.util.fair_util import calculate_fairness_score
from collections import defaultdict
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss
import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
# Threading for memory-efficient parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed
from callmefair.search.optimizer import TrainingOptimizer
# Suppress FutureWarning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Classifiers
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

# Deep learning backend for MLP (GPU-capable when available)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Optional models: LightGBM
try:
    from lightgbm import LGBMClassifier  # https://github.com/microsoft/LightGBM
except Exception:
    LGBMClassifier = None


class KerasMLPWrapper:
    """Minimal sklearn-like wrapper around a Keras binary MLP.

    Exposes predict_proba and classes_ so it can be used by the
    existing bias evaluation pipeline.
    """

    def __init__(self, keras_model, classes_=None):
        self.model = keras_model
        if classes_ is None:
            classes_ = np.array([0, 1])
        self.classes_ = np.asarray(classes_)

    def predict_proba(self, X):
        # Keras model outputs P(y=1); convert to [P(y=0), P(y=1)]
        preds_pos = self.model.predict(X, verbose=0).reshape(-1, 1)
        preds_neg = 1.0 - preds_pos
        return np.hstack([preds_neg, preds_pos])


class CType(Enum):
    """
    Enumeration of attribute combination operations for bias search.
    
    This enum defines the set operations that can be performed when combining
    two binary sensitive attributes to create composite protected groups.
    
    Attributes:
        union: Logical OR operation (either attribute is 1)
        intersection: Logical AND operation (both attributes are 1)
        difference_1_minus_2: Set difference (attribute1=1 AND attribute2=0)
        difference_2_minus_1: Set difference (attribute2=1 AND attribute1=0)
        symmetric_difference: XOR operation (exactly one attribute is 1)
    """
    union = 1
    intersection = 2
    difference_1_minus_2 = 3
    difference_2_minus_1 = 4
    symmetric_difference = 5

    def __str__(self):
        """Return string representation of the operation type."""
        return super().__str__().split('.')[1]

def combine_attributes(df, col1, col2, operation: CType):
    """
    Combines two binary columns in a DataFrame using a specified set operation,
    replacing the original columns with a single combined column.

    This function creates composite protected groups by combining two binary
    sensitive attributes using set operations. The resulting combined attribute
    can be used for more sophisticated bias analysis.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the binary columns
        col1 (str): Name of the first binary column (e.g., 'gender')
        col2 (str): Name of the second binary column (e.g., 'race')
        operation (CType): Set operation to apply ('union', 'intersection', 
                          'difference_1_minus_2', 'difference_2_minus_1', 
                          'symmetric_difference')

    Returns:
        pd.DataFrame: New DataFrame with original columns replaced by combined column

    Raises:
        ValueError: If columns are not binary (contain values other than 0 or 1)

    Example:
        >>> df = pd.DataFrame({'gender': [1, 0, 1, 0], 'race': [1, 1, 0, 0]})
        >>> result = combine_attributes(df, 'gender', 'race', CType.intersection)
        >>> print(result.columns)
        ['gender_race']
    """
    # Check if columns are binary (0 or 1)
    if not all(df[col].isin([0, 1]).all() for col in [col1, col2]):
        raise ValueError("Columns must contain only binary values (0 or 1).")

    # Compute the combined column based on the operation
    if operation == CType.union:                   # and
        combined = df[col1] | df[col2]
    elif operation == CType.intersection:          # or
        combined = df[col1] & df[col2]
    elif operation == CType.difference_1_minus_2:  # col1 - col2
        combined = df[col1] & ~df[col2]
    elif operation == CType.difference_2_minus_1:  # col2 - col1
        combined = df[col2] & ~df[col1]
    elif operation == CType.symmetric_difference:  # xor
        combined = df[col1] ^ df[col2]

    # Create a new DataFrame, dropping original columns and adding the combined column
    new_col_name = f"{col1}_{col2}"
    df_new = df.drop([col1, col2], axis=1).assign(**{new_col_name: combined})

    return df_new

def wrapper_training(train_bld:BinaryLabelDataset,
                     val_bld: BinaryLabelDataset,
                     test_bld:BinaryLabelDataset,
                     attribute:str, model_name:str = 'lr'):
    """
    Train a machine learning model for bias evaluation on a specific attribute.
    
    This function handles the training of different ML models for bias evaluation.
    It supports multiple model types with optimized hyperparameters for fairness
    analysis. The function is designed to work with the multiprocessing wrapper
    for parallel training.

    Parameters:
        train_bld (BinaryLabelDataset): Training dataset with protected attributes
        val_bld (BinaryLabelDataset): Validation dataset for threshold optimization
        test_bld (BinaryLabelDataset): Test dataset for final evaluation
        attribute (str): Name of the sensitive attribute being evaluated
        model_name (str): Type of model to train ('lr', 'mlp', 'xgb', 'cat', 'lgbm', 'tabtransformer')

    Returns:
        tuple: (attribute_name, trained_model)

    Supported Models:
        - 'lr': Logistic Regression with liblinear solver
        - 'cat': CatBoost with optimized parameters for fairness
        - 'xgb': XGBoost with balanced parameters
        - 'mlp': Multi-layer Perceptron with adaptive learning
        - 'lgbm': LightGBM Gradient Boosted Trees (CPU)
        - 'tabtransformer': TabTransformer with attention mechanism for tabular data

    Example:
        >>> result = wrapper_training(train_bld, val_bld, test_bld, 'gender', 'lr')
        >>> attribute, model = result
    """
    try:
        scaler = StandardScaler()
        scaler.fit(train_bld.features)

        x_train = scaler.transform(train_bld.features)
        y_train = train_bld.labels.ravel()

        if model_name == 'lr':
            model = LogisticRegression(solver='liblinear')
            model = model.fit(x_train, y_train, sample_weight=train_bld.instance_weights)

        elif model_name == 'cat':
            model = CatBoostClassifier(
                eval_metric='Accuracy',
                depth=4,
                learning_rate=0.01,
                iterations=10,
                thread_count=1,  # Single thread per worker to avoid oversubscription
                verbose=False)
            model = model.fit(x_train, y_train)

        elif model_name == 'xgb':
            model =  XGBClassifier(
                max_depth=8,
                learning_rate=0.01,
                gamma = 0.25,
                n_estimators = 500,
                subsample = 0.8,
                colsample_bytree = 0.3,
                n_jobs=1)  # Use single thread per model to avoid conflicts with threading
            model = model.fit(x_train, y_train)

        elif model_name == 'mlp':
            # GPU-capable MLP using Keras. Uses a small network and
            # relatively few epochs since we mainly need sufficient
            # accuracy for fairness evaluation, not full convergence.
            input_dim = x_train.shape[1]

            keras_model = keras.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid'),
            ])

            keras_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss='binary_crossentropy',
                metrics=['accuracy'],
            )

            # Train on CPU or GPU depending on the available backend
            keras_model.fit(
                x_train,
                y_train,
                sample_weight=train_bld.instance_weights,
                epochs=50,
                batch_size=1024,
                verbose=0,
            )

            # Wrap into sklearn-like interface expected by the rest
            # of the pipeline (predict_proba, classes_ attribute).
            model = KerasMLPWrapper(keras_model, classes_=np.array([0, 1]))

        elif model_name == 'lgbm':
            if LGBMClassifier is None:
                raise ImportError("LightGBM not installed. Please `pip install lightgbm` to use 'lgbm'.")

            # Create DataFrame so the model is fitted with explicit feature names
            import pandas as pd
            feature_names = [f'feature_{i}' for i in range(x_train.shape[1])]
            x_train_df = pd.DataFrame(x_train, columns=feature_names)

            model = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary',
                n_jobs=1,
                verbose=-1
            )
            model = model.fit(x_train_df, y_train, sample_weight=train_bld.instance_weights)
            # Store feature names on the model so we can reuse them at prediction time
            model._cmf_feature_names = feature_names

        elif model_name == 'tabtransformer':
            try:
                from callmefair.util.models import TabTransformer
            except ImportError:
                raise ImportError("TabTransformer not available. Please ensure callmefair.util.models is properly installed.")

            # Build a compact DataFrame for TabTransformer with a dedicated target column
            import pandas as pd
            feature_names = [f'feature_{i}' for i in range(train_bld.features.shape[1])]
            x_train_df = pd.DataFrame(train_bld.features, columns=feature_names)
            y_series = pd.Series(train_bld.labels.ravel(), name='target')
            df_tab = pd.concat([x_train_df, y_series], axis=1)

            # Minimal config tuned for bias search runs; TabTransformer handles internal optimization
            config = {
                "batch_size": 1024,
                "epochs": 20,
                "learning_rate": 1e-3,
                "verbose": 0
            }

            model = TabTransformer(config=config, target='target')
            model.fit(df_tab)
            # Mark model so prediction pipeline can handle TabTransformer-specific inputs
            model._cmf_is_tabtransformer = True
            model._cmf_feature_names = feature_names
            model._cmf_target_name = 'target'
            # Expose classes_ to align with sklearn-style API (binary case: {0,1})
            model.classes_ = np.array([0, 1])
        
        # Clean up training data to free memory
        del x_train, y_train, scaler
        
    except Exception as e:
        # Clean up on error
        if 'x_train' in locals():
            del x_train
        if 'y_train' in locals():
            del y_train
        if 'scaler' in locals():
            del scaler
        raise e

    return attribute, model

def wrapper(args):
    """
    Threading wrapper for model training.
    
    This function is used by ThreadPoolExecutor to parallelize model training
    across multiple threads. It unpacks the arguments and calls wrapper_training.

    Parameters:
        args (tuple): Packed arguments for wrapper_training

    Returns:
        tuple: Result from wrapper_training
    """
    try:
        result = wrapper_training(*args)
        # Clean up args to free memory
        del args
        return result
    except Exception as e:
        # Clean up on error
        del args
        raise e

def process_wrapper_training(df_new: pd.DataFrame, label_name: str, attribute: str, treat_umbalance: bool, model_name: str):
    """
    Process-safe wrapper for model training.

    Builds datasets inside the worker process to avoid pickling complex objects,
    then delegates to wrapper_training.

    Args:
        df_new (pd.DataFrame): Dataset to use for training and evaluation splits
        label_name (str): Name of the target label column
        attribute (str): Sensitive attribute to evaluate
        treat_umbalance (bool): Whether to apply NearMiss undersampling
        model_name (str): Model identifier ('lr', 'mlp', 'xgb', 'cat', 'lgbm')

    Returns:
        tuple: (attribute_name, trained_model)
    """
    # Arguments provided directly; construct datasets in the worker

    # Split with stratification on attribute and label
    df_train, df_test = train_test_split(
        df_new, test_size=0.3,
        stratify=df_new[[attribute, label_name]],
        random_state=42
    )
    df_test, df_val = train_test_split(
        df_test, test_size=0.5,
        stratify=df_test[[attribute, label_name]],
        random_state=42
    )

    # Optional NearMiss undersampling
    if treat_umbalance:
        nm = NearMiss()
        X_nearmiss, y_nearmiss = nm.fit_resample(
            df_train.drop(columns=[label_name]), df_train[label_name])
        df_train = pd.DataFrame(
            X_nearmiss, columns=df_train.drop(columns=[label_name]).columns
        )
        df_train[label_name] = y_nearmiss

    sensitive_attribute = [attribute]
    train_bld = BinaryLabelDataset(
        df=df_train, label_names=[label_name],
        protected_attribute_names=sensitive_attribute,
        favorable_label=1, unfavorable_label=0
    )
    val_bld = BinaryLabelDataset(
        df=df_val, label_names=[label_name],
        protected_attribute_names=sensitive_attribute,
        favorable_label=1, unfavorable_label=0
    )
    test_bld = BinaryLabelDataset(
        df=df_test, label_names=[label_name],
        protected_attribute_names=sensitive_attribute,
        favorable_label=1, unfavorable_label=0
    )

    return wrapper_training(train_bld, val_bld, test_bld, attribute, model_name)

class BaseSearch:
    """
    Base class for bias search functionality in the CallMeFair framework.
    
    This class provides the core functionality for evaluating bias in machine learning
    models with respect to sensitive attributes. It handles dataset preparation,
    model training, and fairness metric calculation using AIF360.

    The class supports:
    - Individual attribute bias evaluation
    - Multiple ML model types
    - Imbalanced dataset handling with NearMiss
    - Parallel processing for efficient evaluation
    - Comprehensive fairness metrics calculation

    Attributes:
        df (pd.DataFrame): Input dataset with features and target
        label_name (str): Name of the target variable
        scaler (StandardScaler): Feature scaler for model training
        n_threads (int): Number of threads for parallel processing

    Example:
        >>> searcher = BaseSearch(df, 'target')
        >>> results = searcher.evaluate_attribute('gender', iterate=10, model_name='lr')
    """

    def __init__(self, df: pd.DataFrame, label_name: str, n_threads: int = None, backend: str = 'process'):
        """
        Initialize the BaseSearch object.

        Parameters:
            df (pd.DataFrame): Input dataset containing features and target variable
            label_name (str): Name of the target variable column
            n_threads (int, optional): Number of threads for parallel processing.
                                     If None, defaults to min(8, os.cpu_count())
        """
        self.df = df.copy(deep=True)
        self.label_name = label_name
        self.scaler = StandardScaler()
        
        # Set default number of threads if not specified
        if n_threads is None:
            self.n_threads = min(8, os.cpu_count())
        else:
            self.n_threads = max(1, int(n_threads))  # Ensure at least 1 thread

        # Select parallel backend ('thread' or 'process') and optimizer
        self.backend = backend
        self.optimizer = TrainingOptimizer(backend=self.backend, max_workers=self.n_threads)

        # Avoid BLAS/OpenMP oversubscription when using processes
        if self.backend == 'process':
            for key in ['MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                os.environ.setdefault(key, '1')

    def __pre_attribute_bias(self, attribute, apply_nearmiss=False, df_new = None):
        """
        Prepare datasets for bias evaluation on a specific attribute.
        
        This method handles the data preprocessing pipeline for bias evaluation:
        - Splits data into train/validation/test sets with stratification
        - Applies NearMiss undersampling if requested
        - Converts to AIF360 BinaryLabelDataset format
        - Sets up protected attribute groups

        Parameters:
            attribute (str): Name of the sensitive attribute to evaluate
            apply_nearmiss (bool): Whether to apply NearMiss undersampling
            df_new (pd.DataFrame, optional): Alternative dataset to use

        Returns:
            tuple: (train_bld, val_bld, test_bld) - AIF360 datasets for training
        """
        if df_new is None:
            df_new = self.df

        sensitive_attribute = [attribute]

        # Check if we have enough samples for stratification
        # Count samples for each combination of attribute and label
        stratify_cols = [attribute, self.label_name]
        combination_counts = df_new.groupby(stratify_cols).size()
        min_samples = combination_counts.min()
        
        # If any combination has fewer than 2 samples, we can't stratify on both
        if min_samples < 2:
            print(f"Warning: Some combinations of {attribute} and {self.label_name} have fewer than 2 samples.")
            print(f"Minimum samples in any combination: {min_samples}")
            print("Falling back to stratification by label only.")
            
            # Try stratifying by label only
            try:
                df_train, df_test = train_test_split(
                    df_new, test_size=0.3, 
                    stratify=df_new[self.label_name],
                    random_state=42
                )
                
                df_test, df_val = train_test_split(
                    df_test, test_size=0.5,
                    stratify=df_test[self.label_name],
                    random_state=42
                )
            except ValueError as e:
                print(f"Warning: Cannot stratify by label either: {e}")
                print("Using random split without stratification.")
                
                # Fall back to random split without stratification
                df_train, df_test = train_test_split(
                    df_new, test_size=0.3, 
                    random_state=42
                )
                
                df_test, df_val = train_test_split(
                    df_test, test_size=0.5,
                    random_state=42
                )
        else:
            # Normal case: stratify by both attribute and label
            try:
                df_train, df_test = train_test_split(
                    df_new, test_size=0.3, 
                    stratify=df_new[[attribute, self.label_name]],
                    random_state=42
                )
                
                df_test, df_val = train_test_split(
                    df_test, test_size=0.5,
                    stratify=df_test[[attribute, self.label_name]],
                    random_state=42
                )
            except ValueError as e:
                print(f"Warning: Stratification failed: {e}")
                print("Falling back to stratification by label only.")
                
                # Fall back to label-only stratification
                try:
                    df_train, df_test = train_test_split(
                        df_new, test_size=0.3, 
                        stratify=df_new[self.label_name],
                        random_state=42
                    )
                    
                    df_test, df_val = train_test_split(
                        df_test, test_size=0.5,
                        stratify=df_test[self.label_name],
                        random_state=42
                    )
                except ValueError as e2:
                    print(f"Warning: Cannot stratify by label either: {e2}")
                    print("Using random split without stratification.")
                    
                    # Final fallback: random split
                    df_train, df_test = train_test_split(
                        df_new, test_size=0.3, 
                        random_state=42
                    )
                    
                    df_test, df_val = train_test_split(
                        df_test, test_size=0.5,
                        random_state=42
                    )

        if apply_nearmiss:
            nm = NearMiss()
            X_nearmiss, y_nearmiss = nm.fit_resample(
                df_train.drop(columns=[self.label_name]), df_train[self.label_name])
            df_train = pd.DataFrame(
                X_nearmiss, columns = df_train.drop(columns=[self.label_name]).columns
                )
            df_train[self.label_name] = y_nearmiss

        train_bld = BinaryLabelDataset(
            df=df_train, label_names=[self.label_name],
            protected_attribute_names=sensitive_attribute,
            favorable_label=1, unfavorable_label=0
        )
        val_bld = BinaryLabelDataset(
            df=df_val, label_names=[self.label_name],
            protected_attribute_names=sensitive_attribute,
            favorable_label=1, unfavorable_label=0
        )
        test_bld = BinaryLabelDataset(
            df=df_test, label_names=[self.label_name], 
            protected_attribute_names=sensitive_attribute,
            favorable_label=1, unfavorable_label=0
        )

        return train_bld, val_bld, test_bld

        
    def __predict_attribute_bias(self,train_bld:BinaryLabelDataset,
                                    val_bld: BinaryLabelDataset,
                                    test_bld:BinaryLabelDataset,
                                    model,
                                    attribute):
        """
        Evaluate bias metrics for a trained model on a specific attribute.
        
        This method performs comprehensive bias evaluation by:
        - Optimizing classification threshold on validation set
        - Computing fairness metrics on test set
        - Calculating multiple fairness measures (SPD, DI, EOD, AOD, Theil)
        - Returning aggregated fairness scores

        Parameters:
            train_bld (BinaryLabelDataset): Training dataset
            val_bld (BinaryLabelDataset): Validation dataset for threshold optimization
            test_bld (BinaryLabelDataset): Test dataset for final evaluation
            model: Trained machine learning model
            attribute (str): Name of the sensitive attribute

        Returns:
            dict: Dictionary containing raw and overall fairness scores
        """
        privileged_group = [{attribute: 1}]
        unprivileged_group = [{attribute: 0}]

        # Prepare validation/test probabilities depending on model type.
        # TabTransformer consumes DataFrames and owns its own preprocessing,
        # while the other models operate on scaled numpy feature matrices.
        if hasattr(model, '_cmf_is_tabtransformer') and getattr(model, '_cmf_is_tabtransformer'):
            import pandas as pd
            feature_names = getattr(model, '_cmf_feature_names')
            target_name = getattr(model, '_cmf_target_name', 'target')

            # Rebuild DataFrames with the same feature names and target column
            x_val_raw = val_bld.features
            x_test_raw = test_bld.features
            y_val_raw = val_bld.labels.ravel()
            y_test_raw = test_bld.labels.ravel()

            val_df = pd.DataFrame(x_val_raw, columns=feature_names)
            val_df[target_name] = y_val_raw
            test_df = pd.DataFrame(x_test_raw, columns=feature_names)
            test_df[target_name] = y_test_raw

            # TabTransformer.predict_proba returns positive-class probabilities
            import numpy as np
            proba_val_pos = np.asarray(model.predict_proba(val_df)).ravel()
            proba_test_pos = np.asarray(model.predict_proba(test_df)).ravel()

            # Construct 2-column probability matrices [P(class=0), P(class=1)]
            proba_val = np.vstack([1.0 - proba_val_pos, proba_val_pos]).T
            proba_test = np.vstack([1.0 - proba_test_pos, proba_test_pos]).T

            classes_ = getattr(model, 'classes_', np.array([0, 1]))
        else:
            self.scaler.fit(train_bld.features)

            x_val = self.scaler.transform(val_bld.features)
            x_test = self.scaler.transform(test_bld.features)

            # For LightGBM models trained on a DataFrame with feature names, wrap
            # validation and test features in a DataFrame using the same names to
            # avoid feature-name mismatch warnings.
            if LGBMClassifier is not None and isinstance(model, LGBMClassifier) and hasattr(model, '_cmf_feature_names'):
                import pandas as pd
                feature_names = getattr(model, '_cmf_feature_names')
                x_val = pd.DataFrame(x_val, columns=feature_names)
                x_test = pd.DataFrame(x_test, columns=feature_names)

            import numpy as np
            classes_ = model.classes_
            proba_val = model.predict_proba(x_val)
            proba_test = model.predict_proba(x_test)

        pos_idx = np.where(classes_ == train_bld.favorable_label)[0][0]

        valid_bld_pred = val_bld.copy(deepcopy=True)
        valid_bld_pred.scores = proba_val[:, pos_idx].reshape(-1, 1)

        num_thresh = 100
        balanced_acc = np.zeros(num_thresh)
        class_threshold = np.linspace(0.01, 0.99, num_thresh)

        for idx, class_thresh in enumerate(class_threshold):

            fav_idx = valid_bld_pred.scores > class_thresh
            valid_bld_pred.labels[fav_idx] = valid_bld_pred.favorable_label
            valid_bld_pred.labels[~fav_idx] = valid_bld_pred.unfavorable_label

            # computing metrics based on two BinaryLabelDatasets: a dataset containing groud-truth labels and a dataset containing predictions
            classified_metric_orig_valid = ClassificationMetric(val_bld,
                                                                valid_bld_pred,
                                                                unprivileged_groups=unprivileged_group,
                                                                privileged_groups=privileged_group)

            balanced_acc[idx] = 0.5 * (classified_metric_orig_valid.true_positive_rate() + classified_metric_orig_valid.true_negative_rate())

        best_idx = np.where(balanced_acc == np.max(balanced_acc))[0][0]
        best_class_thresh = class_threshold[best_idx]

        test_bld_pred = test_bld.copy(deepcopy=True)
        test_bld_pred.scores = proba_test[:, pos_idx].reshape(-1, 1)

        for thresh in class_threshold:

            fav_idx = test_bld_pred.scores > thresh
            test_bld_pred.labels[fav_idx] = test_bld_pred.favorable_label
            test_bld_pred.labels[~fav_idx] = test_bld_pred.unfavorable_label

            classification_metric_orig_test = ClassificationMetric(test_bld,
                                                                test_bld_pred,
                                                                unprivileged_groups=unprivileged_group,
                                                                privileged_groups=privileged_group)

            spd = classification_metric_orig_test.statistical_parity_difference()
            disparate_impact = classification_metric_orig_test.disparate_impact()
            eq_opp_diff = classification_metric_orig_test.equal_opportunity_difference()
            
            # Handle potential errors in avg_odd_diff calculation due to exploding FPR/TPR
            try:
                avg_odd_diff = classification_metric_orig_test.average_odds_difference()
            except (ValueError, ZeroDivisionError, RuntimeError) as e:
                # Set worst possible value for avg_odd_diff when calculation fails
                # This indicates maximum unfairness (far from optimal value of 0.0)
                avg_odd_diff = 1.0
                print(f"Warning: avg_odd_diff calculation failed in grid search, using worst-case value 1.0. Error: {e}")
            
            theil_idx = classification_metric_orig_test.theil_index()

            if thresh == best_class_thresh:
                return calculate_fairness_score(eq_opp_diff, avg_odd_diff, spd, disparate_impact, theil_idx)


    def evaluate_attribute(self, 
                             attribute,
                             treat_umbalance=False,
                             iterate=10,
                             model_name:str = 'lr',
                             df_new = None) -> dict:
        """
        Evaluate bias for a specific attribute across multiple iterations.
        
        This method performs comprehensive bias evaluation by:
        - Running multiple iterations for statistical robustness
        - Training models with optional class balancing
        - Using parallel processing for efficiency
        - Aggregating results across iterations

        Parameters:
            attribute (str): Name of the sensitive attribute to evaluate
            treat_umbalance (bool): Whether to apply NearMiss undersampling
            iterate (int): Number of iterations for robust evaluation
            model_name (str): Type of model to use ('lr', 'mlp', 'xgb', 'cat', 'lgbm')
            df_new (pd.DataFrame, optional): Alternative dataset to use

        Returns:
            dict: Dictionary containing averaged fairness scores for the attribute

        Note:
            The number of threads for parallel execution is set during object initialization.

        Example:
            >>> searcher = BaseSearch(df, 'target', n_threads=8)
            >>> results = searcher.evaluate_attribute('gender', iterate=5, model_name='lr')
            >>> print(results['gender_raw'], results['gender_overall'])
        """
        if df_new is None:
            df_new = self.df

        # Determine workers to use; allow more workers than iterations
        n_threads = max(1, self.n_threads)

        # Memory-efficient approach: process data in smaller batches
        wrp_out = []
        
        # Initialize single progress bar for the entire execution
        with tqdm(total=iterate, desc=f"Training {model_name} models") as pbar:
            # TabTransformer and Keras-based MLP should run sequentially to
            # allow their internal optimization and avoid TF + multiprocessing
            # issues. Other models can use parallel execution when beneficial.
            if iterate > 1 and n_threads > 1 and model_name not in ('tabtransformer', 'mlp'):
                local_backend = self.backend

                if local_backend == 'process':
                    # Build lightweight args; datasets constructed inside workers
                    args_list = [
                        (df_new, self.label_name, attribute, treat_umbalance, model_name)
                        for _ in range(iterate)
                    ]
                    pbar.set_description(f"Training {model_name} models ({n_threads} processes)")
                    try:
                        # Use a local optimizer honoring the local_backend decision
                        optimizer = TrainingOptimizer(backend=local_backend, max_workers=n_threads)
                        for result in optimizer.map(process_wrapper_training, args_list, max_workers=n_threads):
                            wrp_out.append(result)
                            pbar.update(1)
                    except Exception as e:
                        import traceback
                        print("\n[Error] Parallel training failed (process backend)")
                        print(f"Attribute: {attribute} | Model: {model_name} | Iterate: {iterate} | Workers: {n_threads}")
                        print(f"Exception: {type(e).__name__}: {e}")
                        print('Traceback:\n' + ''.join(traceback.format_exception(type(e), e, e.__traceback__)))
                        print("Hint: Process backend errors may indicate pickling issues or native library crashes.")
                        # Attempt a quick pickling check of first args for diagnostics
                        try:
                            import pickle
                            pickle.dumps(args_list[0])
                            print("Diagnostic: Sample worker arguments appear picklable.")
                        except Exception as pe:
                            print(f"Diagnostic: Pickling worker arguments failed: {pe}")
                        raise
                    gc.collect()
                else:
                    # Thread backend: prepare datasets in main process
                    args_list = []
                    for i in range(iterate):
                        train_bld, val_bld, test_bld = self.__pre_attribute_bias(
                            attribute,
                            apply_nearmiss=treat_umbalance,
                            df_new=df_new)
                        args_list.append((train_bld, val_bld, test_bld, attribute, model_name))
                    pbar.set_description(f"Training {model_name} models ({n_threads} threads)")
                    try:
                        optimizer = TrainingOptimizer(backend='thread', max_workers=n_threads)
                        for result in optimizer.map(wrapper_training, args_list, max_workers=n_threads):
                            wrp_out.append(result)
                            pbar.update(1)
                    except Exception as e:
                        import traceback
                        print("\n[Error] Parallel training failed (thread backend)")
                        print(f"Attribute: {attribute} | Model: {model_name} | Iterate: {iterate} | Workers: {n_threads}")
                        print(f"Exception: {type(e).__name__}: {e}")
                        print('Traceback:\n' + ''.join(traceback.format_exception(type(e), e, e.__traceback__)))
                        raise
                    gc.collect()
            else:
                # Single thread execution for single iteration, when n_threads=1, or for TabTransformer
                if model_name == 'tabtransformer' and iterate > 1 and n_threads > 1:
                    pbar.set_description(f"Training {model_name} models (sequential - self-optimizing)")
                else:
                    pbar.set_description(f"Training {model_name} models (sequential)")
                    
                for i in range(iterate):
                    train_bld, val_bld, test_bld = self.__pre_attribute_bias(
                        attribute,
                        apply_nearmiss=treat_umbalance,
                        df_new=df_new)
                    wrp_out.append(wrapper_training(train_bld, val_bld, test_bld, attribute, model_name))
                    pbar.update(1)  # Update progress bar by 1 for each completed task
                    
                    # Periodic garbage collection for memory management
                    if (i + 1) % 3 == 0:
                        gc.collect()

        att_dic = defaultdict(float)

        # Process results and calculate fairness scores
        for i, wrp in enumerate(wrp_out):
            # Generate test data for evaluation (memory efficient)
            train_bld, val_bld, test_bld = self.__pre_attribute_bias(
                attribute,
                apply_nearmiss=treat_umbalance,
                df_new=df_new)
            
            _, model = wrp
            fair_results_dic = self.__predict_attribute_bias(train_bld, val_bld, test_bld, model, attribute)
            att_dic[f'{attribute}_raw'] += fair_results_dic['raw_score']
            att_dic[f'{attribute}_overall'] += fair_results_dic['overall_score']
            
            # Clean up model to free memory
            del model
            
            # Periodic garbage collection during result processing
            if (i + 1) % 5 == 0:
                gc.collect()
        
        # Calculate averages
        att_dic[f'{attribute}_raw'] = att_dic[f'{attribute}_raw'] / iterate
        att_dic[f'{attribute}_overall'] = att_dic[f'{attribute}_overall'] / iterate

        # Final cleanup
        del wrp_out
        gc.collect()
        
        return att_dic