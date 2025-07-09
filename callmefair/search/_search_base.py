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
from tqdm import tqdm
# Multiprocessing
from multiprocessing import Pool
# Suppress FutureWarning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Classifiers
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier


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
        model_name (str): Type of model to train ('lr', 'cat', 'xgb', 'mlp')

    Returns:
        tuple: (attribute_name, trained_model)

    Supported Models:
        - 'lr': Logistic Regression with liblinear solver
        - 'cat': CatBoost with optimized parameters for fairness
        - 'xgb': XGBoost with balanced parameters
        - 'mlp': Multi-layer Perceptron with adaptive learning

    Example:
        >>> result = wrapper_training(train_bld, val_bld, test_bld, 'gender', 'lr')
        >>> attribute, model = result
    """
    scaler = StandardScaler()
    scaler.fit(train_bld.features)

    x_train = scaler.transform(train_bld.features)
    y_train = train_bld.labels.ravel()

    if model_name == 'lr':
        model = LogisticRegression(solver='liblinear')
        model = model.fit(x_train, y_train, sample_weight=train_bld.instance_weights)

    elif model_name == 'cat':
        model = CatBoostClassifier(eval_metric='Accuracy',
                         depth =  4,
                         learning_rate = 0.01,
                         iterations = 10,
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
            n_jobs=8)
        model = model.fit(x_train, y_train)

    elif model_name == 'mlp':
        model= MLPClassifier(
            max_iter=500,
            hidden_layer_sizes = (100,100,100,100),
            activation = 'logistic',
            solver = 'adam',
            alpha = 0.01,
            learning_rate = 'adaptive')
        model = model.fit(x_train, y_train)

    return attribute, model

def wrapper(args):
    """
    Multiprocessing wrapper for model training.
    
    This function is used by multiprocessing.Pool to parallelize model training
    across multiple processes. It unpacks the arguments and calls wrapper_training.

    Parameters:
        args (tuple): Packed arguments for wrapper_training

    Returns:
        tuple: Result from wrapper_training
    """
    return wrapper_training(*args)

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

    Example:
        >>> searcher = BaseSearch(df, 'target')
        >>> results = searcher.evaluate_attribute('gender', iterate=10, model_name='lr')
    """

    def __init__(self, df: pd.DataFrame, label_name: str):
        """
        Initialize the BaseSearch object.

        Parameters:
            df (pd.DataFrame): Input dataset containing features and target variable
            label_name (str): Name of the target variable column
        """
        self.df = df.copy(deep=True)
        self.label_name = label_name
        self.scaler = StandardScaler()

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

        df_train, df_test = train_test_split(
            df_new, test_size=0.3, 
            stratify = df_new[[attribute, self.label_name]],
            random_state = 42
        )
        
        df_test, df_val = train_test_split(
            df_test, test_size = 0.5,
            stratify = df_test[[attribute, self.label_name]]
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
        self.scaler.fit(train_bld.features)

        privileged_group = [{attribute: 1}]
        unprivileged_group = [{attribute: 0}]

        x_val = self.scaler.transform(val_bld.features)
        x_test = self.scaler.transform(test_bld.features)

        pos_idx = np.where(model.classes_ == train_bld.favorable_label)[0][0]

        valid_bld_pred = val_bld.copy(deepcopy=True)
        valid_bld_pred.scores = model.predict_proba(x_val)[:, pos_idx].reshape(-1, 1)

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
        test_bld_pred.scores = model.predict_proba(x_test)[:, pos_idx].reshape(-1, 1)

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
            avg_odd_diff = classification_metric_orig_test.average_odds_difference()
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
            model_name (str): Type of model to use ('lr', 'cat', 'xgb', 'mlp')
            df_new (pd.DataFrame, optional): Alternative dataset to use

        Returns:
            dict: Dictionary containing averaged fairness scores for the attribute

        Example:
            >>> results = searcher.evaluate_attribute('gender', iterate=5, model_name='lr')
            >>> print(results['gender_raw'], results['gender_overall'])
        """
        if df_new is None:
            df_new = self.df

        bld_list, wrp_out = [], []
        for _ in range(iterate):
            train_bld, val_bld, test_bld = self.__pre_attribute_bias(
                attribute,
                apply_nearmiss=treat_umbalance,
                df_new=df_new)
            bld_list.append([train_bld, val_bld, test_bld, attribute])

        [bld.append(model_name) for bld in bld_list]

        if model_name in ('lr', 'mlp'): 
            # Use multiprocessing.Pool to parallelize training
            with Pool(processes=4) as pool:
                # Pass each model's task to a separate process
                 wrp_out = list(
                     tqdm(
                         pool.imap(wrapper, bld_list),
                         total=len(bld_list)
                        )
                    )
        else:
            for bld, _ in zip(bld_list, tqdm(range(len(bld_list)))):
                wrp_out.append(wrapper_training(*bld))

        att_dic = defaultdict(float)

        for bld, wrp in zip(bld_list, wrp_out):
            train_bld, val_bld, test_bld, attribute, _ = bld
            _, model = wrp
            fair_results_dic = self.__predict_attribute_bias(train_bld, val_bld, test_bld, model, attribute)
            att_dic[f'{attribute}_raw'] += fair_results_dic['raw_score']
            att_dic[f'{attribute}_overall'] += fair_results_dic['overall_score']
        
        att_dic[f'{attribute}_raw'] = att_dic[f'{attribute}_raw'] / iterate
        att_dic[f'{attribute}_overall'] = att_dic[f'{attribute}_overall'] / iterate

        return att_dic