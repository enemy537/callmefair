"""
Utility Functions and Interfaces for Bias Mitigation

This module provides core utility functions and interfaces for bias mitigation
in machine learning systems. It includes the essential `calculate_fairness_score`
function for aggregating multiple fairness metrics into a single score, and the
`BMInterface` class for managing datasets and bias mitigation operations.

The module implements:
- Fairness score calculation with normalization and deviation evaluation
- Dataset management with train/validation/test splits
- Binary label dataset creation and manipulation
- Feature scaling and transformation
- Comprehensive fairness metrics evaluation

Classes:
    BMnames: Data class for bias mitigation attribute configuration
    BMInterface: Main interface for managing datasets and bias mitigation operations
    BMMetrics: Comprehensive fairness metrics evaluation

Functions:
    calculate_fairness_score: Aggregate multiple fairness metrics into a single score

Example:
    >>> from callmefair.util.fair_util import BMInterface, calculate_fairness_score
    >>> import pandas as pd
    >>> 
    >>> # Load your full dataset
    >>> df = pd.read_csv('data.csv')
    >>> 
    >>> # Initialize the interface with a single DataFrame
    >>> bm_interface = BMInterface(df=df, label='label', protected=['gender'])
    >>> 
    >>> # Calculate fairness score
    >>> fairness_result = calculate_fairness_score(
    >>>     EOD=0.05, AOD=0.03, SPD=0.08, DI=0.95, TI=0.12
    >>> )
    >>> print(f"Overall fairness score: {fairness_result['overall_score']}")
"""

from dataclasses import dataclass
# Fairness Dataset
from aif360.datasets import BinaryLabelDataset
# Fairness metrics
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def calculate_fairness_score(EOD: float, AOD: float, SPD: float, DI: float, TI: float) -> dict:
    """
    Calculate a comprehensive fairness score based on multiple fairness metrics.
    
    This function aggregates five key fairness metrics into a single normalized score
    that represents the overall fairness of a machine learning model. The function
    evaluates each metric against predefined acceptable ranges and calculates
    deviations from optimal values to produce a unified fairness assessment.
    
    The function uses a weighted scoring system where:
    - Each metric contributes up to 0.2 for being outside acceptable ranges
    - Each metric contributes up to 0.16 based on deviation from optimal values
    - The final score is normalized to the 0-1 range where 0 = perfect fairness
    
    Args:
        EOD (float): Equal Opportunity Difference - measures difference in true positive
            rates between groups. Optimal value: 0.0, Acceptable range: (-0.1, 0.1)
        AOD (float): Average Odds Difference - measures difference in average of
            true positive and false positive rates between groups. Optimal value: 0.0,
            Acceptable range: (-0.1, 0.1)
        SPD (float): Statistical Parity Difference - measures difference in positive
            prediction rates between groups. Optimal value: 0.0, Acceptable range: (-0.1, 0.1)
        DI (float): Disparate Impact - ratio of positive prediction rates between
            groups. Optimal value: 1.0, Acceptable range: (0.8, 1.2)
        TI (float): Theil Index - measures inequality in prediction distributions.
            Optimal value: 0.0, Acceptable range: (0.0, 0.25)
    
    Returns:
        dict: Dictionary containing fairness evaluation results with keys:
            - 'raw_score' (float): Unnormalized fairness score
            - 'overall_score' (float): Normalized fairness score (0-1, lower is better)
            - 'metric_evaluations' (dict): Boolean evaluation of each metric
            - 'deviations' (dict): Normalized deviations from optimal values
            - 'is_fair' (bool): Whether all metrics are within acceptable ranges
            
    Example:
        >>> # Perfect fairness
        >>> result = calculate_fairness_score(0.0, 0.0, 0.0, 1.0, 0.0)
        >>> print(f"Score: {result['overall_score']}")  # 0.0 (perfect)
        >>> print(f"Is fair: {result['is_fair']}")  # True
        >>> 
        >>> # Moderate unfairness
        >>> result = calculate_fairness_score(0.15, 0.12, 0.18, 0.7, 0.3)
        >>> print(f"Score: {result['overall_score']}")  # ~0.6-0.8
        >>> print(f"Is fair: {result['is_fair']}")  # False
        >>> 
        >>> # Check individual metric evaluations
        >>> for metric, is_acceptable in result['metric_evaluations'].items():
        >>>     print(f"{metric}: {'✓' if is_acceptable else '✗'}")
    """
    # Validate and coerce metric inputs to avoid breaking the score on invalid values
    # Invalid inputs (None, NaN, non-numeric, or implausibly large/small) are mapped
    # to fixed "unfair" defaults that push the overall score towards unfairness
    # instead of raising errors.

    def _validate_metric(name: str, value):
        unfair_defaults = {
            'EOD': 1.0,   # large difference in opportunity
            'AOD': 1.0,   # large odds difference
            'SPD': 1.0,   # large statistical parity difference
            'DI': 0.1,    # strong disparate impact
            'TI': 1.0,    # high inequality
        }

        # Handle obvious invalids and container types early
        if value is None:
            return unfair_defaults[name]

        # Unpack common container types (use first element); empty -> invalid
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            if len(value) == 0:
                return unfair_defaults[name]
            # Recursively validate first element
            return _validate_metric(name, value[0])

        # Booleans should not be treated as numeric metrics
        if isinstance(value, bool):
            return unfair_defaults[name]

        # Treat empty/whitespace strings as invalid
        if isinstance(value, str):
            if value.strip() == "":
                return unfair_defaults[name]
            # Let float() handle things like 'nan', 'inf', etc.

        # Coerce to float when possible
        try:
            v = float(value)
        except (TypeError, ValueError):
            return unfair_defaults[name]

        # Handle NaN/inf and other non-finite outputs from invalid divisions
        if not np.isfinite(v):
            return unfair_defaults[name]

        # Simple plausibility bounds per metric
        if name in ('EOD', 'AOD', 'SPD'):
            # These are differences and should reasonably live in [-1, 1]
            if abs(v) > 1.0:
                return unfair_defaults[name]
        elif name == 'DI':
            # Ratio; negative, zero, extremely small, or extremely large values
            # are considered invalid for disparate impact in this context.
            if v <= 0 or v < 1e-6 or v > 5.0:
                return unfair_defaults[name]
        elif name == 'TI':
            # Inequality index; we expect it within [0, 1] for our use case
            if v < 0 or v > 1.0:
                return unfair_defaults[name]

        return v

    # Apply validation to all metric inputs
    EOD = _validate_metric('EOD', EOD)
    AOD = _validate_metric('AOD', AOD)
    SPD = _validate_metric('SPD', SPD)
    DI = _validate_metric('DI', DI)
    TI = _validate_metric('TI', TI)

    # Define optimal values for each fairness metric
    optimal_values = {
        'EOD': 0.0,  # Equal Opportunity Difference
        'AOD': 0.0,  # Average Odds Difference
        'SPD': 0.0,  # Statistical Parity Difference
        'DI': 1.0,   # Disparate Impact
        'TI': 0.0    # Theil Index
    }
    
    # Define acceptable ranges for each metric
    ranges = {
        'EOD': (-0.1, 0.1),   # ±10% difference acceptable
        'AOD': (-0.1, 0.1),   # ±10% difference acceptable
        'SPD': (-0.1, 0.1),   # ±10% difference acceptable
        'DI': (0.8, 1.2),     # 80-120% ratio acceptable
        'TI': (0.0, 0.25)     # Up to 25% inequality acceptable
    }
    
    # Check if each metric is within its acceptable range
    evaluations = {
        'EOD': ranges['EOD'][0] <= EOD <= ranges['EOD'][1],
        'AOD': ranges['AOD'][0] <= AOD <= ranges['AOD'][1],
        'SPD': ranges['SPD'][0] <= SPD <= ranges['SPD'][1],
        'DI': ranges['DI'][0] <= DI <= ranges['DI'][1],
        'TI': ranges['TI'][0] <= TI <= ranges['TI'][1]
    }
    
    # Calculate deviation from optimal values (normalized by range size)
    deviations = {
        'EOD': abs(EOD - optimal_values['EOD']) / 0.1,  # normalized by range size
        'AOD': abs(AOD - optimal_values['AOD']) / 0.1,
        'SPD': abs(SPD - optimal_values['SPD']) / 0.1,
        'DI': abs(DI - optimal_values['DI']) / 0.2,     # normalized by range size
        'TI': abs(TI - optimal_values['TI']) / 0.25
    }
    
    # Calculate raw unfairness score
    raw_score = 0.0
    max_possible_score = 0.0
    
    # Each metric contributes up to 0.2 to the score for being out of range
    # and up to 0.8 based on its deviation (0.16 * 5 = 0.8 total possible from deviations)
    for metric in evaluations:
        if not evaluations[metric]:
            raw_score += 0.2  # Penalty for being outside acceptable range
        raw_score += deviations[metric] * 0.16  # Contribution based on deviation
        max_possible_score += 0.2 + 0.16  # Maximum possible contribution from each metric
    
    # Normalize score to 0-1 range
    normalized_score = raw_score / max_possible_score
    final_score = min(1.0, max(0.0, normalized_score))
    
    return {
        'raw_score': raw_score,
        'overall_score': round(final_score, 3),
        'metric_evaluations': evaluations,
        'deviations': {k: round(v, 3) for k, v in deviations.items()},
        'is_fair': all(evaluations.values())
    }

def find_best_privileged_threshold(
    df: pd.DataFrame,
    numeric_column: str,
    label_column: str,
    thresholds = range(40, 75),
    favorable_label: int = 1,
    unfavorable_label: int = 0,
    minimize_spd_abs: bool = True,
    privileged_name: str = 'privileged',
    use_ge_operator: bool = True,
    verbose: bool = False,
    di_target: float = 1.0,
    spd_weight: float = 1.0,
    di_weight: float = 1.0,
) -> dict:
    """
    Search for a numeric threshold that defines privileged/unprivileged groups to improve fairness.

    This function labels a DataFrame with a binary protected attribute based on a numeric
    column threshold and evaluates fairness metrics for each threshold using AIF360.
    It selects the threshold that simultaneously maximizes disparate impact and
    minimizes statistical parity difference (optionally by absolute value).

    Args:
        df (pd.DataFrame): Input DataFrame containing features, label, and numeric column.
        numeric_column (str): Name of the numeric column used to define privilege (e.g., 'age').
        label_column (str): Name of the label column (e.g., 'stroke').
        thresholds (Iterable[int|float]): Sequence of thresholds to evaluate (default range(40, 75)).
        favorable_label (int): Value representing the favorable class label (default 1).
        unfavorable_label (int): Value representing the unfavorable class label (default 0).
        minimize_spd_abs (bool): If True, minimize absolute SPD; otherwise minimize raw SPD.
        privileged_name (str): Name for the generated protected attribute column (default 'privileged').
        use_ge_operator (bool): If True, privilege is `>= threshold`; else it is `> threshold`.
        verbose (bool): If True, print the best threshold and metrics.

    Returns:
        dict: Dictionary with keys:
            - 'best_threshold' (int|float): Threshold achieving desired fairness trade-off.
            - 'best_disparate_impact' (float): Disparate impact at the best threshold.
            - 'best_stat_parity_diff' (float): Statistical parity difference at the best threshold.

    Example:
        >>> # Replicates the provided snippet, generalized
        >>> result = find_best_privileged_threshold(
        ...     df=df, numeric_column='age', label_column='stroke', thresholds=range(40, 75)
        ... )
        >>> result['best_threshold']
        >>> result['best_disparate_impact'], result['best_stat_parity_diff']
    """

    best_threshold = None
    best_di = None
    best_spd = None
    best_score = float('inf')

    for threshold in thresholds:
        labeled_df = df.copy()
        if use_ge_operator:
            labeled_df[privileged_name] = (labeled_df[numeric_column] >= threshold).astype(int)
        else:
            labeled_df[privileged_name] = (labeled_df[numeric_column] > threshold).astype(int)

        bld = BinaryLabelDataset(
            df=labeled_df,
            label_names=[label_column],
            protected_attribute_names=[privileged_name],
            favorable_label=favorable_label,
            unfavorable_label=unfavorable_label,
        )

        metric = BinaryLabelDatasetMetric(
            bld,
            privileged_groups=[{privileged_name: 1}],
            unprivileged_groups=[{privileged_name: 0}],
        )

        disparate_impact = metric.disparate_impact()
        stat_parity_diff = metric.statistical_parity_difference()
        spd_dev = abs(stat_parity_diff) if minimize_spd_abs else stat_parity_diff
        di_dev = abs(disparate_impact - di_target)
        score = spd_weight * spd_dev + di_weight * di_dev

        if score < best_score:
            best_score = score
            best_threshold = threshold
            best_di = disparate_impact
            best_spd = stat_parity_diff

    result = {
        'best_threshold': best_threshold,
        'best_disparate_impact': best_di,
        'best_stat_parity_diff': best_spd,
    }

    if verbose:
        print(f"Best threshold on {numeric_column}: {best_threshold}")
        print(f"Disparate Impact at best: {best_di} (target {di_target})")
        print(
            "Statistical Parity Difference at best threshold" +
            (" (abs minimized)" if minimize_spd_abs else "") +
            f": {best_spd}"
        )

    return result

@dataclass
class BMnames:
    """
    Data class for bias mitigation attribute configuration.
    
    This class stores the essential configuration parameters for bias mitigation
    operations, including label names, protected attributes, and favorable/unfavorable
    label values.
    
    Attributes:
        label_names (str): Name of the target/label column in the dataset
        protected_att (list): List of protected attribute column names
        favorable_label (float): Value representing the favorable outcome (default: 1.0)
        unfavorable_label (float): Value representing the unfavorable outcome (default: 0.0)
        
    Example:
        >>> bm_config = BMnames(
        >>>     label_names='income',
        >>>     protected_att=['gender', 'race'],
        >>>     favorable_label=1.0,
        >>>     unfavorable_label=0.0
        >>> )
    """
    label_names: str
    protected_att: list
    favorable_label: float = 1.0
    unfavorable_label: float = 0.0


class BMInterface:
    """
    Main interface for managing datasets and bias mitigation operations.
    
    This class provides a unified interface for managing binary classification
    datasets with bias mitigation capabilities. It handles dataset splitting,
    feature scaling, and provides access to train/validation/test sets in
    various formats required by different bias mitigation techniques.
    
    The interface supports:
    - Automatic dataset splitting and scaling
    - Binary label dataset creation for AIF360 compatibility
    - Feature scaling with train/validation/test consistency
    - Dataset restoration and transformation modes
    
    Attributes:
        data_sets (list): List of DataFrames [train, validation, test]
        BM_attr (BMnames): Bias mitigation attribute configuration
        transform (bool): Whether to apply feature scaling
        biLData (list): List of BinaryLabelDataset objects
        trainXY (tuple): (features, labels) for training data
        valXY (tuple): (features, labels) for validation data
        testXY (tuple): (features, labels) for test data
        
    Example:
        >>> from callmefair.util.fair_util import BMInterface
        >>> import pandas as pd
        >>> 
        >>> # Load your data
        >>> train_df = pd.read_csv('train.csv')
        >>> val_df = pd.read_csv('val.csv')
        >>> test_df = pd.read_csv('test.csv')
        >>> 
        >>> # Initialize interface
        >>> bm_interface = BMInterface(
        >>>     df_train=train_df,
        >>>     df_val=val_df,
        >>>     df_test=test_df,
        >>>     label='income',
        >>>     protected=['gender', 'race']
        >>> )
        >>> 
        >>> # Get data in different formats
        >>> train_bld = bm_interface.get_train_BLD()
        >>> X_train, y_train = bm_interface.get_train_xy()
    """
    
    def __init__(self, 
                 df: pd.DataFrame,
                 label: str,
                 protected: list,
                 random_state: int = 42,
                 shuffle: bool = True,
                 test_size: float = 0.1,
                 val_size: float = 0.1):
        """
        Initialize the Bias Mitigation Interface.
        
        Args:
            df (pd.DataFrame): Single dataset to be split into train/val/test
                with stratified 80/10/10 proportions by the `label` column.
            label (str): Name of the target/label column
            protected (list): List of protected attribute column names
            random_state (int): Random seed used for splitting
            shuffle (bool): Whether to shuffle before splitting
            test_size (float): Proportion for test split from the original dataset
            val_size (float): Proportion for validation split from the original dataset
                
        Example:
            >>> bm_interface = BMInterface(
            >>>     df=full_df,
            >>>     label='income',
            >>>     protected=['gender', 'race']
            >>> )
        """
        # Validate required arguments
        if df is None:
            raise ValueError("`df` must be provided.")
        if label is None or protected is None:
            raise ValueError("`label` and `protected` must be provided.")

        # Enforce balanced splits: ensure each set contains both classes when possible
        labels = df[label]
        unique_classes = pd.unique(labels)
        if len(unique_classes) < 2:
            raise ValueError("BMInterface requires at least two classes in the label column.")

        def attempt(seed: int, data_frame: pd.DataFrame):
            # First split test_size from original for test, stratified by label
            train_val_df, test_df = train_test_split(
                data_frame,
                test_size=test_size,
                random_state=seed,
                shuffle=shuffle,
                stratify=data_frame[label]
            )
            # Then split validation to be val_size of original: compute relative portion
            val_rel = val_size / (1.0 - test_size)
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_rel,
                random_state=seed,
                shuffle=shuffle,
                stratify=train_val_df[label]
            )
            return train_df, val_df, test_df

        def good_split(tr_df: pd.DataFrame, va_df: pd.DataFrame, te_df: pd.DataFrame) -> bool:
            s_tr = set(np.unique(tr_df[label]))
            s_va = set(np.unique(va_df[label]))
            s_te = set(np.unique(te_df[label]))
            return len(s_tr) == 2 and s_tr == s_va == s_te

        final_sets = None
        max_tries = 15
        last_attempt = None
        for i in range(max_tries):
            tr, va, te = attempt(random_state + i, df)
            last_attempt = (tr, va, te)
            if good_split(tr, va, te):
                final_sets = (tr, va, te)
                break

        if final_sets is None:
            # Minimal oversampling of minority class to enable balanced splits when class counts are too small
            vc = df[label].value_counts()
            min_class = vc.idxmin()
            need = max(0, 3 - int(vc[min_class]))
            if need > 0:
                dup = df[df[label] == min_class]
                try:
                    add = dup.sample(n=need, replace=True, random_state=random_state)
                    df_aug = pd.concat([df, add], ignore_index=True)
                except Exception:
                    df_aug = df
            else:
                df_aug = df

            for i in range(max_tries):
                tr, va, te = attempt(random_state + i, df_aug)
                last_attempt = (tr, va, te)
                if good_split(tr, va, te):
                    final_sets = (tr, va, te)
                    break

            if final_sets is None:
                # Fall back to the last attempted split if perfectly balanced splits are not achievable
                final_sets = last_attempt

        train_df, val_df, test_df = final_sets
        self.data_sets = [train_df, val_df, test_df]

        self.BM_attr = BMnames(label_names=label, protected_att=protected)
        self.transform = False
        self.__generate_sets()
        self.__scale_split()

    def __generate_sets(self) -> None:
        """
        Generate BinaryLabelDataset objects for all datasets.
        
        This method creates BinaryLabelDataset objects from the input DataFrames,
        which are required for AIF360 bias mitigation operations.
        
        Example:
            >>> # Called automatically during initialization
            >>> bm_interface = BMInterface(...)
            >>> # BinaryLabelDataset objects are now available
        """
        self.biLData = []
        for data in self.data_sets:
            self.biLData.append(BinaryLabelDataset(
                df=data,
                label_names=[self.BM_attr.label_names],
                protected_attribute_names=self.BM_attr.protected_att,
                favorable_label=self.BM_attr.favorable_label,
                unfavorable_label=self.BM_attr.unfavorable_label
                )
            )

    def __scale_transform(self) -> None:
        """
        Apply feature scaling in transform mode.
        
        This method applies StandardScaler to all datasets using the training
        data's scaling parameters to ensure consistency across train/validation/test.
        
        Example:
            >>> bm_interface.set_transform()
            >>> # Features are now scaled consistently
        """
        scaler = StandardScaler()
        self.biLData[0].features = scaler.fit_transform(self.biLData[0].features)
        self.biLData[1].features = scaler.transform(self.biLData[1].features)
        self.biLData[2].features = scaler.transform(self.biLData[2].features)

    def __scale_split(self) -> None:
        """
        Create scaled feature-label pairs for all datasets.
        
        This method creates (features, labels) tuples for all datasets with
        consistent scaling applied to features.
        
        Example:
            >>> # Called automatically during initialization
            >>> X_train, y_train = bm_interface.get_train_xy()
            >>> # Features are scaled, labels are original
        """
        scaler = StandardScaler()
        scaler.fit(self.biLData[0].features)
        xy_tmp = []
        for i in self.biLData:
            xy_tmp.append(
                (scaler.transform(i.features),
                 i.labels.ravel())
            )
        self.trainXY, self.valXY, self.testXY = xy_tmp

    def set_transform(self) -> None:
        """
        Enable transform mode for feature scaling.
        
        This method enables the transform mode, which applies StandardScaler
        to all datasets. This is useful when working with models that require
        scaled features.
        
        Example:
            >>> bm_interface.set_transform()
            >>> # All subsequent operations will use scaled features
        """
        self.transform = True
        self.__scale_transform()
    
    def get_protected_att(self) -> list:
        """
        Get the list of protected attributes.
        
        Returns:
            list: List of protected attribute column names
            
        Example:
            >>> protected_attrs = bm_interface.get_protected_att()
            >>> print(f"Protected attributes: {protected_attrs}")
        """
        return self.BM_attr.protected_att

    def restore_BLD(self) -> None:
        """
        Restore BinaryLabelDataset objects to their original state.
        
        This method regenerates the BinaryLabelDataset objects and reapplies
        scaling if transform mode is enabled. Useful after bias mitigation
        operations that modify the datasets.
        
        Example:
            >>> # After applying bias mitigation
            >>> bm_interface.pre_BM_set(modified_train_bld)
            >>> # Restore to original state
            >>> bm_interface.restore_BLD()
        """
        self.__generate_sets()
        if self.transform:
            self.__scale_transform()
        self.__scale_split()

    def get_train_BLD(self) -> BinaryLabelDataset:
        """
        Get training dataset as BinaryLabelDataset.
        
        Returns:
            BinaryLabelDataset: Training dataset in AIF360 format
            
        Example:
            >>> train_bld = bm_interface.get_train_BLD()
            >>> print(f"Training samples: {len(train_bld.features)}")
        """
        return self.biLData[0].copy(deepcopy=True)
    
    def get_val_BLD(self) -> BinaryLabelDataset:
        """
        Get validation dataset as BinaryLabelDataset.
        
        Returns:
            BinaryLabelDataset: Validation dataset in AIF360 format
            
        Example:
            >>> val_bld = bm_interface.get_val_BLD()
            >>> print(f"Validation samples: {len(val_bld.features)}")
        """
        return self.biLData[1].copy(deepcopy=True)
    
    def get_test_BLD(self) -> BinaryLabelDataset:
        """
        Get test dataset as BinaryLabelDataset.
        
        Returns:
            BinaryLabelDataset: Test dataset in AIF360 format
            
        Example:
            >>> test_bld = bm_interface.get_test_BLD()
            >>> print(f"Test samples: {len(test_bld.features)}")
        """
        return self.biLData[2].copy(deepcopy=True)

    def get_train_xy(self) -> tuple:
        """
        Get training data as (features, labels) tuple.
        
        Returns:
            tuple: (features, labels) for training data
            
        Example:
            >>> X_train, y_train = bm_interface.get_train_xy()
            >>> print(f"Training features shape: {X_train.shape}")
            >>> print(f"Training labels shape: {y_train.shape}")
        """
        return self.trainXY[0], self.trainXY[1]
    
    def get_test_xy(self) -> tuple:
        """
        Get test data as (features, labels) tuple.
        
        Returns:
            tuple: (features, labels) for test data
            
        Example:
            >>> X_test, y_test = bm_interface.get_test_xy()
            >>> print(f"Test features shape: {X_test.shape}")
        """
        return self.testXY[0], self.testXY[1]
    
    def get_val_xy(self) -> tuple:
        """
        Get validation data as (features, labels) tuple.
        
        Returns:
            tuple: (features, labels) for validation data
            
        Example:
            >>> X_val, y_val = bm_interface.get_val_xy()
            >>> print(f"Validation features shape: {X_val.shape}")
        """
        return self.valXY[0], self.valXY[1]
    
    def pre_BM_set(self, new_train_BLD: BinaryLabelDataset) -> None:
        """
        Set new training dataset after preprocessing bias mitigation.
        
        This method updates the training dataset with the result of preprocessing
        bias mitigation techniques and reapplies scaling if needed.
        
        Args:
            new_train_BLD (BinaryLabelDataset): New training dataset after bias mitigation
            
        Example:
            >>> # After applying reweighing
            >>> modified_train_bld = reweighing.transform(original_train_bld)
            >>> bm_interface.pre_BM_set(modified_train_bld)
        """
        self.biLData[0] = new_train_BLD
        if self.transform:
            self.__scale_transform()
        self.__scale_split()

    def pos_bm_set(self, new_test_BLD: BinaryLabelDataset) -> None:
        """
        Set new test dataset after postprocessing bias mitigation.
        
        This method updates the test dataset with the result of postprocessing
        bias mitigation techniques and reapplies scaling if needed.
        
        Args:
            new_test_BLD (BinaryLabelDataset): New test dataset after bias mitigation
            
        Example:
            >>> # After applying equalized odds
            >>> modified_test_bld = eq_odds.predict(original_test_bld)
            >>> bm_interface.pos_bm_set(modified_test_bld)
        """
        self.biLData[-1] = new_test_BLD
        if self.transform:
            self.__scale_transform()

class BMMetrics:
    """
    Comprehensive fairness metrics evaluation for bias mitigation.
    
    This class provides comprehensive evaluation of both classification performance
    and fairness metrics for machine learning models. It supports both standard
    models and in-processing bias mitigation techniques.
    
    The class evaluates:
    - Classification metrics: accuracy, precision, recall, F1, MCC
    - Fairness metrics: EOD, AOD, SPD, DI, TI
    - Optimal threshold selection based on balanced accuracy
    - Comprehensive fairness scoring
    
    Attributes:
        bmI (BMInterface): Interface for managing datasets
        pos_idx (int): Index of positive class
        in_mode (bool): Whether using in-processing bias mitigation
        pred_test: Test predictions (numpy array or BinaryLabelDataset)
        pred_val: Validation predictions (numpy array or BinaryLabelDataset)
        privileged_group (list[dict]): List of privileged group definitions
        unprivileged_group (list[dict]): List of unprivileged group definitions
        balanced_acc (np.ndarray): Balanced accuracy scores for different thresholds
        class_threshold (np.ndarray): Threshold values for evaluation
        best_class_thresh (float): Optimal threshold based on validation performance
        cmetrics (ClassificationMetric): AIF360 classification metrics object
        
    Example:
        >>> from callmefair.util.fair_util import BMMetrics
        >>> import numpy as np
        >>> 
        >>> # Create metrics evaluator
        >>> metrics = BMMetrics(
        >>>     bmI=bm_interface,
        >>>     class_array=np.array([0, 1]),
        >>>     pred_val=val_predictions,
        >>>     pred_test=test_predictions,
        >>>     privileged_group=privileged_groups,
        >>>     unprivileged_group=unprivileged_groups
        >>> )
        >>> 
        >>> # Get comprehensive report
        >>> report = metrics.get_report()
        >>> print(f"Accuracy: {report['acc']:.4f}")
        >>> print(f"Fairness score: {report['spd']:.4f}")
    """
    
    def __init__(self, bmI: BMInterface, class_array: np.ndarray, 
                 pred_val: np.ndarray | BinaryLabelDataset,
                 pred_test: np.ndarray | BinaryLabelDataset, 
                 privileged_group: list[dict], unprivileged_group: list[dict]):
        """
        Initialize the Bias Mitigation Metrics evaluator.
        
        Args:
            bmI (BMInterface): Interface for managing datasets
            class_array (np.ndarray): Array of class labels [0, 1]
            pred_val: Validation predictions (numpy array or BinaryLabelDataset)
            pred_test: Test predictions (numpy array or BinaryLabelDataset)
            privileged_group (list[dict]): List of privileged group definitions
            unprivileged_group (list[dict]): List of unprivileged group definitions
                
        Example:
            >>> metrics = BMMetrics(
            >>>     bmI=bm_interface,
            >>>     class_array=np.array([0, 1]),
            >>>     pred_val=val_pred,
            >>>     pred_test=test_pred,
            >>>     privileged_group=[{'gender': 1}],
            >>>     unprivileged_group=[{'gender': 0}]
            >>> )
        """
        self.bmI = bmI
        # positive class index
        self.pos_idx = np.where(class_array == bmI.get_train_BLD().favorable_label)[0][0]
        # deal with transform mode
        self.in_mode = False
        if isinstance(pred_val, BinaryLabelDataset):
            self.in_mode = True

        self.pred_test = pred_test
        self.pred_val = pred_val
        
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group

        num_thresh = 100
        self.balanced_acc = np.zeros(num_thresh)
        self.class_threshold = np.linspace(0.01, 0.99, num_thresh)

        self.__get_thresould()
        self.__set_pred_BLD()

    def set_new_pred(self, pred_val: np.ndarray, pred_test: np.ndarray) -> None:
        """
        Update predictions and recalculate metrics.
        
        This method allows updating the predictions and recalculating all
        metrics without recreating the entire BMMetrics object.
        
        Args:
            pred_val (np.ndarray): New validation predictions
            pred_test (np.ndarray): New test predictions
            
        Example:
            >>> # After training a new model
            >>> new_val_pred = model.predict_proba(X_val)
            >>> new_test_pred = model.predict_proba(X_test)
            >>> metrics.set_new_pred(new_val_pred, new_test_pred)
        """
        self.pred_val = pred_val
        self.pred_test = pred_test

        self.__get_thresould()
        self.__set_pred_BLD()
        
    def set_pos_pred(self, test_BLD_pred: BinaryLabelDataset) -> None:
        """
        Set postprocessed predictions and update metrics.
        
        This method is used for postprocessing bias mitigation techniques
        that modify the test predictions directly.
        
        Args:
            test_BLD_pred (BinaryLabelDataset): Postprocessed test predictions
            
        Example:
            >>> # After applying equalized odds
            >>> postprocessed_pred = eq_odds.predict(test_bld)
            >>> metrics.set_pos_pred(postprocessed_pred)
        """
        fav_inds = test_BLD_pred.scores > self.best_class_thresh
        test_BLD_pred.labels[fav_inds] = test_BLD_pred.favorable_label
        test_BLD_pred.labels[~fav_inds] = test_BLD_pred.unfavorable_label

        self.cmetrics = ClassificationMetric(self.bmI.get_test_BLD(), 
                                                test_BLD_pred, 
                                                unprivileged_groups=self.unprivileged_group, 
                                                privileged_groups=self.privileged_group
                                            )

    def get_pred_test(self):
        """
        Get test predictions.
        
        Returns:
            Test predictions (numpy array or BinaryLabelDataset)
            
        Example:
            >>> test_pred = metrics.get_pred_test()
            >>> print(f"Test predictions shape: {test_pred.shape}")
        """
        return self.pred_test
    
    def __set_pred_BLD(self) -> None:
        """
        Set up BinaryLabelDataset for test predictions evaluation.
        
        This method prepares the test predictions for evaluation by creating
        BinaryLabelDataset objects and applying the optimal threshold.
        
        Example:
            >>> # Called automatically during initialization
            >>> metrics = BMMetrics(...)
            >>> # Test predictions are now ready for evaluation
        """
        if not self.in_mode:   
            pred_BLD = self.bmI.get_test_BLD()
            pred_BLD.scores = self.pred_test[:, self.pos_idx].reshape(-1, 1)

        for thresh in self.class_threshold:

            if self.in_mode:
                self.pred_test.labels = (self.pred_test.scores >= thresh).astype(int)
                pred_BLD = self.pred_test
            else:
                fav_idx = pred_BLD.scores > thresh 
                pred_BLD.labels[fav_idx] = pred_BLD.favorable_label
                pred_BLD.labels[~fav_idx] = pred_BLD.unfavorable_label

            self.cmetrics = ClassificationMetric(self.bmI.get_test_BLD(), 
                                                pred_BLD, 
                                                unprivileged_groups=self.unprivileged_group, 
                                                privileged_groups=self.privileged_group
                                                )

            if thresh == self.best_class_thresh:
                break

    def __get_thresould(self) -> None:
        """
        Find optimal classification threshold based on validation performance.
        
        This method evaluates different classification thresholds on validation
        data and selects the threshold that maximizes balanced accuracy.
        
        Example:
            >>> # Called automatically during initialization
            >>> metrics = BMMetrics(...)
            >>> print(f"Optimal threshold: {metrics.best_class_thresh:.3f}")
        """
        if not self.in_mode:
            pred_val_BLD = self.bmI.get_val_BLD()
            pred_val_BLD.scores = self.pred_val[:, self.pos_idx].reshape(-1, 1)

        for idx, class_thresh in enumerate(self.class_threshold):
    
            if self.in_mode:
                self.pred_val.labels = (self.pred_val.scores >= class_thresh).astype(int)
                pred_val_BLD = self.pred_val

            else:
                fav_idx = pred_val_BLD.scores > class_thresh 
                pred_val_BLD.labels[fav_idx] = pred_val_BLD.favorable_label
                pred_val_BLD.labels[~fav_idx] = pred_val_BLD.unfavorable_label
            
            # computing metrics based on two BinaryLabelDatasets: a dataset containing groud-truth labels and a dataset containing predictions
            cm = ClassificationMetric(self.bmI.get_val_BLD(),     
                                      pred_val_BLD,  
                                      unprivileged_groups=self.unprivileged_group,
                                      privileged_groups=self.privileged_group
                                    )

            self.balanced_acc[idx] = 0.5 * (cm.true_positive_rate() + cm.true_negative_rate())

        best_idx = np.where(self.balanced_acc == np.max(self.balanced_acc))[0][0]
        self.best_class_thresh = self.class_threshold[best_idx]

    def __get_classification_metrics(self) -> tuple:
        """
        Calculate comprehensive classification performance metrics.
        
        Returns:
            tuple: (balanced_acc, acc, precision, recall, f1, mcc)
                - balanced_acc (float): Balanced accuracy
                - acc (float): Overall accuracy
                - precision (float): Precision score
                - recall (float): Recall score
                - f1 (float): F1 score
                - mcc (float): Matthews Correlation Coefficient
                
        Example:
            >>> balanced_acc, acc, precision, recall, f1, mcc = metrics.__get_classification_metrics()
            >>> print(f"Balanced accuracy: {balanced_acc:.4f}")
        """
        balanced_acc = 0.5 * (self.cmetrics.true_positive_rate() + self.cmetrics.true_negative_rate())
        acc = self.cmetrics.accuracy()
        precision = self.cmetrics.precision()
        recall = self.cmetrics.recall()
        f1 = 2 * ((precision * recall)/(precision + recall))
        gb_cm = self.cmetrics.generalized_binary_confusion_matrix()
        GTP, GFP, GTN, GFN = gb_cm.values()
        mcc = (GTP*GTN - GFP*GFN) / np.sqrt((GTP+GFP)*(GTP+GFN)*(GTN+GFP)*(GTN+GFN))

        return balanced_acc, acc, precision, recall, f1, mcc

    def __get_fair_metrics(self) -> tuple:
        """
        Calculate comprehensive fairness metrics.
        
        Returns:
            tuple: (eq_opp_diff, avg_odd_diff, spd, disparate_impact, theil_idx)
                - eq_opp_diff (float): Equal Opportunity Difference
                - avg_odd_diff (float): Average Odds Difference
                - spd (float): Statistical Parity Difference
                - disparate_impact (float): Disparate Impact
                - theil_idx (float): Theil Index
                
        Example:
            >>> eod, aod, spd, di, ti = metrics.__get_fair_metrics()
            >>> print(f"Equal Opportunity Difference: {eod:.4f}")
        """
        eq_opp_diff = self.cmetrics.equal_opportunity_difference()
        
        # Handle potential errors in avg_odd_diff calculation due to exploding FPR/TPR
        try:
            avg_odd_diff = self.cmetrics.average_odds_difference()
        except (ValueError, ZeroDivisionError, RuntimeError) as e:
            # Set worst possible value for avg_odd_diff when calculation fails
            # This indicates maximum unfairness (far from optimal value of 0.0)
            avg_odd_diff = 1.0
            print(f"Warning: avg_odd_diff calculation failed, using worst-case value 1.0. Error: {e}")
        
        spd = self.cmetrics.statistical_parity_difference()
        disparate_impact = self.cmetrics.disparate_impact()
        theil_idx = self.cmetrics.theil_index()

        return eq_opp_diff, avg_odd_diff, spd, disparate_impact, theil_idx

    def get_report(self) -> dict:
        """
        Get comprehensive performance and fairness report.
        
        Returns:
            dict: Dictionary containing all classification and fairness metrics
                - Classification metrics: balanced_acc, acc, precision, recall, f1, mcc
                - Fairness metrics: eq_opp_diff, avg_odd_diff, spd, disparate_impact, theil_idx
                
        Example:
            >>> report = metrics.get_report()
            >>> print(f"Accuracy: {report['acc']:.4f}")
            >>> print(f"Statistical Parity Difference: {report['spd']:.4f}")
            >>> print(f"Equal Opportunity Difference: {report['eq_opp_diff']:.4f}")
        """
        eq_opp_diff, avg_odd_diff, spd, disparate_impact, theil_idx = self.__get_fair_metrics()
        balanced_acc, acc, precision, recall, f1, mcc = self.__get_classification_metrics()

        return {'balanced_acc': balanced_acc, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1,
                'mcc': mcc, 'eq_opp_diff': eq_opp_diff, 'avg_odd_diff': avg_odd_diff, 'spd': spd,
                'disparate_impact': disparate_impact, 'theil_idx': theil_idx}

    def get_score(self) -> dict:
        """
        Get comprehensive fairness score using the calculate_fairness_score function.
        
        Returns:
            dict: Fairness score evaluation with overall score and detailed breakdown
                - 'raw_score' (float): Unnormalized fairness score
                - 'overall_score' (float): Normalized fairness score (0-1, lower is better)
                - 'metric_evaluations' (dict): Boolean evaluation of each metric
                - 'deviations' (dict): Normalized deviations from optimal values
                - 'is_fair' (bool): Whether all metrics are within acceptable ranges
                
        Example:
            >>> score_dict = metrics.get_score()
            >>> print(f"Overall fairness score: {score_dict['overall_score']:.3f}")
            >>> print(f"Is fair: {score_dict['is_fair']}")
            >>> 
            >>> # Check individual metric evaluations
            >>> for metric, is_acceptable in score_dict['metric_evaluations'].items():
            >>>     print(f"{metric}: {'✓' if is_acceptable else '✗'}")
        """
        eq_opp_diff, avg_odd_diff, spd, disparate_impact, theil_idx = self.__get_fair_metrics()
        score_dict = calculate_fairness_score(eq_opp_diff, avg_odd_diff, spd, disparate_impact, theil_idx)
        return score_dict

    def get_groups(self) -> tuple:
        """
        Get privileged and unprivileged group definitions.
        
        Returns:
            tuple: (privileged_group, unprivileged_group)
                - privileged_group (list[dict]): List of privileged group definitions
                - unprivileged_group (list[dict]): List of unprivileged group definitions
                
        Example:
            >>> privileged, unprivileged = metrics.get_groups()
            >>> print(f"Privileged groups: {privileged}")
            >>> print(f"Unprivileged groups: {unprivileged}")
        """
        return self.privileged_group, self.unprivileged_group