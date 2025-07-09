"""
Grid Search for Bias Mitigation Combinations

This module provides a comprehensive grid search framework for evaluating different
combinations of bias mitigation techniques. It supports systematic evaluation of
preprocessing, in-processing, and postprocessing bias mitigation methods across
various machine learning models.

The module implements:
- Automatic model adaptation for different ML frameworks
- Systematic evaluation of bias mitigation combinations
- Comprehensive logging and result aggregation
- Support for single sensitive attribute evaluation (extensible to multiple)

Classes:
    dummy_model: Dummy model class for compatibility
    BMGridSearch: Main grid search class for bias mitigation evaluation

Functions:
    get_model_proba: Adapts different ML models for probability prediction

Example:
    >>> from callmefair.mitigation.fair_grid import BMGridSearch
    >>> from callmefair.mitigation.fair_bm import BMType
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> 
    >>> # Define bias mitigation combinations to test
    >>> bm_combinations = [
    >>>     [BMType.preReweighing],
    >>>     [BMType.preDisparate],
    >>>     [BMType.preReweighing, BMType.posCEO],
    >>>     [BMType.inAdversarial]
    >>> ]
    >>> 
    >>> # Initialize grid search
    >>> grid_search = BMGridSearch(
    >>>     bmI=bm_interface,
    >>>     model=RandomForestClassifier(),
    >>>     bm_list=bm_combinations,
    >>>     privileged_group=privileged_groups,
    >>>     unprivileged_group=unprivileged_groups
    >>> )
    >>> 
    >>> # Run grid search
    >>> grid_search.run_single_sensitive()
"""

from callmefair.util.fair_util import BMInterface, BMMetrics
from callmefair.mitigation.fair_bm import BMManager, BMType
from callmefair.mitigation.fair_log import csvLogger
import numpy as np
from datetime import datetime
from dataclasses import dataclass   
try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
except ImportError:
    tf = None

#TODO adapt model fit and predict probabilites to an abstract interface

def get_model_proba(model, bmI: BMInterface) -> tuple[np.ndarray]:
    """
    Adapt different ML models for probability prediction.
    
    This function provides a unified interface for training models and obtaining
    probability predictions across different ML frameworks. It handles various
    model types including scikit-learn, XGBoost, TabNet, and others.
    
    Args:
        model: The machine learning model to train and evaluate
        bmI (BMInterface): Interface for managing binary label datasets
        
    Returns:
        tuple[np.ndarray]: Tuple containing (validation_predictions, test_predictions)
            Both arrays contain probability predictions for the positive class.
            
    Raises:
        ValueError: If the model type is not supported
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> val_pred, test_pred = get_model_proba(model, bm_interface)
        >>> print(f"Validation predictions shape: {val_pred.shape}")
        >>> print(f"Test predictions shape: {test_pred.shape}")
    """
    x_train, y_train = bmI.get_train_xy() 
    x_val , y_val = bmI.get_val_xy()
    x_test, _ = bmI.get_test_xy()

    if model.__str__().startswith('LogisticRegression'):
        model = model.fit(x_train, y_train, sample_weight=bmI.get_train_BLD().instance_weights)
        y_val_pred = model.predict_proba(x_val)
        y_test_pred = model.predict_proba(x_test)

    elif any([model.__str__().startswith(i) for i in ('XGBClassifier', 'MLP')]):
        model = model.fit(x_train, y_train)
        y_val_pred = model.predict_proba(x_val)
        y_test_pred = model.predict_proba(x_test)

    elif model.__str__().startswith('TabNet'):
        model.fit(x_train, y_train,
                 eval_set=[(x_val, y_val)]
                )
        y_val_pred = model.predict_proba(x_val)
        y_test_pred = model.predict_proba(x_test)

    else:
        model = model.fit(x_train, y_train, eval_set=[(x_val , y_val)])
        y_val_pred = model.predict_proba(x_val)
        y_test_pred = model.predict_proba(x_test)
        
    return (y_val_pred, y_test_pred)


@dataclass
class dummy_model:
    """
    Dummy model class for compatibility with bias mitigation metrics.
    
    This class provides a minimal interface required by the BMMetrics class
    for evaluating bias mitigation techniques when no specific model is used
    (e.g., for in-processing techniques that have their own models).
    
    Attributes:
        classes_ (np.ndarray): Array of class labels [0, 1]
    """
    classes_ = np.array([0,1])

class BMGridSearch:
    """
    Grid Search for Bias Mitigation Combinations.
    
    This class provides a systematic framework for evaluating different combinations
    of bias mitigation techniques. It supports preprocessing, in-processing, and
    postprocessing methods, and can work with various machine learning models.
    
    The grid search evaluates each combination of bias mitigation techniques and
    logs the results for comparison. It currently supports single sensitive
    attribute evaluation, with plans for multiple sensitive attributes.
    
    Attributes:
        bmI (BMInterface): Interface for managing binary label datasets
        bmMR (BMManager): Bias mitigation manager for applying techniques
        model: The machine learning model to evaluate
        bm_list (list[list[BMType]]): List of bias mitigation combinations to test
        privileged_group (list[dict]): List of dictionaries defining privileged groups
        unprivileged_group (list[dict]): List of dictionaries defining unprivileged groups
        is_model_in (bool): Whether using in-processing bias mitigation
        
    Example:
        >>> from callmefair.mitigation.fair_grid import BMGridSearch
        >>> from callmefair.mitigation.fair_bm import BMType
        >>> 
        >>> # Define combinations to test
        >>> combinations = [
        >>>     [BMType.preReweighing],
        >>>     [BMType.preDisparate, BMType.posCEO],
        >>>     [BMType.inAdversarial]
        >>> ]
        >>> 
        >>> # Create grid search
        >>> grid_search = BMGridSearch(
        >>>     bmI=bm_interface,
        >>>     model=RandomForestClassifier(),
        >>>     bm_list=combinations,
        >>>     privileged_group=privileged_groups,
        >>>     unprivileged_group=unprivileged_groups
        >>> )
        >>> 
        >>> # Run evaluation
        >>> grid_search.run_single_sensitive()
    """
    
    def __init__(self, bmI: BMInterface, model, bm_list: list[list[BMType]], 
                 privileged_group: list[dict], unprivileged_group: list[dict]):
        """
        Initialize the Bias Mitigation Grid Search.
        
        Args:
            bmI (BMInterface): Interface for managing binary label datasets
            model: The machine learning model to evaluate. Can be None for
                in-processing techniques that have their own models.
            bm_list (list[list[BMType]]): List of bias mitigation combinations to test.
                Each inner list represents one combination of techniques.
            privileged_group (list[dict]): List of dictionaries defining privileged groups.
                Each dict should contain protected attribute names and their privileged values.
            unprivileged_group (list[dict]): List of dictionaries defining unprivileged groups.
                Each dict should contain protected attribute names and their unprivileged values.
                
        Example:
            >>> bm_combinations = [
            >>>     [BMType.preReweighing],
            >>>     [BMType.preDisparate, BMType.posCEO],
            >>>     [BMType.inAdversarial]
            >>> ]
            >>> 
            >>> grid_search = BMGridSearch(
            >>>     bmI=bm_interface,
            >>>     model=RandomForestClassifier(),
            >>>     bm_list=bm_combinations,
            >>>     privileged_group=[{'gender': 1}],
            >>>     unprivileged_group=[{'gender': 0}]
            >>> )
        """
        self.bmI = bmI
        self.bmMR = BMManager(self.bmI, privileged_group, unprivileged_group)
        self.model = model
        if model is None:
            # prepare BMI to deal with transform classifier (scaler)
            self.bmI.set_transform()
            
        self.bm_list = bm_list
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group
        self.is_model_in = False

    def __in_model_run(self) -> tuple:
        """
        Run in-processing model training and prediction.
        
        This method handles the special case of in-processing bias mitigation
        techniques that have their own training procedures.
        
        Returns:
            tuple: Tuple containing (trained_model, validation_predictions, test_predictions)
            
        Example:
            >>> model, val_pred, test_pred = self.__in_model_run()
        """
        infer_model = self.model.fit(self.bmI.get_train_BLD())
        y_val_pred = self.model.predict(self.bmI.get_val_BLD())
        y_test_pred = self.model.predict(self.bmI.get_test_BLD())

        return infer_model, y_val_pred, y_test_pred

    def __warmup(self) -> None:
        """
        Initialize the bias mitigation metrics evaluator.
        
        This method sets up the BMMetrics object for evaluating fairness metrics
        across all bias mitigation combinations. It handles both standard models
        and in-processing models.
        
        Example:
            >>> self.__warmup()
            >>> # BMMetrics object is now ready for evaluation
        """
        if self.is_model_in:
            _, y_val_pred, y_test_pred = self.__in_model_run()
        else:
            y_val_pred, y_test_pred = get_model_proba(self.model, self.bmI)

        self.bmM = BMMetrics(self.bmI, dummy_model.classes_, y_val_pred, y_test_pred, 
                             self.privileged_group, self.unprivileged_group)
        

    def __is_valid_in_processing(self, in_set: list[set[BMType]]) -> tuple[bool, BMType]:
        """
        Check if in-processing bias mitigation is valid in the current combination.
        
        This method validates that in-processing techniques are properly configured
        and returns the specific in-processing type if found.
        
        Args:
            in_set (list[set[BMType]]): List of bias mitigation combinations to check
            
        Returns:
            tuple[bool, BMType]: Tuple containing (is_valid, in_processing_type)
                is_valid: Whether in-processing is valid in the combination
                in_processing_type: The specific in-processing technique found, or None
                
        Example:
            >>> is_valid, in_type = self.__is_valid_in_processing(bm_combinations)
            >>> if is_valid:
            >>>     print(f"Found in-processing technique: {in_type}")
        """
        in_type = None
        enum_count = 0
        for current_set in in_set:
            for item in current_set:
                if item.is_in:
                    enum_count += 1
                    in_type = item
        if enum_count == 0:
                return False, in_type
        return True, in_type

    def run_single_sensitive(self) -> None:
        """
        Run grid search evaluation for single sensitive attribute.
        
        This method performs a comprehensive evaluation of all bias mitigation
        combinations in the grid search. It evaluates each combination and logs
        the results for comparison. Currently supports single sensitive attribute
        evaluation, with plans for multiple sensitive attributes.
        
        The method:
        1. Validates in-processing configurations
        2. Evaluates baseline performance (no bias mitigation)
        3. Evaluates each bias mitigation combination
        4. Logs results to CSV files for analysis
        
        Raises:
            ValueError: If in-processing bias mitigation is defined with a classifier model
            
        Example:
            >>> # Define bias mitigation combinations
            >>> combinations = [
            >>>     [BMType.preReweighing],
            >>>     [BMType.preDisparate, BMType.posCEO],
            >>>     [BMType.inAdversarial]
            >>> ]
            >>> 
            >>> # Run grid search
            >>> grid_search.run_single_sensitive()
            >>> # Results are logged to CSV files
        """
        # check if in processing is possible
        is_in, in_type = self.__is_valid_in_processing(self.bm_list)
        if is_in and self.model is not None:
            raise ValueError('In processing BM defined. Combination with classifier is invalid.')

        if is_in:
            self.is_model_in = True
            if in_type == BMType.inAdversarial:
                self.model = self.bmMR.in_AD()
            elif in_type == BMType.inMeta:
                self.model = self.bmMR.in_Meta(self.bmI.get_protected_att()[0])

        # create BMMetric object
        self.__warmup()

        logger = csvLogger(f'experiment_({datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")})')
        experiment_dict = {'model':self.model.__str__().split('(')[0], 'BM':'baseline'}
        experiment_dict.update(self.bmM.get_report())
        experiment_dict.update({'fair_score':self.bmM.get_score()})

        exp_data_list = [experiment_dict]

        for c_set in self.bm_list:
            bm_name = ''
            pre_in_set = [c for c in c_set if c.is_pre]
            in_in_set = [c for c in c_set if c.is_in]
            pos_in_set = [c for c in c_set if c.is_pos]

            for c in pre_in_set:
                bm_name += f' {c.name}'
                if c == BMType.preReweighing:
                    self.bmMR.pre_Reweighing()
                if c == BMType.preDisparate:
                    self.bmMR.pre_DR(self.bmI.get_protected_att()[0])
                if c == BMType.preLFR:
                    self.bmMR.pre_LFR()

            # check if in-processing is in bm_list
            if is_in:
                # clear memory on AD
                if hasattr(self.model, 'sess'):
                    self.model.sess.close() 
                if any(in_in_set):
                    if in_type == BMType.inAdversarial:
                        self.model = self.bmMR.in_AD(debias=True)
                        bm_name += ' inAD'
                    elif in_type == BMType.inMeta:
                        self.model = self.bmMR.in_Meta(self.bmI.get_protected_att()[0], tau=0.7)
                else:
                    if in_type == BMType.inAdversarial:
                        self.model = self.bmMR.in_AD()
                _, y_val_pred, y_test_pred = self.__in_model_run()

            else:
                y_val_pred, y_test_pred = get_model_proba(self.model, self.bmI)
            
            self.bmM.set_new_pred(y_val_pred, y_test_pred)

            for c in pos_in_set:
                bm_name += f' {c.name}'
                if c == BMType.posCalibrated:
                    self.bmMR.pos_CEO(self.bmI.get_val_BLD(), self.bmI.get_test_BLD())
                elif c == BMType.posEqqOds:
                    self.bmMR.pos_EO(self.bmI.get_val_BLD(), self.bmI.get_test_BLD())
                elif c == BMType.posROC:
                    self.bmMR.pos_ROC(self.bmI.get_val_BLD(), self.bmI.get_test_BLD())

            new_exp_dict = {'model':self.model.__str__().split('(')[0], 'BM':bm_name[1:]}
            new_exp_dict.update(self.bmM.get_report())
            new_exp_dict.update({'fair_score':self.bmM.get_score()})
            exp_data_list.append(new_exp_dict)

            self.bmI.restore_BLD()

        logger(exp_data_list)
        #aggregate_csv_files('./results/', f'./results/experiment_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}.csv')
                
