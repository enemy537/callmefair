"""
Bias Mitigation (BM) Manager Module

This module provides a comprehensive framework for applying various bias mitigation techniques
in machine learning models. It supports preprocessing, in-processing, and postprocessing
approaches to reduce algorithmic bias and promote fairness in AI systems.

The module implements multiple bias mitigation algorithms from the AIF360 library:
- Preprocessing: Reweighing, Disparate Impact Remover (DIR), Learning Fair Representations (LFR)
- In-processing: Adversarial Debiasing, MetaFair Classifier
- Postprocessing: Calibrated Equalized Odds, Equalized Odds, Reject Option Classification

Classes:
    BMType: Enumeration of available bias mitigation techniques
    BMManager: Main class for applying bias mitigation methods to datasets

Example:
    >>> from callmefair.mitigation.fair_bm import BMManager, BMType
    >>> from callmefair.util.fair_util import BMInterface
    >>> 
    >>> # Initialize with your data
    >>> bm_manager = BMManager(bm_interface, privileged_groups, unprivileged_groups)
    >>> 
    >>> # Apply preprocessing bias mitigation
    >>> bm_manager.pre_Reweighing()
    >>> 
    >>> # Apply in-processing bias mitigation
    >>> ad_model = bm_manager.in_AD(debias=True)
    >>> 
    >>> # Apply postprocessing bias mitigation
    >>> mitigated_predictions = bm_manager.pos_CEO(valid_pred, test_pred)
"""

from callmefair.util.fair_util import BMInterface
# Fairness Dataset
from aif360.datasets import BinaryLabelDataset
# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover, LFR
from aif360.algorithms.inprocessing import AdversarialDebiasing, MetaFairClassifier
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import EqOddsPostprocessing, RejectOptionClassification
from aif360.algorithms import Transformer
# ML/Models
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    tf = None

from enum import Enum

import string
import random

def get_random_string(N: int = 7) -> str:
    """
    Generate a random string of specified length.
    
    Args:
        N (int): Length of the random string to generate. Defaults to 7.
        
    Returns:
        str: Random string containing uppercase and lowercase letters.
        
    Example:
        >>> get_random_string(5)
        'AxK9m'
    """
    return ''.join(random.choices(string.ascii_lowercase +
                             string.ascii_uppercase, k=N))

class BMType(Enum):
    """
    Enumeration of bias mitigation techniques.
    
    This enum categorizes bias mitigation methods into three main groups:
    - Preprocessing: Applied to training data before model training
    - In-processing: Applied during model training
    - Postprocessing: Applied to model predictions after training
    
    Attributes:
        preReweighing: Reweighing preprocessing technique
        preDisparate: Disparate Impact Remover preprocessing technique
        preLFR: Learning Fair Representations preprocessing technique
        inAdversarial: Adversarial Debiasing in-processing technique
        inMeta: MetaFair Classifier in-processing technique
        posCalibrated: Calibrated Equalized Odds postprocessing technique
        posEqqOds: Equalized Odds postprocessing technique
        posROC: Reject Option Classification postprocessing technique
    """
    preReweighing = 1
    preDisparate = 2
    preLFR = 3
    inAdversarial = 4
    inMeta = 5
    posCalibrated = 6
    posEqqOds = 7
    posROC = 8

    @property
    def is_pre(self) -> bool:
        """
        Check if the bias mitigation technique is a preprocessing method.
        
        Returns:
            bool: True if the technique is preprocessing, False otherwise.
        """
        return self in frozenset((BMType.preDisparate, BMType.preReweighing, BMType.preLFR))
    
    @property
    def is_in(self) -> bool:
        """
        Check if the bias mitigation technique is an in-processing method.
        
        Returns:
            bool: True if the technique is in-processing, False otherwise.
        """
        return self in frozenset((BMType.inAdversarial, BMType.inMeta))
    
    @property
    def is_pos(self) -> bool:
        """
        Check if the bias mitigation technique is a postprocessing method.
        
        Returns:
            bool: True if the technique is postprocessing, False otherwise.
        """
        return self in frozenset((BMType.posCalibrated, BMType.posEqqOds, BMType.posROC))


class BMManager:
    """
    Bias Mitigation Manager for applying various fairness techniques.
    
    This class provides a unified interface for applying different bias mitigation
    techniques to machine learning datasets. It supports preprocessing, in-processing,
    and postprocessing approaches to reduce algorithmic bias.
    
    The manager works with BinaryLabelDataset objects from the AIF360 library and
    provides methods for each type of bias mitigation technique.
    
    Attributes:
        bmI (BMInterface): Interface for managing binary label datasets
        privileged_group (list[dict]): List of dictionaries defining privileged groups
        unprivileged_group (list[dict]): List of dictionaries defining unprivileged groups
    
    Example:
        >>> from callmefair.util.fair_util import BMInterface
        >>> from callmefair.mitigation.fair_bm import BMManager
        >>> 
        >>> # Initialize with your data
        >>> bm_interface = BMInterface(train_df, val_df, test_df, 'label', ['protected_attr'])
        >>> privileged_groups = [{'protected_attr': 1}]
        >>> unprivileged_groups = [{'protected_attr': 0}]
        >>> 
        >>> bm_manager = BMManager(bm_interface, privileged_groups, unprivileged_groups)
        >>> 
        >>> # Apply preprocessing bias mitigation
        >>> bm_manager.pre_Reweighing()
    """
    
    def __init__(self, bmI: BMInterface, privileged_group: list[dict], unprivileged_group: list[dict]):
        """
        Initialize the Bias Mitigation Manager.
        
        Args:
            bmI (BMInterface): Interface for managing binary label datasets
            privileged_group (list[dict]): List of dictionaries defining privileged groups.
                Each dict should contain protected attribute names and their privileged values.
            unprivileged_group (list[dict]): List of dictionaries defining unprivileged groups.
                Each dict should contain protected attribute names and their unprivileged values.
                
        Example:
            >>> privileged_groups = [{'gender': 1, 'race': 1}]
            >>> unprivileged_groups = [{'gender': 0, 'race': 0}]
            >>> bm_manager = BMManager(bm_interface, privileged_groups, unprivileged_groups)
        """
        self.bmI = bmI
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group

    def pre_Reweighing(self) -> None:
        """
        Apply Reweighing preprocessing bias mitigation.
        
        Reweighing is a preprocessing technique that assigns different weights to
        instances based on their group membership to achieve statistical parity.
        This method modifies the training dataset by reweighting instances.
        
        The method fits the reweighing algorithm on the training data and transforms
        it to create a fairer training dataset.
        
        Example:
            >>> bm_manager.pre_Reweighing()
            >>> # The training dataset is now reweighted for fairness
        """
        rw = Reweighing(unprivileged_groups=self.unprivileged_group,
                        privileged_groups=self.privileged_group)

        rw.fit(self.bmI.get_train_BLD())
        train_BLD_rw = rw.transform(self.bmI.get_train_BLD())
        self.bmI.pre_BM_set(train_BLD_rw)

    def pre_DR(self, sensitive_attribute: str) -> None:
        """
        Apply Disparate Impact Remover (DIR) preprocessing bias mitigation.
        
        DIR is a preprocessing technique that repairs the training data to remove
        disparate impact while preserving the utility of the data. It works by
        learning a repair transformation that minimizes the difference in
        conditional distributions between groups.
        
        Args:
            sensitive_attribute (str): Name of the sensitive attribute to repair.
                This should be one of the protected attributes in your dataset.
                
        Example:
            >>> bm_manager.pre_DR('gender')
            >>> # The training dataset is now repaired for the 'gender' attribute
        """
        di = DisparateImpactRemover(repair_level=0.1,
                                    sensitive_attribute=sensitive_attribute)

        train_BLD_di = di.fit_transform(self.bmI.get_train_BLD())
        self.bmI.pre_BM_set(train_BLD_di)

    def pre_LFR(self) -> None:
        """
        Apply Learning Fair Representations (LFR) preprocessing bias mitigation.
        
        LFR is a preprocessing technique that learns a fair representation of the
        data that removes information about the sensitive attributes while preserving
        the utility for the prediction task. It uses an adversarial approach to
        learn representations that are fair across different groups.
        
        Example:
            >>> bm_manager.pre_LFR()
            >>> # The training dataset now has fair representations
        """
        lfr = LFR(unprivileged_groups=self.unprivileged_group, privileged_groups=self.privileged_group)
        train_BLD_lfr = lfr.fit_transform(self.bmI.get_train_BLD())
        self.bmI.pre_BM_set(train_BLD_lfr)

    def in_AD(self, debias: bool = False) -> AdversarialDebiasing:
        """
        Create an Adversarial Debiasing in-processing model.
        
        Adversarial Debiasing is an in-processing technique that trains a classifier
        while simultaneously training an adversary that tries to predict the sensitive
        attribute from the classifier's predictions. This encourages the classifier
        to make predictions that are independent of the sensitive attribute.
        
        Args:
            debias (bool): Whether to apply debiasing. If True, the model will be
                trained to be fair. If False, it will be a standard classifier.
                Defaults to False.
                
        Returns:
            AdversarialDebiasing: Configured adversarial debiasing model.
            
        Raises:
            ImportError: If TensorFlow is not available.
            
        Example:
            >>> ad_model = bm_manager.in_AD(debias=True)
            >>> # Train the model with your data
            >>> ad_model.fit(train_data, train_labels)
        """
        if tf is None:
            raise ImportError("TensorFlow is required for Adversarial Debiasing")
            
        sess = tf.compat.v1.Session()

        ad = AdversarialDebiasing(privileged_groups=self.privileged_group,
                                    unprivileged_groups=self.unprivileged_group,
                                    scope_name=get_random_string(),
                                    debias=debias,
                                    sess=sess,
                                    adversary_loss_weight=1.2,
                                    batch_size=64)
        return ad
    
    def in_Meta(self, sensitive_attribute: str, tau: float = 0) -> MetaFairClassifier:
        """
        Create a MetaFair Classifier in-processing model.
        
        MetaFair is an in-processing technique that uses a meta-learning approach
        to learn fair classifiers. It optimizes for both accuracy and fairness
        using a regularization term that penalizes unfairness.
        
        Args:
            sensitive_attribute (str): Name of the sensitive attribute to consider
                for fairness. This should be one of the protected attributes.
            tau (float): Fairness parameter that controls the trade-off between
                accuracy and fairness. Higher values prioritize fairness.
                Defaults to 0.
                
        Returns:
            MetaFairClassifier: Configured MetaFair classifier.
            
        Example:
            >>> meta_model = bm_manager.in_Meta('gender', tau=0.1)
            >>> # Train the model with your data
            >>> meta_model.fit(train_data, train_labels)
        """
        meta = MetaFairClassifier(tau=tau, sensitive_attr=sensitive_attribute, type="fdr")
        return meta
    
    def __pos_abstract(self, pos_classifier: Transformer, valid_BLD_pred: BinaryLabelDataset,
                        test_BLD_pred: BinaryLabelDataset) -> BinaryLabelDataset:
        """
        Abstract method for applying postprocessing bias mitigation.
        
        This is a helper method that provides common functionality for all
        postprocessing techniques. It fits the postprocessing classifier on
        validation data and predictions, then applies it to test predictions.
        
        Args:
            pos_classifier (Transformer): The postprocessing classifier to apply
            valid_BLD_pred (BinaryLabelDataset): Validation dataset with predictions
            test_BLD_pred (BinaryLabelDataset): Test dataset with predictions
                
        Returns:
            BinaryLabelDataset: Postprocessed test predictions
        """
        pos_classifier = pos_classifier.fit(self.bmI.get_val_BLD(), valid_BLD_pred)
        test_pos_pred = pos_classifier.predict(test_BLD_pred)
        self.bmI.pos_bm_set(test_pos_pred)
        return test_pos_pred

    def pos_CEO(self, valid_BLD_pred: BinaryLabelDataset, test_BLD_pred: BinaryLabelDataset) -> BinaryLabelDataset:
        """
        Apply Calibrated Equalized Odds (CEO) postprocessing bias mitigation.
        
        CEO is a postprocessing technique that adjusts model predictions to achieve
        equalized odds while maintaining calibration. It ensures that the true
        positive and false positive rates are equal across different groups.
        
        Args:
            valid_BLD_pred (BinaryLabelDataset): Validation dataset with model predictions
            test_BLD_pred (BinaryLabelDataset): Test dataset with model predictions
                
        Returns:
            BinaryLabelDataset: Postprocessed test predictions with equalized odds
            
        Example:
            >>> mitigated_predictions = bm_manager.pos_CEO(valid_pred, test_pred)
            >>> # The predictions now satisfy equalized odds
        """
        cost_constraint = "fpr"

        CEO = CalibratedEqOddsPostprocessing(unprivileged_groups=self.unprivileged_group,
                                            privileged_groups=self.privileged_group,
                                            cost_constraint=cost_constraint)
        return self.__pos_abstract(CEO, valid_BLD_pred, test_BLD_pred)

    def pos_EO(self, valid_BLD_pred: BinaryLabelDataset, test_BLD_pred: BinaryLabelDataset) -> BinaryLabelDataset:
        """
        Apply Equalized Odds (EO) postprocessing bias mitigation.
        
        EO is a postprocessing technique that adjusts model predictions to achieve
        equalized odds. It ensures that the true positive and false positive rates
        are equal across different groups, without considering calibration.
        
        Args:
            valid_BLD_pred (BinaryLabelDataset): Validation dataset with model predictions
            test_BLD_pred (BinaryLabelDataset): Test dataset with model predictions
                
        Returns:
            BinaryLabelDataset: Postprocessed test predictions with equalized odds
            
        Example:
            >>> mitigated_predictions = bm_manager.pos_EO(valid_pred, test_pred)
            >>> # The predictions now satisfy equalized odds
        """
        EO = EqOddsPostprocessing(unprivileged_groups=self.unprivileged_group, 
                                 privileged_groups=self.privileged_group)
        
        return self.__pos_abstract(EO, valid_BLD_pred, test_BLD_pred)

    def pos_ROC(self, valid_BLD_pred: BinaryLabelDataset, test_BLD_pred: BinaryLabelDataset) -> BinaryLabelDataset:
        """
        Apply Reject Option Classification (ROC) postprocessing bias mitigation.
        
        ROC is a postprocessing technique that rejects predictions for instances
        where the model is uncertain, particularly when this uncertainty is
        correlated with the sensitive attribute. This helps reduce bias by
        abstaining from making predictions on potentially unfair cases.
        
        Args:
            valid_BLD_pred (BinaryLabelDataset): Validation dataset with model predictions
            test_BLD_pred (BinaryLabelDataset): Test dataset with model predictions
                
        Returns:
            BinaryLabelDataset: Postprocessed test predictions with reject option
            
        Example:
            >>> mitigated_predictions = bm_manager.pos_ROC(valid_pred, test_pred)
            >>> # Some predictions may be rejected to improve fairness
        """
        ROC = RejectOptionClassification(unprivileged_groups=self.unprivileged_group,
                                         privileged_groups=self.privileged_group)
        
        return self.__pos_abstract(ROC, valid_BLD_pred, test_BLD_pred)