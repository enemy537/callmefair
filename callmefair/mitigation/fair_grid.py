"""
Grid Search for Bias Mitigation Combinations

Systematic evaluation of preprocessing, in-processing and postprocessing bias
mitigation combinations for one model, with CSV logging.

Classes:
    BMGridSearch: Main grid search class for bias mitigation evaluation

Functions:
    get_model_proba: Trains a model and returns probabilities (dispatches to the
        adapters in ``callmefair.mitigation.model_adapters``)

Example:
    >>> from callmefair.mitigation.fair_grid import BMGridSearch
    >>> from callmefair.mitigation.fair_bm import BMType
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> bm_combinations = [
    >>>     [BMType.preReweighing],
    >>>     [BMType.preDisparate],
    >>>     [BMType.preReweighing, BMType.posEqqOds],
    >>> ]
    >>> grid_search = BMGridSearch(bmI=bm_interface,
    >>>                            model=RandomForestClassifier(),
    >>>                            bm_list=bm_combinations,
    >>>                            privileged_group=privileged_groups,
    >>>                            unprivileged_group=unprivileged_groups)
    >>> rows = grid_search.run_single_sensitive()
"""

from datetime import datetime

import numpy as np
from aif360.datasets import BinaryLabelDataset

from callmefair.mitigation.fair_bm import BMManager, BMType, release_model
from callmefair.mitigation.fair_log import csvLogger
from callmefair.mitigation.model_adapters import resolve_adapter
from callmefair.util.fair_util import BMInterface, BMMetrics

# Class labels handed to BMMetrics (binary problems only).
BINARY_CLASSES = np.array([0, 1])


def get_model_proba(model, bmI: BMInterface) -> tuple[np.ndarray, np.ndarray]:
    """Train ``model`` on ``bmI`` and return (validation, test) probability matrices.

    Dispatch is done by :func:`callmefair.mitigation.model_adapters.resolve_adapter`
    (extensible with ``register_adapter``). Instance weights produced by
    ``preReweighing`` reach every model whose ``fit`` accepts ``sample_weight``.

    Raises:
        ValueError: If the model cannot predict probabilities.
    """
    return resolve_adapter(model).fit_predict_proba(model, bmI)


def _model_name(model) -> str:
    return type(model).__name__


def _threshold_labels(bld: BinaryLabelDataset, threshold: float) -> BinaryLabelDataset:
    """Set ``bld.labels`` from ``bld.scores`` (favorable iff score > threshold), in place."""
    favorable = bld.scores > threshold
    bld.labels = np.where(favorable, bld.favorable_label, bld.unfavorable_label)
    return bld


class BMGridSearch:
    """Evaluate combinations of bias mitigation techniques for one model.

    Attributes:
        bmI (BMInterface): Interface for managing binary label datasets
        bmMR (BMManager): Bias mitigation manager for applying techniques
        model: Classifier to evaluate; ``None`` when an in-processing technique
            (inAdversarial / inMeta) supplies its own model.
        bm_list (list[list[BMType]]): Combinations to test
        is_model_in (bool): Whether an in-processing technique is used

    Example:
        >>> grid = BMGridSearch(bmI, RandomForestClassifier(),
        >>>                     [[BMType.preReweighing], [BMType.posEqqOds]],
        >>>                     [{'age': 1}], [{'age': 0}])
        >>> rows = grid.run_single_sensitive()
    """

    def __init__(self, bmI: BMInterface, model, bm_list: list[list[BMType]],
                 privileged_group: list[dict], unprivileged_group: list[dict]):
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

    # ---- model training / prediction -------------------------------------------------

    def _train_predict(self) -> tuple:
        """Train the current model; return (val, test) predictions (arrays, or BLDs in-processing)."""
        if self.is_model_in:
            self.model.fit(self.bmI.get_train_BLD())
            return (self.model.predict(self.bmI.get_val_BLD()),
                    self.model.predict(self.bmI.get_test_BLD()))
        return get_model_proba(self.model, self.bmI)

    @staticmethod
    def _find_in_type(bm_list) -> BMType | None:
        """Last in-processing technique found in ``bm_list`` (None if there is none)."""
        in_types = [item for current in bm_list for item in current if item.is_in]
        return in_types[-1] if in_types else None

    def _new_in_model(self, in_type: BMType, mitigate: bool):
        """Build a fresh in-processing model, releasing the previous one first.

        ``mitigate=False`` yields the unmitigated variant (AD debias=False, Meta tau=0).
        """
        if self.model is not None:
            release_model(self.model)
        if in_type == BMType.inAdversarial:
            return self.bmMR.in_AD(debias=mitigate)
        return self.bmMR.in_Meta(self.bmI.get_protected_att()[0], tau=0.7 if mitigate else 0)

    # ---- one combination, step by step -----------------------------------------------

    def _apply_pre(self, pre_set: list[BMType]) -> list[str]:
        names = []
        for c in pre_set:
            names.append(c.name)
            if c == BMType.preReweighing:
                self.bmMR.pre_Reweighing()
            elif c == BMType.preDisparate:
                self.bmMR.pre_DR(self.bmI.get_protected_att()[0])
            elif c == BMType.preLFR:
                self.bmMR.pre_LFR()
        return names

    def _select_in_model(self, in_set: list[BMType], in_type: BMType) -> list[str]:
        """Replace the in-processing model for this combination; return name parts."""
        self.model = self._new_in_model(in_type, mitigate=bool(in_set))
        if not in_set:
            return []
        return ['inAD' if in_type == BMType.inAdversarial else in_type.name]

    def _prediction_blds(self, y_val_pred, y_test_pred) -> tuple:
        """Prediction datasets: scores = P(favorable), labels thresholded at the best threshold."""
        thresh = self.bmM.best_class_thresh
        if self.is_model_in:  # predictions already are BLDs; work on copies
            val_bld = y_val_pred.copy(deepcopy=True)
            test_bld = y_test_pred.copy(deepcopy=True)
        else:
            idx = self.bmM.pos_idx
            val_bld, test_bld = self.bmI.get_val_BLD(), self.bmI.get_test_BLD()
            val_bld.scores = np.asarray(y_val_pred)[:, idx].reshape(-1, 1)
            test_bld.scores = np.asarray(y_test_pred)[:, idx].reshape(-1, 1)
        return _threshold_labels(val_bld, thresh), _threshold_labels(test_bld, thresh)

    def _apply_post(self, pos_set: list[BMType], y_val_pred, y_test_pred) -> list[str]:
        """Chain post-processors on the predictions and score the final test predictions."""
        if not pos_set:
            return []
        val_pred, test_pred = self._prediction_blds(y_val_pred, y_test_pred)
        for c in pos_set:
            val_pred, test_pred = self.bmMR.pos_apply(c, val_pred, test_pred)
        self.bmM.set_pos_pred(test_pred)
        return [c.name for c in pos_set]

    def _record(self, bm_name: str) -> dict:
        row = {'model': _model_name(self.model), 'BM': bm_name}
        row.update(self.bmM.get_report())
        score = self.bmM.get_score()
        row['fair_score'] = score['overall_score']
        row['fair_score_uncapped'] = score['uncapped_score']
        return row

    def _run_combination(self, c_set: list[BMType], in_type: BMType | None) -> dict:
        pre_set = [c for c in c_set if c.is_pre]
        in_set = [c for c in c_set if c.is_in]
        pos_set = [c for c in c_set if c.is_pos]

        names = self._apply_pre(pre_set)
        if self.is_model_in:
            names += self._select_in_model(in_set, in_set[-1] if in_set else in_type)
        y_val_pred, y_test_pred = self._train_predict()
        self.bmM.set_new_pred(y_val_pred, y_test_pred)
        names += self._apply_post(pos_set, y_val_pred, y_test_pred)
        return self._record(' '.join(names))

    # ---- public ----------------------------------------------------------------------

    def run_single_sensitive(self, results_dir: str | None = 'results') -> list[dict]:
        """Evaluate the baseline and every combination in ``bm_list``.

        Returns one dict per row (baseline first). Rows are also written to
        ``<results_dir>/experiment_<timestamp>.csv`` unless ``results_dir`` is None.

        Raises:
            ValueError: If an in-processing technique is combined with a classifier.
        """
        in_type = self._find_in_type(self.bm_list)
        if in_type is not None and self.model is not None:
            raise ValueError('In processing BM defined. Combination with classifier is invalid.')

        if in_type is not None:
            self.is_model_in = True
            self.model = self._new_in_model(in_type, mitigate=False)

        rows = []
        try:
            y_val_pred, y_test_pred = self._train_predict()
            self.bmM = BMMetrics(self.bmI, BINARY_CLASSES, y_val_pred, y_test_pred,
                                 self.privileged_group, self.unprivileged_group)
            rows.append(self._record('baseline'))
            for c_set in self.bm_list:
                try:
                    rows.append(self._run_combination(c_set, in_type))
                finally:
                    self.bmI.restore_BLD()
        finally:
            if self.is_model_in:
                release_model(self.model)

        if results_dir is not None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            csvLogger(f'experiment_{stamp}', path=results_dir)(rows)
        return rows
