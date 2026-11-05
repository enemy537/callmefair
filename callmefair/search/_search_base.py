"""Core of the bias search: attribute combination and per-attribute evaluation.

``BaseSearch.evaluate_attribute`` runs ``iterate`` independent iterations
(split, train, score - see ``_evaluation.run_iteration``) and averages their
scores. Iteration ``i`` uses split seed ``random_state + i``.
"""

from __future__ import annotations

import warnings
from enum import Enum

import pandas as pd
from tqdm import tqdm

from callmefair.search._evaluation import run_iteration, split_dataset  # noqa: F401  (re-export)
from callmefair.search.model_factory import uses_gpu
from callmefair.search.optimizer import TrainingOptimizer
from callmefair.util import device

warnings.simplefilter(action="ignore", category=FutureWarning)

ROWS_PER_THREAD = 10_000
SCORE_KEYS =("raw", "overall", "balanced_acc")


class CType(Enum):
    """Set operation used to combine two binary attributes."""
    union = 1                 # col1 OR col2
    intersection = 2          # col1 AND col2
    difference_1_minus_2 = 3  # col1 AND NOT col2
    difference_2_minus_1 = 4  # col2 AND NOT col1
    symmetric_difference = 5  # col1 XOR col2

    def __str__(self):
        return self.name


_OPERATIONS = {
    CType.union: lambda a, b: a | b,
    CType.intersection: lambda a, b: a & b,
    CType.difference_1_minus_2: lambda a, b: a & ~b,
    CType.difference_2_minus_1: lambda a, b: b & ~a,
    CType.symmetric_difference: lambda a, b: a ^ b,
}


def combine_attributes(df: pd.DataFrame, col1: str, col2: str, operation: CType) -> pd.DataFrame:
    """Replace binary columns ``col1`` and ``col2`` by one column ``'{col1}_{col2}'``.

    Raises ValueError if either column holds values other than 0/1.
    """
    if not all(df[col].isin([0, 1]).all() for col in (col1, col2)):
        raise ValueError("Columns must contain only binary values (0 or 1).")
    combined = _OPERATIONS[operation](df[col1], df[col2])
    return df.drop([col1, col2], axis=1).assign(**{f"{col1}_{col2}": combined})


class BaseSearch:
    """Evaluates the fairness of a single (possibly combined) binary attribute.

    Parameters:
        df: dataset with numeric features and a binary label.
        label_name: label column.
        n_threads: parallel workers (default: performance cores, at most 8).
        backend: 'process' or 'thread' for parallel iterations.
        random_state: base seed; iteration ``i`` splits with ``random_state + i``.
    """

    def __init__(self, df: pd.DataFrame, label_name: str, n_threads: int | None = None,
                 backend: str = "process", random_state: int = 42):
        self.df = df.copy(deep=True)
        self.label_name = label_name
        self.n_threads = (min(8, device.performance_cores()) if n_threads is None
                          else max(1, int(n_threads)))
        self.backend = backend
        self.random_state = random_state
        self.optimizer = TrainingOptimizer(backend=backend, max_workers=self.n_threads)

    def _runs_in_parallel(self, model_name: str, iterate: int) -> bool:
        if iterate < 2 or self.n_threads < 2:
            return False
        return not uses_gpu(model_name)

    @staticmethod
    def _threads_per_model(n_rows: int) -> int:
        """Threads for a lone (sequential) model: all performance cores, but only one
        per ROWS_PER_THREAD rows - threading tiny fits is slower than a single thread."""
        return max(1, min(device.performance_cores(), n_rows // ROWS_PER_THREAD))

    def evaluate_attribute(self, attribute: str, treat_umbalance: bool = False,
                           iterate: int = 10, model_name: str = "lr",
                           df_new: pd.DataFrame | None = None) -> dict:
        """Average fairness scores of ``attribute`` over ``iterate`` iterations.

        Returns {'{attribute}_raw', '{attribute}_overall', '{attribute}_balanced_acc'}.
        ``df_new`` replaces the dataset (e.g. one holding a combined attribute).
        """
        df = self.df if df_new is None else df_new
        parallel = self._runs_in_parallel(model_name, iterate)
        n_jobs = 1 if parallel else self._threads_per_model(len(df))
        args_list = [(df, self.label_name, attribute, treat_umbalance, model_name,
                      self.random_state + i, n_jobs) for i in range(iterate)]

        if parallel:
            results = self.optimizer.map(run_iteration, args_list)
            mode = f"{self.n_threads} {self.backend}s"
        else:
            results = (run_iteration(*args) for args in args_list)
            mode = "sequential"

        totals = dict.fromkeys(SCORE_KEYS, 0.0)
        for scores in tqdm(results, total=iterate, desc=f"Training {model_name} models ({mode})"):
            totals["raw"] += scores["raw_score"]
            totals["overall"] += scores["overall_score"]
            totals["balanced_acc"] += scores["balanced_accuracy"]
        return {f"{attribute}_{key}": total / iterate for key, total in totals.items()}
