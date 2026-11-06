"""
CallMeFair: A Comprehensive Framework for Automatic Bias Mitigation in AI Systems

Tools to identify, measure and reduce algorithmic bias in machine learning
models through preprocessing, in-processing and postprocessing methods.

More information: https://github.com/uofc-ai2lab/callmefair/

Public names are imported lazily (PEP 562) so ``import callmefair`` does not
pull in TensorFlow, PyTorch or AIF360 until a class is actually used.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.1.0"  # single source of truth; read by hatch (see pyproject.toml)
__author__ = "CallMeFair Team"
__description__ = "A comprehensive framework for automatic bias mitigation in AI systems"

_EXPORTS = {
    "BMManager": ".mitigation.fair_bm",
    "BMType": ".mitigation.fair_bm",
    "BMGridSearch": ".mitigation.fair_grid",
    "BiasSearch": ".search.fair_search",
    "BaseSearch": ".search._search_base",
    "CType": ".search._search_base",
    "combine_attributes": ".search._search_base",
    "BMInterface": ".util.fair_util",
    "BMMetrics": ".util.fair_util",
    "calculate_fairness_score": ".util.fair_util",
    "audit_predictions": ".audit",
    "audit_file": ".audit",
    "BiasReport": ".audit",
}

__all__ = list(_EXPORTS)

if TYPE_CHECKING:
    from .audit import BiasReport, audit_file, audit_predictions
    from .mitigation.fair_bm import BMManager, BMType
    from .mitigation.fair_grid import BMGridSearch
    from .search._search_base import BaseSearch, CType, combine_attributes
    from .search.fair_search import BiasSearch
    from .util.fair_util import BMInterface, BMMetrics, calculate_fairness_score


def __getattr__(name):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted([*globals(), *_EXPORTS])
