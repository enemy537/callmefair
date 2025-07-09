"""
CallMeFair: A Comprehensive Framework for Automatic Bias Mitigation in AI Systems

CallMeFair provides tools and techniques to identify, measure, and reduce algorithmic bias
in machine learning models through preprocessing, in-processing, and postprocessing methods.

For more information, visit: https://github.com/your-repo/callmefair
"""

__version__ = "0.1.0"
__author__ = "CallMeFair Team"
__email__ = "support@callmefair.org"
__description__ = "A comprehensive framework for automatic bias mitigation in AI systems"

# Import main classes for easy access
from .mitigation.fair_bm import BMManager, BMType
from .mitigation.fair_grid import BMGridSearch
from .search.fair_search import BiasSearch
from .search._search_base import BaseSearch, CType, combine_attributes
from .util.fair_util import BMInterface, BMMetrics, calculate_fairness_score

__all__ = [
    'BMManager',
    'BMType', 
    'BMGridSearch',
    'BiasSearch',
    'BaseSearch',
    'CType',
    'combine_attributes',
    'BMInterface',
    'BMMetrics',
    'calculate_fairness_score'
]
