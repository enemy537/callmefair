"""Plots for bias-mitigation results (matplotlib is imported lazily, on first use).

Every ``plot_*`` function returns the matplotlib Figure and never shows or saves it.
"""

from importlib import import_module

_EXPORTS = {
    "pareto_front": ".plots",
    "plot_tradeoff": ".plots",
    "plot_metric_heatmap": ".plots",
    "plot_mitigation_effect": ".plots",
    "plot_fair_ranges": ".plots",
    "plot_bias_search": ".plots",
    "plot_group_rates": ".plots",
    "plot_threshold_sweep": ".plots",
    "plot_tradeoff_map": ".tradeoff_map",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
