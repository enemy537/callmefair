"""Helpers shared by the example notebooks (not part of the callmefair package)."""

import logging
import os
import warnings


def setup() -> None:
    """Quiet, reproducible notebook output: hide library noise, compact tables."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    import numpy as np
    import pandas as pd

    pd.set_option("display.precision", 3)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    np.random.seed(0)
