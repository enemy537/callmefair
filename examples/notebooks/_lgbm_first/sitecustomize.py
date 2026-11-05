# Imported automatically by every Python process whose PYTHONPATH contains this folder (see Bias_Search_Example.ipynb).
# On macOS arm64, LightGBM segfaults if it is imported AFTER torch/tensorflow in the same process. callmefair imports
# tensorflow/torch before lightgbm, so spawned worker processes (BiasSearch uses a process pool) crash with
# BrokenProcessPool for model_name='lgbm'. Importing lightgbm first in each worker avoids this.
try:
    import lightgbm  # noqa: F401
except Exception:
    pass
