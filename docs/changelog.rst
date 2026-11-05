Changelog
=========

0.1.0 (unreleased)
------------------

Fixed
~~~~~

* Post-processing mitigations (``posCalibrated``, ``posEqqOds``, ``posROC``) were no-ops; results for ``pos*`` rows from earlier versions are invalid.
* ``BiasSearch`` iterations reused the same data split; iteration ``i`` now uses seed ``random_state + i``.
* ``evaluate_combination_average`` divided by ``iterate`` twice.
* ``preReweighing`` weights now reach any model whose ``fit`` accepts ``sample_weight``.
* TabTransformer used different encodings in training and prediction.
* F1 and MCC return 0.0 instead of NaN when the denominator is zero.
* ``BMMetrics`` no longer mutates its inputs.
* ``csvLogger`` writes a single header and no index column.
* ``inMeta`` naming and state reset.
* LightGBM segfault on macOS when torch was loaded first.
* Typo in the packaging dependencies.

Changed
~~~~~~~

* ``csvLogger`` output uses a new file name format (``experiment_<timestamp>``).
* ``BlackBoxAuditing`` is a required dependency (needed by ``preDisparate``).
* ``import callmefair`` is lazy and no longer loads TensorFlow, PyTorch or AIF360.
* TabNet runs on CUDA or CPU; MPS is not used.

Added
~~~~~

* ``callmefair.util.device`` (CUDA, MPS or CPU detection; ``CALLMEFAIR_DEVICE`` override).
* ``register_model`` for the bias search model registry.
* ``register_adapter`` for the grid search model adapter registry.
* ``delete_sources`` argument of ``aggregate_csv_files``.
* Installation extras ``dl``, ``mac``, ``cuda``, ``examples`` and ``dev``.

Removed
~~~~~~~

* ``find_best_privileged_threshold``.
