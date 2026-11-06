Changelog
=========

0.1.0 (unreleased)
------------------

Added
~~~~~

* ``callmefair.viz`` plotting module (trade-off scatter, heatmap, mitigation effect, fair ranges, bias search, group rates, threshold sweep, trade-off map).
* ``callmefair.tradeoff``: paper-faithful Fairea trade-off map (Hort et al., 2021): mutation baseline averaged over splits and repetitions (``create_baseline``), strategy selection, normalisation by the baseline's range, the paper's five regions and Fig. 3 areas (``tradeoff_regions``). Defaults: balanced accuracy and ``|SPD|``; ``'acc'``, ``'aod'``, ``'eod'`` and ``'fair_score'`` are options.
* ``BMMetrics.threshold_curve()``.
* ``METRIC_RANGES`` and ``METRIC_OPTIMA`` in ``callmefair.util.fair_util``.

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
* Example trade-off code (``examples/Fairea``, now ``callmefair.tradeoff``): regions were drawn with an inverted accuracy axis, so they were flipped (a method with the original accuracy and much lower bias was labelled "bad trade-off"); baseline and methods were normalised over different ranges; the composite fairness score made the baseline non-linear and saturating; plain-accuracy baselines on imbalanced data were invalid (mutating to the majority label raises accuracy; ``baseline_curve`` now warns); ``compute_area`` raised on most inputs. Trade-off maps made with the old code should be re-examined.
* ``shapely`` is no longer a dependency of the library (only a dev/test extra).
* Stroke data pipeline: the old ``stroke.csv`` merged a NearMiss-undersampled training split with untouched validation/test rows (leakage), used an age cut-off (73) tuned on the outcome labels, and integer-coded nominal features. ``work_type`` and ``smoking_status`` are now one-hot and ``age`` is 1 for 65 or older.

Changed
~~~~~~~

* ``csvLogger`` output uses a new file name format (``experiment_<timestamp>``).
* CatBoost no longer writes a ``catboost_info/`` folder (``allow_writing_files=False``).
* ``examples/data/stroke_pre/stroke.csv`` was replaced: 5,109 rows x 18 columns, 4.87% positive (was 1,883 x 11, 13.2%). ``stroke_age_ever_married*.csv`` were removed. Documentation numbers were regenerated.
* ``BlackBoxAuditing`` is a required dependency (needed by ``preDisparate``).
* ``import callmefair`` is lazy and no longer loads TensorFlow, PyTorch or AIF360.
* TabNet runs on CUDA or CPU; MPS is not used.

Added
~~~~~

* ``callmefair.util.device`` (CUDA, MPS or CPU detection; ``CALLMEFAIR_DEVICE`` override).
* ``register_model`` for the bias search model registry.
* ``register_adapter`` for the grid search model adapter registry.
* ``delete_sources`` argument of ``aggregate_csv_files``.
* ``BMGridSearch.run_single_sensitive(results_dir=None)`` skips CSV logging.
* ``progress=False`` on ``BiasSearch`` disables the tqdm bars.
* Example notebook series ``00`` to ``05`` and the ``fhg_examples`` helper package (datasets, preprocessing, trade-off map); the preprocessing reproduces the shipped CSVs from raw files (``examples/data/raw/README.md``).
* Models registered in a notebook are not visible to worker processes; use ``BiasSearch(..., backend="thread")`` for them.
* Installation extras ``dl``, ``mac``, ``cuda``, ``examples`` and ``dev``.

Removed
~~~~~~~

* ``find_best_privileged_threshold``.
