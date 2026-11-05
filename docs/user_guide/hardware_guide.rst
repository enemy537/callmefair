Hardware Guide
==============

:mod:`callmefair.util.device` is the single place that decides where models run. Installing
GPU-enabled packages is covered in :doc:`/user_guide/installation`.

Detection
---------

``device.detect()`` returns ``'cuda'``, ``'mps'`` or ``'cpu'``, in this order:

1. ``CALLMEFAIR_DEVICE`` (``cuda``, ``mps`` or ``cpu``; anything else raises ``ValueError``);
2. CUDA, if ``torch.cuda.is_available()`` or ``nvidia-smi`` succeeds;
3. MPS, on Apple Silicon with PyTorch's MPS available;
4. CPU.

The result is cached (``lru_cache``): set the environment variable before the first call,
i.e. before starting the process.

.. code-block:: console

    $ python -c "from callmefair.util import device as d; print(d.detect(), d.performance_cores(), d.xgboost_params(), d.tabnet_device())"
    mps 8 {'tree_method': 'hist', 'device': 'cpu'} cpu

Settings per library
--------------------

=============  ===========================  ================  ================
Library        cuda                         mps               cpu
=============  ===========================  ================  ================
XGBoost        ``device='cuda'``            ``device='cpu'``  ``device='cpu'``
CatBoost       ``task_type='GPU'``          CPU               CPU
LightGBM       CPU (``device='gpu'`` if     CPU               CPU
               ``CALLMEFAIR_LGBM_GPU=1``)
TabNet         ``cuda``                     ``cpu``           ``cpu``
=============  ===========================  ================  ================

TabNet stays on CPU under MPS: no speed-up was measured and MPS rejects float64 inputs.
LightGBM wheels are CPU builds, so GPU use is opt-in for a GPU-enabled build. Keras models
(``mlp``, ``tabtransformer``) use a GPU only if TensorFlow sees one: CUDA on Linux, or on
macOS the ``mac`` extra (``tensorflow-metal``); otherwise they run on CPU.

Forcing a device (checked on this machine; ``CALLMEFAIR_DEVICE=cuda`` yields
``{'tree_method': 'hist', 'device': 'cuda'}`` and CatBoost ``{'task_type': 'GPU'}``):

.. code-block:: console

    $ CALLMEFAIR_DEVICE=cpu python your_script.py

``device.performance_cores()`` returns the Apple performance-core count (via ``sysctl``),
else ``os.cpu_count()``; :class:`callmefair.search.fair_search.BiasSearch` uses
``min(8, performance_cores())`` workers by default.

Threading and processes
-----------------------

* In a process pool each worker trains with one thread, and the pool sets ``OMP_NUM_THREADS``,
  ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` and ``NUMEXPR_NUM_THREADS`` to 1. A lone
  (sequential) model uses up to all performance cores, one per 10,000 rows.
* GPU-backed models run sequentially, one at a time.
* macOS starts workers with *spawn*, so scripts must put the entry point under
  ``if __name__ == "__main__":``.

macOS: LightGBM and PyTorch
---------------------------

PyTorch ships its own ``libomp``. If it loads before LightGBM's, multi-threaded LightGBM
training segfaults later in the same process. In code that uses both, call
``device.import_torch()`` instead of ``import torch`` (it imports LightGBM first on macOS), or
import ``lightgbm`` before ``torch`` yourself.

.. code-block:: python

    from callmefair.util import device

    torch = device.import_torch()

AdversarialDebiasing
--------------------

AdversarialDebiasing (from AIF360) uses TensorFlow 1 graph mode and its own session; it does
not use the settings above, and runs on GPU only if TensorFlow sees one.
