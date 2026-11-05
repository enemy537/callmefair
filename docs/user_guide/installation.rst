Installation
============

Requires Python 3.11 or newer.

.. code-block:: bash

   uv venv --python 3.11 && source .venv/bin/activate
   uv pip install "callmefair[dl]"

.. note::

   Until 0.1.0 is published, PyPI serves 0.0.1 (no declared dependencies and
   the bugs listed in :doc:`/changelog`). Install from source instead:
   ``uv pip install "callmefair[dl] @ git+https://github.com/uofc-ai2lab/callmefair.git"``.

With pip: ``pip install "callmefair[dl]"``. The core install covers
scikit-learn, XGBoost, LightGBM, CatBoost and AIF360; ``dl`` adds the deep
models (TabTransformer, MLP, AdversarialDebiasing, TabNet).

For development, from a clone of
`the repository <https://github.com/uofc-ai2lab/callmefair/>`_:

.. code-block:: bash

   uv pip install -e ".[dev]"

Extras
------

* ``dl``: tensorflow, torch, pytorch-tabnet, einops
* ``mac``: tensorflow-metal (optional Apple GPU support for TensorFlow)
* ``cuda``: tensorflow[and-cuda] (Linux)
* ``examples``: shapely, plotly, kaleido, statsmodels, more-itertools, threadpoolctl
* ``dev``: ``dl`` and ``examples`` plus pytest, jupyter, ipykernel

macOS
-----

LightGBM and XGBoost need OpenMP:

.. code-block:: bash

   brew install libomp

Linux with NVIDIA GPU
---------------------

Install PyTorch from the CUDA index first, then the extras:

.. code-block:: bash

   uv pip install torch --index-url https://download.pytorch.org/whl/cu121
   uv pip install "callmefair[dl,cuda]"

Device selection (including ``CALLMEFAIR_DEVICE``) is described in
:doc:`/user_guide/hardware_guide`.

Notebooks and verification
--------------------------

.. code-block:: bash

   python -m ipykernel install --user --name callmefair

.. code-block:: python

   import callmefair
   print(callmefair.__version__)
