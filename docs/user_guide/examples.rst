Examples
========

A series of notebooks in ``examples/notebooks/`` walks through the library. Run them in
order; each builds on the previous one. The shared helpers live in the
``fhg_examples`` package next to them (``datasets``, ``preprocessing``, ``tradeoff``).

1. ``00_data_and_preprocessing``: turn the raw datasets into model-ready tables, define the
   privileged and unprivileged groups, and check the result against the shipped CSVs.
2. ``01_fairness_fundamentals``: split data and train with ``BMInterface``, compute SPD and
   DI by hand and with ``BMMetrics``, and combine five metrics into one fairness score.
3. ``02_bias_search``: rank sensitive attributes and their intersections with ``BiasSearch``.
4. ``03_bias_mitigation``: apply pre-, in- and post-processing mitigation by hand and with
   ``BMGridSearch``.
5. ``04_tradeoff_map``: compare mitigations with the degraded-model baseline
   (:doc:`/user_guide/tradeoff_map_guide`).
6. ``05_advanced_models_and_hardware``: compare model families, register your own, and
   choose hardware.

To run them, from a clone of the repository:

.. code-block:: bash

   uv pip install -e ".[dev]"
   python -m ipykernel install --user --name callmefair
   cd examples/notebooks && jupyter lab

Select the ``callmefair`` kernel. Raw data files are optional and only needed to reproduce
the preprocessing; see ``examples/data/raw/README.md``.
