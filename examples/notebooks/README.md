# Example notebooks

A step-by-step series on real healthcare data. Each notebook builds on the previous one.

| # | Notebook | Dataset | What it shows |
|---|---|---|---|
| 00 | [Data and preprocessing](00_data_and_preprocessing.ipynb) | all | Raw data to model-ready tables; labels; privileged / unprivileged groups and why |
| 01 | [Fairness fundamentals](01_fairness_fundamentals.ipynb) | diabetes | Group metrics (SPD, DI, EOD, AOD, TI), threshold selection, the composite score 𝓕 |
| 02 | [Bias search](02_bias_search.ipynb) | WiDS | Which sensitive attribute (or intersection) shows the most bias |
| 03 | [Bias mitigation](03_bias_mitigation.ipynb) | diabetes | Pre-, in- and post-processing, then a grid search over models × mitigations |
| 04 | [Trade-off map](04_tradeoff_map.ipynb) | diabetes | Is a mitigation better than simply degrading the model? |
| 05 | [Advanced models and hardware](05_advanced_models_and_hardware.ipynb) | stroke | Model zoo, custom models and adapters, CUDA / MPS / CPU, intersectional grid |
| 06 | [Audit external predictions](06_audit_external_predictions.ipynb) | stroke | Check the bias of any classifier's predictions from a CSV, in Python or from the command line |

## Running

```bash
uv pip install -e ".[dev]"                       # from the repository root
python -m ipykernel install --user --name callmefair
cd examples/notebooks && jupyter lab
```

Notebooks run from this folder (they import the small `fhg_examples` helper package) and
write no files. To re-run the preprocessing from raw data, see
[`../data/raw/README.md`](../data/raw/README.md).
