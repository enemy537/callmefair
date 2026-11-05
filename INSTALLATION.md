# Installation

Requires **Python 3.11+**. Examples use [uv](https://docs.astral.sh/uv/); plain `pip` works the same way.

## Quick start

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install "callmefair[dl]"
```

The core install (no extras) covers scikit-learn, XGBoost, LightGBM, CatBoost and AIF360. The
`dl` extra adds TensorFlow (TabTransformer, MLP, AdversarialDebiasing), PyTorch and TabNet.

From a clone (development):

```bash
git clone https://github.com/uofc-ai2lab/callmefair.git && cd callmefair
uv pip install -e ".[dev]"      # dl + examples + pytest + jupyter
pytest tests/test_models.py
```

## Extras

| Extra | Contents |
|-------|----------|
| `dl` | tensorflow, torch, pytorch-tabnet, einops |
| `mac` | tensorflow-metal (Apple GPU for TensorFlow, optional) |
| `cuda` | tensorflow[and-cuda] (Linux) |
| `examples` | shapely, plotly, kaleido, statsmodels, more-itertools, threadpoolctl |
| `dev` | `dl` + `examples` + pytest, jupyter, ipykernel |

`requirements.txt` mirrors core + `dl` + `examples` for a pinned-range environment.

## macOS (Apple Silicon)

```bash
brew install libomp                  # required by LightGBM and XGBoost
uv pip install "callmefair[dl]"
```

PyTorch (TabNet) uses the MPS GPU automatically. TensorFlow runs on CPU unless you add
`uv pip install "callmefair[mac]"` (tensorflow-metal); it works with TensorFlow 2.17, but for small
tabular models CPU is often just as fast.

## Linux + NVIDIA GPU

Install PyTorch from the CUDA index first, then the rest:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install "callmefair[dl,cuda]"       # cuda = tensorflow[and-cuda]
```

XGBoost and CatBoost are switched to GPU automatically when CUDA is detected.

## Choosing the device

Detection is automatic (CUDA, then MPS, then CPU). Override it with:

```bash
export CALLMEFAIR_DEVICE=cpu     # cuda | mps | cpu
```

## Verify

```python
import callmefair
print(callmefair.__version__)
```

`import callmefair` is lightweight; TensorFlow and PyTorch load only when a class that needs them is used.

## Docker

```bash
docker build -t callmefair .
docker run --rm -it -v "$PWD":/workspace callmefair python
```

The image is CPU-only. Docs: https://callmefair.readthedocs.io/
