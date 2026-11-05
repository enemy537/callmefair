# Installation

Requires Python 3.11+. Quick start with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"        # from a clone of this repository
brew install libomp               # macOS only (LightGBM/XGBoost)
```

Extras (`dl`, `mac`, `cuda`, `examples`, `dev`), Linux + CUDA setup and device
selection are documented in
[docs/user_guide/installation.rst](docs/user_guide/installation.rst) and
[docs/user_guide/hardware_guide.rst](docs/user_guide/hardware_guide.rst).
