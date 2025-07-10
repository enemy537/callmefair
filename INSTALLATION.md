# Installation Guide

## Installing callmefair

`callmefair` is a comprehensive framework for automatic bias mitigation in AI systems. You can install it using pip.

### Basic Installation

```bash
pip install callmefair
```

### Installation with Optional Dependencies

For full functionality including all bias mitigation algorithms and visualization tools:

```bash
pip install callmefair[all]
```

### Development Installation

If you want to install the latest development version from GitHub:

```bash
pip install git+https://github.com/uofc-ai2lab/callmefair.git
```

### Requirements

- Python 3.9 or higher
- See `requirements.txt` for full dependency list

### Dependencies

The package includes the following key dependencies:

- **Machine Learning**: scikit-learn, xgboost, catboost, tensorflow
- **Fairness**: aif360, BlackBoxAuditing, pytorch-tabnet
- **Data Processing**: pandas, numpy, imblearn
- **Visualization**: matplotlib, seaborn
- **Utilities**: tqdm, prettytable, shapely

### Verification

After installation, you can verify the installation by running:

```python
import callmefair
print(callmefair.__version__)
```

### Troubleshooting

If you encounter installation issues:

1. **Upgrade pip**: `pip install --upgrade pip`
2. **Install build tools**: `pip install build wheel setuptools`
3. **Check Python version**: Ensure you're using Python 3.9+

### Examples

See the `examples/` directory for comprehensive usage examples and tutorials.

### Documentation

For detailed documentation, visit: [https://callmefair.readthedocs.io/](https://callmefair.readthedocs.io/) 