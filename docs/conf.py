import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

project = 'callmefair'
author = 'Pedro Paiva et al., University of Calgary AI2Lab'
copyright = '2025, ' + author

# Read the version without importing the package.
release = re.search(r'^__version__\s*=\s*"([^"]+)"',
                    (ROOT / 'callmefair' / '__init__.py').read_text(), re.M).group(1)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

autodoc_member_order = 'bysource'
# Mock the ML stack so the docs build without installing it.
autodoc_mock_imports = [
    'tensorflow', 'keras', 'torch', 'pytorch_tabnet', 'einops', 'aif360',
    'xgboost', 'catboost', 'lightgbm', 'imblearn', 'BlackBoxAuditing',
    'sklearn', 'scipy', 'matplotlib', 'seaborn', 'prettytable', 'tqdm',
]
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

exclude_patterns = ['_build']
html_theme = 'sphinx_rtd_theme'

# Types from mocked or third-party libraries cannot be resolved.
nitpick_ignore_regex = [('py:class', r'.*')]
