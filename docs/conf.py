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
    'matplotlib.sphinxext.plot_directive',
]

autodoc_member_order = 'bysource'
# Mock the ML stack so the docs build without installing it. numpy, pandas and matplotlib
# are real: the plot directive draws figures with them.
autodoc_mock_imports = [
    'tensorflow', 'keras', 'torch', 'pytorch_tabnet', 'einops', 'aif360',
    'xgboost', 'catboost', 'lightgbm', 'imblearn', 'BlackBoxAuditing',
    'sklearn', 'scipy', 'seaborn', 'prettytable', 'tqdm',
]
# Plot directive: figures are generated at build time into _build only.
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [('png', 110)]
plot_rcparams = {'figure.max_open_warning': 0}

# ``callmefair.viz`` imports callmefair.util.fair_util, which needs aif360 and
# scikit-learn only for other code paths; mock them for the plot scripts as well.
from sphinx.ext.autodoc.mock import mock as _mock  # noqa: E402
_mock(['aif360', 'sklearn', 'scipy']).__enter__()

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
