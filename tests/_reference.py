"""Loads the ORIGINAL search implementation (pre-refactor snapshot) as package ``refcmf``.

The sources are read from git (``ORIGINAL_COMMIT``) into a temp dir with their
imports rewritten, so they can be compared against the refactored code.
"""

import importlib
import re
import subprocess
import sys
from pathlib import Path

ORIGINAL_COMMIT = "e2490ef"
ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "callmefair/search/_search_base.py": "_search_base.py",
    "callmefair/search/optimizer.py": "optimizer.py",
    "callmefair/search/fair_search.py": "fair_search.py",
    "callmefair/util/fair_util.py": "fair_util.py",
}
REWRITES = [("callmefair.search.optimizer", "refcmf.optimizer"),
            ("callmefair.util.fair_util", "refcmf.fair_util"),
            ("callmefair.search._search_base", "refcmf._search_base")]


def load_original(target_dir: Path):
    """Return the module ``refcmf._search_base`` / ``refcmf.fair_search`` pair."""
    pkg_dir = Path(target_dir) / "refcmf"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("")
    for src, dst in FILES.items():
        text = subprocess.run(["git", "show", f"{ORIGINAL_COMMIT}:{src}"], cwd=ROOT,
                              check=True, capture_output=True, text=True).stdout
        for old, new in REWRITES:
            text = text.replace(old, new)
        (pkg_dir / dst).write_text(text)
    sys.path.insert(0, str(target_dir))
    importlib.invalidate_caches()
    return (importlib.import_module("refcmf._search_base"),
            importlib.import_module("refcmf.fair_search"))
