"""The example preprocessing must reproduce the processed CSVs shipped in examples/data."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "notebooks"))

from fhg_examples.datasets import RAW_DIR, SPECS  # noqa: E402
from fhg_examples.preprocessing import PREPROCESSORS, matches_processed  # noqa: E402


@pytest.mark.parametrize("name", sorted(PREPROCESSORS))
def test_preprocessing_reproduces_shipped_csv(name):
    if not (RAW_DIR / SPECS[name].raw_file).exists():
        pytest.skip(f"raw file for '{name}' not downloaded (see examples/data/raw/README.md)")
    assert matches_processed(name)
