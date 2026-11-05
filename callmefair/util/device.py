"""Hardware detection and per-library accelerator settings.

Single source of truth for "where should this model run?". Every library gets
its parameters from here, so platform logic (CUDA on Linux, Apple Silicon MPS
on macOS, plain CPU elsewhere) lives in exactly one place.

Override detection with the ``CALLMEFAIR_DEVICE`` environment variable
(``cuda``, ``mps`` or ``cpu``), e.g. to benchmark or to force CPU in CI.

Heavy frameworks (torch, tensorflow) are imported lazily so importing
callmefair stays cheap - important for process pools on macOS, which spawn
fresh interpreters for every worker.
"""

from __future__ import annotations

import os
import platform
import subprocess
from functools import lru_cache

CUDA, MPS, CPU = "cuda", "mps", "cpu"
_VALID = (CUDA, MPS, CPU)


def _torch_has(kind: str) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if kind == CUDA:
        return torch.cuda.is_available()
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def _nvidia_smi_present() -> bool:
    try:
        subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, timeout=5)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


@lru_cache(maxsize=1)
def detect() -> str:
    """Return the best available accelerator: 'cuda', 'mps' or 'cpu'."""
    forced = os.environ.get("CALLMEFAIR_DEVICE", "").strip().lower()
    if forced:
        if forced not in _VALID:
            raise ValueError(f"CALLMEFAIR_DEVICE must be one of {_VALID}, got {forced!r}")
        return forced
    if _torch_has(CUDA) or _nvidia_smi_present():
        return CUDA
    if platform.system() == "Darwin" and platform.machine() == "arm64" and _torch_has(MPS):
        return MPS
    return CPU


def is_cuda() -> bool:
    return detect() == CUDA


def torch_device() -> str:
    """Device string for PyTorch-based models (TabNet)."""
    return detect()


def xgboost_params() -> dict:
    """XGBoost >= 2.0 runs on CUDA only; Apple GPUs are unsupported."""
    return {"tree_method": "hist", "device": CUDA if is_cuda() else CPU}


def catboost_params() -> dict:
    """CatBoost GPU training is CUDA-only."""
    return {"task_type": "GPU"} if is_cuda() else {}


def lightgbm_params() -> dict:
    """LightGBM pip wheels are CPU builds; GPU needs a custom build, so opt-in only.

    Set ``CALLMEFAIR_LGBM_GPU=1`` on a machine with a GPU-enabled LightGBM build.
    """
    if is_cuda() and os.environ.get("CALLMEFAIR_LGBM_GPU") == "1":
        return {"device": "gpu"}
    return {}


def performance_cores() -> int:
    """Number of cores worth scheduling CPU-bound work on.

    On Apple Silicon this is the performance-core count (efficiency cores slow
    down synchronous parallel work); elsewhere it is ``os.cpu_count()``.
    """
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                                 check=True, capture_output=True, text=True, timeout=5)
            return max(1, int(out.stdout.strip()))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return os.cpu_count() or 1
