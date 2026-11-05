"""Executor wrapper to run tasks on threads or processes.

Results are returned in submission order, so downstream floating-point sums do
not depend on completion order. Process workers cap BLAS/OpenMP threads to 1
to avoid oversubscription.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Iterable, Iterator

THREAD_CAP_VARS = ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS")


def _init_worker_env(caps: dict[str, int]) -> None:
    for key, val in caps.items():
        os.environ[key] = str(val)


class TrainingOptimizer:
    """Run ``func(*args)`` for each args tuple on a 'thread' or 'process' pool."""

    def __init__(self, backend: str = "process", max_workers: int | None = None,
                 worker_thread_caps: dict[str, int] | None = None) -> None:
        if backend not in ("thread", "process"):
            raise ValueError("backend must be 'thread' or 'process'")
        self.backend = backend
        self.max_workers = max_workers
        self._executor = None
        self.worker_thread_caps = worker_thread_caps or dict.fromkeys(THREAD_CAP_VARS, 1)

    def _get_executor(self):
        if self._executor is None:
            workers = self.max_workers or os.cpu_count() or 1
            if self.backend == "thread":
                self._executor = ThreadPoolExecutor(max_workers=workers)
            else:
                self._executor = ProcessPoolExecutor(
                    max_workers=workers, initializer=_init_worker_env,
                    initargs=(self.worker_thread_caps,))
        return self._executor

    def map(self, func: Callable[..., Any], args_list: Iterable[tuple]) -> Iterator[Any]:
        """Yield results in the order of ``args_list``.

        The pool is created on first use and reused (worker start-up is costly
        under macOS 'spawn'); release it with ``shutdown()``.
        """
        executor = self._get_executor()
        futures = [executor.submit(func, *args) for args in args_list]
        for future in futures:
            yield future.result()

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
