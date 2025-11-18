"""
Parallel Training Optimizer

Provides a unified executor to run CPU-bound training tasks across threads or
processes, with controls to avoid oversubscription by limiting BLAS/OpenMP
threads inside each worker.

Example:
    >>> from callmefair.search.optimizer import TrainingOptimizer
    >>> opt = TrainingOptimizer(backend='process', max_workers=8)
    >>> results_iter = opt.map(func, args_list)
    >>> for res in results_iter:
    ...     handle(res)
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import os
from typing import Iterable, Callable, Any, Dict


def _init_worker_env(caps: Dict[str, int]):
    """Initializer for process workers to set thread caps for BLAS/OpenMP libraries."""
    for key, val in caps.items():
        try:
            os.environ[key] = str(val)
        except Exception:
            pass


class TrainingOptimizer:
    """
    A pluggable optimizer to execute training tasks using threads or processes.

    Args:
        backend (str): 'thread' or 'process'. Use 'process' for CPU-bound tasks.
        max_workers (int): Maximum workers to use.
        worker_thread_caps (dict): Env caps for MKL/OMP/OpenBLAS/NumExpr in workers.
            Defaults to 1 to avoid oversubscription.
    """

    def __init__(
        self,
        backend: str = 'process',
        max_workers: int | None = None,
        worker_thread_caps: Dict[str, int] | None = None,
    ) -> None:
        self.backend = backend
        self.max_workers = max_workers
        self.worker_thread_caps = worker_thread_caps or {
            'MKL_NUM_THREADS': 1,
            'OMP_NUM_THREADS': 1,
            'OPENBLAS_NUM_THREADS': 1,
            'NUMEXPR_NUM_THREADS': 1,
        }

    def map(self, func: Callable[..., Any], args_list: Iterable[tuple], max_workers: int | None = None):
        """
        Execute tasks and yield results as they complete.

        Returns an iterator over results, enabling incremental consumption.
        """
        workers = max_workers or self.max_workers or os.cpu_count() or 1

        if self.backend == 'thread':
            executor_cls = ThreadPoolExecutor
            executor_kwargs = {'max_workers': workers}
        elif self.backend == 'process':
            executor_cls = ProcessPoolExecutor
            # Ensure worker env caps are set to avoid oversubscription
            executor_kwargs = {
                'max_workers': workers,
                'initializer': _init_worker_env,
                'initargs': (self.worker_thread_caps,),
            }
        else:
            raise ValueError("backend must be 'thread' or 'process'")

        executor = executor_cls(**executor_kwargs)
        futures = {}
        for args in args_list:
            fut = executor.submit(func, *args)
            futures[fut] = args

        for future in as_completed(futures):
            try:
                yield future.result()
            except Exception as e:
                import traceback
                args = futures.get(future)
                func_name = getattr(func, '__name__', repr(func))
                print("\n[Executor Error] Task failed in parallel execution")
                print(f"Backend: {self.backend} | Workers: {workers}")
                print(f"Function: {func_name}")
                print(f"Args (truncated repr): {repr(args)[:500]}")
                print(f"Exception: {type(e).__name__}: {e}")
                print('Traceback:\n' + ''.join(traceback.format_exception(type(e), e, e.__traceback__)))
                raise

        executor.shutdown(wait=True)