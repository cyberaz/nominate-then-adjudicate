"""
Parallel Executor

Wraps concurrent execution of SCIP solves across multiple instances/configs.
Uses ProcessPoolExecutor for CPU-bound MIP solving.
"""

import concurrent.futures
from typing import List, Callable, Any, Optional
from tqdm import tqdm


class ParallelExecutor:
    """
    Execute solver calls in parallel across multiple workers.
    """

    def __init__(self, max_workers: int = 8):
        """
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers

    def map(
        self,
        func: Callable,
        items: List[Any],
        desc: str = "Processing",
        show_progress: bool = True,
    ) -> List[Any]:
        """
        Apply func to each item in parallel.

        Args:
            func: Function to apply to each item
            items: List of inputs
            desc: Progress bar description
            show_progress: Whether to show tqdm progress bar

        Returns:
            List of results in same order as items
        """
        results = [None] * len(items)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(func, item): i
                for i, item in enumerate(items)
            }

            if show_progress:
                pbar = tqdm(total=len(items), desc=desc)

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = None
                    print(f"  [WARN] Worker failed for item {idx}: {e}")

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        return results

    def solve_batch(
        self,
        scip_wrapper,
        models_and_configs: List[dict],
        time_limit: float = 60,
    ) -> List[Any]:
        """
        Solve a batch of (model, config) pairs in parallel.

        Args:
            scip_wrapper: SCIPWrapper instance
            models_and_configs: List of {"model": Model, "config": dict}
            time_limit: Time limit per solve

        Returns:
            List of SolveResult objects
        """
        def solve_one(item):
            return scip_wrapper.solve(item["model"], item["config"], time_limit)

        return self.map(solve_one, models_and_configs, desc="Solving")
