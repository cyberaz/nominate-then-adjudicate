"""
Scoring Function

Computes the performance score of a configuration relative to default.
"""

import numpy as np
from typing import Optional
from .scip_wrapper import SolveResult


def compute_score(
    result_config: SolveResult,
    result_default: SolveResult,
) -> Optional[float]:
    """
    Compute the performance score of a configuration relative to default.

    Returns:
        Score in range [-1, 1], or None if instance should be discarded
    """
    rc = result_config
    rd = result_default

    # Case 1: Both infeasible -> discard
    if not rc.has_solution and not rd.has_solution:
        return None

    # Case 2: Config failed, default succeeded -> worst score
    if not rc.has_solution and rd.has_solution:
        return -1.0

    # Case 3: Config succeeded, default failed -> best score
    if rc.has_solution and not rd.has_solution:
        return 1.0

    # Case 4: Both found feasible solutions.
    # Optimality has priority: a config that failed to prove optimality within the
    # time limit (i.e. timed out) LOSES to one that solved to optimality. Test
    # instances are pre-filtered so the default solves well within the limit, so
    # this makes "recommended config timed out" a clean loss for the recommendation.
    if rc.is_optimal and not rd.is_optimal:
        return 1.0
    if not rc.is_optimal and rd.is_optimal:
        return -1.0

    # Both optimal -> faster wins
    if rc.is_optimal and rd.is_optimal:
        tc, td = rc.time, rd.time
        score = (td - tc) / (td + tc + 1e-10)
        return float(np.clip(score, -1.0, 1.0))

    # Neither optimal (both timed out) -> compare feasible objectives
    oc, od = rc.objective, rd.objective
    score = (od - oc) / (abs(od) + abs(oc) + 1e-10)
    return float(np.clip(score, -1.0, 1.0))


def compare_results(
    result_a: SolveResult,
    result_b: SolveResult,
) -> Optional[float]:
    """
    Compare two solve results with strict priority:
      1. Optimal solution always beats non-optimal feasible solution
      2. Among both optimal: faster time wins
      3. Among both feasible non-optimal: better objective wins

    Returns:
        Score in [-1, 1]: positive means A is better than B,
        negative means B is better, or None if both are infeasible.
    """
    # Both infeasible -> incomparable
    if not result_a.has_solution and not result_b.has_solution:
        return None

    # A has solution, B doesn't -> A wins
    if result_a.has_solution and not result_b.has_solution:
        return 1.0

    # A has no solution, B does -> B wins
    if not result_a.has_solution and result_b.has_solution:
        return -1.0

    # Both have solutions

    # Priority 1: Optimal beats non-optimal
    if result_a.is_optimal and not result_b.is_optimal:
        return 1.0
    if not result_a.is_optimal and result_b.is_optimal:
        return -1.0

    # Priority 2: Both optimal -> compare time (faster is better)
    if result_a.is_optimal and result_b.is_optimal:
        ta, tb = result_a.time, result_b.time
        score = (tb - ta) / (tb + ta + 1e-10)
        return float(np.clip(score, -1.0, 1.0))

    # Priority 3: Both feasible, neither optimal -> compare objective (minimization)
    oa, ob = result_a.objective, result_b.objective
    score = (ob - oa) / (abs(ob) + abs(oa) + 1e-10)
    return float(np.clip(score, -1.0, 1.0))


def compute_score_from_times(
    time_config: float,
    time_default: float,
) -> float:
    """Simplified scoring when both found optimal solutions."""
    score = (time_default - time_config) / (time_default + time_config + 1e-10)
    return float(np.clip(score, -1.0, 1.0))
