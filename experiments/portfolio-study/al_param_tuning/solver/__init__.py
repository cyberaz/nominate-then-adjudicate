# src/solver/__init__.py
from .scip_wrapper import SCIPWrapper, SolveResult
from .scoring import compute_score, compute_score_from_times, compare_results
