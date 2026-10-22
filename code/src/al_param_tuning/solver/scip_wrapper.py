"""
SCIP Solver Wrapper

Encapsulates PySCIPOpt for:
- Solving MIP instances with custom parameter configurations
- Extracting solve statistics (time, nodes, gap, objective)
"""

import time
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SolveResult:
    """Result of a single SCIP solve."""
    status: str           # "optimal", "feasible", "infeasible", "timelimit", "nodelimit"
    objective: float      # Best objective value found
    time: float           # Wall clock solve time (seconds)
    nodes: int            # Number of branch-and-bound nodes explored
    gap: float            # Final optimality gap (0 if optimal)
    has_solution: bool    # Whether a feasible solution was found
    is_optimal: bool      # Whether optimality was proven


class SCIPWrapper:
    """
    Wrapper around PySCIPOpt for solving MIP instances with custom configs.
    """

    def __init__(self, param_config: Optional[Dict] = None):
        """
        Args:
            param_config: Dict of parameter definitions (from param_config.json).
        """
        self.param_config = param_config

    def solve(
        self,
        model,
        config: Optional[Dict[str, Any]] = None,
        time_limit: float = 60.0,
    ) -> SolveResult:
        """
        Solve a MIP instance with the given configuration.

        Args:
            model: PySCIPOpt Model object
            config: Parameter configuration dict {param_name: value}
            time_limit: Time limit in seconds

        Returns:
            SolveResult with solve statistics
        """
        # Reset model state to ensure clean solve
        try:
            model.freeTransform()
        except Exception:
            pass

        # Apply configuration
        if config:
            self._apply_config(model, config)

        # Suppress SCIP output
        model.hideOutput()

        # Set time limit
        model.setRealParam("limits/time", time_limit)

        # Fix SCIP random seeds for reproducibility
        try:
            model.setIntParam("randomization/randomseedshift", 42)
            model.setIntParam("randomization/permutationseed", 42)
            model.setIntParam("randomization/lpseed", 42)
        except Exception:
            pass  # Parameters may not exist in all SCIP versions

        # Pin SCIP to single-threaded execution so wall-clock timing is stable
        # and reproducible (no LP / concurrent-solver thread contention).
        try:
            model.setIntParam("lp/threads", 1)
            model.setIntParam("parallel/maxnthreads", 1)
        except Exception:
            pass  # Parameters may not exist in all SCIP versions

        # Solve
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time

        # Extract results
        status = model.getStatus()
        obj_val = model.getObjVal() if model.getNSols() > 0 else float("inf")
        gap = model.getGap() if model.getNSols() > 0 else float("inf")
        nodes = model.getNNodes()
        has_sol = model.getNSols() > 0
        is_opt = status == "optimal"

        return SolveResult(
            status=status,
            objective=obj_val,
            time=solve_time,
            nodes=nodes,
            gap=gap,
            has_solution=has_sol,
            is_optimal=is_opt,
        )

    def _apply_config(self, model, config: Dict[str, Any]):
        """Apply parameter configuration to SCIP model."""
        for param_name, value in config.items():
            try:
                # Check for bool first (including numpy bool)
                # Note: isinstance(True, int) is True in Python, so bool must be checked before int
                if isinstance(value, (bool, np.bool_)):
                    model.setBoolParam(param_name, bool(value))
                elif isinstance(value, str) and len(value) == 1:
                    # Char parameters (single character)
                    model.setCharParam(param_name, value)
                elif isinstance(value, str):
                    model.setStringParam(param_name, value)
                elif isinstance(value, (float, np.floating)):
                    model.setRealParam(param_name, float(value))
                elif isinstance(value, (int, np.integer)):
                    try:
                        model.setIntParam(param_name, int(value))
                    except Exception:
                        model.setLongintParam(param_name, int(value))
                else:
                    # Fallback: try to infer type
                    try:
                        model.setIntParam(param_name, int(value))
                    except Exception:
                        try:
                            model.setRealParam(param_name, float(value))
                        except Exception:
                            model.setBoolParam(param_name, bool(value))
            except Exception:
                # Parameter doesn't exist or invalid value - skip
                pass

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get SCIP default configuration for selected parameters.

        Returns:
            Dict of default parameter values
        """
        if self.param_config is None:
            return {}

        defaults = {}
        for param in self.param_config.get("parameters", []):
            defaults[param["name"]] = param["default"]
        return defaults

    def get_param_names(self) -> list:
        """Get list of parameter names from config."""
        if self.param_config is None:
            return []
        return [p["name"] for p in self.param_config.get("parameters", [])]
