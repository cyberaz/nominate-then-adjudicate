"""
Instance Descriptor

Generates raw constraint matrix and variable information for LLM consumption.
Outputs the complete planning formulation without summarization.
"""

import numpy as np
from typing import Dict, List, Any
from pyscipopt import Model


class InstanceDescriptor:
    """
    Generate raw textual descriptions of MIP instances.
    """

    def __init__(self, max_constraints: int = 10000, max_variables: int = 10000):
        """
        Args:
            max_constraints: Maximum number of constraints to show (default: no limit)
            max_variables: Maximum number of variables to show (default: no limit)
        """
        self.max_constraints = max_constraints
        self.max_variables = max_variables

    def describe(self, model: Model) -> str:
        """
        Generate a raw text description of the MIP instance.

        Args:
            model: PySCIPOpt Model object

        Returns:
            Multi-line string with complete instance information
        """
        lines = []

        # Basic dimensions
        n_vars = model.getNVars()
        n_cons = model.getNConss()
        lines.append(f"## Instance Dimensions")
        lines.append(f"- Variables: {n_vars}")
        lines.append(f"- Constraints: {n_cons}")
        lines.append("")

        # Variable details
        lines.append(self._describe_variables(model))
        lines.append("")

        # Constraint details
        lines.append(self._describe_constraints(model))
        lines.append("")

        # Objective function
        lines.append(self._describe_objective(model))
        lines.append("")

        return "\n".join(lines)

    def _describe_variables(self, model: Model) -> str:
        """Describe all variables."""
        variables = model.getVars()
        n_vars = len(variables)

        # Type counts
        type_counts = {"BINARY": 0, "INTEGER": 0, "CONTINUOUS": 0}
        for v in variables:
            vtype = v.vtype()
            if vtype in type_counts:
                type_counts[vtype] += 1

        lines = ["## Variables"]
        lines.append(f"Total: {n_vars}")
        lines.append(f"- Binary: {type_counts['BINARY']}")
        lines.append(f"- Integer: {type_counts['INTEGER']}")
        lines.append(f"- Continuous: {type_counts['CONTINUOUS']}")
        lines.append("")

        # All variables
        sample_vars = variables[:min(self.max_variables, n_vars)]
        lines.append(f"Variable list (showing {len(sample_vars)}/{n_vars}):")
        for v in sample_vars:
            lb = v.getLbOriginal()
            ub = v.getUbOriginal()
            obj = v.getObj()
            lines.append(f"  {v.name}: type={v.vtype()}, bounds=[{lb}, {ub}], obj={obj:.6f}")

        return "\n".join(lines)

    def _describe_constraints(self, model: Model) -> str:
        """Describe all constraints."""
        constraints = model.getConss()
        n_cons = len(constraints)

        lines = ["## Constraints"]
        lines.append(f"Total: {n_cons}")
        lines.append("")

        # All constraints
        sample_cons = constraints[:min(self.max_constraints, n_cons)]
        lines.append(f"Constraint list (showing {len(sample_cons)}/{n_cons}):")
        for i, c in enumerate(sample_cons):
            lhs = model.getLhs(c)
            rhs = model.getRhs(c)

            # Get variables in this constraint
            try:
                cons_vars = model.getConsVars(c)
                cons_vals = model.getConsVals(c)

                # Build constraint expression
                terms = []
                for v, coeff in zip(cons_vars, cons_vals):
                    if abs(coeff) > 1e-10:
                        if abs(coeff - 1.0) < 1e-10:
                            terms.append(f"+ {v.name}")
                        elif abs(coeff + 1.0) < 1e-10:
                            terms.append(f"- {v.name}")
                        else:
                            terms.append(f"{coeff:+.6f}*{v.name}")

                expr = " ".join(terms)

                # Determine constraint type
                if lhs is not None and rhs is not None and abs(lhs - rhs) < 1e-9:
                    ctype = f"= {rhs:.6f}"
                elif lhs is not None and lhs > -1e20:
                    ctype = f"in [{lhs:.6f}, {rhs:.6f}]" if rhs < 1e20 else f">= {lhs:.6f}"
                else:
                    ctype = f"<= {rhs:.6f}" if rhs < 1e20 else "(no bounds)"

                lines.append(f"  [{i+1}] {c.name}: {expr} {ctype}")

            except Exception as e:
                lines.append(f"  [{i+1}] {c.name}: (error: {e})")

        return "\n".join(lines)

    def _describe_objective(self, model: Model) -> str:
        """Describe the objective function."""
        variables = model.getVars()

        # Count non-zero objective coefficients
        obj_nonzeros = [(v.name, v.getObj()) for v in variables if abs(v.getObj()) > 1e-10]
        n_obj = len(obj_nonzeros)

        lines = ["## Objective Function"]
        lines.append(f"Type: {'Minimize' if model.getObjectiveSense() == 'minimize' else 'Maximize'}")
        lines.append(f"Non-zero terms: {n_obj}/{len(variables)}")
        lines.append("")

        if obj_nonzeros:
            lines.append("Objective terms:")
            for name, coeff in obj_nonzeros:
                lines.append(f"  {coeff:+.6f} * {name}")

        return "\n".join(lines)
