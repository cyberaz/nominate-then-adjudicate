"""
Instance Feature Extractor

Extracts 20-dimensional numerical features from MIP instances,
organized into 5 modules:
  Module 1: Problem size (4 features)
  Module 2: Variable type distribution (4 features)
  Module 3: Constraint type distribution (4 features)
  Module 4: Coefficient and numerical statistics (5 features)
  Module 5: Lightweight graph topology (3 features)
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class InstanceFeatureExtractor:
    """
    Extract 20-dimensional feature vector from a MIP instance.
    """

    FEATURE_NAMES = [
        # Module 1: Problem size (4)
        "row_count",
        "col_count",
        "total_nonzeros",
        "matrix_density",
        # Module 2: Variable type distribution (4)
        "binary_ratio",
        "general_integer_ratio",
        "continuous_ratio",
        "unbounded_count",
        # Module 3: Constraint type distribution (4)
        "equality_ratio",
        "inequality_ratio",
        "set_operation_count",
        "special_constraint_count",
        # Module 4: Coefficient and numerical statistics (5)
        "mean_nonzeros_per_row",
        "mean_nonzeros_per_col",
        "matrix_coeff_cv",
        "objective_density",
        "rhs_cv",
        # Module 5: Lightweight graph topology (3)
        "num_connected_components",
        "largest_component_ratio",
        "isolated_variable_count",
    ]

    def __init__(self):
        self.dimension = len(self.FEATURE_NAMES)  # 20

    def extract(self, model) -> np.ndarray:
        """
        Extract all 20 features from a PySCIPOpt Model.

        Args:
            model: PySCIPOpt Model (before solve)

        Returns:
            Numpy array of shape (20,)
        """
        feats = {}

        # Extract constraint matrix info
        matrix_data = self._extract_constraint_matrix(model)

        # Module 1: Problem size
        feats.update(self._module1_size(model, matrix_data))

        # Module 2: Variable types
        feats.update(self._module2_variables(model))

        # Module 3: Constraint types
        feats.update(self._module3_constraints(matrix_data))

        # Module 4: Coefficient statistics
        feats.update(self._module4_coefficients(model, matrix_data))

        # Module 5: Graph topology
        feats.update(self._module5_topology(model, matrix_data))

        # Build vector in canonical order
        vector = np.zeros(self.dimension, dtype=np.float64)
        for i, name in enumerate(self.FEATURE_NAMES):
            vector[i] = feats.get(name, 0.0)

        return vector

    def _extract_constraint_matrix(self, model) -> Dict:
        """
        Extract constraint matrix coefficients, RHS, and types.

        Returns dict with:
          - coeffs: list of (row, col, val) triples
          - rhs_values: list of RHS per constraint
          - lhs_values: list of LHS per constraint
          - constraint_types: list of 'eq', 'le', 'ge'
          - row_nnz: list of non-zeros per row
          - col_nnz: list of non-zeros per col
        """
        cons = model.getConss()
        variables = model.getVars()
        # Use variable names as keys since Variable objects aren't hashable
        var_idx = {v.name: i for i, v in enumerate(variables)}
        n_vars = len(variables)
        n_cons = len(cons)

        coeffs = []
        rhs_values = []
        lhs_values = []
        constraint_types = []
        row_nnz = [0] * n_cons
        col_nnz = [0] * n_vars

        for i, c in enumerate(cons):
            try:
                row_vars = model.getConsVars(c)
                row_coeffs = model.getConsVals(c)
            except Exception:
                row_vars = []
                row_coeffs = []

            for v, coeff in zip(row_vars, row_coeffs):
                j = var_idx.get(v.name, -1)
                if j >= 0:
                    coeffs.append((i, j, float(coeff)))
                    row_nnz[i] += 1
                    col_nnz[j] += 1

            lhs = model.getLhs(c)
            rhs = model.getRhs(c)
            rhs_values.append(float(rhs) if rhs is not None else 0.0)
            lhs_values.append(float(lhs) if lhs is not None else float("-inf"))

            # Determine constraint type
            if lhs is not None and rhs is not None and abs(float(lhs) - float(rhs)) < 1e-9:
                constraint_types.append("eq")
            elif lhs is not None and float(lhs) > float("-inf") + 1e9:
                # Range constraint (has both lower and upper bound)
                constraint_types.append("range")
            else:
                constraint_types.append("ineq")

        return {
            "coeffs": coeffs,
            "rhs_values": rhs_values,
            "lhs_values": lhs_values,
            "constraint_types": constraint_types,
            "row_nnz": row_nnz,
            "col_nnz": col_nnz,
            "n_cons": n_cons,
            "n_vars": n_vars,
        }

    def _module1_size(self, model, md: Dict) -> Dict[str, float]:
        """Module 1: Problem size and basic dimensions."""
        n_rows = md["n_cons"]
        n_cols = md["n_vars"]
        total_nnz = len(md["coeffs"])
        density = total_nnz / max(n_rows * n_cols, 1)

        return {
            "row_count": float(n_rows),
            "col_count": float(n_cols),
            "total_nonzeros": float(total_nnz),
            "matrix_density": float(density),
        }

    def _module2_variables(self, model) -> Dict[str, float]:
        """Module 2: Variable type distribution."""
        variables = model.getVars()
        n_vars = len(variables)
        if n_vars == 0:
            return {
                "binary_ratio": 0.0,
                "general_integer_ratio": 0.0,
                "continuous_ratio": 0.0,
                "unbounded_count": 0.0,
            }

        n_binary = 0
        n_integer = 0
        n_continuous = 0
        n_unbounded = 0

        for v in variables:
            vtype = v.vtype()
            if vtype == "BINARY":
                n_binary += 1
            elif vtype == "INTEGER":
                n_integer += 1
            else:
                n_continuous += 1

            # Check unbounded
            lb = v.getLbOriginal()
            ub = v.getUbOriginal()
            if (lb <= -1e20 or ub >= 1e20):
                n_unbounded += 1

        return {
            "binary_ratio": n_binary / n_vars,
            "general_integer_ratio": n_integer / n_vars,
            "continuous_ratio": n_continuous / n_vars,
            "unbounded_count": float(n_unbounded),
        }

    def _module3_constraints(self, md: Dict) -> Dict[str, float]:
        """Module 3: Constraint type distribution."""
        n_cons = md["n_cons"]
        if n_cons == 0:
            return {
                "equality_ratio": 0.0,
                "inequality_ratio": 0.0,
                "set_operation_count": 0.0,
                "special_constraint_count": 0.0,
            }

        n_eq = 0
        n_ineq = 0
        n_set_op = 0

        # Build per-row info for set operation detection
        row_coeffs = defaultdict(list)
        for row, col, val in md["coeffs"]:
            row_coeffs[row].append(val)

        INF = 1e19  # SCIP represents unbounded sides with ~1e20
        for i in range(n_cons):
            ctype = md["constraint_types"][i]
            if ctype == "eq":
                n_eq += 1
            else:
                n_ineq += 1

            # Set-operation constraint (set packing / covering / partitioning):
            # all coefficients are 0/1, and the active finite bound is a positive
            # integer. Packing (<=k) stores the bound in rhs; covering (>=k) stores
            # it in lhs (rhs=+inf); partitioning (=k) has both. Checking only rhs
            # (the previous logic) missed every covering constraint.
            coeffs_in_row = row_coeffs.get(i, [])
            if not coeffs_in_row:
                continue
            all_binary_coeffs = all(
                abs(c) < 1e-9 or abs(c - 1.0) < 1e-9 for c in coeffs_in_row
            )
            if not all_binary_coeffs:
                continue
            rhs = md["rhs_values"][i]
            lhs = md["lhs_values"][i]
            bound = None
            if rhs is not None and abs(rhs) < INF:
                bound = rhs
            elif lhs is not None and abs(lhs) < INF:
                bound = lhs
            if (
                bound is not None
                and bound >= 1.0 - 1e-9
                and abs(bound - round(bound)) < 1e-9
            ):
                n_set_op += 1

        return {
            "equality_ratio": n_eq / n_cons,
            "inequality_ratio": n_ineq / n_cons,
            "set_operation_count": float(n_set_op),
            "special_constraint_count": 0.0,  # No SOS/indicator/quad in our generators
        }

    def _module4_coefficients(self, model, md: Dict) -> Dict[str, float]:
        """Module 4: Coefficient and numerical statistics."""
        n_rows = md["n_cons"]
        n_cols = md["n_vars"]

        # Mean non-zeros per row and column
        row_nnz = md["row_nnz"]
        col_nnz = md["col_nnz"]
        mean_nnz_row = float(np.mean(row_nnz)) if row_nnz else 0.0
        mean_nnz_col = float(np.mean(col_nnz)) if col_nnz else 0.0

        # Matrix coefficient CV (coefficient of variation = std / mean)
        all_coeffs = [abs(v) for _, _, v in md["coeffs"]]
        if len(all_coeffs) > 1:
            coeff_arr = np.array(all_coeffs)
            coeff_mean = np.mean(coeff_arr)
            coeff_std = np.std(coeff_arr)
            matrix_cv = float(coeff_std / max(coeff_mean, 1e-10))
        else:
            matrix_cv = 0.0

        # Objective density
        variables = model.getVars()
        obj_nonzeros = sum(1 for v in variables if abs(v.getObj()) > 1e-10)
        obj_density = obj_nonzeros / max(len(variables), 1)

        # RHS CV
        rhs_vals = np.array([r for r in md["rhs_values"] if abs(r) > 1e-10])
        if len(rhs_vals) > 1:
            rhs_mean = np.mean(np.abs(rhs_vals))
            rhs_std = np.std(rhs_vals)
            rhs_cv = float(rhs_std / max(rhs_mean, 1e-10))
        else:
            rhs_cv = 0.0

        return {
            "mean_nonzeros_per_row": mean_nnz_row,
            "mean_nonzeros_per_col": mean_nnz_col,
            "matrix_coeff_cv": matrix_cv,
            "objective_density": float(obj_density),
            "rhs_cv": rhs_cv,
        }

    def _module5_topology(self, model, md: Dict) -> Dict[str, float]:
        """Module 5: Lightweight graph topology features."""
        n_rows = md["n_cons"]
        n_cols = md["n_vars"]

        if n_cols == 0:
            return {
                "num_connected_components": 0.0,
                "largest_component_ratio": 0.0,
                "isolated_variable_count": 0.0,
            }

        # Build bipartite graph: variable <-> constraint
        # Use Union-Find on variables: two variables are connected if they
        # appear in the same constraint.
        parent = list(range(n_cols))
        rank = [0] * n_cols

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        # Group variables by constraint
        row_vars = defaultdict(list)
        for row, col, val in md["coeffs"]:
            row_vars[row].append(col)

        for row, cols in row_vars.items():
            for j in range(1, len(cols)):
                union(cols[0], cols[j])

        # Count connected components and sizes
        comp_size = defaultdict(int)
        for j in range(n_cols):
            comp_size[find(j)] += 1

        n_components = len(comp_size)
        largest_comp = max(comp_size.values()) if comp_size else 0

        # Isolated variables: columns with all-zero coefficients
        col_has_nonzero = set()
        for _, col, _ in md["coeffs"]:
            col_has_nonzero.add(col)
        n_isolated = n_cols - len(col_has_nonzero)

        return {
            "num_connected_components": float(n_components),
            "largest_component_ratio": float(largest_comp) / max(n_cols, 1),
            "isolated_variable_count": float(n_isolated),
        }

    def get_feature_names(self) -> List[str]:
        """Get canonical feature names."""
        return self.FEATURE_NAMES.copy()
