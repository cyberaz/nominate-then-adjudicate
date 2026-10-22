"""
TSP — Miller-Tucker-Zemlin (MTZ) Formulation Generator

min sum(c_ij * x_ij)
s.t. sum(x_ij) = 1   for all i (one outgoing edge per city)
     sum(x_ij) = 1   for all j (one incoming edge per city)
     u_i - u_j + n * x_ij <= n - 1   for i,j >= 1 (MTZ subtour elim)
     x_ij in {0, 1}, u_i >= 0
"""

from pyscipopt import Model
from .base import set_quiet, make_rng


def generate_instance(n_cities=12, seed=None):
    rng = make_rng(seed)
    model = Model("tsp")
    set_quiet(model)

    # Random distance matrix (symmetric)
    coords = rng.rand(n_cities, 2) * 100
    dist = {}
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                dist[i, j] = round((dx**2 + dy**2) ** 0.5, 2)

    n = n_cities

    # Edge variables
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = model.addVar(
                    vtype="B", obj=dist[i, j], name=f"x_{i}_{j}"
                )

    # Subtour elimination variables (MTZ)
    u = {}
    for i in range(1, n):
        u[i] = model.addVar(vtype="C", lb=1, ub=n - 1, name=f"u_{i}")

    # Degree constraints: each city has exactly one outgoing and incoming edge
    for i in range(n):
        model.addCons(sum(x[i, j] for j in range(n) if i != j) == 1, name=f"out_{i}")
        model.addCons(sum(x[j, i] for j in range(n) if i != j) == 1, name=f"in_{i}")

    # MTZ subtour elimination
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addCons(
                    u[i] - u[j] + n * x[i, j] <= n - 1,
                    name=f"mtz_{i}_{j}",
                )

    model.setMinimize()
    return model
