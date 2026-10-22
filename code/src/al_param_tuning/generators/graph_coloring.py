"""
Graph Coloring Problem Generator (MTZ-like)

min sum(w_k)
s.t. sum(x_ik) = 1             for all vertices i (each vertex gets one color)
     x_ik + x_jk <= w_k        for all edges (i,j), all colors k
     x_ik in {0,1}, w_k in {0,1}
"""

from pyscipopt import Model
from .base import set_quiet, make_rng


def generate_instance(n_vertices=30, edge_prob=0.3, n_colors_ub=10, seed=None):
    """n_vertices / edge_prob / n_colors_ub are UPPER BOUNDS. Scale, edge
    density and the symmetry-breaking toggle heavily randomized per-seed."""
    rng = make_rng(seed)
    model = Model("graph_coloring")
    set_quiet(model)

    n_vertices = int(rng.randint(3, n_vertices + 1))
    edge_prob = float(rng.uniform(0.05, max(edge_prob, 0.06)))

    edges = []
    degree = [0] * n_vertices
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if rng.random() < edge_prob:
                edges.append((i, j))
                degree[i] += 1
                degree[j] += 1

    # Guarantee feasibility: greedy coloring always succeeds with (max_degree + 1) colors.
    max_degree = max(degree) if degree else 0
    feasible_floor = min(max_degree + 1, n_vertices)
    K = int(rng.randint(2, n_colors_ub + 1))
    K = max(K, feasible_floor)

    w = {}
    x = {}
    for k in range(K):
        w[k] = model.addVar(vtype="B", obj=1, name=f"w_{k}")
    for i in range(n_vertices):
        for k in range(K):
            x[i, k] = model.addVar(vtype="B", obj=0, name=f"x_{i}_{k}")

    for i in range(n_vertices):
        model.addCons(sum(x[i, k] for k in range(K)) == 1, name=f"color_{i}")

    for e_idx, (i, j) in enumerate(edges):
        for k in range(K):
            model.addCons(
                x[i, k] + x[j, k] <= w[k],
                name=f"adj_{e_idx}_{k}",
            )

    # Symmetry breaking randomly enabled
    if rng.random() < 0.5:
        for k in range(K - 1):
            model.addCons(w[k] >= w[k + 1], name=f"sym_w_{k}")

    model.setMinimize()
    return model
