"""
Vertex Cover Problem Generator

min sum(c_i * x_i)
s.t. x_i + x_j >= 1   for all edges (i, j)
     x_i in {0, 1}
"""

from pyscipopt import Model
from .base import set_quiet, make_rng


def generate_instance(n_vertices=50, edge_prob=0.3, seed=None):
    """n_vertices / edge_prob are UPPER BOUNDS. Scale, edge density and
    objective weights are heavily randomized per-seed."""
    rng = make_rng(seed)
    model = Model("vertex_cover")
    set_quiet(model)

    n_vertices = int(rng.randint(5, n_vertices + 1))
    edge_prob = float(rng.uniform(0.02, max(edge_prob, 0.03)))

    edges = []
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if rng.random() < edge_prob:
                edges.append((i, j))

    mode = rng.randint(0, 3)
    w_hi = int(rng.randint(1, 200))

    x = {}
    for i in range(n_vertices):
        if mode == 0:
            obj = 1
        elif mode == 1:
            obj = int(rng.randint(1, w_hi + 1))
        else:
            obj = int(rng.lognormal(1.0, rng.uniform(0.3, 1.2)) + 1)
        x[i] = model.addVar(vtype="B", obj=obj, name=f"x_{i}")

    for k, (i, j) in enumerate(edges):
        model.addCons(x[i] + x[j] >= 1, name=f"edge_{k}")

    model.setMinimize()
    return model
