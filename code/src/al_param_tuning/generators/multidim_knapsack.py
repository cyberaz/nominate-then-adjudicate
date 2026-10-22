"""
Multi-dimensional Knapsack Problem Generator

max sum(v_j * x_j)
s.t. sum(a_ij * x_j) <= b_i   for i=1..d
     x_j in {0, 1}
"""

from pyscipopt import Model
from .base import set_quiet, make_rng


def generate_instance(n=30, d=5, seed=None):
    """n (items) and d (dimensions) are UPPER BOUNDS. Scale, coefficient
    ranges/shape and capacity tightness heavily randomized per-seed."""
    rng = make_rng(seed)
    model = Model("multidim_knapsack")
    set_quiet(model)

    n = int(rng.randint(5, n + 1))
    d = int(rng.randint(1, d + 1))

    v_hi = int(rng.randint(10, 300))
    w_hi = int(rng.randint(10, 200))
    if rng.random() < 0.5:
        values = rng.randint(1, v_hi + 1, size=n)
        weights = rng.randint(1, w_hi + 1, size=(d, n))
    else:
        values = (rng.lognormal(1.5, rng.uniform(0.3, 1.2), size=n) + 1).astype(int)
        weights = (rng.lognormal(1.0, rng.uniform(0.3, 1.2), size=(d, n)) + 1).astype(int)

    cap_frac = float(rng.uniform(0.1, 0.9))
    capacities = (weights.sum(axis=1) * cap_frac).astype(int)
    capacities = [max(int(capacities[i]), int(weights[i].max())) for i in range(d)]

    x = {}
    for j in range(n):
        x[j] = model.addVar(vtype="B", obj=-int(values[j]), name=f"x_{j}")

    for i in range(d):
        model.addCons(
            sum(int(weights[i, j]) * x[j] for j in range(n)) <= capacities[i],
            name=f"dim_{i}",
        )

    model.setMinimize()
    return model
