"""
0-1 Knapsack Problem Generator

max sum(v_j * x_j)
s.t. sum(w_j * x_j) <= W
     x_j in {0, 1}
"""

from pyscipopt import Model
from .base import set_quiet, make_rng


def generate_instance(n=50, seed=None):
    """n is an UPPER BOUND. Scale sampled from a small floor to the cap;
    coefficient ranges, distribution shape and capacity tightness heavily
    randomized per-seed."""
    rng = make_rng(seed)
    model = Model("knapsack")
    set_quiet(model)

    n = int(rng.randint(5, n + 1))

    # Randomize weight/value ranges widely, and distribution shape
    w_hi = int(rng.randint(10, 200))
    v_hi = int(rng.randint(10, 300))
    if rng.random() < 0.5:
        weights = rng.randint(1, w_hi + 1, size=n)
        values = rng.randint(1, v_hi + 1, size=n)
    else:
        weights = (rng.lognormal(1.0, rng.uniform(0.3, 1.2), size=n) + 1).astype(int)
        values = (rng.lognormal(1.5, rng.uniform(0.3, 1.2), size=n) + 1).astype(int)

    # Randomize capacity tightness across a wide band
    cap_frac = float(rng.uniform(0.1, 0.9))
    capacity = max(int(weights.sum() * cap_frac), int(weights.max()))

    x = {}
    for j in range(n):
        x[j] = model.addVar(vtype="B", obj=-int(values[j]), name=f"x_{j}")

    model.addCons(sum(int(weights[j]) * x[j] for j in range(n)) <= capacity, name="capacity")

    model.setMinimize()
    return model
