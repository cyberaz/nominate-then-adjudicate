"""
Cutting Stock Problem Generator (Pattern-based formulation)

min sum(x_k)
s.t. sum(a_ik * x_k) >= d_i   for all item types i
     x_k >= 0, integer

where a_ik is the number of items of type i in pattern k.
Patterns are pre-generated (random feasible patterns).
"""

from pyscipopt import Model
from .base import set_quiet, make_rng


def generate_instance(roll_length=100, n_item_types=8, n_patterns=20, seed=None):
    rng = make_rng(seed)
    model = Model("cutting_stock")
    set_quiet(model)

    # Item lengths and demands
    item_lengths = rng.randint(10, int(roll_length * 0.6), size=n_item_types)
    demands = rng.randint(10, 50, size=n_item_types)

    # Generate feasible cutting patterns
    patterns = []
    for _ in range(n_patterns):
        pattern = [0] * n_item_types
        remaining = roll_length
        # Greedy fill
        order = rng.permutation(n_item_types)
        for i in order:
            max_cuts = remaining // item_lengths[i]
            if max_cuts > 0:
                count = rng.randint(1, max_cuts + 1)
                pattern[i] = int(count)
                remaining -= count * item_lengths[i]
        # Only keep non-empty patterns
        if sum(pattern) > 0:
            patterns.append(pattern)

    # Ensure we have enough patterns (add single-item patterns)
    for i in range(n_item_types):
        max_per_roll = roll_length // item_lengths[i]
        if max_per_roll > 0:
            p = [0] * n_item_types
            p[i] = int(max_per_roll)
            patterns.append(p)

    n_actual = len(patterns)

    # Pattern usage variables (integer)
    x = {}
    for k in range(n_actual):
        x[k] = model.addVar(vtype="I", lb=0, obj=1, name=f"x_{k}")

    # Demand satisfaction
    for i in range(n_item_types):
        model.addCons(
            sum(patterns[k][i] * x[k] for k in range(n_actual)) >= demands[i],
            name=f"demand_{i}",
        )

    model.setMinimize()
    return model
