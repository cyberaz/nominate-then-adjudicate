"""
Bin Packing Problem Generator

min sum(y_j)
s.t. sum(s_i * x_ij) <= C * y_j   for all bins j
     sum(x_ij) = 1                 for all items i
     x_ij in {0,1}, y_j in {0,1}
"""

from pyscipopt import Model
from .base import set_quiet, make_rng


def generate_instance(n_items=20, n_bins_ub=10, seed=None):
    """n_items / n_bins_ub are UPPER BOUNDS. Scale, capacity, item sizes and
    the symmetry-breaking toggle heavily randomized per-seed."""
    rng = make_rng(seed)
    model = Model("bin_packing")
    set_quiet(model)

    n_items = int(rng.randint(3, n_items + 1))
    n_bins = int(rng.randint(2, n_bins_ub + 1))

    # Randomize capacity and item size range widely
    capacity = int(rng.randint(40, 200))
    size_lo = int(rng.randint(1, capacity // 2 + 1))
    size_hi = min(size_lo + int(rng.randint(5, capacity)), capacity)
    sizes = rng.randint(size_lo, size_hi + 1, size=n_items)

    # Guarantee feasibility: with n_bins >= n_items, each item can occupy its own
    # bin (every size <= capacity), so a feasible packing always exists. The
    # ceil(total/capacity) bound is only necessary, not sufficient (fragmentation).
    n_bins = max(n_bins, n_items)

    y = {}
    x = {}
    for j in range(n_bins):
        y[j] = model.addVar(vtype="B", obj=1, name=f"y_{j}")
    for i in range(n_items):
        for j in range(n_bins):
            x[i, j] = model.addVar(vtype="B", obj=0, name=f"x_{i}_{j}")

    for i in range(n_items):
        model.addCons(sum(x[i, j] for j in range(n_bins)) == 1, name=f"assign_{i}")

    for j in range(n_bins):
        model.addCons(
            sum(int(sizes[i]) * x[i, j] for i in range(n_items)) <= capacity * y[j],
            name=f"cap_{j}",
        )

    # Symmetry breaking ALWAYS enabled: bins are interchangeable, so without
    # ordering constraints the branch-and-bound tree explodes and hits the time
    # limit. Forcing y_j >= y_{j+1} removes this symmetry and avoids 60s timeouts.
    for j in range(n_bins - 1):
        model.addCons(y[j] >= y[j + 1], name=f"sym_{j}")

    model.setMinimize()
    return model
