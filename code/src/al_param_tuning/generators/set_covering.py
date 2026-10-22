"""
Set Covering Problem Generator

Generate random set covering instances following Balas & Ho (1980).
min sum(c_j * x_j)
s.t. sum(a_ij * x_j) >= 1  for all i
     x_j in {0, 1}
"""

from pyscipopt import Model
from .base import set_quiet, make_rng


def generate_instance(m=200, n=200, density=0.1, seed=None):
    """Signature params are UPPER BOUNDS. Scale is sampled uniformly from a
    small floor up to the cap, and all coefficient distributions are
    heavily randomized per-seed to blur the family fingerprint."""
    rng = make_rng(seed)
    model = Model("set_covering")
    set_quiet(model)

    # Scale: uniform from a small floor up to the cap
    m = int(rng.randint(5, m + 1))
    n = int(rng.randint(5, n + 1))
    # Density: wide range up to the cap
    density = float(rng.uniform(0.02, max(density, 0.03)))

    # Randomize cost distribution: range AND shape
    cost_hi = int(rng.randint(10, 500))
    if rng.random() < 0.5:
        costs = rng.randint(1, cost_hi + 1, size=n)
    else:
        # log-normal-ish heavy tail via exponential of uniform
        costs = (rng.lognormal(mean=rng.uniform(0.5, 3.0), sigma=rng.uniform(0.3, 1.5), size=n) + 1).astype(int)

    # Random coverage requirement (RHS heterogeneity): mostly >=1, sometimes >=b
    max_req = 1 + int(rng.random() < 0.4) * int(rng.randint(1, 4))

    x = {}
    for j in range(n):
        x[j] = model.addVar(vtype="B", obj=int(costs[j]), name=f"x_{j}")

    for i in range(m):
        covered_by = [j for j in range(n) if rng.random() < density]
        req = int(rng.randint(1, max_req + 1))
        if len(covered_by) < req:
            # ensure feasibility: pad with random distinct sets
            extra = list(rng.choice(n, size=min(req, n), replace=False))
            covered_by = list(set(covered_by) | set(extra))
        model.addCons(
            sum(x[j] for j in covered_by) >= req,
            name=f"cover_{i}",
        )

    model.setMinimize()
    return model
