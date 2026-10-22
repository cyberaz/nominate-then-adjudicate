"""
Capacitated Facility Location Problem Generator

min sum(f_j * y_j) + sum(c_ij * x_ij)
s.t. sum(x_ij) = 1               for all clients i
     sum(x_ij) <= cap_j * y_j    for all facilities j
     x_ij >= 0, y_j in {0, 1}
"""

from pyscipopt import Model
from .base import set_quiet, make_rng


def generate_instance(n_facilities=10, n_clients=30, seed=None):
    """n_facilities / n_clients are UPPER BOUNDS. Scale, cost and capacity
    distributions heavily randomized per-seed."""
    rng = make_rng(seed)
    model = Model("facility_location")
    set_quiet(model)

    n_facilities = int(rng.randint(2, n_facilities + 1))
    n_clients = int(rng.randint(3, n_clients + 1))

    # Randomize coordinate grid scale (affects transport cost magnitude)
    grid = float(rng.uniform(20, 200))
    fac_coords = rng.rand(n_facilities, 2) * grid
    cli_coords = rng.rand(n_clients, 2) * grid

    c = {}
    for i in range(n_clients):
        for j in range(n_facilities):
            dx = fac_coords[j, 0] - cli_coords[i, 0]
            dy = fac_coords[j, 1] - cli_coords[i, 1]
            c[i, j] = (dx**2 + dy**2) ** 0.5

    # Randomize fixed opening cost range widely
    f_lo = int(rng.randint(10, 150))
    f_hi = f_lo + int(rng.randint(20, 400))
    f = rng.randint(f_lo, f_hi, size=n_facilities).astype(float)

    # Randomize capacity tightness widely
    base = max(n_clients // max(n_facilities, 1), 2)
    cap_mult = float(rng.uniform(1.1, 4.0))
    cap = rng.randint(base, max(int(base * cap_mult), base + 1), size=n_facilities)

    # Guarantee feasibility: total capacity must cover all clients (each demand = 1)
    shortfall = n_clients - int(cap.sum())
    if shortfall > 0:
        cap[int(cap.argmax())] += shortfall

    y = {}
    x = {}
    for j in range(n_facilities):
        y[j] = model.addVar(vtype="B", obj=f[j], name=f"y_{j}")
    for i in range(n_clients):
        for j in range(n_facilities):
            x[i, j] = model.addVar(vtype="C", obj=c[i, j], lb=0, ub=1, name=f"x_{i}_{j}")

    for i in range(n_clients):
        model.addCons(sum(x[i, j] for j in range(n_facilities)) == 1, name=f"serve_{i}")

    for j in range(n_facilities):
        model.addCons(
            sum(x[i, j] for i in range(n_clients)) <= cap[j] * y[j],
            name=f"cap_{j}",
        )

    model.setMinimize()
    return model
