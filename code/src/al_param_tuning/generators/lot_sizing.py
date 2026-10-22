"""
Capacitated Lot Sizing Problem Generator

min sum(p_t * x_t + h_t * s_t + f_t * y_t)
s.t. s_t = s_{t-1} + x_t - d_t   for all t (inventory balance)
     x_t <= M * y_t               for all t (setup forcing)
     s_0 = 0, s_T >= 0
     x_t, s_t >= 0, y_t in {0, 1}

Multi-product variant with shared capacity.
"""

from pyscipopt import Model
from .base import set_quiet, make_rng


def generate_instance(n_periods=12, n_products=3, seed=None):
    rng = make_rng(seed)
    model = Model("lot_sizing")
    set_quiet(model)

    T = n_periods
    P = n_products

    # Costs
    prod_cost = rng.randint(5, 20, size=(P, T))
    hold_cost = rng.randint(1, 5, size=(P, T))
    setup_cost = rng.randint(50, 200, size=(P, T))

    # Demand
    demand = rng.randint(10, 50, size=(P, T))

    # Capacity per period (shared across products)
    capacity = rng.randint(
        int(demand.sum(axis=0).max() * 1.2),
        int(demand.sum(axis=0).max() * 2.0) + 1,
        size=T,
    )
    big_M = int(capacity.max())

    # Variables
    x = {}  # production
    s = {}  # inventory
    y = {}  # setup

    for p in range(P):
        for t in range(T):
            x[p, t] = model.addVar(
                vtype="C", lb=0, obj=prod_cost[p, t], name=f"x_{p}_{t}"
            )
            s[p, t] = model.addVar(
                vtype="C", lb=0, obj=hold_cost[p, t], name=f"s_{p}_{t}"
            )
            y[p, t] = model.addVar(
                vtype="B", obj=setup_cost[p, t], name=f"y_{p}_{t}"
            )

    # Inventory balance
    for p in range(P):
        for t in range(T):
            prev_inv = s[p, t - 1] if t > 0 else 0
            model.addCons(
                s[p, t] == prev_inv + x[p, t] - demand[p, t],
                name=f"inv_{p}_{t}",
            )

    # Setup forcing
    for p in range(P):
        for t in range(T):
            model.addCons(
                x[p, t] <= big_M * y[p, t],
                name=f"setup_{p}_{t}",
            )

    # Shared capacity
    for t in range(T):
        model.addCons(
            sum(x[p, t] for p in range(P)) <= capacity[t],
            name=f"cap_{t}",
        )

    model.setMinimize()
    return model
