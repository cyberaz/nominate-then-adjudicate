"""
Generalized Assignment Problem Generator

min sum(c_ij * x_ij)
s.t. sum(x_ij) = 1           for all jobs j (each job assigned once)
     sum(r_ij * x_ij) <= b_i for all agents i (capacity)
     x_ij in {0, 1}
"""

from pyscipopt import Model
import numpy as np
from .base import set_quiet, make_rng


def generate_instance(n_agents=5, n_jobs=20, seed=None):
    rng = make_rng(seed)
    model = Model("assignment")
    set_quiet(model)

    resource = rng.randint(5, 25, size=(n_agents, n_jobs))
    # Inverse cost-resource correlation (OR-library type-D style): on each agent,
    # jobs that are CHEAP tend to consume MORE capacity. This tension is what makes
    # GAP genuinely hard — with uniform-random independent costs (the previous
    # scheme) the LP relaxation is integral and SCIP closes the gap at the root in
    # 1 node (~0.1-0.4s regardless of size), so no instance ever lands in the
    # [0.5,5]s window. Anti-correlated costs force real branch-and-bound.
    noise = rng.randint(-4, 5, size=(n_agents, n_jobs))
    cost = np.clip(30 - resource + noise, 1, None).astype(int)

    # Capacity set so total supply only modestly exceeds the expected per-agent
    # load (jobs_per_agent x mean_resource), keeping the capacity constraints
    # BINDING rather than slack.
    jobs_per_agent = n_jobs / n_agents
    mean_resource = float(resource.mean())
    base_cap = jobs_per_agent * mean_resource
    capacity = rng.randint(
        int(base_cap * 1.1),
        int(base_cap * 1.4) + 1,
        size=n_agents,
    )

    x = {}
    for i in range(n_agents):
        for j in range(n_jobs):
            x[i, j] = model.addVar(
                vtype="B", obj=cost[i, j], name=f"x_{i}_{j}"
            )

    # Each job assigned exactly once
    for j in range(n_jobs):
        model.addCons(
            sum(x[i, j] for i in range(n_agents)) == 1,
            name=f"assign_{j}",
        )

    # Agent capacity
    for i in range(n_agents):
        model.addCons(
            sum(resource[i, j] * x[i, j] for j in range(n_jobs)) <= capacity[i],
            name=f"cap_{i}",
        )

    model.setMinimize()
    return model
