"""
Single Machine Job Scheduling (Weighted Tardiness) Generator

min sum(w_j * T_j)
where T_j = max(0, C_j - d_j), C_j = completion time of job j

Uses big-M formulation with binary precedence variables.
"""

from pyscipopt import Model
from .base import set_quiet, make_rng


def generate_instance(n_jobs=10, seed=None):
    rng = make_rng(seed)
    model = Model("job_scheduling")
    set_quiet(model)

    proc_times = rng.randint(5, 30, size=n_jobs)
    due_dates = rng.randint(20, int(proc_times.sum() * 0.8) + 1, size=n_jobs)
    weights = rng.randint(1, 10, size=n_jobs)
    big_M = int(proc_times.sum()) + 1

    # Start time variables
    s = {}
    for j in range(n_jobs):
        s[j] = model.addVar(vtype="C", lb=0, ub=big_M, name=f"s_{j}")

    # Precedence variables: z_ij = 1 if job i precedes job j
    z = {}
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i != j:
                z[i, j] = model.addVar(vtype="B", name=f"z_{i}_{j}")

    # Tardiness variables
    T = {}
    for j in range(n_jobs):
        T[j] = model.addVar(vtype="C", lb=0, name=f"T_{j}")

    # Objective: minimize weighted tardiness
    model.setObjective(sum(weights[j] * T[j] for j in range(n_jobs)), "minimize")

    # Non-overlap: either i before j or j before i
    for i in range(n_jobs):
        for j in range(i + 1, n_jobs):
            model.addCons(
                s[i] + proc_times[i] <= s[j] + big_M * (1 - z[i, j]),
                name=f"prec_{i}_{j}_a",
            )
            model.addCons(
                s[j] + proc_times[j] <= s[i] + big_M * z[i, j],
                name=f"prec_{i}_{j}_b",
            )

    # Tardiness definition
    for j in range(n_jobs):
        C_j = s[j] + proc_times[j]
        model.addCons(T[j] >= C_j - due_dates[j], name=f"tard_{j}")

    return model
