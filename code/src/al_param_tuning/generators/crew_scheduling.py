"""
Crew Scheduling Problem Generator

Assign crew members to tasks over a planning horizon.
min sum(c_ij * x_ij) + penalty for unassigned tasks
s.t. each task assigned to at most one crew (or uncovered with penalty)
     crew availability / skill constraints
     rest time between consecutive tasks
"""

from pyscipopt import Model
from .base import set_quiet, make_rng


def generate_instance(n_crew=8, n_tasks=20, seed=None):
    rng = make_rng(seed)
    model = Model("crew_scheduling")
    set_quiet(model)

    # Task attributes: start time, duration, required skill level
    task_starts = sorted(rng.randint(0, 100, size=n_tasks))
    task_durations = rng.randint(5, 20, size=n_tasks)
    task_skills = rng.randint(1, 4, size=n_tasks)  # skill level 1-3

    # Crew attributes: skill level, max hours
    crew_skills = rng.randint(1, 4, size=n_crew)
    crew_max_hours = rng.randint(40, 80, size=n_crew)

    # Cost matrix (crew-task compatibility)
    cost = rng.randint(10, 100, size=(n_crew, n_tasks))

    # Penalty for unassigned tasks
    penalty = 500

    # Variables
    x = {}
    for i in range(n_crew):
        for j in range(n_tasks):
            x[i, j] = model.addVar(vtype="B", obj=cost[i, j], name=f"x_{i}_{j}")

    # Uncovered task variables
    u = {}
    for j in range(n_tasks):
        u[j] = model.addVar(vtype="B", obj=penalty, name=f"u_{j}")

    # Each task: covered by one crew or uncovered
    for j in range(n_tasks):
        model.addCons(
            sum(x[i, j] for i in range(n_crew)) + u[j] == 1,
            name=f"cover_{j}",
        )

    # Skill constraint: crew can only do tasks at or below their skill level
    for i in range(n_crew):
        for j in range(n_tasks):
            if crew_skills[i] < task_skills[j]:
                model.addCons(x[i, j] == 0, name=f"skill_{i}_{j}")

    # Time overlap constraint: crew cannot do overlapping tasks
    for i in range(n_crew):
        for a in range(n_tasks):
            for b in range(a + 1, n_tasks):
                # Check if tasks a and b overlap in time
                end_a = task_starts[a] + task_durations[a]
                end_b = task_starts[b] + task_durations[b]
                if task_starts[b] < end_a and task_starts[a] < end_b:
                    model.addCons(
                        x[i, a] + x[i, b] <= 1,
                        name=f"overlap_{i}_{a}_{b}",
                    )

    # Max hours constraint per crew (simplified: sum of durations)
    for i in range(n_crew):
        model.addCons(
            sum(task_durations[j] * x[i, j] for j in range(n_tasks)) <= crew_max_hours[i],
            name=f"hours_{i}",
        )

    model.setMinimize()
    return model
