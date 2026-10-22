"""
Network Design Problem Generator

Given a graph and commodity demands, select edges and route flows.
min sum(f_e * y_e) + sum(c_e * f_e^k)
s.t. flow conservation for each commodity at each node
     f_e^k <= cap_e * y_e    for all edges e, commodities k
     y_e in {0, 1}
"""

from pyscipopt import Model
from .base import set_quiet, make_rng


def generate_instance(n_nodes=8, n_commodities=5, seed=None):
    rng = make_rng(seed)
    model = Model("network_design")
    set_quiet(model)

    # Generate random edges (not complete graph — random sparse graph)
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < 0.5:
                edges.append((i, j))

    # Always union a spanning chain so the graph is connected and no node is
    # isolated. Isolated nodes break flow conservation (empty balance) and make
    # commodities with an isolated source/sink infeasible.
    edge_set = set(edges)
    for i in range(n_nodes - 1):
        if (i, i + 1) not in edge_set:
            edges.append((i, i + 1))
            edge_set.add((i, i + 1))

    n_edges = len(edges)
    edge_idx = {e: k for k, e in enumerate(edges)}

    # Fixed installation costs and capacities
    fixed_cost = rng.randint(10, 50, size=n_edges)
    capacity = rng.randint(20, 80, size=n_edges)
    flow_cost = rng.randint(1, 10, size=n_edges)

    # Generate commodities: random (source, sink, demand)
    commodities = []
    for _ in range(n_commodities):
        src = rng.randint(0, n_nodes)
        sink = rng.randint(0, n_nodes)
        while sink == src:
            sink = rng.randint(0, n_nodes)
        demand = rng.randint(5, 20)
        commodities.append((src, sink, demand))

    # Variables
    y = {}
    f = {}   # forward flow u->v
    g = {}   # reverse flow v->u  (edges are undirected: flow allowed both ways)
    for k in range(n_edges):
        y[k] = model.addVar(vtype="B", obj=fixed_cost[k], name=f"y_{k}")
    for k in range(n_edges):
        for c in range(n_commodities):
            f[k, c] = model.addVar(
                vtype="C", lb=0, obj=flow_cost[k], name=f"f_{k}_{c}"
            )
            g[k, c] = model.addVar(
                vtype="C", lb=0, obj=flow_cost[k], name=f"g_{k}_{c}"
            )

    # Flow conservation at each node for each commodity.
    # For edge (u, v): forward flow f leaves u and arrives at v; reverse flow g
    # leaves v and arrives at u.
    for c, (src, sink, demand) in enumerate(commodities):
        for node in range(n_nodes):
            flow_out = []
            flow_in = []
            for e_idx, (u, v) in enumerate(edges):
                if u == node:
                    flow_out.append(f[e_idx, c])
                    flow_in.append(g[e_idx, c])
                if v == node:
                    flow_out.append(g[e_idx, c])
                    flow_in.append(f[e_idx, c])

            if node == src:
                rhs = demand
            elif node == sink:
                rhs = -demand
            else:
                rhs = 0

            model.addCons(
                sum(flow_out) - sum(flow_in) == rhs,
                name=f"flow_{node}_{c}",
            )

    # Capacity constraints: total flow across the edge (both directions, all
    # commodities) is bounded by the installed capacity.
    for k in range(n_edges):
        model.addCons(
            sum(f[k, c] + g[k, c] for c in range(n_commodities)) <= capacity[k] * y[k],
            name=f"cap_{k}",
        )

    model.setMinimize()
    return model
