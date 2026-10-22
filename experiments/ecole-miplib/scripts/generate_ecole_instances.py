"""
Generate Ecole-style benchmark instances as .lp files, calibrated to a target
default-solve-time window.

Ecole (BSD-3, github.com/ds4dm/ecole) publishes no Windows build -- its setup.py
raises `NotImplementedError: OS Windows is not supported`, and conda-forge only
ships linux-64/aarch64 and osx-64/arm64 binaries. This script therefore ports the
four built-in Ecole instance generators to NumPy + PySCIPOpt, transcribed
line-by-line from the upstream C++ sources at ds4dm/ecole@master:

    libecole/src/instance/set-cover.cpp
    libecole/src/instance/combinatorial-auction.cpp
    libecole/src/instance/capacitated-facility-location.cpp
    libecole/src/instance/independent-set.cpp
    libecole/src/utility/graph.cpp            (barabasi_albert, greedy_clique_partition)

Default parameters are taken verbatim from the corresponding
libecole/include/ecole/instance/*.hpp `Parameters` structs.

FIDELITY NOTE. The control flow, formulas, constraint sets and defaults follow the
upstream sources. What cannot be reproduced is the random *stream*: Ecole draws from
a C++ RandomGenerator through xtensor's samplers, which no NumPy Generator
replicates bit-for-bit. Instances are therefore drawn from the same distributions
by the same procedure, but a given seed does not reproduce a given Ecole instance.
The only intentional addition is an adaptive size search, so that SCIP's default
configuration solves each instance within [min-time, max-time].

Files are written to data/results/instances/ as
    <family>_<tag>_<N>.lp         e.g. set_covering_ecole_1.lp
so that scripts/epm_with_selection.py finds them with
    families: [set_covering, ...]   variants: ['ecole']   seed_offset: 1

SEED CONVENTION (same as ensure_lp elsewhere in this repo): the number in the
file name IS the nominal RNG seed, written unpadded. If that seed's instance falls
outside the solve-time window, the attempt is retried with the derived seed
N*1000+attempt,
while the file keeps its nominal number. The actual seed used, the calibrated
size and the measured default time are recorded in the manifest.

Usage:
    python scripts/generate_ecole_instances.py
    python scripts/generate_ecole_instances.py --families set_covering independent_set -n 10
    # append instances 11..40 without touching 01..10:
    python scripts/generate_ecole_instances.py --tag ecole_train --start-index 11 -n 30
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pyscipopt import Model, quicksum


# ───────────────────────────── shared utilities ──────────────────────────────
def arg_choice_without_replacement(n_samples, weights, rng):
    """Ecole utility::arg_choice_without_replacement: sample `n_samples` distinct
    indices with probability proportional to `weights`."""
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total <= 0:
        # degenerate weights: fall back to uniform over the support
        return rng.choice(len(w), size=n_samples, replace=False)
    return rng.choice(len(w), size=n_samples, replace=False, p=w / total)


# ───────────────── Ecole generator 1: set cover (set-cover.cpp) ──────────────
def gen_set_covering(rng, n_rows=500, n_cols=1000, density=0.05, max_coef=100):
    """Port of SetCoverGenerator::generate_instance (set-cover.cpp).
    Defaults from set-cover.hpp: 500 / 1000 / 0.05 / 100.

    The construction seeds `indices` with 2 rows per column and 1 column per row,
    so it needs nnzrs >= max(2*n_cols, n_rows), i.e. n_rows*density >= 2 and
    n_cols*density >= 1. Ecole's defaults satisfy this comfortably (500*0.05 = 25);
    randomized draws can violate it, and numpy then fails with an opaque
    "could not broadcast" error instead of naming the cause.
    """
    nnzrs = int(n_rows * n_cols * density)
    need = max(2 * n_cols, n_rows)
    if nnzrs < need:
        raise ValueError(
            f"set_covering needs nnzrs >= max(2*n_cols, n_rows) = {need}, got {nnzrs} "
            f"from n_rows={n_rows}, n_cols={n_cols}, density={density:.4f}; "
            f"density must be at least {(need + 1) / (n_rows * n_cols):.4f}")

    indices = np.zeros(nnzrs, dtype=np.int64)

    # force at least 2 rows per col:  xt::arange(0, 2*n_cols) % n_cols
    indices[: 2 * n_cols] = np.arange(2 * n_cols) % n_cols

    # assign remaining column indexes at random:
    #   get_choice_in_range(0, n_cols*(n_rows-2), nnzrs - 2*n_cols, /*replace=*/false) % n_cols
    n_tail = nnzrs - 2 * n_cols
    if n_tail > 0:
        hi = n_cols * (n_rows - 2)
        indices[2 * n_cols :] = rng.choice(hi, size=n_tail, replace=False) % n_cols

    # get_counts(indices, n_cols): per-column counts, zeros included
    col_n_rows = np.bincount(indices, minlength=n_cols)[:n_cols]

    # ensure at least 1 column per row
    indices[:n_rows] = rng.permutation(n_rows)

    i = 0
    indptr = [0]
    for n in col_n_rows:
        n = int(n)
        if i + n <= n_rows:
            pass                                  # upstream first branch is empty
        elif i >= n_rows:
            indices[i : i + n] = rng.choice(n_rows, size=n, replace=False)
        else:                                     # i + n > n_rows
            remaining = np.setdiff1d(np.arange(n_rows), indices[i:n_rows])
            indices[n_rows : i + n] = rng.choice(remaining, size=i + n - n_rows, replace=False)
        i += n
        indptr.append(i)

    # csc -> csr: rows -> covering columns
    rows = [[] for _ in range(n_rows)]
    for col in range(n_cols):
        for jj in range(indptr[col], indptr[col + 1]):
            rows[int(indices[jj])].append(col)

    # sample coefficients: randint(0, max_coef) + 1
    c = rng.randint(0, max_coef, size=n_cols) + 1

    m = Model(f"SetCover-{n_rows}-{n_cols}")
    m.hideOutput()
    x = [m.addVar(vtype="B", lb=0.0, ub=1.0, obj=float(c[j]), name=f"x_{j}")
         for j in range(n_cols)]
    for i_row, cols in enumerate(rows):
        if cols:
            m.addCons(quicksum(x[j] for j in cols) >= 1.0, name=f"c_{i_row}")
    m.setMinimize()
    return m


# ───── Ecole generator 2: combinatorial auction (combinatorial-auction.cpp) ──
def _ca_bundle_price(bundle, private_values, integers, additivity):
    """get_bundle_price"""
    price = private_values[bundle].sum() + float(len(bundle)) ** (1.0 + additivity)
    return np.floor(price) if integers else price


def _ca_choose_next_item(bundle_mask, interests, compats, rng):
    """choose_next_item:
        compats_masked      = compats * bundle_mask      (mask broadcast over columns)
        compats_masked_mean = mean(compats_masked, axis=1)
        probs               = (1 - bundle_mask) * interests * compats_masked_mean
    """
    compats_masked = compats * bundle_mask
    compats_masked_mean = compats_masked.mean(axis=1)
    probs = (1 - bundle_mask) * interests * compats_masked_mean
    return int(arg_choice_without_replacement(1, probs, rng)[0])


def gen_combinatorial_auction(rng, n_items=100, n_bids=500, min_value=1, max_value=100,
                              value_deviation=0.5, add_item_prob=0.65, max_n_sub_bids=5,
                              additivity=0.2, budget_factor=1.5, resale_factor=0.5,
                              integers=False):
    """Port of CombinatorialAuctionGenerator::generate_instance.
    Defaults from combinatorial-auction.hpp (note add_item_prob = 0.65)."""
    if max_value < min_value:
        raise ValueError("min_value <= max_value required")
    if not 0.0 <= add_item_prob <= 1.0:
        raise ValueError("add_item_prob must be in [0,1]")

    values = min_value + (max_value - min_value) * rng.rand(n_items)

    # compats = triu(rand,1); compats += transpose(compats); compats /= sum(compats, 1)
    compats = np.triu(rng.rand(n_items, n_items), 1)
    compats = compats + compats.T
    compats = compats / compats.sum(axis=1)      # broadcasts over the LAST axis, as upstream

    bids = []                                    # list of (bundle, price)
    n_dummy_items = 0

    while len(bids) < n_bids:
        private_interests = rng.rand(n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)

        # ---- get_bundle ----
        item = int(arg_choice_without_replacement(1, private_interests, rng)[0])
        bundle_mask = np.zeros(n_items, dtype=np.int64)
        bundle_mask[item] = 1
        while True:
            if rng.rand() >= add_item_prob:
                break
            if bundle_mask.sum() == n_items:
                break
            item = _ca_choose_next_item(bundle_mask, private_interests, compats, rng)
            bundle_mask[item] = 1
        bundle = np.nonzero(bundle_mask)[0]
        price = _ca_bundle_price(bundle, private_values, integers, additivity)

        if price < 0:                            # negatively priced bundle -> restart bidder
            continue

        bidder_bids = {tuple(bundle.tolist()): price}

        # ---- get_substitute_bundles ----
        sub_bundles = []
        for it in bundle:
            sub_mask = np.zeros(n_items, dtype=np.int64)
            sub_mask[it] = 1
            while sub_mask.sum() < len(bundle):
                nxt = _ca_choose_next_item(sub_mask, private_interests, compats, rng)
                sub_mask[nxt] = 1
            sub_bundle = np.nonzero(sub_mask)[0]
            sub_bundles.append((sub_bundle,
                                _ca_bundle_price(sub_bundle, private_values, integers, additivity)))

        # ---- add_bundles ----
        budget = budget_factor * price
        min_resale_value = resale_factor * values[bundle].sum()
        sub_bundles.sort(key=lambda t: -t[1])    # highest price first
        for sub_bundle, sub_price in sub_bundles:
            if len(bidder_bids) >= max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= n_bids:
                break
            if sub_price < 0:
                continue
            if sub_price > budget:
                continue
            if values[sub_bundle].sum() < min_resale_value:
                continue
            key = tuple(sub_bundle.tolist())
            if key in bidder_bids:
                continue
            bidder_bids[key] = sub_price

        # ---- add_dummy_item ----  (0 means "no dummy", as upstream)
        dummy_item = 0
        if len(bidder_bids) > 2:
            dummy_item = n_items + n_dummy_items
            n_dummy_items += 1

        # ---- add_bids ----  (std::map iterates in sorted key order)
        for key in sorted(bidder_bids):
            b = list(key) + ([dummy_item] if dummy_item else [])
            bids.append((b, bidder_bids[key]))

    bids = bids[:n_bids]

    m = Model(f"CombinatorialAuction-{n_items}-{n_bids}")
    m.hideOutput()
    x = [m.addVar(vtype="B", lb=0.0, ub=1.0, obj=float(p), name=f"x_{i}")
         for i, (_, p) in enumerate(bids)]
    bids_per_item = [[] for _ in range(n_items + n_dummy_items)]
    for i, (bundle, _) in enumerate(bids):
        for it in bundle:
            bids_per_item[it].append(i)
    for it, idxs in enumerate(bids_per_item):
        if idxs:
            m.addCons(quicksum(x[i] for i in idxs) <= 1.0, name=f"c_{it}")
    m.setMaximize()
    return m


# ── Ecole generator 3: capacitated facility location (capacitated-...cpp) ────
def gen_facility_location(rng, n_customers=100, n_facilities=100,
                          continuous_assignment=True, ratio=5.0,
                          demand_interval=(5, 35 + 1), capacity_interval=(10, 160 + 1),
                          fixed_cost_cste_interval=(0, 90 + 1),
                          fixed_cost_scale_interval=(100, 110 + 1)):
    """Port of CapacitatedFacilityLocationGenerator::generate_instance.
    Defaults from capacitated-facility-location.hpp."""
    demands = rng.randint(demand_interval[0], demand_interval[1], size=n_customers)
    capacities = rng.randint(capacity_interval[0], capacity_interval[1], size=n_facilities)
    fixed_costs = (rng.randint(fixed_cost_scale_interval[0], fixed_cost_scale_interval[1],
                               size=n_facilities) * np.sqrt(capacities)
                   + rng.randint(fixed_cost_cste_interval[0], fixed_cost_cste_interval[1],
                                 size=n_facilities))

    # unit_transportation_costs: euclidean distance between random points, times 10
    c_x, c_y = rng.rand(n_customers), rng.rand(n_customers)
    f_x, f_y = rng.rand(n_facilities), rng.rand(n_facilities)
    unit_trans = 10.0 * np.sqrt((c_x.reshape(-1, 1) - f_x.reshape(1, -1)) ** 2
                               + (c_y.reshape(-1, 1) - f_y.reshape(1, -1)) ** 2)
    trans_costs = unit_trans * demands.reshape(-1, 1)

    # scale capacities according to ratio AFTER sampling (Cornuejols et al. 1991)
    total_demand = float(demands.sum())
    capacities = capacities * ratio * total_demand / float(capacities.sum())

    m = Model(f"CapacitatedFacilityLocation-{n_customers}-{n_facilities}")
    m.hideOutput()
    y = [m.addVar(vtype="B", lb=0.0, ub=1.0, obj=float(fixed_costs[j]), name=f"f_{j}")
         for j in range(n_facilities)]
    vt = "C" if continuous_assignment else "B"
    x = {(i, j): m.addVar(vtype=vt, lb=0.0, ub=1.0, obj=float(trans_costs[i, j]),
                          name=f"s_{i}_{j}")
         for i in range(n_customers) for j in range(n_facilities)}

    # add_demand_cons: for each customer, sum_j x_ij >= 1
    for i in range(n_customers):
        m.addCons(quicksum(x[i, j] for j in range(n_facilities)) >= 1.0, name=f"d_{i}")

    # add_capacity_cons: sum_i demand_i * x_ij - capacity_j * y_j <= 0
    for j in range(n_facilities):
        m.addCons(quicksum(float(demands[i]) * x[i, j] for i in range(n_customers))
                  - float(capacities[j]) * y[j] <= 0.0, name=f"c_{j}")

    # total demand cons: sum_j capacity_j * y_j >= total_demand
    m.addCons(quicksum(float(capacities[j]) * y[j] for j in range(n_facilities))
              >= total_demand, name="t_total_demand")

    # add_tightening_cons: x_ij - y_j <= 0
    for i in range(n_customers):
        for j in range(n_facilities):
            m.addCons(x[i, j] - y[j] <= 0.0, name=f"t_{i}_{j}")

    m.setMinimize()
    return m


# ─── Ecole generator 4: independent set (independent-set.cpp, graph.cpp) ─────
def _barabasi_albert(rng, n_nodes, affinity):
    """Port of Graph::barabasi_albert (graph.cpp): the first `affinity` nodes are
    all connected to node 0 (star), then each new node attaches to `affinity`
    existing nodes with probability proportional to degree."""
    if not 1 <= affinity < n_nodes:
        raise ValueError("Affinity must be between 1 and the number of nodes.")
    neighbors = {v: set() for v in range(n_nodes)}
    degrees = np.zeros(n_nodes, dtype=np.int64)

    def add_edge(u, v):
        if v in neighbors[u]:
            return
        neighbors[u].add(v)
        neighbors[v].add(u)
        degrees[u] += 1
        degrees[v] += 1

    for n in range(1, affinity + 1):
        add_edge(0, n)
    for n in range(affinity + 1, n_nodes):
        for nb in arg_choice_without_replacement(affinity, degrees[:n], rng):
            add_edge(n, int(nb))
    return neighbors, degrees


def _erdos_renyi(rng, n_nodes, edge_probability):
    """Port of Graph::erdos_renyi: each of the n(n-1)/2 pairs is independently
    sampled with probability edge_probability."""
    neighbors = {v: set() for v in range(n_nodes)}
    degrees = np.zeros(n_nodes, dtype=np.int64)
    # Vectorized to keep n_nodes up to ~1000 fast
    upper = np.triu(rng.rand(n_nodes, n_nodes) < edge_probability, k=1)
    us, vs = np.nonzero(upper)
    for u, v in zip(us.tolist(), vs.tolist()):
        neighbors[u].add(int(v))
        neighbors[v].add(int(u))
        degrees[u] += 1
        degrees[v] += 1
    return neighbors, degrees


def _greedy_clique_partition(n_nodes, neighbors, degrees):
    """Port of Graph::greedy_clique_partition (graph.cpp): repeatedly start a clique
    from the leftover node with maximum degree, then greedily absorb leftover
    neighbours that stay connected to every current clique member."""
    leftover = {v: int(degrees[v]) for v in range(n_nodes)}
    partition = []
    while leftover:
        # extract_node_with_max_degree
        center = max(leftover, key=lambda v: (leftover[v], -v))
        del leftover[center]
        # best_clique_candidates: neighbours still leftover, highest degree first
        candidates = sorted((v for v in neighbors[center] if v in leftover),
                            key=lambda v: (-leftover[v], v))
        clique = [center]
        for node in candidates:
            if all(node in neighbors[c] for c in clique):
                clique.append(node)
                leftover.pop(node, None)
        partition.append(clique)
    return partition


def gen_independent_set(rng, n_nodes=500, graph_type="barabasi_albert",
                        affinity=4, edge_probability=0.25):
    """Port of IndependentSetGenerator::generate_instance. Defaults from
    independent-set.hpp: n_nodes=500, graph_type=barabasi_albert, affinity=4,
    edge_probability=0.25."""
    if graph_type == "barabasi_albert":
        neighbors, degrees = _barabasi_albert(rng, n_nodes, affinity)
    elif graph_type == "erdos_renyi":
        neighbors, degrees = _erdos_renyi(rng, n_nodes, edge_probability)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    partition = _greedy_clique_partition(n_nodes, neighbors, degrees)
    clique_id = np.empty(n_nodes, dtype=np.int64)
    for cid, clique in enumerate(partition):
        for v in clique:
            clique_id[v] = cid

    m = Model(f"IndependentSet-{n_nodes}")
    m.hideOutput()
    v = [m.addVar(vtype="B", lb=0.0, ub=1.0, obj=1.0, name=f"n_{i}") for i in range(n_nodes)]

    idx = 0
    # one constraint per clique in the partition (including singletons, as upstream)
    for clique in partition:
        m.addCons(quicksum(v[u] for u in clique) <= 1.0, name=f"c_{idx}")
        idx += 1
    # constraints for edges whose endpoints are in different cliques
    for u in range(n_nodes):
        for w in neighbors[u]:
            if w > u and clique_id[u] != clique_id[w]:
                m.addCons(v[u] + v[w] <= 1.0, name=f"c_{idx}")
                idx += 1
    # unconnected nodes, otherwise SCIP complains
    for u in range(n_nodes):
        if len(neighbors[u]) == 0:
            m.addCons(v[u] <= 1.0, name=f"c_{idx}")
            idx += 1

    m.setMaximize()
    return m


# ───────────────────────────── size-adaptive driver ──────────────────────────
GENERATORS = {
    "set_covering": dict(
        fn=gen_set_covering,
        base=dict(n_rows=500, n_cols=1000, density=0.05, max_coef=100),
        scale=("n_rows", "n_cols"),
    ),
    "combinatorial_auction": dict(
        fn=gen_combinatorial_auction,
        base=dict(n_items=100, n_bids=500),
        scale=("n_items", "n_bids"),
    ),
    "facility_location": dict(
        fn=gen_facility_location,
        base=dict(n_customers=100, n_facilities=100),
        scale=("n_customers", "n_facilities"),
    ),
    "independent_set": dict(
        fn=gen_independent_set,
        base=dict(n_nodes=500, affinity=4),
        scale=("n_nodes",),
    ),
}


def sample_kwargs_random(fam, rng, factor):
    """Per-instance parameter sampling within Ecole's exposed parameter space.
    Ranges are wide enough to strongly heterogenize the resulting instances
    (variable size, density, cost ranges, capacity/demand intervals, graph
    types), matching the randomization style of the self-generators without
    modifying any of Ecole's algorithms."""
    if fam == "set_covering":
        # Ecole exposes n_rows / n_cols / density / max_coef only. Constraint
        # coefficients and RHS are hardwired to 1 in Ecole, so matrix_coeff_cv
        # and rhs_cv still cannot vary; the rest can.
        n_rows = max(30, int(rng.randint(100, 801) * factor))
        n_cols = max(60, int(rng.randint(200, 1501) * factor))
        density = float(rng.uniform(0.02, 0.15))
        # The construction needs nnzrs >= max(2*n_cols, n_rows); with factor
        # shrinking the sizes toward their floors, a low density draw falls below
        # that. Clamp density rather than fixing up nnzrs inside the generator, so
        # the kwargs recorded in the manifest describe the instance actually built.
        density = max(density, (max(2 * n_cols, n_rows) + 1) / (n_rows * n_cols))
        return dict(
            n_rows=n_rows,
            n_cols=n_cols,
            density=density,
            max_coef=int(rng.randint(20, 501)),
        )
    if fam == "combinatorial_auction":
        return dict(
            n_items=max(20, int(rng.randint(40, 151) * factor)),
            n_bids=max(60, int(rng.randint(200, 801) * factor)),
            min_value=1,
            max_value=int(rng.randint(50, 201)),
            value_deviation=float(rng.uniform(0.3, 0.8)),
            add_item_prob=float(rng.uniform(0.4, 0.85)),
            max_n_sub_bids=int(rng.randint(3, 9)),
            additivity=float(rng.uniform(0.05, 0.40)),
            budget_factor=float(rng.uniform(1.2, 2.0)),
            resale_factor=float(rng.uniform(0.3, 0.7)),
            integers=bool(rng.random() < 0.3),
        )
    if fam == "facility_location":
        dlo = int(rng.randint(3, 12)); dhi = dlo + int(rng.randint(15, 45))
        clo = int(rng.randint(5, 30)); chi = clo + int(rng.randint(70, 180))
        slo = int(rng.randint(80, 130)); shi = slo + int(rng.randint(15, 60))
        return dict(
            n_customers=max(15, int(rng.randint(30, 151) * factor)),
            n_facilities=max(15, int(rng.randint(30, 151) * factor)),
            continuous_assignment=bool(rng.random() < 0.75),
            ratio=float(rng.uniform(3.0, 8.0)),
            demand_interval=(dlo, dhi + 1),
            capacity_interval=(clo, chi + 1),
            fixed_cost_cste_interval=(0, int(rng.randint(60, 121)) + 1),
            fixed_cost_scale_interval=(slo, shi + 1),
        )
    if fam == "independent_set":
        gt = "erdos_renyi" if rng.random() < 0.5 else "barabasi_albert"
        n = max(60, int(rng.randint(100, 801) * factor))
        if gt == "barabasi_albert":
            return dict(n_nodes=n, graph_type=gt, affinity=int(rng.randint(2, 9)))
        return dict(n_nodes=n, graph_type=gt,
                    edge_probability=float(rng.uniform(0.03, 0.30)))
    raise ValueError(f"unknown family: {fam}")


def scaled_kwargs(spec, factor, min_size=20):
    kw = dict(spec["base"])
    for k in spec["scale"]:
        kw[k] = max(min_size, int(round(spec["base"][k] * factor)))
    return kw


def solve_default(model, time_limit):
    """Solve under the SCIP default, single-threaded; return (time, has_sol, status)."""
    model.hideOutput()
    model.setParam("limits/time", time_limit)
    model.setParam("lp/threads", 1)
    model.setParam("parallel/maxnthreads", 1)
    model.setParam("randomization/randomseedshift", 42)
    t0 = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - t0
    has_sol = model.getNSols() > 0
    status = model.getStatus()
    model.freeProb()
    return elapsed, has_sol, status


def main():
    ap = argparse.ArgumentParser(
        description="Generate Ecole-ported .lp instances inside a solve-time window")
    ap.add_argument("--families", nargs="*", default=list(GENERATORS),
                    choices=list(GENERATORS))
    ap.add_argument("-n", "--n-instances", type=int, default=10)
    ap.add_argument("--tag", default="ecole", help="Middle name token (the yaml 'variant')")
    ap.add_argument("--min-time", type=float, default=0.5)
    ap.add_argument("--max-time", type=float, default=5.0)
    ap.add_argument("--max-attempts", type=int, default=40)
    ap.add_argument("--out-dir", default="data/results/instances")
    ap.add_argument("--start-index", type=int, default=1,
                    help="First instance number; the number in the file name IS the "
                         "nominal RNG seed, as elsewhere in this repo. Use this to append "
                         "instances without regenerating existing ones")
    ap.add_argument("--manifest", default="data/results/instances/ecole_manifest.json",
                    help="Records file -> (actual seed, size, solve time); '' to disable")
    ap.add_argument("--seed-offset", type=int, default=None,
                    help="Added to the filename number to form the actual RNG seed base. "
                         "Purpose: keep separate splits from ever drawing the same "
                         "instance under the same nominal seed. Default: 500000 for tag "
                         "'ecole_train', 0 otherwise. An md5 audit against test files "
                         "found 7 of 32 collisions before this was added.")
    ap.add_argument("--randomize", type=lambda s: s.lower() not in ("0","false","no"),
                    default=True,
                    help="If True (default), sample Ecole parameters per instance "
                         "(size, density, cost ranges, intervals, graph type, ...) "
                         "from wide distributions. If False, use Ecole's fixed defaults "
                         "and only adapt sizes to the time window.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.seed_offset is None:
        args.seed_offset = 500000 if args.tag == "ecole_train" else 0

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    limit = args.max_time + 2.0
    report = {}
    manifest = {}
    man_path = Path(args.manifest) if args.manifest else None
    if man_path and man_path.exists():
        try:
            manifest = json.load(open(man_path, encoding="utf-8"))
        except Exception:
            manifest = {}

    for fam in args.families:
        spec = GENERATORS[fam]
        factor = 1.0                      # persists across instances: converges fast
        print(f"\n=== {fam} (Ecole defaults {spec['base']}) ===", flush=True)
        made = 0
        for idx in range(args.start_index, args.start_index + args.n_instances):
            path = out / f"{fam}_{args.tag}_{idx}.lp"
            if path.exists() and not args.overwrite:
                print(f"  {path.name}: exists, skip")
                made += 1
                continue

            best = None
            accepted = False
            base = idx + args.seed_offset
            for att in range(args.max_attempts):
                # repo convention (cf. ensure_lp): the file's number IS the nominal
                # seed; a miss on the time window derives base*1000 + attempt.
                # seed_offset (default 500000 for ecole_train) shifts base so that
                # different splits never draw the same instance at the same number.
                seed = base if att == 0 else base * 1000 + att
                rng = np.random.RandomState(seed)
                kw = (sample_kwargs_random(fam, rng, factor)
                      if args.randomize else scaled_kwargs(spec, factor))
                try:
                    model = spec["fn"](rng, **kw)
                except ValueError as e:
                    # A draw the construction rejects costs one attempt (the next
                    # one uses a different derived seed), rather than aborting a
                    # multi-hour run. Printed, so a systematic problem stays visible.
                    print(f"    attempt {att}: rejected draw ({e})", flush=True)
                    continue
                t, has_sol, _ = solve_default(model, limit)

                if has_sol and args.min_time <= t <= args.max_time:
                    accepted = True
                    break
                if has_sol:
                    target = np.sqrt(args.min_time * args.max_time)
                    err = abs(np.log(max(t, 1e-3) / target))
                    if best is None or err < best[0]:
                        best = (err, t, kw, seed)
                if (not has_sol) or t > args.max_time:
                    factor *= 0.75
                elif t < args.min_time:
                    factor *= 1.35

            if not accepted:
                if best is None:
                    print(f"  {path.name}: FAILED (no solvable candidate in "
                          f"{args.max_attempts} attempts)")
                    continue
                _, t, kw, seed = best
                tag = f"fallback {t:.2f}s"
            else:
                tag = f"{t:.2f}s in-window"

            rng = np.random.RandomState(seed)
            if args.randomize:
                sample_kwargs_random(fam, rng, factor)
            model = spec["fn"](rng, **kw)
            n_var, n_con = model.getNVars(), model.getNConss()
            model.writeProblem(str(path))
            model.freeProb()
            kw_serializable = {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in kw.items()}
            manifest[path.name] = dict(family=fam, tag=args.tag, index=idx,
                                       seed_offset=int(args.seed_offset),
                                       actual_seed=int(seed), derived=(seed != base),
                                       default_time=round(float(t), 4),
                                       n_vars=int(n_var), n_cons=int(n_con),
                                       size={k: int(kw[k]) for k in spec["scale"]},
                                       kwargs=kw_serializable,
                                       randomized=bool(args.randomize),
                                       in_window=bool(accepted))
            made += 1
            mode = "rand" if args.randomize else "scaled"
            print(f"  {path.name}: {tag} [{mode}] | vars={n_var} cons={n_con} | "
                  f"size={ {k: kw[k] for k in spec['scale']} }", flush=True)
        report[fam] = dict(made=made, final_factor=round(factor, 3))

    if man_path:
        man_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(manifest, open(man_path, "w", encoding="utf-8"), indent=1)
        print(f"\nmanifest: {len(manifest)} entries -> {man_path}")

    print("\n" + "=" * 64)
    print("Done. To evaluate these instances, set in config/experiment.yaml:")
    print("  evaluation:")
    print(f"    families: [{', '.join(args.families)}]")
    print(f"    variants: ['{args.tag}']")
    print(f"    seed_offset: {args.start_index}")
    print(f"    n_test_instances: {args.n_instances}")
    print("then run:  python scripts/epm_with_selection.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
