"""
Combinatorial Auction Problem Generator (Winner Determination Problem)

A set of items is auctioned; bidders submit bids on bundles (subsets) of items.
Select a subset of bids to accept, maximizing total revenue, such that every
item is allocated to at most one accepted bid (set-packing structure).

    max  sum_b  price_b * x_b
    s.t. sum_{b : i in bundle_b} x_b <= 1   for every item i
         x_b in {0, 1}

This is a weighted set-packing problem: large LP relaxation gap, no strong
symmetry, and real branch-and-bound search space. Bundles are generated in the
CATS "arbitrary" spirit (items grouped by popularity/complementarity).
"""

from pyscipopt import Model
from .base import set_quiet, make_rng


def generate_instance(n_items=200, n_bids=600, seed=None):
    """n_items / n_bids are UPPER BOUNDS. Scale, item popularity, bundle sizes,
    price distribution and price-bundle correlation are heavily randomized
    per-seed to blur the family fingerprint."""
    rng = make_rng(seed)
    model = Model("combinatorial_auction")
    set_quiet(model)

    # Scale: uniform from a small floor up to the cap
    n_items = int(rng.randint(20, n_items + 1))
    n_bids = int(rng.randint(max(30, n_items), n_bids + 1))

    # Item popularity weights (skewed): some items appear in many bundles
    pop_alpha = rng.uniform(0.4, 1.5)
    popularity = rng.dirichlet(pop_alpha * (rng.rand(n_items) + 0.05))

    # Bundle size range (bundles are subsets of items)
    bundle_lo = 1 + int(rng.randint(0, 3))
    bundle_hi = max(bundle_lo + 1, int(rng.randint(bundle_lo + 1, max(bundle_lo + 2, n_items // 4 + 2))))

    # Price model: base value per item, plus a super/sub-additive bundle effect
    per_item_value_hi = int(rng.randint(5, 200))
    additivity = rng.uniform(-0.3, 0.6)  # >0 super-additive (complements), <0 sub-additive
    lognormal_prices = rng.random() < 0.5

    x = {}
    # item -> list of bids that include it
    item_to_bids = [[] for _ in range(n_items)]

    for b in range(n_bids):
        size = int(rng.randint(bundle_lo, bundle_hi + 1))
        size = min(size, n_items)
        # Sample items by popularity without replacement
        bundle = rng.choice(n_items, size=size, replace=False, p=popularity)
        bundle = [int(i) for i in bundle]

        # Base price: sum of per-item values (randomized distribution)
        if lognormal_prices:
            item_vals = rng.lognormal(mean=rng.uniform(0.5, 2.5), sigma=rng.uniform(0.3, 1.2), size=size) + 1.0
        else:
            item_vals = rng.randint(1, per_item_value_hi + 1, size=size).astype(float)
        base = float(item_vals.sum())
        # Bundle (dis)economies of scale
        price = base * (1.0 + additivity) * (size ** rng.uniform(0.0, 0.3))
        price = max(1, int(round(price)))

        x[b] = model.addVar(vtype="B", obj=price, name=f"bid_{b}")
        for i in bundle:
            item_to_bids[i].append(b)

    # Each item allocated to at most one accepted bid (set-packing constraint)
    for i in range(n_items):
        if item_to_bids[i]:
            model.addCons(
                sum(x[b] for b in item_to_bids[i]) <= 1,
                name=f"item_{i}",
            )

    model.setMaximize()
    return model
