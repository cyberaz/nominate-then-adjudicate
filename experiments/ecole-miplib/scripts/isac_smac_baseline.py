"""
ISAC + SMAC baseline: identical to scripts/isac_roar_baseline.py except that the
per-cluster configuration search is model-based SMAC instead of model-free ROAR.

Everything else --- feature extraction, k-means clustering, the k sweep and its
selection on held-out TRAINING instances, the PAR-penalized log-runtime cost, the
solve cache, the solver-call budget, and the one-pass test protocol --- is imported
from that module and therefore shared verbatim. The two arms differ only in the
search, which is what makes the comparison isolate search strength: SMAC adds a
random-forest surrogate and expected-improvement proposals but still yields ONE
configuration per cluster, so the piecewise-constant restriction is untouched.

SMAC is driven through its ask/tell interface rather than handed the target
function, so that its trials go through the shared Runner and are counted and cached
exactly like ROAR's. Configuration space: all 30 parameters are finite candidate
lists, hence Categorical; values are carried as strings and mapped back to their
native types before reaching the solver.

Usage:
    python scripts/isac_smac_baseline.py --fresh
    python scripts/isac_smac_baseline.py --load-model      # inference only
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# SMAC's dask-based parallelism is unused here (we drive it with ask/tell) and its
# worker startup is slow on Windows; keep everything in-process.
os.environ.setdefault("SMAC_DISABLE_DASK", "1")

import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))


def _patch_sklearn_for_smac():
    """SMAC 2.4 imports sklearn.tree._tree.DTYPE / DOUBLE, which scikit-learn
    removed in 1.9. Re-expose them with their historical values instead of pinning
    an older scikit-learn: this project's ISAC clustering uses KMeans from the same
    install, and changing that version would change the clusters and invalidate
    every trained checkpoint."""
    import sklearn.tree._tree as t
    if not hasattr(t, "DTYPE"):
        t.DTYPE = np.float32
    if not hasattr(t, "DOUBLE"):
        t.DOUBLE = np.float64


_patch_sklearn_for_smac()

import isac_roar_baseline as base

# SMAC 2.4 reduces instance features by PCA as soon as there are at least
# `pca_components` (default 7) of them, but `predict_marginalized` then asserts
# against the RAW feature width and dies with an AssertionError once the surrogate
# takes over from the initial design. Measured on this install: 4 features work,
# 7 / 8 / 20 crash.
#
# We therefore pass no instance features. This is not a workaround that weakens the
# baseline: the per-cluster objective is the aggregate PAR-penalized cost over that
# cluster's instances, a scalar function of the CONFIGURATION alone. Instance
# features would only help a surrogate predict per instance, which is precisely what
# a piecewise-constant per-cluster baseline does not do. SMAC still fits
# f(config) -> cost and proposes by expected improvement, so the contrast against
# ROAR's uniform proposals -- the thing this arm exists to test -- is intact.
# Set to a value in 1..6 to sensitivity-check with a truncated feature vector.
N_INSTANCE_FEATURES = 0


def _make_space(parameters, seed):
    """One Categorical per parameter over its candidate list.

    Choices are strings because a ConfigSpace Categorical requires a single type
    while our candidates mix int, bool and char. `back` maps each string to the
    native value the solver expects."""
    from ConfigSpace import ConfigurationSpace, Categorical

    space = ConfigurationSpace(seed=int(seed))
    back = {}
    for p in parameters:
        name = p["name"]
        choices, mapping = [], {}
        for cand in p["candidates"]:
            s = str(cand)
            if s not in mapping:            # keep list and map consistent
                mapping[s] = cand
                choices.append(s)
        back[name] = mapping
        space.add(Categorical(name, choices))
    return space, back


def smac_search(runner, instances, parameters, default_config, budget_cap, rng,
                verbose=False):
    """Model-based per-cluster search. Drop-in replacement for base.roar with the
    same signature, budget accounting and return value."""
    from smac import AlgorithmConfigurationFacade, Scenario
    from smac.runhistory.dataclasses import TrialValue

    lps = [i["lp"] for i in instances]
    start_used = runner.budget.used

    def spent():
        return runner.budget.used - start_used

    space, back = _make_space(parameters, rng.randint(1 << 30))
    out_dir = Path(tempfile.mkdtemp(prefix="smac_isac_"))
    try:
        extra = {}
        if N_INSTANCE_FEATURES:
            if N_INSTANCE_FEATURES >= 7:
                raise ValueError("N_INSTANCE_FEATURES must be <= 6; SMAC 2.4 crashes "
                                 "once PCA engages (see note at top of this file)")
            extra["instance_features"] = {
                i["lp"]: [float(v) for v in i["features"][:N_INSTANCE_FEATURES]]
                for i in instances}
        scenario = Scenario(
            configspace=space,
            output_directory=out_dir,
            deterministic=True,          # the solver seed is pinned in SCIPWrapper
            objectives="cost",
            n_trials=int(budget_cap),
            instances=lps,
            seed=int(rng.randint(1 << 30)),
            **extra,
        )

        def _unused(config, instance=None, seed=0, budget=None):
            # ask/tell is used instead; SMAC never calls this.
            raise AssertionError("target function must not be invoked under ask/tell")

        smac = AlgorithmConfigurationFacade(
            scenario, _unused,
            overwrite=True,
            logging_level=40,            # ERROR: keep the solver log readable
        )

        n_trials = 0
        while spent() < budget_cap and not runner.budget.roar_exhausted(spent()):
            info = smac.ask()
            cfg = {k: back[k][str(v)] for k, v in dict(info.config).items()}
            lp = info.instance if info.instance is not None else lps[0]
            cost, _ = runner.cost(lp, cfg)
            smac.tell(info, TrialValue(cost=float(cost)))
            n_trials += 1

        inc = smac.intensifier.get_incumbent()
        if inc is None:
            incumbent = dict(default_config)
        else:
            incumbent = {k: back[k][str(v)] for k, v in dict(inc).items()}
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    runner.budget.roar_used += spent()
    if verbose:
        print(f"      SMAC: {n_trials} trials, {spent()} solver calls "
              f"over {len(lps)} instances")
    return incumbent, dict(n_trials=n_trials, calls=spent())


base.SEARCHERS["smac"] = smac_search

if __name__ == "__main__":
    # Default this entry point to the SMAC arm; everything else is the shared driver.
    if not any(a == "--searcher" or a.startswith("--searcher=") for a in sys.argv[1:]):
        sys.argv += ["--searcher", "smac"]
    base.main()
