"""
ISAC + ROAR baseline, evaluated under the same one-pass protocol as
scripts/epm_with_selection.py so the two are directly comparable.

ARCHITECTURE (Kadioglu et al., ECAI 2010 shell; Hutter et al., LION 2011 kernel)
  Training (solver-in-the-loop, offline):
    1. extract the same 20 static features from every training instance;
    2. k-means cluster the (standardized) feature vectors;
    3. for each cluster run ROAR -- Random Online Aggressive Racing -- over the
       FULL 30-parameter space to obtain that cluster's configuration.
    4. sweep k, and pick k by held-out validation on TRAINING instances only
       (never on the test set).
  Inference (one-pass, ZERO solver calls):
    features -> nearest cluster centroid -> that cluster's configuration.

ROAR kernel (model-free ablation of SMAC):
    incumbent <- SCIP default
    while budget remains:
        theta <- uniform random configuration
        N <- 1
        loop:                                        # aggressive racing
            run theta on N instances the incumbent has already been run on
            compare the two on their COMMON instance subset
            if theta is worse            -> reject, break
            if theta covers all of the incumbent's instances and is better
                                         -> theta becomes the incumbent, break
            else                         -> N <- 2N and continue
Cost of a run is log(runtime), with a non-optimal run charged PAR x time_limit
(PAR10 by default), aggregated as the mean over the common instance subset --
i.e. a shifted geometric mean of runtime, consistent with the main pipeline.

Budget is counted in SOLVER CALLS and printed, so it can be matched against the
main pipeline's offline cost.

Usage (see --help for all knobs):
    # 1. training instances (disjoint seeds from the test set)
    python scripts/generate_ecole_instances.py --tag ecole_train -n 30
    # 2. train + test
    python scripts/isac_roar_baseline.py
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from al_param_tuning.solver import SCIPWrapper, compute_score
from al_param_tuning.features import InstanceFeatureExtractor

FEATURE_NAMES = InstanceFeatureExtractor.FEATURE_NAMES


# ─────────────────────────────── helpers ─────────────────────────────────────
def load_instances(cache_dir, families, tag, idx_range):
    """Collect (family, variant, seed, lp_path) for <family>_<tag>_<NN>.lp files."""
    out = []
    for fam in families:
        for i in idx_range:
            p = cache_dir / f"{fam}_{tag}_{i}.lp"
            if p.exists():
                out.append(dict(family_id=fam, variant=tag, seed=i, lp=str(p)))
    return out


def extract_features(instances, fx):
    from pyscipopt import Model
    feats = []
    for inst in instances:
        m = Model()
        m.hideOutput()
        m.readProblem(inst["lp"])
        f = np.asarray(fx.extract(m), dtype=float)
        m.freeProb()
        inst["features"] = f.tolist()
        feats.append(f)
    return np.vstack(feats)


def random_config(parameters, rng):
    return {p["name"]: p["candidates"][rng.randint(len(p["candidates"]))] for p in parameters}


class Budget:
    """Counts solver calls so the offline cost can be reported and matched.

    `total` caps only the ROAR search (`roar_used`); validation and test solves
    are counted in `used` for reporting but do not starve later k values."""

    def __init__(self, total):
        self.total = total
        self.used = 0
        self.roar_used = 0

    def spend(self, n=1):
        self.used += n

    def roar_exhausted(self, pending=0):
        return self.roar_used + pending >= self.total


class Runner:
    """Solves (instance, configuration) pairs and caches them, so a repeated
    comparison never costs a second solver call.

    The cache is also persisted to disk. Note its scope honestly: ROAR samples
    from a ~5^30 space, so two runs with DIFFERENT seeds essentially never draw
    the same configuration and the cache will not help. It pays off for
    (a) resuming an interrupted run with unchanged settings, and (b) the default
    configuration, whose runs are seed-independent and reused everywhere."""

    def __init__(self, scip, time_limit, par, budget, cache_path=None):
        self.scip = scip
        self.time_limit = time_limit
        self.par = par
        self.budget = budget
        self.cache = {}          # str key -> dict(cost, time, nodes, status, is_optimal)
        self.cache_path = Path(cache_path) if cache_path else None
        self.hits = 0
        self.n_since_flush = 0
        if self.cache_path and self.cache_path.exists():
            try:
                raw = json.load(open(self.cache_path, encoding="utf-8"))
                if (abs(raw.get("time_limit", -1) - time_limit) < 1e-9
                        and abs(raw.get("par", -1) - par) < 1e-9):
                    self.cache = raw.get("entries", {})
                    print(f"solve cache: loaded {len(self.cache)} entries from "
                          f"{self.cache_path}")
                else:
                    print("solve cache: time_limit/par changed -> cache discarded")
            except Exception:
                self.cache = {}

    @staticmethod
    def _key(lp, cfg):
        fp = "|".join(f"{k}={cfg[k]}" for k in sorted(cfg))
        return f"{Path(lp).name}::{fp}"

    def flush(self):
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(time_limit=self.time_limit, par=self.par, entries=self.cache),
                  open(self.cache_path, "w", encoding="utf-8"))
        self.n_since_flush = 0

    def cost(self, lp, cfg):
        """Return (cost, entry-dict); cost = log(runtime) with PAR penalty."""
        key = self._key(lp, cfg)
        if key in self.cache:
            self.hits += 1
            e = self.cache[key]
            return e["cost"], e
        from pyscipopt import Model
        m = Model()
        m.hideOutput()
        m.readProblem(lp)
        res = self.scip.solve(m, cfg, time_limit=self.time_limit)
        self.budget.spend()
        t = res.time if res.is_optimal else self.par * self.time_limit
        entry = dict(cost=float(np.log(max(t, 1e-3))), time=res.time, nodes=res.nodes,
                     status=res.status, is_optimal=bool(res.is_optimal))
        self.cache[key] = entry
        self.n_since_flush += 1
        if self.n_since_flush >= 25:
            self.flush()
        return entry["cost"], entry


# ────────────────────────────────── ROAR ─────────────────────────────────────
def roar(runner, instances, parameters, default_config, budget_cap, rng, verbose=False):
    """Random Online Aggressive Racing on one cluster.

    `budget_cap` is the number of solver calls this call may spend (checked
    against the shared Budget object as well)."""
    lps = [i["lp"] for i in instances]
    start_used = runner.budget.used

    def spent():
        return runner.budget.used - start_used

    incumbent = dict(default_config)
    inc_runs = {}                       # lp -> cost, for the incumbent

    # seed the incumbent with one run so comparisons have a basis
    first = lps[rng.randint(len(lps))]
    c, _ = runner.cost(first, incumbent)
    inc_runs[first] = c

    n_challengers = 0
    n_wins = 0
    while spent() < budget_cap and not runner.budget.roar_exhausted(spent()):
        theta = random_config(parameters, rng)
        n_challengers += 1
        th_runs = {}
        N = 1
        rejected = False
        while True:
            # instances the incumbent has been run on, that theta has not
            pool = [lp for lp in inc_runs if lp not in th_runs]
            if not pool:
                # extend the incumbent's evidence base onto a fresh instance
                fresh = [lp for lp in lps if lp not in inc_runs]
                if not fresh or spent() >= budget_cap or runner.budget.roar_exhausted(spent()):
                    break
                lp = fresh[rng.randint(len(fresh))]
                c, _ = runner.cost(lp, incumbent)
                inc_runs[lp] = c
                pool = [lp]
            take = pool[: max(1, N)]
            for lp in take:
                if spent() >= budget_cap or runner.budget.roar_exhausted(spent()):
                    rejected = True
                    break
                c, _ = runner.cost(lp, theta)
                th_runs[lp] = c
            if rejected:
                break
            common = set(th_runs) & set(inc_runs)
            th_mean = float(np.mean([th_runs[lp] for lp in common]))
            inc_mean = float(np.mean([inc_runs[lp] for lp in common]))
            if th_mean > inc_mean:                      # aggressive rejection
                rejected = True
                break
            if set(inc_runs).issubset(set(th_runs)) and th_mean < inc_mean:
                incumbent = theta                        # challenger takes over
                inc_runs = dict(th_runs)
                n_wins += 1
                break
            N *= 2
    runner.budget.roar_used += spent()
    if verbose:
        print(f"      ROAR: {n_challengers} challengers, {n_wins} incumbent changes, "
              f"{spent()} solver calls, evidence on {len(inc_runs)} instances")
    return incumbent, dict(n_challengers=n_challengers, n_wins=n_wins, calls=spent())


# Per-cluster searchers. A searcher takes
# (runner, instances, parameters, default_config, budget_cap, rng, verbose)
# and returns (best_config, info_dict). Registering a new one is the ONLY
# difference between the ROAR and SMAC baselines; every other part of the
# protocol below is shared, so the comparison isolates the search.
SEARCHERS = {"roar": roar}


# ───────────────────────────────── driver ────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="ISAC + ROAR baseline under a one-pass protocol")
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--searcher", default="roar", choices=sorted(SEARCHERS),
                    help="Per-cluster configuration search. Everything else in the "
                         "protocol is identical, so the two arms differ only here")
    ap.add_argument("--families", nargs="*", default=None,
                    help="Default: evaluation.families from --config")
    ap.add_argument("--train-tag", default="ecole_train")
    ap.add_argument("--train-n", type=int, default=30, help="Training instances per family")
    ap.add_argument("--test-tag", nargs="*", default=None,
                    help="Default: evaluation.variants from --config")
    ap.add_argument("--test-n", type=int, default=None,
                    help="Test instances per family. Default: evaluation.n_test_instances")
    ap.add_argument("--test-seed-start", type=int, default=None,
                    help="First test seed. Default: evaluation.seed_offset")
    ap.add_argument("--k-sweep", nargs="*", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--total-budget", type=int, default=3000,
                    help="Total ROAR solver calls, split evenly over all (k, cluster) pairs")
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="Fraction of training instances held out to choose k")
    ap.add_argument("--par", type=float, default=10.0, help="Penalty factor for non-optimal runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None,
                    help="Default: data/results/isac_<searcher>_results.json")
    ap.add_argument("--model", default=None,
                    help="Trained model (centroids, standardization, per-k configs). "
                         "Written after every k; reused by --load-model. "
                         "Default: data/results/isac_<searcher>_model.json")
    ap.add_argument("--load-model", action="store_true",
                    help="Skip training entirely: load --model and only run inference "
                         "+ evaluation on the test set (zero solver calls for training)")
    ap.add_argument("--solve-cache", default=None,
                    help="Persistent (instance, config) -> runtime cache; '' to disable. "
                         "Default: data/results/isac_<searcher>_solve_cache.json")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore any existing model checkpoint and retrain all k")
    args = ap.parse_args()

    # Searcher-specific artifacts: the arms must never clobber each other's model
    tag = args.searcher
    if args.output is None:
        args.output = f"data/results/isac_{tag}_results.json"
    if args.model is None:
        args.model = f"data/results/isac_{tag}_model.json"
    if args.solve_cache is None:
        args.solve_cache = f"data/results/isac_{tag}_solve_cache.json"
    search_fn = SEARCHERS[args.searcher]

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    param_config = json.load(open(cfg["paths"]["param_config"], encoding="utf-8"))
    parameters = param_config["parameters"]
    eval_cfg = cfg["evaluation"]
    time_limit = float(eval_cfg["time_limit"])
    cache_dir = project_root / "data" / "results" / "instances"

    # Test set comes from config['evaluation'], like epm_no_llm.py / epm_with_selection.py
    families = args.families if args.families else list(eval_cfg["families"])
    test_tags = args.test_tag if args.test_tag else list(eval_cfg["variants"])
    test_n = args.test_n if args.test_n is not None else int(eval_cfg["n_test_instances"])
    test_start = (args.test_seed_start if args.test_seed_start is not None
                  else int(eval_cfg["seed_offset"]))
    test_seeds = range(test_start, test_start + test_n)

    scip = SCIPWrapper(param_config)
    default_config = scip.get_default_config()
    fx = InstanceFeatureExtractor()
    rng = np.random.RandomState(args.seed)

    # Shells disagree on how to pass an empty string (PowerShell needs '""' and
    # swallows a bare ""), so a literal quotes-only path means "disabled" as well;
    # otherwise it is an invalid filename that only fails in flush(), at the end.
    solve_cache = (args.solve_cache or "").strip("\"'") or None

    # ── data ──
    train = load_instances(cache_dir, families, args.train_tag, range(1, args.train_n + 1))
    test = []
    for tag in test_tags:
        test += load_instances(cache_dir, families, tag, test_seeds)
    if not train and not args.load_model:
        sys.exit(f"ERROR: no training instances '<family>_{args.train_tag}_NN.lp' found. "
                 f"Generate them first:\n"
                 f"  python scripts/generate_ecole_instances.py --tag {args.train_tag} "
                 f"-n {args.train_n}")
    if not test:
        sys.exit(f"ERROR: no test instances found for families={families}, "
                 f"variants={test_tags}, seeds {test_start}-{test_start + test_n - 1}.\n"
                 f"Generate them first:\n"
                 f"  python scripts/generate_ecole_instances.py --tag {test_tags[0]} "
                 f"--start-index {test_start} -n {test_n}")

    print("=" * 72)
    print(f"ISAC + {args.searcher.upper()} baseline")
    print("=" * 72)
    print(f"train: {len(train)} instances ({dict(Counter(i['family_id'] for i in train))})")
    # auto_iterate advances evaluation.seed_offset every round, so state the window
    print(f"test : {len(test)} instances, variants={test_tags}, "
          f"seeds {test_start}-{test_start + test_n - 1} "
          f"({dict(Counter(i['family_id'] for i in test))})")

    Xte = extract_features(test, fx)
    if train:
        Xtr = extract_features(train, fx)
        # standardize on the training set (ISAC normalizes features before clustering)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-12] = 1.0
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

        # train / validation split, used ONLY to choose k
        perm = rng.permutation(len(train))
        n_val = max(1, int(round(args.val_frac * len(train))))
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
        print(f"cluster-fitting instances: {len(fit_idx)} | "
              f"validation (for choosing k): {n_val}")
    else:
        mu = sd = Ztr = Zte = None
        val_idx = fit_idx = np.array([], dtype=int)

    budget = Budget(args.total_budget)
    runner = Runner(scip, time_limit, args.par, budget,
                    cache_path=solve_cache)
    n_cluster_runs = sum(args.k_sweep)
    per_cluster = max(20, args.total_budget // max(1, n_cluster_runs))

    model_path = Path(args.model)
    saved = {}
    if model_path.exists() and not args.fresh:
        try:
            saved = json.load(open(model_path, encoding="utf-8"))
        except Exception:
            saved = {}
    # a checkpoint is only reusable if the training setup is identical
    # Deliberately excludes the test window: the model is a function of the
    # TRAINING data only, so it stays reusable across test batches via --load-model.
    # families is sorted because it is a set - its order does not affect the model.
    setup = dict(families=sorted(families), train_tag=args.train_tag,
                 train_n=args.train_n, val_frac=args.val_frac, par=args.par,
                 time_limit=time_limit, seed=args.seed, searcher=args.searcher)

    def norm(s):
        if not isinstance(s, dict):
            return None
        s = dict(s)
        s["families"] = sorted(s.get("families") or [])
        # checkpoints written before --searcher existed were all roar
        s.setdefault("searcher", "roar")
        return s

    if norm(saved.get("setup")) != setup:
        if saved:
            print("model checkpoint: training setup changed -> starting fresh")
        saved = {}
    trained = {int(k): v for k, v in saved.get("per_k", {}).items()} if saved else {}

    def save_model(per_k):
        json.dump(dict(setup=setup, mu=mu.tolist(), sd=sd.tolist(),
                       feature_names=list(FEATURE_NAMES),
                       offline_solver_calls=budget.used,
                       per_k={str(k): v for k, v in per_k.items()}),
                  open(model_path, "w", encoding="utf-8"), indent=1)

    from sklearn.cluster import KMeans

    def fit_k(k):
        """Return dict(centroids, configs, val_score) for this k, training if needed."""
        if k in trained:
            b = trained[k].get("per_cluster_budget")
            note = "" if b == per_cluster else f", trained with budget {b} != {per_cluster}"
            print(f"\n--- k = {k} --- (loaded from checkpoint, "
                  f"val_score={trained[k]['val_score']:+.4f}{note})")
            return trained[k]
        print(f"\n--- k = {k} ---", flush=True)
        km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(Ztr[fit_idx])
        labels = km.labels_
        configs = {}
        for c in range(k):
            members = [train[fit_idx[j]] for j in range(len(fit_idx)) if labels[j] == c]
            if not members:
                configs[c] = dict(default_config)
                continue
            print(f"  cluster {c}: {len(members)} instances "
                  f"({dict(Counter(m['family_id'] for m in members))})", flush=True)
            theta, _ = search_fn(runner, members, parameters, default_config,
                                 per_cluster, rng, verbose=True)
            configs[c] = theta
        val_labels = km.predict(Ztr[val_idx])
        gains = []
        for j, vi in enumerate(val_idx):
            inst = train[vi]
            c_cfg, _ = runner.cost(inst["lp"], configs[int(val_labels[j])])
            c_def, _ = runner.cost(inst["lp"], default_config)
            gains.append(c_def - c_cfg)          # positive = faster than default
        entry = dict(centroids=km.cluster_centers_.tolist(),
                     configs={str(c): configs[c] for c in configs},
                     val_score=float(np.mean(gains)),
                     per_cluster_budget=per_cluster)
        print(f"  k={k}: validation mean log-speedup vs default = {entry['val_score']:+.4f}")
        trained[k] = entry
        save_model(trained)                       # checkpoint after every k
        runner.flush()
        return entry

    if args.load_model:
        if not trained:
            sys.exit(f"ERROR: --load-model given but no reusable checkpoint in {model_path}")
        # standardization MUST come from the model, or centroids are meaningless
        mu = np.asarray(saved["mu"], float)
        sd = np.asarray(saved["sd"], float)
        Zte = (Xte - mu) / sd
        print(f"\n--load-model: skipping training, reusing {len(trained)} trained k values "
              f"({sorted(trained)}); standardization taken from the model")
        sweep = trained
    else:
        print(f"ROAR budget: {args.total_budget} solver calls over {n_cluster_runs} "
              f"(k, cluster) pairs -> {per_cluster} calls each")
        if trained:
            print(f"model checkpoint: reusing k = {sorted(trained)}")
        sweep = {k: fit_k(k) for k in args.k_sweep}

    best_k = max(sweep, key=lambda k: sweep[k]["val_score"])
    centroids = np.asarray(sweep[best_k]["centroids"], float)
    configs = {int(c): cfg_ for c, cfg_ in sweep[best_k]["configs"].items()}
    print(f"\nSelected k = {best_k} (by held-out training validation, not test)")
    print(f"Solver calls used offline this run: {budget.used} "
          f"(of which ROAR search: {budget.roar_used}/{args.total_budget}; "
          f"cache hits: {runner.hits})")
    runner.flush()

    # ── test: one-pass inference, then solve chosen config vs default ──
    print(f"\nEvaluating {len(test)} test instances (one-pass: features -> centroid -> config)")
    # nearest centroid, exactly what KMeans.predict does
    assign = np.argmin(((Zte[:, None, :] - centroids[None, :, :]) ** 2).sum(-1), axis=1)
    param_names = [p["name"] for p in parameters]
    results = []
    for i, inst in enumerate(test):
        cid = int(assign[i])
        theta = configs[cid]
        from pyscipopt import Model

        def solve(c):
            m = Model()
            m.hideOutput()
            m.readProblem(inst["lp"])
            return scip.solve(m, c, time_limit=time_limit)

        res_rec = solve(theta)
        res_def = solve(default_config)
        score = compute_score(res_rec, res_def)
        speedup = res_def.time / max(res_rec.time, 1e-3)
        winner = ("RECOMMENDED" if score and score > 0
                  else ("DEFAULT" if score and score < 0 else "TIE"))
        n_changed = sum(1 for p in param_names if theta.get(p) != default_config.get(p))
        results.append(dict(
            family_id=inst["family_id"], variant=inst["variant"], seed=inst["seed"],
            features=inst["features"], cluster_id=cid,
            predicted_config=theta, n_params_changed=n_changed,
            recommended_time=res_rec.time, recommended_nodes=res_rec.nodes,
            recommended_status=res_rec.status,
            recommended_objective=res_rec.objective if res_rec.has_solution else None,
            recommended_gap=res_rec.gap if res_rec.has_solution else None,
            default_time=res_def.time, default_nodes=res_def.nodes,
            default_status=res_def.status,
            default_objective=res_def.objective if res_def.has_solution else None,
            default_gap=res_def.gap if res_def.has_solution else None,
            score=score, speedup=speedup, winner=winner,
        ))
        print(f"  [{len(results)}/{len(test)}] {inst['family_id']}/{inst['seed']}: "
              f"cluster={cid} def={res_def.time:.2f}s rec={res_rec.time:.2f}s "
              f"speedup={speedup:.2f}x score={score:+.3f} {winner}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(
        meta=dict(method=f"ISAC+{args.searcher.upper()}", searcher=args.searcher,
                  best_k=best_k, k_sweep=args.k_sweep,
                  offline_solver_calls=budget.used, total_budget=args.total_budget,
                  train_instances=len(train), test_instances=len(test),
                  test_variants=list(test_tags), test_seed_start=test_start,
                  test_n_per_family=test_n, families=list(families),
                  par=args.par, time_limit=time_limit,
                  val_scores={str(k): sweep[k]["val_score"] for k in sweep},
                  cluster_configs={str(c): configs[c] for c in configs}),
        results=results,
    ), open(out, "w", encoding="utf-8"), indent=2)

    # ── summary, in the same shape as epm_with_selection.py ──
    sc = np.array([r["score"] for r in results if r["score"] is not None], float)
    sp = np.array([r["default_time"] / max(r["recommended_time"], 1e-3) for r in results])
    rec_t = np.array([r["recommended_time"] for r in results])
    def_t = np.array([r["default_time"] for r in results])

    def sgm(x, s=1.0):
        return float(np.exp(np.mean(np.log(np.asarray(x) + s))) - s)

    print("\n" + "=" * 72)
    print(f"ISAC + {args.searcher.upper()} SUMMARY (comparable to epm_with_selection.py)")
    print("=" * 72)
    print(f"Instances: {len(results)} | selected k = {best_k} | "
          f"offline solver calls = {budget.used}")
    print(f"  Mean score        : {sc.mean():+.3f}")
    print(f"  Median score      : {np.median(sc):+.3f}")
    print(f"  Win rate          : {int((sc > 0).sum())}/{len(sc)} "
          f"({100 * (sc > 0).mean():.1f}%)")
    print(f"  Speedup med/GM/SGM: {np.median(sp):.3f}x / "
          f"{float(np.exp(np.mean(np.log(sp)))):.3f}x / {sgm(sp):.3f}x")
    print(f"  Time mean rec/def : {rec_t.mean():.3f}s / {def_t.mean():.3f}s")
    print(f"  SGM-time ratio    : {sgm(def_t) / max(sgm(rec_t), 1e-9):.3f}x")
    n_opt = sum(1 for r in results if r["recommended_status"] == "optimal")
    print(f"  Recommended optimal: {n_opt}/{len(results)}")

    print("\nPer-family:")
    print(f"  {'family':<24}{'n':>4}{'mean score':>12}{'win':>8}{'med speedup':>13}")
    by = defaultdict(list)
    for r in results:
        by[r["family_id"]].append(r)
    for fam in families:
        rs = by.get(fam, [])
        if not rs:
            continue
        s = np.array([r["score"] for r in rs], float)
        u = np.array([r["default_time"] / max(r["recommended_time"], 1e-3) for r in rs])
        print(f"  {fam:<24}{len(rs):>4}{s.mean():>12.3f}"
              f"{f'{int((s > 0).sum())}/{len(s)}':>8}{np.median(u):>12.2f}x")

    print("\nCluster usage on the test set:", dict(Counter(r["cluster_id"] for r in results)))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
