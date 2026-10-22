"""
ISAC+ baseline: the per-cluster configurations are pooled into a portfolio and a
per-instance selector chooses among them, discarding ISAC's piecewise-constant
restriction (Ansotegui et al., AAAI 2014, in spirit).

This is the baseline that matches OUR granularity: like our pipeline it may give any
instance any portfolio member, chosen from static features in a single pass with no
solver calls. With granularity, feature set and model class all matched, the only
remaining difference from our method is HOW THE PORTFOLIO WAS BUILT -- by
configuration search (here SMAC, k clusters) or by random-search distillation.

Why the selector is a CLASSIFIER and not our EPM regressor: with only k distinct
configurations the 30-dim configuration encoding takes k values and carries no usable
signal, so the learning problem is not "regress log-speedup over configuration
space" but "given instance features, which of the k is fastest" -- k-class
classification. Hydra-MIP reports exactly this substitution as an improvement. The
classifier is cost-sensitive by default: each training instance is weighted by how
much choosing wrong actually costs there (the regret of the runner-up), so instances
where all members tie contribute almost nothing and instances with a real spread
dominate.

Stage 1 measures the k x N performance matrix on the TRAINING instances, which is
evidence ISAC itself never needs (it only measures each cluster's configuration
inside its own cluster). That extra offline cost is reported and must be counted
against this arm.

Stage 1 does NOT reuse the shared ROAR/SMAC solve cache, even though that cache
already holds 420 of the 960 cells. The label for an instance is the argmin across
its row of members, so every cell in a row has to be measured under the same machine
conditions. This machine has been observed drifting 1.1x-4x within a day; on cached
cells re-measured a day later the log-cost shift was +0.11 to +0.21, which is the same
order as the median runner-up regret (0.13-0.22) that the classifier actually learns
from. A row mixing yesterday's and today's cells can therefore have a flipped label.
Measuring all 960 cells in one session costs 78% more solves and removes the problem.

Usage:
    python scripts/isac_plus_baseline.py                    # k=8 from the SMAC model
    python scripts/isac_plus_baseline.py --k 4 --source-model data/results/isac_roar_model.json
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from al_param_tuning.solver import SCIPWrapper, compute_score
from al_param_tuning.features import InstanceFeatureExtractor
from isac_roar_baseline import Budget, Runner, load_instances, extract_features


def sgm(x, s=1.0):
    return float(np.exp(np.mean(np.log(np.asarray(x, dtype=float) + s))) - s)


def main():
    ap = argparse.ArgumentParser(
        description="ISAC+ : per-instance selector over the pooled cluster configurations")
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--source-model", default="data/results/isac_smac_model.json",
                    help="Trained ISAC model supplying the portfolio (its per-k configs)")
    ap.add_argument("--k", type=int, default=8,
                    help="Which k of the source model to pool into the portfolio")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--train-tag", default="ecole_train")
    ap.add_argument("--train-n", type=int, default=30)
    ap.add_argument("--test-tag", nargs="*", default=None)
    ap.add_argument("--test-n", type=int, default=None)
    ap.add_argument("--test-seed-start", type=int, default=None)
    ap.add_argument("--par", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cost-sensitive", type=lambda s: s.lower() not in ("0", "false", "no"),
                    default=True,
                    help="Weight each training instance by the regret of getting it "
                         "wrong. Off = plain multiclass")
    ap.add_argument("--matrix", default=None,
                    help="Where to cache the k x N training performance matrix. "
                         "Default: data/results/isac_plus_matrix_k<k>.json")
    ap.add_argument("--output", default=None,
                    help="Default: data/results/isac_plus_k<k>_results.json")
    ap.add_argument("--log", default="data/results/isac_plus_baseline.log",
                    help="Append progress here as well as to stdout; '' to disable")
    ap.add_argument("--solve-cache", default="data/results/isac_plus_solve_cache.json",
                    help="Cache for THIS arm only. Deliberately NOT the shared "
                         "ROAR/SMAC cache: the labels are an argmin across each row of "
                         "members, so all cells in a row must be measured under the "
                         "same machine conditions (see note in the module docstring)")
    args = ap.parse_args()

    log_fh = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.log, "a", encoding="utf-8")
        log_fh.write(f"\n{'=' * 72}\nrun start {datetime.now():%Y-%m-%d %H:%M:%S} | "
                     f"{' '.join(sys.argv[1:]) or '(no args)'}\n")
        log_fh.flush()

    def say(msg):
        print(msg, flush=True)
        if log_fh:
            log_fh.write(msg + "\n")
            log_fh.flush()

    if args.matrix is None:
        args.matrix = f"data/results/isac_plus_matrix_k{args.k}.json"
    if args.output is None:
        args.output = f"data/results/isac_plus_k{args.k}_results.json"
    solve_cache = (args.solve_cache or "").strip("\"'") or None

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    param_config = json.load(open(cfg["paths"]["param_config"], encoding="utf-8"))
    eval_cfg = cfg["evaluation"]
    time_limit = float(eval_cfg["time_limit"])
    cache_dir = project_root / "data" / "results" / "instances"

    families = args.families if args.families else list(eval_cfg["families"])
    test_tags = args.test_tag if args.test_tag else list(eval_cfg["variants"])
    test_n = args.test_n if args.test_n is not None else int(eval_cfg["n_test_instances"])
    test_start = (args.test_seed_start if args.test_seed_start is not None
                  else int(eval_cfg["seed_offset"]))

    # ── the portfolio ──
    src = json.load(open(args.source_model, encoding="utf-8"))
    if str(args.k) not in src.get("per_k", {}):
        sys.exit(f"ERROR: k={args.k} not in {args.source_model} "
                 f"(has {sorted(src.get('per_k', {}))})")
    pool = [src["per_k"][str(args.k)]["configs"][c]
            for c in sorted(src["per_k"][str(args.k)]["configs"], key=int)]
    K = len(pool)

    scip = SCIPWrapper(param_config)
    default_config = scip.get_default_config()
    fx = InstanceFeatureExtractor()

    train = load_instances(cache_dir, families, args.train_tag, range(1, args.train_n + 1))
    test = []
    for tag in test_tags:
        test += load_instances(cache_dir, families, tag, range(test_start, test_start + test_n))
    if not train:
        sys.exit(f"ERROR: no training instances '<family>_{args.train_tag}_NN.lp'")
    if not test:
        sys.exit(f"ERROR: no test instances for seeds "
                 f"{test_start}-{test_start + test_n - 1}")

    say("=" * 72)
    say(f"ISAC+ : {K}-member portfolio from {args.source_model} (k={args.k})")
    say("=" * 72)
    say(f"train: {len(train)} instances | test: {len(test)} instances, "
          f"variants={test_tags}, seeds {test_start}-{test_start + test_n - 1}")
    say(f"selector: XGBClassifier over {K} classes, "
          f"cost-sensitive={args.cost_sensitive}")

    Xtr = extract_features(train, fx)
    Xte = extract_features(test, fx)

    # ── Stage 1: the K x N training performance matrix ──
    budget = Budget(10 ** 9)
    runner = Runner(scip, time_limit, args.par, budget, cache_path=solve_cache)
    mat_path = Path(args.matrix)
    cost = {}
    if mat_path.exists():
        try:
            raw = json.load(open(mat_path, encoding="utf-8"))
        except Exception:
            raw = {}
        # A matrix built while a DIFFERENT solve cache was active silently carries
        # that cache's measurements forward, which is exactly how a previous run
        # re-imported an earlier day's timings. Rows must be internally
        # contemporaneous or the argmin labels are wrong, so refuse rather than resume.
        mismatch = None
        if raw:
            if raw.get("k") != args.k:
                mismatch = f"k={raw.get('k')} != {args.k}"
            elif abs(raw.get("time_limit", -1) - time_limit) > 1e-9:
                mismatch = f"time_limit={raw.get('time_limit')} != {time_limit}"
            elif str(raw.get("source_model")) != str(args.source_model):
                mismatch = f"source_model={raw.get('source_model')}"
            elif str(raw.get("solve_cache")) != str(solve_cache):
                mismatch = (f"built with solve_cache={raw.get('solve_cache')}, "
                            f"now {solve_cache}")
        if mismatch:
            sys.exit(f"ERROR: {mat_path} was built under different settings "
                     f"({mismatch}).\nResuming would mix measurement sessions inside "
                     f"single rows, and the label for each instance is the argmin "
                     f"across its row.\nDelete it and start clean:\n  del {mat_path}")
        if raw:
            cost = dict(raw.get("cost", {}))
            say(f"matrix: resumed {len(cost)} cells from {mat_path} "
                f"(same k / time_limit / source model / solve cache)")

    def flush_matrix():
        mat_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(k=args.k, time_limit=time_limit, par=args.par,
                       source_model=str(args.source_model),
                       solve_cache=str(solve_cache), cost=cost),
                  open(mat_path, "w", encoding="utf-8"))

    total = len(train) * K
    say(f"\nStage 1: measuring {total} (instance, member) cells "
        f"({len(cost)} cached in the matrix, plus whatever the shared solve cache has)")
    t0 = time.perf_counter()
    n_worked = 0
    for i, inst in enumerate(train):
        before = budget.used
        for j, theta in enumerate(pool):
            key = f"{Path(inst['lp']).name}::{j}"
            if key in cost:
                continue
            c, _ = runner.cost(inst["lp"], theta)
            cost[key] = float(c)
        new = budget.used - before
        if new:
            n_worked += 1
        el = time.perf_counter() - t0
        # ETA from instances that actually solved; cache-only ones cost nothing
        left = sum(1 for m in train[i + 1:]
                   if any(f"{Path(m['lp']).name}::{j}" not in cost for j in range(K)))
        eta = (el / max(n_worked, 1) * left / 60) if left else 0.0
        cell = [cost[f"{Path(inst['lp']).name}::{j}"] for j in range(K)]
        say(f"  [{i + 1}/{len(train)}] {Path(inst['lp']).name}: {new} new solves | "
            f"best/worst member cost {min(cell):.2f}/{max(cell):.2f} | "
            f"calls {budget.used} hits {runner.hits} | "
            f"elapsed {el / 60:.1f}min ETA {eta:.0f}min")
        if (i + 1) % 10 == 0 or i + 1 == len(train):
            flush_matrix()
    flush_matrix()
    runner.flush()
    say(f"Stage 1 done: {budget.used} NEW solver calls "
        f"(+{runner.hits} served from the shared cache)")

    # ── Stage 2: the selector ──
    C = np.array([[cost[f"{Path(inst['lp']).name}::{j}"] for j in range(K)]
                  for inst in train], dtype=float)
    y = C.argmin(1)
    srt = np.sort(C, axis=1)
    regret = srt[:, 1] - srt[:, 0]          # cost of picking the runner-up instead
    say(f"\nStage 2: label distribution {dict(Counter(y.tolist()))}")
    say(f"  regret of the runner-up: mean {regret.mean():.3f}, "
          f"median {np.median(regret):.3f} (log-cost units)")

    from xgboost import XGBClassifier
    present = sorted(set(y.tolist()))
    remap = {c: i for i, c in enumerate(present)}
    inv = {i: c for c, i in remap.items()}
    y_fit = np.array([remap[v] for v in y])
    w = None
    if args.cost_sensitive:
        # instances where every member ties teach nothing; scale so mean weight is 1
        w = regret / max(regret.mean(), 1e-12)
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                        reg_alpha=0.1, reg_lambda=1.0, random_state=args.seed,
                        objective="multi:softprob", num_class=len(present))
    clf.fit(Xtr, y_fit, sample_weight=w)
    tr_acc = float((np.array([inv[v] for v in clf.predict(Xtr)]) == y).mean())
    say(f"  classes used {present} | training accuracy {100 * tr_acc:.1f}% "
          f"(in-sample, diagnostic only)")

    # ── Stage 3: one-pass test ──
    pred = np.array([inv[v] for v in clf.predict(Xte)])
    say(f"\nStage 3: evaluating {len(test)} test instances "
          f"(features -> class -> solve; no solver calls to choose)")
    from pyscipopt import Model
    results = []
    for i, inst in enumerate(test):
        j = int(pred[i])

        def solve(c):
            m = Model()
            m.hideOutput()
            m.readProblem(inst["lp"])
            return scip.solve(m, c, time_limit=time_limit)

        res_rec = solve(pool[j])
        res_def = solve(default_config)
        score = compute_score(res_rec, res_def)
        speedup = res_def.time / max(res_rec.time, 1e-3)
        results.append(dict(
            family_id=inst["family_id"], variant=inst["variant"], seed=inst["seed"],
            features=inst["features"], chosen_member=j,
            predicted_config=pool[j],
            recommended_time=res_rec.time, recommended_nodes=res_rec.nodes,
            recommended_status=res_rec.status,
            default_time=res_def.time, default_nodes=res_def.nodes,
            default_status=res_def.status,
            score=score, speedup=speedup,
            winner=("RECOMMENDED" if score and score > 0
                    else ("DEFAULT" if score and score < 0 else "TIE")),
        ))
        say(f"  [{i + 1}/{len(test)}] {inst['family_id']}/{inst['seed']}: "
              f"member={j} def={res_def.time:.2f}s "
              f"rec={res_rec.time:.2f}s speedup={speedup:.2f}x "
            f"score={score:+.3f}")

    sc = np.array([r["score"] for r in results], float)
    sp = np.array([r["default_time"] / max(r["recommended_time"], 1e-3) for r in results])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(meta=dict(
        method=f"ISAC+ (k={args.k})", source_model=str(args.source_model),
        portfolio_size=K, cost_sensitive=bool(args.cost_sensitive),
        stage1_new_solver_calls=budget.used, stage1_cache_hits=runner.hits,
        source_offline_solver_calls=src.get("offline_solver_calls"),
        test_variants=list(test_tags), test_seed_start=test_start,
        test_n_per_family=test_n, families=list(families),
        par=args.par, time_limit=time_limit, portfolio=pool),
        results=results), open(out, "w", encoding="utf-8"), indent=2)

    say("\n" + "=" * 72)
    say(f"ISAC+ SUMMARY (k={args.k}, {K} members)")
    say("=" * 72)
    say(f"  Mean score          : {sc.mean():+.3f}")
    say(f"  Median score        : {np.median(sc):+.3f}")
    say(f"  Win rate vs default : {int((sc > 0).sum())}/{len(sc)} "
          f"({100 * (sc > 0).mean():.1f}%)")
    say(f"  SGM / med speedup   : {sgm(sp):.3f}x / {np.median(sp):.3f}x")
    say(f"  Offline cost        : {src.get('offline_solver_calls')} (search) "
          f"+ {budget.used} (this stage-1 matrix) = "
          f"{(src.get('offline_solver_calls') or 0) + budget.used}")
    say("\nPer-family:")
    by = defaultdict(list)
    for r in results:
        by[r["family_id"]].append(r)
    for fam in families:
        rs = by.get(fam, [])
        if not rs:
            continue
        s = np.array([r["score"] for r in rs], float)
        say(f"  {fam:<24}{len(rs):>4}{s.mean():>+10.3f}"
              f"{f'{int((s > 0).sum())}/{len(s)}':>8}")
    say(f"\nMember usage on the test set: "
          f"{dict(Counter(r['chosen_member'] for r in results))}")
    say(f"Saved to {out}")
    if log_fh:
        log_fh.close()


if __name__ == "__main__":
    main()
