"""
EPM ablation: NO LLM direction (50-dim input).

Identical to scripts/epm.py except the input drops the 30-dim LLM direction:
        X = [20 instance features | 30 encoded config values] = 50-dim
        y = log(default_time / config_time)

Candidates: the FIXED pool of 100 configs from config_100.json (ids 0..99 =
position in that file). All eval instances score the same pool; per-instance
top-K ids are logged and directly comparable.

Eval instances: 16 families x small x seeds 1024-1033 (LP files pre-generated
in data/results/instances/); the 20-dim features are computed on the fly.

K comes from --top-k and is written into the default output name
(data/results/epm_no_llm_top{K}_config{pool_size}_seed{seed_start}_results.json),
so runs with different K never overwrite each other.

Because the EPM is deterministic, a large-K run is a SUPERSET of every smaller one:
ranks 1..k of a --top-k 20 file are exactly what --top-k k would have nominated.
scripts/summarize_selection_skill.py --top-k k therefore reuses one big run for any
k <= K without re-solving. Cost scales with K: 16 x 10 instances x (K + 1) solves.

Usage:
    python scripts/epm_no_llm.py
    python scripts/epm_no_llm.py --top-k 20     # superset run, reusable for any k <= 20
    python scripts/epm_no_llm.py --n-eval 0     # train + diagnostic only
"""

import os

# Pin BLAS/OpenMP to a single thread for reproducible wall-clock timing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import sys
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from al_param_tuning.solver import SCIPWrapper, compute_score

N_REPEATS = 1


# ── Config encoding ────────────────────────────────────────────────────
def make_encoder(parameters):
    """Encode a config dict into a 30-dim float vector.
    int/real -> raw value; bool -> 0/1; char -> index in candidate list."""
    def encode(cfg):
        v = []
        for p in parameters:
            val = cfg[p['name']]
            t = p['type']
            if t == 'char':
                v.append(float(p['candidates'].index(val)))
            elif t == 'bool':
                b = val if isinstance(val, bool) else str(val).lower() in ('true', '1')
                v.append(1.0 if b else 0.0)
            else:
                v.append(float(val))
        return v
    return encode


def main():
    parser = argparse.ArgumentParser(description="EPM no-LLM: f(features, config) -> log(speedup)")
    parser.add_argument("--config", type=str, default="config/experiment.yaml")
    parser.add_argument("--cache", type=str, default="data/results/training_cache_multi.json")
    parser.add_argument("--config-pool", type=str, default="config_100.json",
                        help="Fixed candidate pool JSON (ids 0..N-1 = position in file)")
    parser.add_argument("--output", type=str, default=None,
                        help="Results JSON. Default: data/results/epm_no_llm_top{k}_"
                             "config{pool_size}_seed{seed_start}_results.json")
    parser.add_argument("--seed-start", type=int, default=1024)
    parser.add_argument("--seed-end", type=int, default=1033, help="Inclusive")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Solve the top-K EPM-ranked candidates; report the ACTUAL fastest "
                             "(oracle) as recommended. A larger K is a superset: any k <= K can "
                             "be derived from the same file without re-solving")
    parser.add_argument("--n-eval", type=int, default=None, help="Evaluate only first N instances (default all; 0 = train only)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.top_k < 1:
        sys.exit(f"ERROR: --top-k must be >= 1 (got {args.top_k}).")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    param_config_path = config['paths']['param_config']
    with open(param_config_path, 'r', encoding='utf-8') as f:
        param_config = json.load(f)
    parameters = param_config['parameters']
    encode = make_encoder(parameters)
    param_names = [p['name'] for p in parameters]

    # =================================================================
    # Step 1: Build training rows from the cache (NO direction)
    # =================================================================
    print("=" * 70)
    print(f"EPM no-LLM: train f(20 feat + 30 config) -> log(speedup) | Top-{args.top_k}")
    print("=" * 70)
    cache = json.load(open(args.cache, 'r'))
    X_all = cache['X_all']            # 400 x 50 (only first 20 used here)
    raw = cache['raw_data_all']
    n_inst = len(X_all)

    X_rows, y_rows, groups = [], [], []
    for i in range(n_inst):
        rec = raw[i]
        dt = rec.get('default_time', 0)
        if dt <= 0:
            continue
        feats = np.array(X_all[i][:20], dtype=float)
        for cfg in rec.get('configs', []):
            t = cfg.get('time', 0)
            if t <= 0:
                continue
            X_rows.append(np.concatenate([feats, encode(cfg['values'])]))
            y_rows.append(np.log(dt / t))
            groups.append(i)
    X_rows = np.array(X_rows)
    y_rows = np.array(y_rows)
    groups = np.array(groups)
    print(f"Training rows: {X_rows.shape} from {n_inst} instances "
          f"(pos {int((y_rows > 0).sum())} / neg {int((y_rows <= 0).sum())})")

    from xgboost import XGBRegressor
    xgb_params = dict(n_estimators=400, max_depth=7, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                      reg_alpha=0.1, reg_lambda=1.0, random_state=42)

    # ── Holdout ranking diagnostic (grouped by instance, same seed as epm.py) ──
    rng = np.random.RandomState(0)
    inst_ids = np.unique(groups)
    test_ids = rng.choice(inst_ids, size=max(1, int(0.2 * len(inst_ids))), replace=False)
    mtest = np.isin(groups, test_ids)
    diag = XGBRegressor(**xgb_params)
    diag.fit(X_rows[~mtest], y_rows[~mtest])
    pred = diag.predict(X_rows[mtest])
    from scipy.stats import spearmanr
    sps, regrets, top1_beats_def = [], [], []
    for gid in test_ids:
        m = groups[mtest] == gid
        if m.sum() < 3:
            continue
        yt, yp = y_rows[mtest][m], pred[m]
        sp = spearmanr(yt, yp).statistic
        if not np.isnan(sp):
            sps.append(sp)
        pick = int(np.argmax(yp))
        regrets.append(float(yt.max() - yt[pick]))
        top1_beats_def.append(bool(yt[pick] > 0))
    print(f"[Holdout diagnostic] {len(test_ids)} held-out instances: "
          f"mean Spearman={np.mean(sps):.3f}, mean top-1 regret={np.mean(regrets):.3f} (log pts), "
          f"picked-config beats default: {int(np.sum(top1_beats_def))}/{len(top1_beats_def)} "
          f"({100*np.mean(top1_beats_def):.0f}%)")

    # ── Retrain on ALL rows for the real evaluation ──
    model = XGBRegressor(**xgb_params)
    model.fit(X_rows, y_rows)
    print("EPM (no-LLM) trained on full data.")

    # =================================================================
    # Step 2: Evaluate on the 160 eval instances
    # =================================================================
    families = config['evaluation']['families']
    seeds = list(range(args.seed_start, args.seed_end + 1))
    eval_data = [{'family_id': fid, 'variant': 'small', 'seed': s}
                 for fid in families for s in seeds]
    if args.n_eval is not None:
        eval_data = eval_data[:args.n_eval]

    # FIXED candidate pool from config_100.json; ids 0..N-1 = position in file
    pool = json.load(open(args.config_pool, 'r', encoding='utf-8'))
    cands = [entry['config'] for entry in pool]
    cand_vecs = np.array([encode(c) for c in cands])
    if args.top_k > len(cands):
        sys.exit(f"ERROR: --top-k {args.top_k} exceeds the {len(cands)}-config pool "
                 f"in {args.config_pool}.")
    if args.output is None:
        args.output = (f"data/results/epm_no_llm_top{args.top_k}_config{len(cands)}_"
                       f"seed{args.seed_start}_results.json")
    time_limit = config['evaluation']['time_limit']
    print(f"\nEvaluating {len(eval_data)} instances "
          f"(seeds {args.seed_start}-{args.seed_end}, {len(cands)} fixed candidates from "
          f"{args.config_pool}, ids 0..{len(cands)-1}, {N_REPEATS}x serial solves)...")
    print(f"Top-{args.top_k} solved per instance -> "
          f"{len(eval_data)} x ({args.top_k} + 1 default) = "
          f"{len(eval_data) * (args.top_k + 1)} solves, "
          f"<= {time_limit}s each; output -> {args.output}")

    from al_param_tuning.features import InstanceFeatureExtractor
    feat_extractor = InstanceFeatureExtractor()

    scip = SCIPWrapper(param_config)
    default_config = scip.get_default_config()
    cache_dir = Path('data/results/instances')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    done = set()
    if output_path.exists() and not args.no_resume:
        try:
            results = json.load(open(output_path, 'r', encoding='utf-8'))
            done = {(r['family_id'], r['variant'], r['seed']) for r in results}
            print(f"Resuming: {len(done)} instances already done.")
        except Exception:
            results = []
        found = sorted({len(r.get('topk_detail', [])) for r in results})
        if found and found != [args.top_k]:
            sys.exit(f"ERROR: {output_path} already holds records with {found} candidates per "
                     f"instance, not {args.top_k}. Pick a different --output, or --no-resume "
                     f"to overwrite.")

    from pyscipopt import Model
    from dataclasses import replace as _dc_replace

    def solve_repeated(lp_file, cfg):
        times, nodes = [], []
        last = None
        for _ in range(N_REPEATS):
            m = Model()
            m.readProblem(lp_file)
            m.hideOutput()
            r = scip.solve(m, cfg, time_limit=time_limit)
            times.append(r.time)
            nodes.append(r.nodes)
            last = r
        return _dc_replace(last, time=float(np.mean(times)), nodes=int(np.mean(nodes))), times

    for idx, e in enumerate(eval_data):
        key = (e['family_id'], e['variant'], e['seed'])
        if key in done:
            continue
        lp_file = cache_dir / f"{e['family_id']}_{e['variant']}_{e['seed']}.lp"
        if not lp_file.exists():
            print(f"  WARNING: missing {lp_file}, skip")
            continue

        # Compute the 20-dim features from the LP file
        fm = Model()
        fm.hideOutput()
        fm.readProblem(str(lp_file))
        feats = np.array(feat_extractor.extract(fm), dtype=float)
        fm.freeProb()

        # Score the FIXED candidate pool for this instance
        Xc = np.hstack([np.tile(feats, (len(cands), 1)), cand_vecs])
        preds = model.predict(Xc)
        order = np.argsort(-preds)[:args.top_k]

        # Solve ALL top-K; oracle = the actually fastest one
        solved = []
        for rank, j in enumerate(order):
            res, raw = solve_repeated(str(lp_file), cands[j])
            solved.append({'rank': rank + 1, 'cand_idx': int(j),
                           'pred_log_speedup': float(preds[j]),
                           'result': res, 'times_raw': raw, 'config': cands[j]})
        best = min(solved, key=lambda s: s['result'].time)
        chosen = best['config']
        pred_ls = best['pred_log_speedup']
        oracle_rank = best['rank']
        result_rec, rec_raw = best['result'], best['times_raw']
        top1 = solved[0]

        result_def, def_raw = solve_repeated(str(lp_file), default_config)
        score = compute_score(result_rec, result_def)
        top1_score = compute_score(top1['result'], result_def)
        speedup = result_def.time / max(result_rec.time, 0.001) if result_rec.time > 0 else 0
        winner = "RECOMMENDED" if score and score > 0 else ("DEFAULT" if score and score < 0 else "TIE")
        n_changed = sum(1 for p in param_names if chosen.get(p) != default_config.get(p))

        results.append({
            'family_id': e['family_id'], 'variant': e['variant'], 'seed': e['seed'],
            'features': feats.tolist(),
            'top_k': len(solved),
            'predicted_config': chosen, 'predicted_log_speedup': pred_ls,
            'n_params_changed': n_changed,
            'oracle_rank': oracle_rank,
            'topk_cand_ids': [int(j) for j in order],
            'recommended_cand_id': int(best['cand_idx']),
            'topk_detail': [{'rank': s['rank'], 'cand_id': s['cand_idx'],
                             'pred': s['pred_log_speedup'],
                             'time': s['result'].time, 'nodes': s['result'].nodes,
                             'status': s['result'].status} for s in solved],
            'top1_time': top1['result'].time, 'top1_nodes': top1['result'].nodes,
            'top1_status': top1['result'].status, 'top1_score': top1_score,
            'recommended_time': result_rec.time, 'recommended_times_raw': rec_raw,
            'recommended_nodes': result_rec.nodes, 'recommended_status': result_rec.status,
            'recommended_objective': result_rec.objective if result_rec.has_solution else None,
            'recommended_gap': result_rec.gap if result_rec.has_solution else None,
            'default_time': result_def.time, 'default_times_raw': def_raw,
            'default_nodes': result_def.nodes, 'default_status': result_def.status,
            'default_objective': result_def.objective if result_def.has_solution else None,
            'default_gap': result_def.gap if result_def.has_solution else None,
            'score': score, 'speedup': speedup, 'winner': winner,
        })
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        print(f"[{len(results)}/{len(eval_data)}] {e['family_id']}/{e['seed']}: "
              f"top{args.top_k}_ids={[int(j) for j in order]} chosen_id={int(best['cand_idx'])} "
              f"oracle_rank={oracle_rank}/{args.top_k} "
              f"speedup={speedup:.2f}x score={score:+.3f} (top1 {top1_score:+.3f}) {winner}", flush=True)

    # =================================================================
    # Step 3: Summary (same metrics as evaluate.py)
    # =================================================================
    scores = [r['score'] for r in results if r['score'] is not None]
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY (EPM no-LLM: 20 feat + 30 config -> log speedup, Top-{args.top_k})")
    print("=" * 80)
    print(f"Instances evaluated: {len(results)}")
    print(f"Instances with valid scores: {len(scores)}")

    if scores:
        print(f"\nOverall Metrics:")
        print(f"  Mean score:   {np.mean(scores):.3f}")
        print(f"  Median score: {np.median(scores):.3f}")
        print(f"  Std score:    {np.std(scores):.3f}")
        n_rec = sum(1 for s in scores if s > 0)
        n_def = sum(1 for s in scores if s < 0)
        n_tie = sum(1 for s in scores if s == 0)
        print(f"  Recommended better: {n_rec}/{len(scores)} ({100*n_rec/len(scores):.1f}%)")
        print(f"  Default better:     {n_def}/{len(scores)} ({100*n_def/len(scores):.1f}%)")
        print(f"  Tie:                {n_tie}/{len(scores)}")

        # ── Oracle ceiling analysis (top-1 vs oracle within top-K) ──
        if any('oracle_rank' in r for r in results):
            from collections import Counter
            rank_dist = Counter(r['oracle_rank'] for r in results if 'oracle_rank' in r)
            print(f"\nOracle Ceiling Analysis (LLM-selector upper bound):")
            print(f"  Oracle rank distribution: " +
                  ", ".join(f"rank{k}={rank_dist[k]}" for k in sorted(rank_dist)))
            t1 = [r['top1_score'] for r in results if r.get('top1_score') is not None]
            print(f"  Top-1 (pure EPM):  mean score={np.mean(t1):.3f}, "
                  f"win rate={100*np.mean([s > 0 for s in t1]):.1f}%")
            print(f"  Oracle (best of {args.top_k}): mean score={np.mean(scores):.3f}, "
                  f"win rate={100*n_rec/len(scores):.1f}%")
            print(f"  -> Max selector gain: score +{np.mean(scores)-np.mean(t1):.3f}, "
                  f"win +{100*(n_rec/len(scores)-np.mean([s > 0 for s in t1])):.1f}pp")

        # ── Shared-pool candidate ID usage ──
        if any('topk_cand_ids' in r for r in results):
            from collections import Counter
            topk_ct = Counter(cid for r in results for cid in r.get('topk_cand_ids', []))
            chosen_ct = Counter(r['recommended_cand_id'] for r in results if 'recommended_cand_id' in r)
            print(f"\nShared Candidate Pool Usage:")
            print(f"  Distinct configs in any top-K: {len(topk_ct)}; distinct chosen: {len(chosen_ct)}")
            print(f"  Most frequent in top-K: " +
                  ", ".join(f"id{c}x{n}" for c, n in topk_ct.most_common(10)))
            print(f"  Most frequently chosen:  " +
                  ", ".join(f"id{c}x{n}" for c, n in chosen_ct.most_common(10)))

        rec_times = [r['recommended_time'] for r in results]
        def_times = [r['default_time'] for r in results]
        print(f"\nTime Statistics (wall-clock):")
        print(f"  Recommended: mean={np.mean(rec_times):.3f}s, median={np.median(rec_times):.3f}s")
        print(f"  Default:     mean={np.mean(def_times):.3f}s, median={np.median(def_times):.3f}s")
        print(f"  Speedup (mean/median): {np.mean(def_times)/max(np.mean(rec_times),0.001):.2f}x / "
              f"{np.median(def_times)/max(np.median(rec_times),0.001):.2f}x")

        rec_nodes = [r['recommended_nodes'] for r in results]
        def_nodes = [r['default_nodes'] for r in results]
        print(f"\nNode Statistics (branch-and-bound nodes):")
        print(f"  Recommended: mean={np.mean(rec_nodes):.1f}, median={np.median(rec_nodes):.1f}")
        print(f"  Default:     mean={np.mean(def_nodes):.1f}, median={np.median(def_nodes):.1f}")
        print(f"  Node reduction (mean/median): "
              f"{np.mean(def_nodes)/max(np.mean(rec_nodes),1):.2f}x / "
              f"{np.median(def_nodes)/max(np.median(rec_nodes),1):.2f}x")

        rec_opt = sum(1 for r in results if r['recommended_status'] == 'optimal')
        def_opt = sum(1 for r in results if r['default_status'] == 'optimal')
        rec_feas = sum(1 for r in results if r['recommended_status'] in ('optimal', 'timelimit', 'nodelimit'))
        def_feas = sum(1 for r in results if r['default_status'] in ('optimal', 'timelimit', 'nodelimit'))
        print(f"\nSolve Status:")
        print(f"  Recommended: optimal={rec_opt}, feasible={rec_feas}, other={len(results)-rec_feas}")
        print(f"  Default:     optimal={def_opt}, feasible={def_feas}, other={len(results)-def_feas}")

        print(f"\nPer-Family Breakdown:")
        print(f"  {'Family':<25s} {'Mean Score':>10s} {'Win Rate':>10s} {'Avg Rec Time':>12s} {'Avg Def Time':>12s} {'Count':>6s}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*6}")
        families = config['evaluation']['families']
        for fid in families:
            fr = [r for r in results if r['family_id'] == fid]
            fs = [r['score'] for r in fr if r['score'] is not None]
            if fs:
                print(f"  {fid:<25s} {np.mean(fs):>10.3f} {f'{sum(1 for s in fs if s>0)}/{len(fs)}':>10s} "
                      f"{np.mean([r['recommended_time'] for r in fr]):>11.3f}s "
                      f"{np.mean([r['default_time'] for r in fr]):>11.3f}s {len(fr):>6d}")

        print(f"\nMost Frequently Changed Parameters:")
        from collections import Counter
        pc = Counter()
        for r in results:
            for pname, pval in r['predicted_config'].items():
                if pval != default_config.get(pname):
                    pc[pname] += 1
        for param, count in pc.most_common(10):
            print(f"  {param:50s}: {count:3d} ({100*count/len(results):.1f}%)")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
