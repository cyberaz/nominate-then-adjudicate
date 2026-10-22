"""
EPM ablation: NO LLM (50-dim input). Reports the Top-1 floor and the oracle ceiling.

        X = [20 instance features | 30 encoded config values] = 50-dim
        y = log(default_time / config_time)

Candidates: the FIXED pool from --config-pool (ids 0..N-1 = position in that
file). Every eval instance scores the same pool, so candidate ids are comparable
across instances, batches and runs.

With the default --top-k 5 all five EPM-ranked candidates are solved, which gives
both bounds on the 5-choose-1 task in one run:
  - Top-1  = what the EPM alone achieves (no LLM). The floor the selector must beat.
  - Oracle = the actually fastest of the five. The ceiling a perfect selector reaches.
The gap between them is exactly the headroom available to the LLM selector, and
the oracle rank distribution shows how often Top-1 was already the best choice.
Use --top-k 1 to measure only the floor (1 solve per instance instead of 5).

Eval instances come from config/experiment.yaml `evaluation`
(families x variants x [seed_offset, seed_offset + n_test_instances)); the LP
files must already exist in data/results/instances/.

Usage:
    python scripts/epm_no_llm.py
    python scripts/epm_no_llm.py --top-k 1      # Top-1 floor only, 5x cheaper
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


def sgm(x, s=1.0):
    """Shifted geometric mean, as used in the round reports and the ISAC baseline."""
    return float(np.exp(np.mean(np.log(np.asarray(x, dtype=float) + s))) - s)


def main():
    parser = argparse.ArgumentParser(description="EPM no-LLM: f(features, config) -> log(speedup)")
    parser.add_argument("--config", type=str, default="config/experiment.yaml")
    parser.add_argument("--cache", type=str,
                        default="data/results/training_cache_pool1000.json",
                        help="EPM training cache. Must be collected over the same pool as "
                             "--config-pool, and must match the cache used by "
                             "epm_with_selection.py or the two arms are not comparable")
    parser.add_argument("--config-pool", type=str, default="config_pool_1000.json",
                        help="Fixed candidate pool JSON (ids 0..N-1 = position in file); "
                             "every instance scores this same pool, so ids are comparable "
                             "across instances, batches and runs")
    parser.add_argument("--output", type=str,
                        default="data/results/epm_no_llm_ecole_results.json")
    parser.add_argument("--seed-start", type=int, default=None,
                        help="Override evaluation.seed_offset")
    parser.add_argument("--n-instances", type=int, default=None,
                        help="Override evaluation.n_test_instances")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Solve the top-K EPM-ranked candidates. K>1 yields both the "
                             "Top-1 floor and the best-of-K oracle ceiling; K=1 yields "
                             "only the floor. Keep equal to epm_with_selection.py --top-k")
    parser.add_argument("--n-eval", type=int, default=None, help="Evaluate only first N instances (default all; 0 = train only)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

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
    print("EPM no-LLM: train f(20 feat + 30 config) -> log(speedup)")
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
    # Step 2: Evaluate on the config-driven eval set
    # =================================================================
    eval_cfg = config['evaluation']
    families = eval_cfg['families']
    seed_start = args.seed_start if args.seed_start is not None else eval_cfg['seed_offset']
    n_inst_per_fam = (args.n_instances if args.n_instances is not None
                      else eval_cfg['n_test_instances'])
    seeds = list(range(seed_start, seed_start + n_inst_per_fam))
    eval_data = [{'family_id': fid, 'variant': v, 'seed': s}
                 for fid in families for v in eval_cfg['variants'] for s in seeds]
    if args.n_eval is not None:
        eval_data = eval_data[:args.n_eval]

    cache_dir = Path('data/results/instances')
    absent = [f"{e['family_id']}_{e['variant']}_{e['seed']}.lp" for e in eval_data
              if not (cache_dir / f"{e['family_id']}_{e['variant']}_{e['seed']}.lp").exists()]
    if absent:
        sys.exit(f"ERROR: {len(absent)} eval LP file(s) missing, e.g. "
                 f"{', '.join(absent[:5])}{' ...' if len(absent) > 5 else ''}\n"
                 f"Generate them first:\n"
                 f"  python scripts/generate_ecole_instances.py --tag {eval_cfg['variants'][0]} "
                 f"--start-index {seed_start} -n {n_inst_per_fam}")

    # FIXED candidate pool; report each candidate's own cand_id rather than its
    # position, so distilling a pool down to a subset does not renumber the ids
    # that experience rules cite.
    pool = json.load(open(args.config_pool, 'r', encoding='utf-8'))
    cands = [entry['config'] for entry in pool]
    cand_ids = [int(entry.get('cand_id', i)) for i, entry in enumerate(pool)]
    cand_vecs = np.array([encode(c) for c in cands])
    mode = "pure EPM Top-1" if args.top_k == 1 else f"oracle within Top-{args.top_k}"
    print(f"\nEvaluating {len(eval_data)} instances [{mode}] "
          f"(seeds {seeds[0]}-{seeds[-1]}, variants {eval_cfg['variants']}, "
          f"{len(cands)} fixed candidates from {args.config_pool}, "
          f"ids 0..{len(cands)-1}, {N_REPEATS}x serial solves)...")

    from al_param_tuning.features import InstanceFeatureExtractor
    feat_extractor = InstanceFeatureExtractor()

    scip = SCIPWrapper(param_config)
    default_config = scip.get_default_config()
    time_limit = eval_cfg['time_limit']

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
            solved.append({'rank': rank + 1, 'cand_idx': cand_ids[j],
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
            'predicted_config': chosen, 'predicted_log_speedup': pred_ls,
            'n_params_changed': n_changed,
            'oracle_rank': oracle_rank,
            'topk_cand_ids': [cand_ids[j] for j in order],
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
              f"top{args.top_k}_ids={[cand_ids[j] for j in order]} chosen_id={int(best['cand_idx'])} "
              f"oracle_rank={oracle_rank}/{args.top_k} "
              f"speedup={speedup:.2f}x score={score:+.3f} (top1 {top1_score:+.3f}) {winner}", flush=True)

    # =================================================================
    # Step 3: Summary (same metrics as evaluate.py)
    # =================================================================
    scores = [r['score'] for r in results if r['score'] is not None]
    mode = ("pure EPM Top-1" if args.top_k == 1
            else f"Top-1 floor + best-of-{args.top_k} oracle ceiling")
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY (EPM no-LLM, 20 feat + 30 config -> log speedup) [{mode}]")
    print("=" * 80)
    print(f"Instances evaluated: {len(results)}")
    print(f"Instances with valid scores: {len(scores)}")

    if scores:
        or_sp = [r['default_time'] / max(r['recommended_time'], 1e-3) for r in results]
        t1s = [r['top1_score'] for r in results if r.get('top1_score') is not None]
        t1_sp = [r['default_time'] / max(r['top1_time'], 1e-3) for r in results
                 if r.get('top1_time') is not None]

        def row(label, sc, sp):
            n_win = sum(1 for s in sc if s > 0)
            return (f"  {label:<22}{np.mean(sc):>+9.3f}{np.median(sc):>10.3f}"
                    f"{f'{n_win}/{len(sc)}':>10s}{100 * n_win / len(sc):>7.1f}%"
                    f"{sgm(sp):>10.3f}x{np.median(sp):>10.2f}x")

        print(f"\nOverall Metrics:")
        print(f"  {'':<22}{'mean':>9}{'median':>10}{'win':>10}{'':>8}"
              f"{'SGM sp':>11}{'med sp':>11}")
        if args.top_k > 1:
            print(row("Top-1 (EPM alone)", t1s, t1_sp))
            print(row(f"Oracle (best of {args.top_k})", scores, or_sp))
            n_t1, n_or = sum(1 for s in t1s if s > 0), sum(1 for s in scores if s > 0)
            print(f"\n  LLM selector headroom: score {np.mean(scores) - np.mean(t1s):+.3f}, "
                  f"win {100 * (n_or - n_t1) / len(scores):+.1f}pp, "
                  f"SGM speedup {sgm(or_sp) - sgm(t1_sp):+.3f}x "
                  f"({sgm(or_sp) / max(sgm(t1_sp), 1e-9):.2f}x of the Top-1 speedup)")
            print(f"  -> a PERFECT 5-choose-1 selector would land at the Oracle row; the "
                  f"LLM's value is whatever\n     fraction of this headroom it actually "
                  f"harvests.")
        else:
            print(row("Top-1 (EPM alone)", scores, or_sp))

        # ── Where in the ranking the oracle sat; meaningless when K=1 ──
        if args.top_k > 1 and any('oracle_rank' in r for r in results):
            from collections import Counter
            rank_dist = Counter(r['oracle_rank'] for r in results if 'oracle_rank' in r)
            n_t1_optimal = rank_dist.get(1, 0)
            print(f"\nWhere the oracle sat in the EPM ranking:")
            print(f"  rank distribution: " +
                  ", ".join(f"rank{k}={rank_dist[k]}" for k in sorted(rank_dist)))
            print(f"  Top-1 was already the fastest of the {args.top_k}: "
                  f"{n_t1_optimal}/{len(results)} "
                  f"({100 * n_t1_optimal / len(results):.1f}%) - nothing for a selector "
                  f"to gain on these")
            gains = [r['score'] - r['top1_score'] for r in results
                     if r.get('top1_score') is not None and r['oracle_rank'] != 1]
            if gains:
                print(f"  on the remaining {len(gains)}, a perfect selector gains "
                      f"{np.mean(gains):+.3f} score each (max {max(gains):+.3f})")

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
        families = config['evaluation']['families']
        if args.top_k > 1:
            print(f"  {'Family':<24s}{'T1 score':>10s}{'OR score':>10s}{'gap':>8s}"
                  f"{'T1 win':>9s}{'OR win':>9s}{'T1 SGM':>9s}{'OR SGM':>9s}{'n':>4s}")
            print(f"  {'-' * 24}{'-' * 68}")
            for fid in families:
                fr = [r for r in results if r['family_id'] == fid]
                if not fr:
                    continue
                fo = [r['score'] for r in fr]
                ft = [r['top1_score'] for r in fr]
                osp = [r['default_time'] / max(r['recommended_time'], 1e-3) for r in fr]
                tsp = [r['default_time'] / max(r['top1_time'], 1e-3) for r in fr]
                print(f"  {fid:<24s}{np.mean(ft):>+10.3f}{np.mean(fo):>+10.3f}"
                      f"{np.mean(fo) - np.mean(ft):>+8.3f}"
                      f"{f'{sum(1 for s in ft if s > 0)}/{len(ft)}':>9s}"
                      f"{f'{sum(1 for s in fo if s > 0)}/{len(fo)}':>9s}"
                      f"{sgm(tsp):>8.2f}x{sgm(osp):>8.2f}x{len(fr):>4d}")
        else:
            print(f"  {'Family':<25s} {'Mean Score':>10s} {'Win Rate':>10s} "
                  f"{'Avg Rec Time':>12s} {'Avg Def Time':>12s} {'Count':>6s}")
            print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*6}")
            for fid in families:
                fr = [r for r in results if r['family_id'] == fid]
                fs = [r['score'] for r in fr if r['score'] is not None]
                if fs:
                    print(f"  {fid:<25s} {np.mean(fs):>10.3f} "
                          f"{f'{sum(1 for s in fs if s>0)}/{len(fs)}':>10s} "
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
