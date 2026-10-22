"""
EPM + LLM selection: no-LLM EPM nominates Top-K, LLM makes the final pick (K-choose-1).

K is set by --top-k (default 5) and drives everything downstream: how many candidates
are solved, what the prompt asks for, and the accepted range of the LLM's answer.

Pipeline (mirrors scripts/epm_no_llm.py):
  1. Train the 50-dim EPM  f(20 features + 30 config) -> log(speedup)  (no direction).
  2. Per eval instance (from config/experiment.yaml evaluation section:
     families x variants x seed_offset..seed_offset+n_test_instances, features
     computed on the fly): score the FIXED pool from config_100.json (ids 0..99)
     -> EPM Top-K (identical to the epm_no_llm.py run) -> ONE LLM call selects one.
  3. LLM input: the skill file + full formulation matrix + 20-dim features
     + the K candidates (each with its CONFIG ID + full 30-param listing)
     + risk-averse instructions: do NOT chase the fastest; ELIMINATE any candidate
     that could plausibly be SLOWER than default; pick the most RELIABLE winner.
  4. Solve chosen config vs default, report the same metrics as evaluate.py.

Fallback: on LLM error / unparsable reply -> Top-1 (pure EPM choice).

Usage:
    python scripts/epm_with_selection.py
    python scripts/epm_with_selection.py --top-k 3
    python scripts/epm_with_selection.py --skill skills/selection_skill.md --n-eval 5
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import re
import sys
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from al_param_tuning.solver import SCIPWrapper, compute_score
from al_param_tuning.llm.client import LLMClient

N_REPEATS = 1

FEATURE_NAMES = [
    "row_count", "col_count", "total_nonzeros", "matrix_density",
    "binary_ratio", "general_integer_ratio", "continuous_ratio", "unbounded_count",
    "equality_ratio", "inequality_ratio", "set_operation_count", "special_constraint_count",
    "mean_nonzeros_per_row", "mean_nonzeros_per_col", "matrix_coeff_cv",
    "objective_density", "rhs_cv",
    "num_connected_components", "largest_component_ratio", "isolated_variable_count",
]


def make_encoder(parameters):
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


def config_full_text(cfg, param_names):
    """Full 30-parameter listing of a configuration."""
    return "\n".join(f"  {p} = {cfg.get(p)}" for p in param_names)


def slim_objective(model, max_terms=40, n_example=20):
    """Objective text; if many nonzero terms, summarize instead of listing all."""
    variables = model.getVars()
    nz = [(v.name, v.getObj()) for v in variables if abs(v.getObj()) > 1e-10]
    sense = 'Minimize' if model.getObjectiveSense() == 'minimize' else 'Maximize'
    lines = ["## Objective Function", f"Type: {sense}", f"Non-zero terms: {len(nz)}/{len(variables)}"]
    if not nz:
        return "\n".join(lines)
    if len(nz) <= max_terms:
        lines.append("")
        lines.append("Objective terms:")
        for name, coeff in nz:
            lines.append(f"  {coeff:+.6f} * {name}")
        return "\n".join(lines)
    coeffs = [c for _, c in nz]
    lines.append(f"系数: 范围[{min(coeffs):.6g}, {max(coeffs):.6g}], 均值 {sum(coeffs)/len(coeffs):.6g} "
                 f"(目标项过多,仅列前 {n_example} 项作示例,其余同类省略)")
    lines.append("示例项:")
    for name, coeff in nz[:n_example]:
        lines.append(f"  {coeff:+.6f} * {name}")
    lines.append(f"  ... (还有 {len(nz) - n_example} 个非零项,系数在上述范围内)")
    return "\n".join(lines)


def slim_describe(model):
    """Compressed formulation: bounds/binaries omitted; one representative
    constraint per named group (count + LHS/RHS ranges); long rows truncated;
    long objectives summarized."""
    n_vars = model.getNVars()
    n_cons = model.getNConss()
    tc = {"BINARY": 0, "INTEGER": 0, "CONTINUOUS": 0}
    for v in model.getVars():
        t = v.vtype()
        if t in tc:
            tc[t] += 1

    groups, order = {}, []
    for c in model.getConss():
        key = '_'.join(t for t in c.name.split('_') if not t.isdigit()) or c.name
        if key not in groups:
            groups[key] = {'count': 0, 'first': c, 'lhs': [], 'rhs': []}
            order.append(key)
        g = groups[key]
        g['count'] += 1
        lhs, rhs = model.getLhs(c), model.getRhs(c)
        if lhs is not None and lhs > -1e20:
            g['lhs'].append(lhs)
        if rhs is not None and rhs < 1e20:
            g['rhs'].append(rhs)

    def cons_text(c, max_terms=50):
        try:
            pairs = [(v, k) for v, k in zip(model.getConsVars(c), model.getConsVals(c)) if abs(k) > 1e-10]

            def fmt(v, k):
                if abs(k - 1.0) < 1e-10:
                    return f"+ {v.name}"
                if abs(k + 1.0) < 1e-10:
                    return f"- {v.name}"
                return f"{k:+.6f}*{v.name}"

            if len(pairs) <= max_terms:
                expr = " ".join(fmt(v, k) for v, k in pairs)
            else:
                shown = " ".join(fmt(v, k) for v, k in pairs[:max_terms])
                ks = [k for _, k in pairs]
                expr = (f"{shown} ... [该行共 {len(pairs)} 项,仅展示前 {max_terms} 项;"
                        f"全行系数范围 [{min(ks):.4g}, {max(ks):.4g}]]")
            lhs, rhs = model.getLhs(c), model.getRhs(c)
            if lhs is not None and rhs is not None and abs(lhs - rhs) < 1e-9:
                sense = f"= {rhs:.6f}"
            elif lhs is not None and lhs > -1e20:
                sense = f"in [{lhs:.6f}, {rhs:.6f}]" if (rhs is not None and rhs < 1e20) else f">= {lhs:.6f}"
            else:
                sense = f"<= {rhs:.6f}" if (rhs is not None and rhs < 1e20) else "(no bounds)"
            return f"{c.name}: {expr} {sense}"
        except Exception as ex:
            return f"{c.name}: (error: {ex})"

    lines = [
        "## Instance Dimensions",
        f"- Variables: {n_vars} (binary={tc['BINARY']}, integer={tc['INTEGER']}, "
        f"continuous={tc['CONTINUOUS']}; per-variable bounds/binaries omitted — binary vars are [0,1])",
        f"- Constraints: {n_cons}, grouped into {len(order)} named groups",
        "",
        "## Constraints (每个命名约束组只展示第 1 条作为代表;同组其余约束结构相同,仅下标不同,系数/RHS 可能逐行变化,范围见标注)",
    ]
    for key in order:
        g = groups[key]
        rng = []
        if g['lhs']:
            lo, hi = min(g['lhs']), max(g['lhs'])
            rng.append(f"LHS范围[{lo:.4g}, {hi:.4g}]" if lo != hi else f"LHS={lo:.4g}")
        if g['rhs']:
            lo, hi = min(g['rhs']), max(g['rhs'])
            rng.append(f"RHS范围[{lo:.4g}, {hi:.4g}]" if lo != hi else f"RHS={lo:.4g}")
        lines.append(f"### 组 `{key}` — 共 {g['count']} 条{'; ' + ', '.join(rng) if rng else ''}")
        lines.append(f"  [代表] {cons_text(g['first'])}")
    lines += ["", slim_objective(model)]
    return "\n".join(lines)


def build_selection_prompt(skill, feature_text, description, default_text, cand_entries):
    k = len(cand_entries)
    cand_block = "\n\n".join(
        f"### Candidate {i+1} (EPM rank {i+1}, CONFIG ID {cid})\n{t}"
        for i, (cid, t) in enumerate(cand_entries))
    id_list = ", ".join(str(cid) for cid, _ in cand_entries)
    return f"""You are an expert in SCIP solver parameter tuning.

## Task
A machine-learning model (EPM) nominated {k} candidate parameter configurations for the MIP instance
described below (its Top-{k} by predicted speedup). The candidates come from a FIXED numbered
configuration pool; their CONFIG IDS are: {id_list}. The accumulated experience below may refer to
these same config ids — match them directly. Your job: SELECT EXACTLY ONE of the {k} candidates.

## Selection principle (RISK-AVERSE — read carefully)
Your objective has an EXPLICIT priority order:
- **Priority 1 (dominant): reliably beat the SCIP default.** Never pick a candidate that has a
  plausible risk of losing to the default — a stable 2x is worth more than a gamble at 10x.
- **Priority 2 (secondary but REAL): speedup still matters.** Among the candidates that safely beat
  the default, you should pick the one you expect to be FASTEST — do NOT simply take the most
  conservative/closest-to-default survivor. Safety first, but never waste speedup that costs no risk.

Procedure:
- FIRST, ELIMINATE every candidate that has a plausible risk of being SLOWER than the SCIP DEFAULT
  configuration on this instance — e.g. it hits a known dealbreaker for this problem type in the
  accumulated experience, uses risky extreme values (unlimited rounds like -1, disabling presolve or
  propagation where this structure depends on them), or gambles on mechanisms known to be unstable
  for this structure — EVEN IF that candidate might sometimes be the fastest.
- THEN, among the surviving candidates, pick the one with the HIGHEST expected speedup (use the
  accumulated experience: which changes typically yield large gains for this structure).
- The EPM cannot rank reliably WITHIN these {k} (its ordering here is close to random). Base your
  decision on the accumulated experience and the instance structure, not on the EPM rank.
- If several survivors look equally fast, prefer the one whose changes match the type-level
  recommendations for this problem class in the accumulated experience.
- DEFAULT RULE: if you are NOT sufficiently confident that another candidate is clearly better,
  choose Candidate 1 (the EPM top-1). Only deviate from Candidate 1 when you have a CONCRETE,
  experience-backed reason (a known dealbreaker in Candidate 1, or a clearly safer+faster
  alternative). Uncertain deviation is worse than staying with Candidate 1.

## Accumulated Experience
{skill}

## Instance: 20 numerical features
{feature_text}

## Instance: formulation (compressed — each named constraint group shows ONE representative row; the rest are structurally identical up to indices, with counts and RHS/LHS ranges noted inline)
{description}

## SCIP DEFAULT configuration (reference baseline)
{default_text}

## Candidate configurations ({k} candidates, FULL 30-parameter listing each)
{cand_block}

## Output
Return STRICT JSON only, no other text:
{{"choice": <integer 1-{k}>, "excluded": [<candidate numbers you eliminated>], "reason": "<one short sentence>"}}"""


def parse_choice(response, k):
    """Extract {"choice": i} (1 <= i <= k) from the LLM reply;
    return (choice, excluded, reason) or None."""
    if not response:
        return None
    m = re.search(r'\{.*\}', response, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        c = int(obj.get('choice'))
        if 1 <= c <= k:
            return c, obj.get('excluded', []), str(obj.get('reason', ''))[:300]
    except Exception:
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description="EPM Top-k + LLM risk-averse k-choose-1 selection")
    parser.add_argument("--config", type=str, default="config/experiment.yaml")
    parser.add_argument("--cache", type=str, default="data/results/training_cache_multi.json")
    parser.add_argument("--config-pool", type=str, default="config_100.json",
                        help="Fixed candidate pool JSON (ids 0..N-1 = position in file)")
    parser.add_argument("--output", type=str, default="data/results/epm_llm_selection_config100_results.json")
    parser.add_argument("--skill", type=str, default="skills/skill.md")
    parser.add_argument("--top-k", type=int, default=5,
                        help="k for the k-choose-1 task: the EPM nominates its Top-k from the "
                             "fixed pool and the LLM picks exactly one of them (default 5)")
    parser.add_argument("--n-eval", type=int, default=None, help="Evaluate only first N instances (0 = train only)")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Concurrent LLM calls in Phase B")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.top_k < 2:
        sys.exit(f"ERROR: --top-k must be >= 2 (got {args.top_k}); "
                 f"for the pure-EPM Top-1 baseline use scripts/epm_no_llm.py.")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    param_config_path = config['paths']['param_config']
    with open(param_config_path, 'r', encoding='utf-8') as f:
        param_config = json.load(f)
    parameters = param_config['parameters']
    encode = make_encoder(parameters)
    param_names = [p['name'] for p in parameters]

    # =================================================================
    # Step 1: Train the 50-dim EPM (identical to epm_no_llm.py)
    # =================================================================
    print("=" * 70)
    print(f"EPM Top-{args.top_k} + LLM risk-averse selection "
          f"({args.top_k}-choose-1; base: no-LLM EPM, 50-dim)")
    print("=" * 70)
    cache = json.load(open(args.cache, 'r'))
    X_all = cache['X_all']
    raw = cache['raw_data_all']
    n_inst = len(X_all)

    X_rows, y_rows = [], []
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
    X_rows = np.array(X_rows)
    y_rows = np.array(y_rows)
    print(f"Training rows: {X_rows.shape}")

    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=400, max_depth=7, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                         reg_alpha=0.1, reg_lambda=1.0, random_state=42)
    model.fit(X_rows, y_rows)
    print("EPM (no-LLM) trained.")

    # =================================================================
    # Step 2: LLM client + skill + descriptor
    # =================================================================
    llm_cfg = config['llm']
    llm = LLMClient(provider=llm_cfg['provider'], model=llm_cfg['model'],
                    base_url=llm_cfg.get('base_url'), api_key=llm_cfg.get('api_key_env'),
                    temperature=llm_cfg['temperature'], max_tokens=llm_cfg['max_tokens'])
    skill = Path(args.skill).read_text(encoding='utf-8')
    print(f"Skill: {args.skill} ({len(skill)} chars)")

    # =================================================================
    # Step 3: Evaluate
    # =================================================================
    scip = SCIPWrapper(param_config)
    default_config = scip.get_default_config()
    time_limit = config['evaluation']['time_limit']
    cache_dir = Path('data/results/instances')
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Eval set from config['evaluation']: families x variants x seeds (pre-generated LPs)
    eval_cfg = config['evaluation']
    families = eval_cfg['families']
    seed_offset = eval_cfg['seed_offset']
    n_test = eval_cfg['n_test_instances']
    seeds = list(range(seed_offset, seed_offset + n_test))
    eval_data = [{'family_id': fid, 'variant': v, 'seed': s}
                 for fid in families for v in eval_cfg['variants'] for s in seeds]
    if args.n_eval is not None:
        eval_data = eval_data[:args.n_eval]

    # On-demand instance generation for missing LP files (same accept window as
    # evaluate.py: keep resampling the seed until the default solve time lands in
    # [min_solve_time, max_solve_time]; otherwise write the fastest solvable candidate).
    from al_param_tuning.generators import ProblemRegistry
    registry = ProblemRegistry()
    _inst_cfg = config.get('instances', {}) or {}
    MIN_T = float(_inst_cfg.get('min_solve_time', 0.5))
    MAX_T = float(_inst_cfg.get('max_solve_time', 5.0))
    MAX_ATTEMPTS = int(_inst_cfg.get('max_resample_attempts', 40))

    def ensure_lp(inst):
        """Return the LP path for an instance, generating it if absent."""
        lp = cache_dir / f"{inst['family_id']}_{inst['variant']}_{inst['seed']}.lp"
        if lp.exists():
            return lp
        # Externally generated benchmark files use a zero-padded index
        # (e.g. set_covering_ecole_01.lp); accept those as-is.
        padded = cache_dir / f"{inst['family_id']}_{inst['variant']}_{int(inst['seed']):02d}.lp"
        if padded.exists():
            return padded
        if inst['family_id'] not in registry.list_families():
            sys.exit(f"ERROR: no LP file for {lp.name} (nor {padded.name}) and "
                     f"'{inst['family_id']}' has no registered generator. "
                     f"Generate the instances first, e.g. with "
                     f"scripts/generate_ecole_instances.py")
        base = inst['seed']
        accepted = None
        fb_seed, fb_time, fb_has = base, float('inf'), False
        for attempt in range(MAX_ATTEMPTS):
            cand_seed = base if attempt == 0 else base * 1000 + attempt
            m = registry.generate_instance(family_id=inst['family_id'],
                                           variant=inst['variant'], seed=cand_seed)
            r = scip.solve(m, default_config, time_limit=MAX_T + 2.0)
            if (r.has_solution and not fb_has) or \
               (r.has_solution == fb_has and r.time < fb_time):
                fb_seed, fb_time, fb_has = cand_seed, r.time, r.has_solution
            if r.has_solution and MIN_T <= r.time <= MAX_T:
                accepted = cand_seed
                break
        write_seed = accepted if accepted is not None else fb_seed
        m = registry.generate_instance(family_id=inst['family_id'],
                                       variant=inst['variant'], seed=write_seed)
        m.writeProblem(str(lp))
        tag = "in-window" if accepted is not None else f"fallback (best {fb_time:.2f}s)"
        print(f"  generated {lp.name} [{tag}]", flush=True)
        return lp

    # FIXED candidate pool from config_100.json; ids 0..N-1 = position in file
    pool = json.load(open(args.config_pool, 'r', encoding='utf-8'))
    cands = [entry['config'] for entry in pool]
    cand_vecs = np.array([encode(c) for c in cands])
    if args.top_k > len(cands):
        sys.exit(f"ERROR: --top-k {args.top_k} exceeds the {len(cands)}-config pool "
                 f"in {args.config_pool}.")

    from al_param_tuning.features import InstanceFeatureExtractor
    feat_extractor = InstanceFeatureExtractor()
    print(f"\nEvaluating {len(eval_data)} instances "
          f"(seeds {seeds[0]}-{seeds[-1]}, {len(cands)} fixed candidates from "
          f"{args.config_pool}, Top-{args.top_k} -> LLM pick, {N_REPEATS}x solve)...")

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

    # ── Phase A: compute Top-k + build prompts for all pending instances ──
    pending = []
    default_text = config_full_text(default_config, param_names)
    print(f"\n[Phase A] Computing Top-{args.top_k} and building prompts...")
    for idx, e in enumerate(tqdm(eval_data, desc=f"  Top-{args.top_k}")):
        key = (e['family_id'], e['variant'], e['seed'])
        if key in done:
            continue
        lp_file = ensure_lp(e)

        # Compute the 20-dim features from the LP file
        m0 = Model()
        m0.hideOutput()
        m0.readProblem(str(lp_file))
        feats = np.array(feat_extractor.extract(m0), dtype=float)
        e['features'] = feats.tolist()

        # Score the FIXED pool; identical Top-k as epm_no_llm.py
        Xc = np.hstack([np.tile(feats, (len(cands), 1)), cand_vecs])
        preds = model.predict(Xc)
        order = np.argsort(-preds)[:args.top_k]
        topk_ids = [int(j) for j in order]
        topk = [cands[j] for j in order]
        topk_pred = [float(preds[j]) for j in order]

        feature_text = "\n".join(f"- {n}: {feats[i]:.4f}" for i, n in enumerate(FEATURE_NAMES))
        description = slim_describe(m0)
        m0.freeProb()
        cand_entries = [(cid, config_full_text(c, param_names)) for cid, c in zip(topk_ids, topk)]
        prompt = build_selection_prompt(skill, feature_text, description, default_text, cand_entries)

        pending.append({'e': e, 'key': key, 'lp_file': str(lp_file),
                        'topk': topk, 'topk_ids': topk_ids, 'topk_pred': topk_pred,
                        'prompt': prompt})

    # ── Phase B: batch LLM selection (concurrent), with sidecar cache ──
    sel_path = Path(str(output_path) + ".selections.json")
    selections = {}
    if sel_path.exists() and not args.no_resume:
        try:
            selections = {tuple(k.split('|')): v for k, v in
                          json.load(open(sel_path, 'r', encoding='utf-8')).items()}
            print(f"[Phase B] Loaded {len(selections)} cached LLM selections.")
        except Exception:
            selections = {}
    # The cache is keyed by instance only, so entries made under a larger k would
    # point past the current Top-k list; drop them and re-ask.
    stale = [key for key, v in selections.items() if v.get('choice', 1) > args.top_k]
    for key in stale:
        del selections[key]
    if stale:
        print(f"[Phase B] Discarded {len(stale)} cached selections with choice > k={args.top_k}.")
    need = [p for p in pending if tuple(map(str, p['key'])) not in selections]
    print(f"[Phase B] LLM selection for {len(need)}/{len(pending)} instances "
          f"({args.max_concurrent} concurrent)...")
    if need:
        responses = llm.batch_ask([p['prompt'] for p in need],
                                  max_concurrent=args.max_concurrent)
        for p, resp in zip(need, responses):
            parsed = parse_choice(resp, args.top_k)
            if parsed is not None:
                c, ex, reason = parsed
                selections[tuple(map(str, p['key']))] = {
                    'choice': c, 'excluded': ex, 'reason': reason, 'fallback': False}
            else:
                selections[tuple(map(str, p['key']))] = {
                    'choice': 1, 'excluded': [], 'reason': 'parse/LLM failure', 'fallback': True}
        with open(sel_path, 'w', encoding='utf-8') as f:
            json.dump({'|'.join(k): v for k, v in selections.items()}, f, indent=2, ensure_ascii=False)
    n_fallback = sum(1 for v in selections.values() if v.get('fallback'))
    print(f"[Phase B] Done. Fallbacks: {n_fallback}")

    # ── Phase C: serial SCIP evaluation (clean timing; solve ALL k + default) ──
    print(f"\n[Phase C] Solving all Top-{args.top_k} + default ({N_REPEATS}x each, serial)...")
    for p in pending:
        e = p['e']
        sel = selections.get(tuple(map(str, p['key'])),
                             {'choice': 1, 'excluded': [], 'reason': 'missing', 'fallback': True})
        llm_choice, llm_excluded = sel['choice'], sel['excluded']
        llm_reason, fallback = sel['reason'], sel['fallback']
        topk, topk_ids, topk_pred = p['topk'], p['topk_ids'], p['topk_pred']

        # solve default + all k candidates
        result_def, def_raw = solve_repeated(p['lp_file'], default_config)
        solved = []
        for rank, cfg in enumerate(topk):
            res, raw = solve_repeated(p['lp_file'], cfg)
            solved.append({'rank': rank + 1, 'config': cfg, 'result': res, 'raw': raw,
                           'score': compute_score(res, result_def)})

        chosen_entry = solved[llm_choice - 1]
        chosen = chosen_entry['config']
        result_rec, rec_raw = chosen_entry['result'], chosen_entry['raw']
        oracle_entry = min(solved, key=lambda s: s['result'].time)
        oracle_rank, oracle_score = oracle_entry['rank'], oracle_entry['score']
        top1_score = solved[0]['score']
        rank_scores = [s['score'] for s in solved if s['score'] is not None]
        random_exp_score = float(np.mean(rank_scores)) if rank_scores else None

        score = chosen_entry['score']
        if llm_choice == 1:
            h2h = 'SAME'
        elif score is not None and top1_score is not None:
            h2h = 'LLM' if score > top1_score else ('TOP1' if score < top1_score else 'TIE')
        else:
            h2h = 'NA'
        speedup = result_def.time / max(result_rec.time, 0.001) if result_rec.time > 0 else 0
        winner = "RECOMMENDED" if score and score > 0 else ("DEFAULT" if score and score < 0 else "TIE")
        n_changed = sum(1 for pn in param_names if chosen.get(pn) != default_config.get(pn))

        results.append({
            'family_id': e['family_id'], 'variant': e['variant'], 'seed': e['seed'],
            'features': e['features'],
            'top_k': len(topk),
            'llm_choice': llm_choice, 'llm_excluded': llm_excluded,
            'llm_reason': llm_reason, 'llm_fallback': fallback,
            'topk_cand_ids': topk_ids,
            'chosen_cand_id': topk_ids[llm_choice - 1],
            'oracle_cand_id': topk_ids[oracle_rank - 1],
            'topk_configs': topk, 'topk_pred': topk_pred,
            'topk_results': [{'rank': s['rank'], 'cand_id': topk_ids[s['rank'] - 1],
                              'pred': topk_pred[s['rank'] - 1],
                              'time': s['result'].time, 'nodes': s['result'].nodes,
                              'status': s['result'].status, 'score': s['score']} for s in solved],
            'oracle_rank': oracle_rank, 'oracle_score': oracle_score,
            'top1_score': top1_score, 'random_exp_score': random_exp_score,
            'h2h_vs_top1': h2h,
            'predicted_config': chosen, 'predicted_log_speedup': topk_pred[llm_choice - 1],
            'n_params_changed': n_changed,
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

        rank_str = " ".join(
            f"r{s['rank']}={s['result'].time:.2f}s{'!' if s['result'].time >= result_def.time else ''}"
            + ("*" if s['rank'] == llm_choice else "")
            for s in solved)
        n_w = sum(1 for x in results if x.get('h2h_vs_top1') == 'LLM')
        n_l = sum(1 for x in results if x.get('h2h_vs_top1') == 'TOP1')
        n_t = sum(1 for x in results if x.get('h2h_vs_top1') == 'TIE')
        n_s = sum(1 for x in results if x.get('h2h_vs_top1') == 'SAME')
        print(f"[{len(results)}/{len(eval_data)}] {e['family_id']}/{e['seed']}: "
              f"topk_ids={topk_ids} def={result_def.time:.2f}s | {rank_str} | oracle=r{oracle_rank} "
              f"LLM=r{llm_choice}(id{topk_ids[llm_choice - 1]}){' (FALLBACK)' if fallback else ''} H2H={h2h} "
              f"speedup={speedup:.2f}x score={score:+.3f} {winner} "
              f"| 累计 {n_w}胜 {n_l}负 {n_t}平 {n_s}同", flush=True)

    # =================================================================
    # Step 4: Summary
    # =================================================================
    scores = [r['score'] for r in results if r['score'] is not None]
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY (EPM Top-{args.top_k} + LLM risk-averse {args.top_k}-choose-1)")
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

        # ── Same-run four-way selector comparison ──
        if any('oracle_score' in r for r in results):
            from collections import Counter
            t1 = [r['top1_score'] for r in results if r.get('top1_score') is not None]
            rd = [r['random_exp_score'] for r in results if r.get('random_exp_score') is not None]
            oc = [r['oracle_score'] for r in results if r.get('oracle_score') is not None]
            print(f"\nSelector Comparison (same-run, no cross-run contamination):")
            print(f"  {'LLM choice':14s}: mean score={np.mean(scores):.3f}, win={100*n_rec/len(scores):.1f}%")
            print(f"  {'Top-1':14s}: mean score={np.mean(t1):.3f}, win={100*np.mean([s > 0 for s in t1]):.1f}%")
            print(f"  {'Random (E)':14s}: mean score={np.mean(rd):.3f}")
            print(f"  {'Oracle':14s}: mean score={np.mean(oc):.3f}, win={100*np.mean([s > 0 for s in oc]):.1f}%")
            rank_dist = Counter(r['oracle_rank'] for r in results if 'oracle_rank' in r)
            print(f"  Oracle rank distribution: " +
                  ", ".join(f"rank{k}={rank_dist[k]}" for k in sorted(rank_dist)))
            hit = sum(1 for r in results if r.get('llm_choice') == r.get('oracle_rank'))
            print(f"  LLM hit oracle: {hit}/{len(results)} ({100*hit/len(results):.0f}%, "
                  f"chance={100/args.top_k:.0f}%)")

        from collections import Counter
        cdist = Counter(r['llm_choice'] for r in results)
        print(f"\nLLM Selection Behavior:")
        print(f"  Choice-rank distribution: " + ", ".join(f"rank{k}={cdist[k]}" for k in sorted(cdist)))
        print(f"  Fallbacks (LLM failed -> top1): {sum(1 for r in results if r.get('llm_fallback'))}")

        def _h2h(r):
            if r.get('h2h_vs_top1'):
                return r['h2h_vs_top1']
            if r.get('llm_choice') == 1:
                return 'SAME'
            s, t = r.get('score'), r.get('top1_score')
            if s is None or t is None:
                return 'NA'
            return 'LLM' if s > t else ('TOP1' if s < t else 'TIE')

        print(f"\nHead-to-Head LLM vs Top-1 (per family; SAME = LLM chose rank 1):")
        print(f"  {'Family':<25s} {'LLM':>4s} {'TOP1':>5s} {'TIE':>4s} {'SAME':>5s} {'Mean dScore':>12s}")
        tot = Counter()
        for fid in config['evaluation']['families']:
            fr = [r for r in results if r['family_id'] == fid]
            if not fr:
                continue
            fc = Counter(_h2h(r) for r in fr)
            tot.update(fc)
            deltas = [r['score'] - r['top1_score'] for r in fr
                      if r.get('score') is not None and r.get('top1_score') is not None]
            print(f"  {fid:<25s} {fc['LLM']:>4d} {fc['TOP1']:>5d} {fc['TIE']:>4d} {fc['SAME']:>5d} "
                  f"{np.mean(deltas) if deltas else 0.0:>+12.3f}")
        all_d = [r['score'] - r['top1_score'] for r in results
                 if r.get('score') is not None and r.get('top1_score') is not None]
        print(f"  {'TOTAL':<25s} {tot['LLM']:>4d} {tot['TOP1']:>5d} {tot['TIE']:>4d} {tot['SAME']:>5d} "
              f"{np.mean(all_d) if all_d else 0.0:>+12.3f}")

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

        rec_opt = sum(1 for r in results if r['recommended_status'] == 'optimal')
        print(f"\nSolve Status: recommended optimal={rec_opt}/{len(results)}")

        print(f"\nPer-Family Breakdown:")
        print(f"  {'Family':<25s} {'Mean Score':>10s} {'Win Rate':>10s} {'Avg Rec Time':>12s} {'Avg Def Time':>12s}")
        families = config['evaluation']['families']
        for fid in families:
            fr = [r for r in results if r['family_id'] == fid]
            fs = [r['score'] for r in fr if r['score'] is not None]
            if fs:
                print(f"  {fid:<25s} {np.mean(fs):>10.3f} {f'{sum(1 for s in fs if s>0)}/{len(fs)}':>10s} "
                      f"{np.mean([r['recommended_time'] for r in fr]):>11.3f}s "
                      f"{np.mean([r['default_time'] for r in fr]):>11.3f}s")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
