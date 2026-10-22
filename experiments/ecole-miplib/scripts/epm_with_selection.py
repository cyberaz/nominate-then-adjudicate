"""
EPM + LLM selection: no-LLM EPM nominates Top-k, LLM makes the final pick.

Pipeline (mirrors scripts/epm_no_llm.py):
  1. Train the 50-dim EPM  f(20 features + 30 config) -> log(speedup)  (no direction).
  2. Per eval instance (from config/experiment.yaml evaluation section:
     families x variants x seed_offset..seed_offset+n_test_instances, features
     computed on the fly): score the FIXED pool from config_100.json (ids 0..99)
     -> EPM Top-k (identical to the epm_no_llm.py run) -> ONE LLM call selects one.
  3. LLM input: the skill file + full formulation matrix + 20-dim features
     + the 5 candidates (each with its CONFIG ID + full 30-param listing)
     + risk-averse instructions: do NOT chase the fastest; ELIMINATE any candidate
     that could plausibly be SLOWER than default; pick the most RELIABLE winner.
  4. Solve chosen config vs default, report the same metrics as evaluate.py.

Fallback: on LLM error / unparsable reply -> Top-1 (pure EPM choice).

Usage:
    python scripts/epm_with_selection.py
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
import subprocess
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


_INF = 1e20
_EPS = 1e-10


def _row_facts(model, cons):
    """Nonzeros, sense and finite bounds of one row."""
    pairs = [(v, k) for v, k in zip(model.getConsVars(cons), model.getConsVals(cons))
             if abs(k) > _EPS]
    lhs, rhs = model.getLhs(cons), model.getRhs(cons)
    lo = lhs if lhs is not None and lhs > -_INF else None
    hi = rhs if rhs is not None and rhs < _INF else None
    if lo is not None and hi is not None:
        sense = "EQ" if abs(lo - hi) < 1e-9 else "RANGE"
    elif lo is not None:
        sense = "GE"
    elif hi is not None:
        sense = "LE"
    else:
        sense = "FREE"
    return pairs, sense, lo, hi


def _coeff_class(coeffs):
    """Coarse coefficient class used for grouping; exact ranges go in the header.

    Kept coarse on purpose: including the numeric range in the grouping key would
    split rows that are structurally identical but numerically scaled, which is what
    made an earlier version emit more classes than the name-based renderer did.
    """
    u = {round(c, 6) for c in coeffs}
    if u == {1.0}:
        return "all +1"
    if u <= {-1.0, 1.0}:
        return "all ±1"
    if all(float(c).is_integer() for c in u):
        return "integer"
    return "fractional"


def _semantic_label(pairs, sense, lo, hi, vtypes):
    """Name the row's structure from the row itself, never from its name."""
    ks = [k for _, k in pairs]
    all_bin = vtypes == {"BINARY"}
    unit = all(abs(k - 1.0) < _EPS for k in ks)
    rhs = hi if hi is not None else lo
    if all_bin and unit and rhs is not None and abs(rhs - 1.0) < 1e-9:
        return {"EQ": "set partitioning", "LE": "set packing",
                "GE": "set covering"}.get(sense, "cardinality")
    if all_bin and unit and rhs is not None:
        return "cardinality"
    if all_bin and sense == "LE" and all(k > 0 for k in ks):
        return "binary knapsack"
    if len(pairs) == 2 and sense in ("LE", "GE") and any(
            v.vtype() == "BINARY" for v, _ in pairs) and any(
            v.vtype() != "BINARY" for v, _ in pairs):
        return "variable bound (big-M)"
    if sense == "EQ" and {round(k, 6) for k in ks} <= {-1.0, 1.0} and abs(sum(ks)) < _EPS:
        return "flow conservation"
    return "general equality" if sense == "EQ" else "general inequality"


def describe_structural(model, max_classes=20, max_terms=14, max_rows_scanned=50000):
    """Compressed formulation grouped by DERIVED row structure, not by row name.

    slim_describe keys its groups on the constraint name with digits stripped, which
    collapses generator output (`c_1`, `c_2`, ...) into one group but degenerates to one
    group per row on instances whose names carry no group structure (`R0001`, `CNS..001`),
    where it emits the entire matrix row by row. Grouping on a signature computed from
    the row itself is name-independent, so it also removes the last channel through
    which generator metadata could reach the model.
    """
    conss = model.getConss()
    n_cons = len(conss)
    scanned = conss if n_cons <= max_rows_scanned else conss[:max_rows_scanned]
    tc = {"BINARY": 0, "INTEGER": 0, "CONTINUOUS": 0}
    for v in model.getVars():
        if v.vtype() in tc:
            tc[v.vtype()] += 1

    classes = {}
    for c in scanned:
        try:
            pairs, sense, lo, hi = _row_facts(model, c)
        except Exception:
            continue
        if not pairs:
            continue
        vtypes = {v.vtype() for v, _ in pairs}
        ks = [k for _, k in pairs]
        key = (_semantic_label(pairs, sense, lo, hi, vtypes), sense,
               _coeff_class(ks), "".join(sorted(t[0] for t in vtypes)))
        g = classes.setdefault(key, dict(n=0, rep=(pairs, sense, lo, hi),
                                         nnz=[], rhs=[], lo_k=[], hi_k=[]))
        g["n"] += 1
        g["nnz"].append(len(pairs))
        g["lo_k"].append(min(ks))
        g["hi_k"].append(max(ks))
        if hi is not None:
            g["rhs"].append(hi)
        elif lo is not None:
            g["rhs"].append(lo)

    out = ["## Instance Dimensions",
           f"- Variables: {model.getNVars()} (binary={tc['BINARY']}, "
           f"integer={tc['INTEGER']}, continuous={tc['CONTINUOUS']}; per-variable "
           f"bounds omitted, binary vars are [0,1])",
           f"- Constraints: {n_cons}, falling into {len(classes)} structural classes",
           ""]
    if n_cons > max_rows_scanned:
        out.append(f"NOTE: class structure was derived from the first "
                   f"{max_rows_scanned} of {n_cons} rows; counts below are over that "
                   f"sample, not the whole matrix.")
        out.append("")
    out.append("## Constraint classes (grouped by structure derived from each row: "
               "sense, coefficient pattern, row length, variable types. Row NAMES are "
               "not used, so classes reflect the matrix itself.)")

    ordered = sorted(classes.items(), key=lambda kv: -kv[1]["n"])
    for (label, sense, cclass, vsig), g in ordered[:max_classes]:
        nnz, rhs = g["nnz"], g["rhs"]
        span = f"{min(nnz)}" if min(nnz) == max(nnz) else f"{min(nnz)}-{max(nnz)}"
        bits = [sense, f"coeffs {cclass}", f"{span} nonzeros/row", f"vars {vsig}"]
        if cclass in ("integer", "fractional"):
            bits.append(f"coeff range [{min(g['lo_k']):.4g}, {max(g['hi_k']):.4g}]")
        if rhs:
            bits.append(f"RHS={min(rhs):.4g}" if min(rhs) == max(rhs)
                        else f"RHS in [{min(rhs):.4g}, {max(rhs):.4g}]")
        out.append(f"### {label} x{g['n']}  [{'; '.join(bits)}]")
        pairs, s2, lo2, hi2 = g["rep"]
        shown = pairs[:max_terms]
        expr = " ".join(("+ " if k > 0 else "- ") +
                        (f"{abs(k):.6g}*" if abs(abs(k) - 1.0) > _EPS else "") + v.name
                        for v, k in shown)
        if len(pairs) > max_terms:
            expr += f" ... (+{len(pairs) - max_terms} further terms in this row)"
        if s2 == "EQ":
            rel = f"= {hi2:.6g}"
        elif s2 == "LE":
            rel = f"<= {hi2:.6g}"
        elif s2 == "GE":
            rel = f">= {lo2:.6g}"
        else:
            rel = f"in [{lo2:.6g}, {hi2:.6g}]"
        out.append(f"  [representative] {expr} {rel}")
    if len(ordered) > max_classes:
        rest = sum(g["n"] for _, g in ordered[max_classes:])
        out.append(f"### ... {len(ordered) - max_classes} further classes covering "
                   f"{rest} constraints, each with fewer rows than those above")
    out += ["", slim_objective(model)]
    return "\n".join(out)


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

## Selection principle
Your objective is SINGLE: **pick the candidate you expect to solve this instance FASTEST.**
There is no secondary objective. Maximize expected speedup over the SCIP default — an expected
8x is a better pick than an expected 2x even if the 8x is less certain. Do NOT bias toward the
candidate that stays closest to the default, and do NOT fall back on the EPM's ordering.

Procedure:
- ELIMINATE candidates whose configurations the accumulated experience records as actually
  SLOWING DOWN this problem structure (a documented dealbreaker for this family, a value known
  to blow up runtime on this structure). Exclude them because they are SLOW on this structure,
  not because they look risky.
- Among the rest, pick the one with the HIGHEST expected speedup: which changes does the
  accumulated experience show to yield large gains on instances with this structure?
- The EPM cannot rank reliably WITHIN these {k} (its ordering here is close to random). Base your
  decision on the accumulated experience and the instance structure, not on the EPM rank.
- If several candidates look equally fast, prefer the one whose changes match the type-level
  recommendations for this problem class in the accumulated experience.

## Accumulated Experience
{skill}

## Instance: 20 numerical features
{feature_text}

## Instance: formulation (compressed — each named constraint group shows ONE representative row; the rest are structurally identical up to indices, with counts and RHS/LHS ranges noted inline)
{description}

## SCIP DEFAULT configuration (reference baseline)
{default_text}

## Candidate configurations (FULL 30-parameter listing each)
{cand_block}

## Output
Return STRICT JSON only, no other text:
{{"choice": <integer 1-{k}>, "excluded": [<candidate numbers you judged clearly slower>], "reason": "<one short sentence>"}}"""


def parse_choice(response, k):
    """Extract {"choice": i} from the LLM reply; return (choice, excluded, reason) or None."""
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
    parser = argparse.ArgumentParser(description="EPM Top-k + LLM risk-averse selection")
    parser.add_argument("--config", type=str, default="config/experiment.yaml")
    parser.add_argument("--cache", type=str,
                        default="data/results/training_cache_pool1000.json",
                        help="EPM training cache. Must be collected over the same pool as "
                             "--config-pool, and must match the cache used by "
                             "epm_no_llm.py or the two arms are not comparable")
    parser.add_argument("--config-pool", type=str, default="config_pool_1000.json",
                        help="Fixed candidate pool JSON (ids 0..N-1 = position in file); "
                             "every instance scores this same pool, so ids are comparable "
                             "across instances, batches and runs")
    parser.add_argument("--output", type=str, default="data/results/epm_llm_selection_config100_results.json")
    parser.add_argument("--skill", type=str, default="skills/skill.md")
    parser.add_argument("--renderer", choices=["name", "structural"], default="name",
                        help="How the constraint matrix is rendered into the prompt. "
                             "'name' groups rows by constraint name with digits stripped, "
                             "which only compresses when the generator encoded groups into "
                             "the names; 'structural' derives the groups from each row and "
                             "works on instances with arbitrary names")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Shortlist size handed to the LLM (k-choose-1). The EPM ranks "
                             "near-randomly within its own shortlist, so a wider k raises "
                             "the ceiling the selector can actually reach; it also costs k "
                             "solves per instance in Phase C")
    parser.add_argument("--n-eval", type=int, default=None, help="Evaluate only first N instances (0 = train only)")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Concurrent LLM calls in Phase B")
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
    # Step 1: Train the 50-dim EPM (identical to epm_no_llm.py)
    # =================================================================
    print("=" * 70)
    print(f"EPM Top-{args.top_k} + LLM risk-averse selection (base: no-LLM EPM, 50-dim)")
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

    # Missing LP files are filled in by scripts/generate_ecole_instances.py, the
    # only generator that calibrates instance SIZE into the
    # [min_solve_time, max_solve_time] window. Never fall back to ProblemRegistry
    # here: it has no 'ecole' entry in scale_variants, so it would silently drop to
    # each generator's toy function defaults (e.g. independent_set n_vertices=50)
    # and mix self-generated 1ms instances into an Ecole-trained pipeline.
    _inst_cfg = config.get('instances', {}) or {}
    MIN_T = float(_inst_cfg.get('min_solve_time', 0.5))
    MAX_T = float(_inst_cfg.get('max_solve_time', 5.0))
    GEN_SCRIPT = Path(__file__).resolve().parent / 'generate_ecole_instances.py'
    MANIFEST = cache_dir / 'ecole_manifest.json'

    def lp_path(inst):
        """Resolve an instance to its LP file, or None if absent.

        Legacy compatibility: older benchmark files used a zero-padded index
        (set_covering_ecole_01.lp). Current generators write them unpadded, but
        accept the padded form so a stale file is never silently replaced by a
        freshly generated instance from a different generator.
        """
        stem = f"{inst['family_id']}_{inst['variant']}_"
        for name in (f"{stem}{inst['seed']}.lp", f"{stem}{int(inst['seed']):02d}.lp"):
            p = cache_dir / name
            if p.exists():
                return p
        return None

    def generate_missing(insts):
        """Fill in absent LP files with the calibrated Ecole generator."""
        missing = [e for e in insts if lp_path(e) is None]
        if not missing:
            print(f"[gen] all {len(insts)} eval instances present")
            return
        groups = {}
        for e in missing:
            groups.setdefault(e['variant'], {}).setdefault(e['family_id'], []).append(
                int(e['seed']))
        print(f"[gen] {len(missing)} of {len(insts)} eval instances missing, "
              f"generating with {GEN_SCRIPT.name}", flush=True)
        for variant, fams in groups.items():
            # The generator skips files that already exist, so one call per family
            # spanning min..max of its missing seeds is idempotent.
            for fam, fam_seeds in sorted(fams.items()):
                lo, hi = min(fam_seeds), max(fam_seeds)
                cmd = [sys.executable, str(GEN_SCRIPT),
                       '--families', fam, '--tag', variant,
                       '--start-index', str(lo), '-n', str(hi - lo + 1),
                       '--out-dir', str(cache_dir), '--manifest', str(MANIFEST),
                       '--min-time', str(MIN_T), '--max-time', str(MAX_T),
                       '--randomize', 'True']
                print(f"[gen] {fam}/{variant} seeds {lo}-{hi} "
                      f"({len(fam_seeds)} missing)", flush=True)
                if subprocess.run(cmd).returncode != 0:
                    sys.exit(f"ERROR: {GEN_SCRIPT.name} failed for {fam}/{variant} "
                             f"seeds {lo}-{hi}; eval aborted.")
        # The generator prints FAILED but still exits 0 when no solvable candidate
        # is found, so verify rather than trust the exit code.
        still = [f"{e['family_id']}/{e['variant']}/{e['seed']}"
                 for e in insts if lp_path(e) is None]
        if still:
            sys.exit(f"ERROR: {len(still)} instances still missing after generation: "
                     f"{still[:10]}{' ...' if len(still) > 10 else ''}. Eval aborted "
                     f"rather than falling back to the self-generators.")

    def warn_out_of_window(insts):
        """Surface instances the generator could not fit into the time window."""
        if not MANIFEST.exists():
            return
        try:
            manifest = json.load(open(MANIFEST, encoding='utf-8'))
        except Exception:
            return
        bad, unknown = [], []
        for e in insts:
            p = lp_path(e)
            entry = manifest.get(p.name) if p else None
            if entry is None:
                unknown.append(p.name if p else f"{e['family_id']}_{e['variant']}_{e['seed']}.lp")
            elif not entry.get('in_window', True):
                bad.append(f"{p.name} ({entry.get('default_time')}s)")
        if unknown:
            print(f"[warn] {len(unknown)} eval instances have no manifest entry, so "
                  f"their solve time was never verified against "
                  f"[{MIN_T}, {MAX_T}]s: {unknown[:10]}"
                  f"{' ...' if len(unknown) > 10 else ''}", flush=True)
        if bad:
            print(f"[warn] {len(bad)} eval instances are OUTSIDE the "
                  f"[{MIN_T}, {MAX_T}]s window: {bad[:10]}"
                  f"{' ...' if len(bad) > 10 else ''}", flush=True)

    generate_missing(eval_data)
    warn_out_of_window(eval_data)

    def ensure_lp(inst):
        """Return the LP path for an instance; all files exist by this point."""
        lp = lp_path(inst)
        if lp is None:
            sys.exit(f"ERROR: no LP file for {inst['family_id']}/{inst['variant']}/"
                     f"{inst['seed']}. Generate it first with {GEN_SCRIPT.name}.")
        return lp

    # FIXED candidate pool; report each candidate's own cand_id rather than its
    # position, so distilling a pool down to a subset does not renumber the ids
    # that experience rules cite.
    pool = json.load(open(args.config_pool, 'r', encoding='utf-8'))
    cands = [entry['config'] for entry in pool]
    cand_ids = [int(entry.get('cand_id', i)) for i, entry in enumerate(pool)]
    cand_vecs = np.array([encode(c) for c in cands])

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
        top5_ids = [cand_ids[j] for j in order]
        top5 = [cands[j] for j in order]
        top5_pred = [float(preds[j]) for j in order]

        feature_text = "\n".join(f"- {n}: {feats[i]:.4f}" for i, n in enumerate(FEATURE_NAMES))
        description = (slim_describe(m0) if args.renderer == "name"
                       else describe_structural(m0))
        m0.freeProb()
        cand_entries = [(cid, config_full_text(c, param_names)) for cid, c in zip(top5_ids, top5)]
        prompt = build_selection_prompt(skill, feature_text, description, default_text, cand_entries)

        pending.append({'e': e, 'key': key, 'lp_file': str(lp_file),
                        'top5': top5, 'top5_ids': top5_ids, 'top5_pred': top5_pred,
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
    need = [p for p in pending if tuple(map(str, p['key'])) not in selections]
    print(f"[Phase B] LLM selection for {len(need)}/{len(pending)} instances "
          f"({args.max_concurrent} concurrent)...")
    if need:
        responses = llm.batch_ask([p['prompt'] for p in need],
                                  max_concurrent=args.max_concurrent)
        for p, resp in zip(need, responses):
            parsed = parse_choice(resp, len(p['top5_ids']))
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

    # ── Phase C: serial SCIP evaluation (clean timing; solve ALL 5 + default) ──
    print(f"\n[Phase C] Solving all Top-{args.top_k} + default ({N_REPEATS}x each, serial)...")
    for p in pending:
        e = p['e']
        sel = selections.get(tuple(map(str, p['key'])),
                             {'choice': 1, 'excluded': [], 'reason': 'missing', 'fallback': True})
        llm_choice, llm_excluded = sel['choice'], sel['excluded']
        llm_reason, fallback = sel['reason'], sel['fallback']
        top5, top5_ids, top5_pred = p['top5'], p['top5_ids'], p['top5_pred']

        # solve default + all five candidates
        result_def, def_raw = solve_repeated(p['lp_file'], default_config)
        solved = []
        for rank, cfg in enumerate(top5):
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
            'llm_choice': llm_choice, 'llm_excluded': llm_excluded,
            'llm_reason': llm_reason, 'llm_fallback': fallback,
            'topk_cand_ids': top5_ids,
            'chosen_cand_id': top5_ids[llm_choice - 1],
            'oracle_cand_id': top5_ids[oracle_rank - 1],
            'top5_configs': top5, 'top5_pred': top5_pred,
            'topk_results': [{'rank': s['rank'], 'cand_id': top5_ids[s['rank'] - 1],
                              'pred': top5_pred[s['rank'] - 1],
                              'time': s['result'].time, 'nodes': s['result'].nodes,
                              'status': s['result'].status, 'score': s['score']} for s in solved],
            'oracle_rank': oracle_rank, 'oracle_score': oracle_score,
            'top1_score': top1_score, 'random_exp_score': random_exp_score,
            'h2h_vs_top1': h2h,
            'predicted_config': chosen, 'predicted_log_speedup': top5_pred[llm_choice - 1],
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
              f"top5_ids={top5_ids} def={result_def.time:.2f}s | {rank_str} | oracle=r{oracle_rank} "
              f"LLM=r{llm_choice}(id{top5_ids[llm_choice - 1]}){' (FALLBACK)' if fallback else ''} H2H={h2h} "
              f"speedup={speedup:.2f}x score={score:+.3f} {winner} "
              f"| 累计 {n_w}胜 {n_l}负 {n_t}平 {n_s}同", flush=True)

    # =================================================================
    # Step 4: Summary
    # =================================================================
    scores = [r['score'] for r in results if r['score'] is not None]
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY (EPM Top-{args.top_k} + LLM risk-averse selection)")
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
            print(f"  LLM hit oracle: {hit}/{len(results)} ({100*hit/len(results):.0f}%, chance=20%)")

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
