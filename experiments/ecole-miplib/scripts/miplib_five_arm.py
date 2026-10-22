"""
Five-arm comparison on MIPLIB instances the nominator's training distribution covers,
scored on the optimality gap under a short cap.

Why gap and not time. On MIPLIB almost nothing finishes in the window we calibrated the
generated families to, so time-to-optimality is undefined for most instances and the
lexicographic score collapses every arm to -1. The gap remaining when the cap expires is
defined for every run and is the metric the cutting-plane configuration literature uses
in this regime. Arms are still compared per instance against a default measured in the
same session.

Comparison rule, applied per instance and per arm against the shared default:
  tier 0  proved optimality        -> compare solve time
  tier 1  feasible, finite gap     -> compare gap
  tier 2  no feasible solution     -> compare dual bound if available, else tie
A lower tier always beats a higher one, so proving optimality beats any gap and finding
a solution beats finding none.

Instance set. Selected by whether all 20 features fall inside the ranges of the EPM's
training cache, which is checkable before any solve and does not look at outcomes. With
the 16-family cache 26 of 213 instances qualify; with the 4-family cache only 1 does.

Config pool and skill memory. The skill was iterated against config_100.json and its rules
cite candidates by POSITION in that file (0-99), not by the `cand_id` field it also carries,
which holds ids from the larger pool it was distilled from. This script therefore presents
positions as the config ids, so the memory's numeric rules resolve to the configurations
they were written about.

Usage:
    python scripts/miplib_five_arm.py
    python scripts/miplib_five_arm.py --time-limit 60 --repeats 2
    python scripts/miplib_five_arm.py --names-file data/results/miplib_cands16.json --no-llm
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import gzip
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from al_param_tuning.features import InstanceFeatureExtractor
from al_param_tuning.solver import SCIPWrapper
from epm_with_selection import (FEATURE_NAMES, config_full_text, describe_structural,
                                make_encoder, parse_choice)
from isac_roar_baseline import extract_features, load_instances

ARMS = ["ours", "isacplus", "smac", "roar"]


def build_generalization_prompt(skill, feature_text, description, default_text,
                                cand_entries, time_limit):
    """Selection prompt for an instance outside every family the memory describes.

    Differs from the in-distribution prompt in two ways that matter. The instance belongs
    to none of the memory's classes, so the model is asked to first decide which classes
    the derived structure resembles and then transfer those rules rather than look up a
    class by name. And the objective is the gap left at the cap rather than raw speedup,
    because most instances here will not finish.
    """
    k = len(cand_entries)
    cand_block = "\n\n".join(
        f"### Candidate {i+1} (EPM rank {i+1}, CONFIG ID {cid})\n{t}"
        for i, (cid, t) in enumerate(cand_entries))
    id_list = ", ".join(str(cid) for cid, _ in cand_entries)
    return f"""You are an expert in SCIP solver parameter tuning.

## Task
A machine-learning model (EPM) nominated {k} candidate parameter configurations for the MIP
instance below. The candidates come from the SAME fixed numbered pool the accumulated
experience was written against; their CONFIG IDS are: {id_list}. When the experience names a
config id, it means that same configuration — match ids directly.

Select EXACTLY ONE of the {k} candidates.

## What is different about this instance
This instance is NOT drawn from any of the problem classes the experience describes. It comes
from a public library of real-world models, so there is no class label to look up and no
guarantee that any class in the experience matches it. You must GENERALIZE.

Work in this order:
1. Read the derived structural classes in the formulation below. They are computed from the
   matrix itself (constraint sense, coefficient pattern, row length, variable types), not
   from any name or metadata, so they are trustworthy evidence about what this model IS.
2. Decide which of the experience's problem classes this structure most resembles, and say
   which in your reason. Examples of the mapping you should be making: rows that are
   set partitioning / packing / covering over binaries resemble set_covering,
   independent_set or crew_scheduling; big-M variable bounds linking a binary to a
   continuous variable resemble facility_location or network_design; flow conservation
   rows resemble network_design or lot_sizing; knapsack rows over binaries resemble
   knapsack or multidim_knapsack. A single instance may mix several.
3. Transfer the reasoning those classes record. Prefer rules stated at the PARAMETER level
   (which parameter values help or hurt on which structure), because those generalize across
   classes. Treat class-specific config-id preferences as weaker evidence here, since this
   instance is not in that class.
4. Apply the experience's veto rules whenever the structural condition they name is present.

## Selection objective
The solver runs with a {time_limit:.0f} second limit, and most instances of this kind will NOT
finish. Your objective, in order:
  (a) if you judge the instance can be closed within the limit, pick the candidate most
      likely to PROVE OPTIMALITY inside it;
  (b) otherwise pick the candidate expected to leave the SMALLEST optimality gap at the
      limit, and never pick one at risk of finding NO feasible solution at all.
Configurations that disable presolve, cut generation or primal heuristics are especially
dangerous under (b): they often destroy the primal side and leave an enormous or infinite
gap. Weigh that risk explicitly.
The EPM cannot rank reliably WITHIN these {k}; do not defer to its ordering.

## Accumulated Experience (16 problem classes)
{skill}

## Instance: 20 numerical features
{feature_text}

## Instance: formulation (compressed; rows grouped by structure derived from the matrix)
{description}

## SCIP DEFAULT configuration (reference baseline)
{default_text}

## Candidate configurations (FULL 30-parameter listing each)
{cand_block}

## Output
Return STRICT JSON only, no other text:
{{"choice": <integer 1-{k}>, "excluded": [<candidate numbers you judged clearly worse>], "reason": "<one sentence naming the classes you generalized from>"}}"""


def tier_of(res):
    """0 proved optimal, 1 feasible with a finite gap, 2 no feasible solution."""
    if res["status"] == "optimal":
        return 0
    if res["has_solution"] and res["gap"] is not None and np.isfinite(res["gap"]):
        return 1
    return 2


def verdict(arm, dfl):
    """Lexicographic comparison of one arm against the default on the same instance."""
    ta, td = tier_of(arm), tier_of(dfl)
    if ta != td:
        return "better" if ta < td else "worse"
    if ta == 0:
        if arm["time"] < 0.8 * dfl["time"]:
            return "better"
        if arm["time"] > 1.25 * dfl["time"]:
            return "worse"
        return "tie"
    if ta == 1:
        if arm["gap"] < 0.95 * dfl["gap"]:
            return "better"
        if arm["gap"] > 1.05 * dfl["gap"]:
            return "worse"
        return "tie"
    return "tie"


def resolve(spec, tmp):
    """Expand .gz and the directory-wrapped entries into a plain file SCIP can read."""
    p = Path(spec)
    if p.is_dir():
        p = p / p.name
    if p.suffix == ".gz":
        out = tmp / p.name[:-3]
        if not out.exists() or out.stat().st_size == 0:
            with gzip.open(p, "rb") as fi, open(out, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        return out
    return p


def main():
    ap = argparse.ArgumentParser(description="Five-arm MIPLIB comparison scored on gap")
    ap.add_argument("--dir", default="D:/MIPLIB_benchmark")
    ap.add_argument("--names-file", default="data/results/miplib_cands16.json")
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--cache", default="data/results/training_cache_multi.json",
                    help="EPM training cache. The 16-family cache leaves only 1 of 20 "
                         "features constant, against 7 for the 4-family cache, which is "
                         "why far more MIPLIB instances fall inside its ranges")
    ap.add_argument("--config-pool", default="config_100.json",
                    help="Pool the skill memory was written against")
    ap.add_argument("--skill", default="skills/skill_multi16.md")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--only", default="",
                    help="Comma-separated instance names to restrict the run to")
    ap.add_argument("--target-gap", type=float, default=0.0,
                    help="If > 0, measure TIME TO REACH this relative gap instead of the "
                         "gap left at the cap; --time-limit then acts as a safety cap and "
                         "a run that hits it is reported as censored")
    ap.add_argument("--reuse-choices", default="",
                    help="Previous results file to read each instance's adjudicated choice "
                         "from, so a re-measurement costs no model calls")
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--roar-model", default="data/results/isac_roar_model.json")
    ap.add_argument("--smac-model", default="data/results/isac_smac_model.json")
    ap.add_argument("--isacplus-matrix", default="data/results/isac_plus_matrix_k8.json")
    ap.add_argument("--isacplus-result", default="data/results/FINAL_isac_plus.json")
    ap.add_argument("--no-llm", action="store_true",
                    help="Use the top nominee instead of calling the model")
    ap.add_argument("--output", default="data/results/miplib_five_arm.json")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    from pyscipopt import Model
    tmp = project_root / "data" / "miplib_raw"
    tmp.mkdir(parents=True, exist_ok=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    names = json.load(open(args.names_file, encoding="utf-8"))
    if args.only:
        want = [x.strip() for x in args.only.split(",") if x.strip()]
        missing = [w for w in want if w not in names]
        if missing:
            sys.exit(f"ERROR: --only names not in {args.names_file}: {missing}")
        names = want
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    param_config = json.load(open(config["paths"]["param_config"], encoding="utf-8"))
    parameters = param_config["parameters"]
    param_names = [p["name"] for p in parameters]
    encode = make_encoder(parameters)
    scip = SCIPWrapper(param_config)
    default_config = scip.get_default_config()
    fx = InstanceFeatureExtractor()

    print("=" * 96)
    print("MIPLIB FIVE-ARM COMPARISON, SCORED ON GAP")
    print(f"  instances : {len(names)} (all 20 features inside the training ranges)")
    print(f"  cap       : {args.time_limit:.0f}s x {args.repeats} repeat(s) x 5 arms")
    print(f"  pool      : {args.config_pool} (ids = POSITION, matching the skill memory)")
    print(f"  cache     : {args.cache}")
    print(f"  skill     : {args.skill}")
    print(f"  worst case: {len(names) * 5 * args.time_limit * args.repeats / 60:.0f} min "
          f"of solving")
    print("=" * 96, flush=True)

    # ── EPM over the same pool the memory cites ──────────────────────────────
    cache = json.load(open(args.cache, encoding="utf-8"))
    Xtr_f = np.asarray([r[:20] for r in cache["X_all"]], float)
    lo, hi = Xtr_f.min(0), Xtr_f.max(0)
    X, y = [], []
    for i in range(len(cache["X_all"])):
        rec = cache["raw_data_all"][i]
        dt = rec.get("default_time", 0)
        if dt <= 0:
            continue
        f20 = np.asarray(cache["X_all"][i][:20], float)
        for cfg in rec.get("configs", []):
            t = cfg.get("time", 0)
            if t > 0:
                X.append(np.concatenate([f20, encode(cfg["values"])]))
                y.append(np.log(dt / t))
    from xgboost import XGBRegressor
    epm = XGBRegressor(n_estimators=400, max_depth=7, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1,
                       reg_lambda=1.0, random_state=42)
    epm.fit(np.asarray(X), np.asarray(y))
    print(f"[epm] trained on {len(X)} rows from {len(cache['X_all'])} instances")

    pool = json.load(open(args.config_pool, encoding="utf-8"))
    cands = [e["config"] for e in pool]
    cand_ids = list(range(len(pool)))          # POSITION, matching the memory
    cand_vecs = np.asarray([encode(c) for c in cands])
    skill = Path(args.skill).read_text(encoding="utf-8")
    default_text = config_full_text(default_config, param_names)

    # ── cluster-level arms, inference only ───────────────────────────────────
    cluster_models = {}
    for tag, path in (("smac", args.smac_model), ("roar", args.roar_model)):
        m = json.load(open(path, encoding="utf-8"))
        key = max(m["per_k"], key=lambda k: m["per_k"][k]["val_score"])
        cluster_models[tag] = dict(
            mu=np.asarray(m["mu"], float), sd=np.asarray(m["sd"], float),
            centroids=np.asarray(m["per_k"][key]["centroids"], float),
            configs={int(c): cfg for c, cfg in m["per_k"][key]["configs"].items()}, k=key)
        print(f"[{tag}] k={key}, {len(cluster_models[tag]['configs'])} clusters")

    portfolio = json.load(open(args.isacplus_result, encoding="utf-8"))["meta"]["portfolio"]
    cost = json.load(open(args.isacplus_matrix, encoding="utf-8"))["cost"]
    K = len(portfolio)
    train = load_instances(project_root / "data" / "results" / "instances",
                           config["evaluation"]["families"], "ecole_train", range(1, 121))
    Xtr = extract_features(train, fx)
    rws, ytr = [], []
    for i, inst in enumerate(train):
        keys = [f"{Path(inst['lp']).name}::{j}" for j in range(K)]
        if all(kk in cost for kk in keys):
            rws.append(Xtr[i])
            ytr.append(int(np.argmin([cost[kk] for kk in keys])))
    from xgboost import XGBClassifier
    present = sorted(set(ytr))
    remap = {c: i for i, c in enumerate(present)}
    inv = {i: c for c, i in remap.items()}
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1,
                        reg_lambda=1.0, random_state=42, objective="multi:softprob",
                        num_class=len(present))
    clf.fit(np.asarray(rws), np.asarray([remap[v] for v in ytr]))
    print(f"[isacplus] {K}-member portfolio, classifier refit on {len(rws)} instances")

    done = {}
    if out.exists() and not args.no_resume:
        try:
            done = {r["name"]: r for r in json.load(open(out, encoding="utf-8"))["results"]}
            print(f"[resume] {len(done)} instances already measured")
        except Exception as e:
            print(f"[resume] could not read {out} ({e}); starting fresh")

    # ── Phase A: features, nomination, prompts ───────────────────────────────
    print("\n[Phase A] features, EPM nomination, structural rendering", flush=True)
    pending = []
    for name in names:
        if name in done:
            continue
        lp = resolve(Path(args.dir) / f"{name}.mps.gz", tmp)
        try:
            m0 = Model()
            m0.hideOutput()
            m0.readProblem(str(lp))
            feats = np.asarray(fx.extract(m0), float)
            n_oob = int(sum(1 for i in range(20) if feats[i] < lo[i] or feats[i] > hi[i]))
            preds = epm.predict(np.hstack([np.tile(feats, (len(cands), 1)), cand_vecs]))
            order = np.argsort(-preds)[:args.top_k]
            top_ids = [cand_ids[j] for j in order]
            top_cfgs = [cands[j] for j in order]
            desc = describe_structural(m0)
            feature_text = "\n".join(f"- {n}: {feats[i]:.4f}"
                                     for i, n in enumerate(FEATURE_NAMES))
            prompt = build_generalization_prompt(
                skill, feature_text, desc, default_text,
                [(cid, config_full_text(c, param_names))
                 for cid, c in zip(top_ids, top_cfgs)], args.time_limit)
            m0.freeProb()
            pending.append(dict(name=name, lp=str(lp), features=feats.tolist(),
                                n_out_of_range=n_oob, top_ids=top_ids, top_cfgs=top_cfgs,
                                top_pred=[float(preds[j]) for j in order],
                                describe_classes=desc.count("### "),
                                prompt_chars=len(prompt), prompt=prompt))
            print(f"  {name:<26} oob={n_oob}/20  classes={desc.count('### '):<3} "
                  f"prompt={len(prompt):>7,}ch  top-{args.top_k}={top_ids}", flush=True)
        except Exception as ex:
            print(f"  {name:<26} SKIPPED: {type(ex).__name__}: {ex}", flush=True)

    # ── Phase B: concurrent model calls ──────────────────────────────────────
    if args.reuse_choices:
        prev = json.load(open(args.reuse_choices, encoding="utf-8"))
        prior = {r["name"]: r for r in prev["results"]}
        for p in pending:
            q = prior.get(p["name"])
            if q is None or q.get("choice") is None:
                sys.exit(f"ERROR: {args.reuse_choices} has no recorded choice for "
                         f"{p['name']}; rerun without --reuse-choices")
            p["choice"], p["reason"] = int(q["choice"]), q.get("reason", "")
            p["fallback"] = bool(q.get("fallback", False))
        print(f"\n[Phase B] reused {len(pending)} adjudicated choices from "
              f"{args.reuse_choices}; no model calls")
    elif args.no_llm:
        for p in pending:
            p["choice"], p["reason"], p["fallback"] = 1, "no-llm flag", True
        print("\n[Phase B] skipped (--no-llm); using the top nominee")
    else:
        print(f"\n[Phase B] {len(pending)} model calls, {args.max_concurrent} concurrent",
              flush=True)
        from al_param_tuning.llm.client import LLMClient
        lc = config["llm"]
        llm = LLMClient(provider=lc["provider"], model=lc["model"], base_url=lc.get('base_url'),
                        api_key=lc.get('api_key_env'), temperature=lc["temperature"],
                        max_tokens=lc["max_tokens"])
        t0 = time.time()
        responses = llm.batch_ask([p["prompt"] for p in pending],
                                  max_concurrent=args.max_concurrent) if pending else []
        print(f"  {len(pending)} calls in {time.time() - t0:.0f}s")
        for p, resp in zip(pending, responses):
            parsed = parse_choice(resp, len(p["top_ids"]))
            if parsed is None:
                p["choice"], p["reason"], p["fallback"] = 1, "parse failure", True
            else:
                p["choice"], _, p["reason"] = parsed
                p["fallback"] = False
            print(f"  {p['name']:<26} -> #{p['choice']} "
                  f"(cfg {p['top_ids'][p['choice']-1]})"
                  f"{'  [fallback]' if p['fallback'] else ''}  {str(p['reason'])[:70]}")

    # ── Phase C: solve five arms per instance ────────────────────────────────
    if args.target_gap > 0:
        print(f"\n[Phase C] solving to gap <= {args.target_gap:g}, "
              f"safety cap {args.time_limit:.0f}s\n", flush=True)
    else:
        print(f"\n[Phase C] solving, {args.time_limit:.0f}s cap\n", flush=True)
    t_start = time.time()
    for n, p in enumerate(pending, 1):
        feats = np.asarray(p["features"], float)
        cfgs = {"default": default_config, "ours": p["top_cfgs"][p["choice"] - 1]}
        notes = {"ours": f"cfg {p['top_ids'][p['choice']-1]}"
                         f"{' (fallback)' if p['fallback'] else ''}"}
        for tag in ("smac", "roar"):
            cm = cluster_models[tag]
            z = (feats - cm["mu"]) / cm["sd"]
            cid = int(np.argmin(((z[None, :] - cm["centroids"]) ** 2).sum(-1)))
            cfgs[tag] = cm["configs"][cid]
            notes[tag] = f"k={cm['k']} cluster {cid}"
        j = inv[int(clf.predict(feats.reshape(1, -1))[0])]
        cfgs["isacplus"] = portfolio[j]
        notes["isacplus"] = f"member {j}/{K}"

        rec = dict(name=p["name"], features=p["features"],
                   n_out_of_range=p["n_out_of_range"],
                   describe_classes=p["describe_classes"],
                   prompt_chars=p["prompt_chars"], top_ids=p["top_ids"],
                   top_pred=p["top_pred"], choice=p["choice"], reason=p["reason"],
                   fallback=p["fallback"], notes=notes, arms={})
        for tag in ["default"] + ARMS:
            times, last = [], None
            for _ in range(args.repeats):
                mm = Model()
                mm.hideOutput()
                mm.readProblem(p["lp"])
                if args.target_gap > 0:
                    # limits/gap survives freeTransform and is not one of the tuned 30
                    mm.setRealParam("limits/gap", args.target_gap)
                r = scip.solve(mm, cfgs[tag], time_limit=args.time_limit)
                times.append(r.time)
                last = r
                mm.freeProb()
            g = last.gap
            rec["arms"][tag] = dict(
                time=float(np.median(times)), times=[round(t, 3) for t in times],
                status=last.status, nodes=last.nodes, objective=last.objective,
                gap=(float(g) if g is not None and np.isfinite(g) else None),
                has_solution=bool(last.has_solution))
            if args.target_gap > 0:
                reached = g is not None and np.isfinite(g) and g <= args.target_gap + 1e-9
                rec["arms"][tag]["reached_target"] = bool(reached)
                rec["arms"][tag]["censored"] = bool(
                    not reached and float(np.median(times)) >= args.time_limit - 5)
        dfl = rec["arms"]["default"]
        for tag in ARMS:
            rec["arms"][tag]["verdict"] = verdict(rec["arms"][tag], dfl)

        gtxt = lambda a: ("opt" if a["status"] == "optimal"
                          else (f"{a['gap']:.3f}" if a["gap"] is not None else "nosol"))
        if args.target_gap > 0:
            ttxt = lambda a: f"{a['time']:.0f}s{'*' if a.get('censored') else ''}"
            print(f"  [{n}/{len(pending)}] {p['name']:<24} "
                  f"def={ttxt(dfl):<9} " +
                  "  ".join(f"{t}={ttxt(rec['arms'][t]):<9}"
                            f"({dfl['time']/max(rec['arms'][t]['time'],1e-9):.2f}x)"
                            for t in ARMS) +
                  f"   [{(time.time()-t_start)/60:.0f}min]", flush=True)
        else:
            print(f"  [{n}/{len(pending)}] {p['name']:<24} "
                  f"def={dfl['time']:>6.1f}s/{gtxt(dfl):<6} " +
                  "  ".join(f"{t}={gtxt(rec['arms'][t]):<6}"
                            f"{rec['arms'][t]['verdict'][0].upper()}" for t in ARMS) +
                  f"   [{(time.time()-t_start)/60:.0f}min]", flush=True)

        done[p["name"]] = rec
        json.dump(dict(meta=dict(
            names_file=args.names_file, cache=args.cache, pool=args.config_pool,
            skill=args.skill, top_k=args.top_k, time_limit=args.time_limit,
            target_gap=args.target_gap,
            repeats=args.repeats, id_convention="pool position",
            at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            results=list(done.values())), open(out, "w", encoding="utf-8"), indent=1)

    # ── summary ──────────────────────────────────────────────────────────────
    rows = list(done.values())
    print("\n" + "=" * 96)
    print(f"{len(rows)} instances, {args.time_limit:.0f}s cap")
    print(f"\n{'arm':<10}{'better':>8}{'tie':>6}{'worse':>7}{'optimal':>9}"
          f"{'nosol':>7}{'mean gap':>10}")
    print("-" * 60)
    for tag in ["default"] + ARMS:
        a = [r["arms"][tag] for r in rows]
        v = [x.get("verdict", "-") for x in a]
        gaps = [x["gap"] for x in a if x["gap"] is not None]
        print(f"{tag:<10}{v.count('better'):>8}{v.count('tie'):>6}{v.count('worse'):>7}"
              f"{sum(1 for x in a if x['status'] == 'optimal'):>9}"
              f"{sum(1 for x in a if not x['has_solution']):>7}"
              f"{(np.mean(gaps) if gaps else float('nan')):>10.4f}")
    print("\ninstances where an arm beat the default:")
    for r in rows:
        w = [t for t in ARMS if r["arms"][t].get("verdict") == "better"]
        if w:
            d = r["arms"]["default"]
            print(f"  {r['name']:<24} default {d['time']:.1f}s "
                  f"gap={d['gap'] if d['gap'] is not None else 'nosol'} | " +
                  ", ".join(f"{t}({r['arms'][t]['gap'] if r['arms'][t]['gap'] is not None else r['arms'][t]['status']})"
                            for t in w))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
