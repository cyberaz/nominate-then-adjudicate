"""
Exhaustively measure every configuration in a candidate pool on ONE instance.

Purpose: establish the true per-instance ceiling of a candidate pool, and check
whether pool distillation discarded the configs that actually win on this
instance's family. The EPM/LLM pipeline can never do better than the best config
in its pool, so this is the ceiling that bounds the whole arm.

The reference config (ISAC+ portfolio member, --isac-member) is measured LAST, in
the same session, so its rank among the pool is directly comparable.

DRIFT: this machine has been observed drifting 1.1x-4x within a day, and a full
1000-config sweep takes ~35 minutes. Speedup is defined against the default, so a
drifting default silently rescales every number after it. The default is therefore
re-measured every --recheck-every configs and the observed drift is reported; each
config's speedup uses the default interpolated from the surrounding checkpoints.

Usage:
    python scripts/probe_pool_on_instance.py                       # combinatorial_auction/102
    python scripts/probe_pool_on_instance.py --family set_covering --seed 105
    python scripts/probe_pool_on_instance.py --limit 5             # smoke test
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
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from al_param_tuning.solver import SCIPWrapper
from pyscipopt import Model


def solve_once(scip, lp, config, time_limit):
    m = Model()
    m.hideOutput()
    m.readProblem(str(lp))
    return scip.solve(m, config, time_limit=time_limit)


def main():
    ap = argparse.ArgumentParser(
        description="Measure every pool config on one instance to find the pool ceiling")
    ap.add_argument("--lp", default="",
                    help="Probe this instance file directly (.mps/.lp/.mps.gz), "
                         "bypassing --family/--seed/--variant")
    ap.add_argument("--family", default="combinatorial_auction")
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("--variant", default="ecole")
    ap.add_argument("--instance-dir", default="data/results/instances")
    ap.add_argument("--pool", default="config_pool_1000.json")
    ap.add_argument("--distilled", default="",
                    help="Optional: mark configs belonging to a DISTILLED pool, e.g. "
                         "config_100_ecole.json. Only meaningful if the eval is actually "
                         "run with --config-pool pointing at that file; "
                         "epm_with_selection.py defaults to the full config_pool_1000.json, "
                         "in which case every config is selectable and this adds nothing")
    ap.add_argument("--isac-results", default="data/results/isac_plus_k8_results.json",
                    help="Source of the reference portfolio (meta.portfolio)")
    ap.add_argument("--isac-member", type=int, default=7,
                    help="Which portfolio member to measure as the reference")
    ap.add_argument("--param-config", default="config/param_config.json")
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--default-repeats", type=int, default=5,
                    help="Repeats per default measurement (median is used)")
    ap.add_argument("--recheck-every", type=int, default=100,
                    help="Re-measure the default every N configs to track drift")
    ap.add_argument("--limit", type=int, default=None, help="Only test the first N configs")
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    if args.lp:
        src = Path(args.lp)
        if src.is_dir():
            src = src / src.name
        if src.suffix == ".gz":
            import gzip
            import shutil
            expanded = Path("data/miplib_raw") / src.name[:-3]
            expanded.parent.mkdir(parents=True, exist_ok=True)
            if not expanded.exists() or expanded.stat().st_size == 0:
                with gzip.open(src, "rb") as fi, open(expanded, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            src = expanded
        lp = src
        if not lp.exists():
            sys.exit(f"ERROR: no instance file at {lp}")
        tag = lp.stem
    else:
        lp = Path(args.instance_dir) / f"{args.family}_{args.variant}_{args.seed}.lp"
        if not lp.exists():
            sys.exit(f"ERROR: no LP file at {lp}. Generate it first with "
                     f"scripts/generate_ecole_instances.py")
        tag = f"{args.family}_{args.seed}"

    out = Path(args.output) if args.output else Path(f"data/results/pool_probe_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    param_config = json.load(open(args.param_config, encoding="utf-8"))
    scip = SCIPWrapper(param_config)
    default_config = scip.get_default_config()

    pool = json.load(open(args.pool, encoding="utf-8"))
    entries = [(int(e.get("cand_id", i)), e.get("config", e)) for i, e in enumerate(pool)]
    if args.limit:
        entries = entries[:args.limit]

    kept = set()
    if args.distilled and Path(args.distilled).exists():
        kept = {int(e["cand_id"]) for e in json.load(open(args.distilled, encoding="utf-8"))}

    state = {"meta": {}, "default_checks": [], "configs": {}, "reference": None}
    if out.exists() and not args.no_resume:
        try:
            state = json.load(open(out, encoding="utf-8"))
            print(f"Resuming: {len(state['configs'])} configs already measured")
        except Exception as e:
            print(f"WARNING: could not read {out} ({e}); starting fresh")

    def measure_default(tag):
        ts = [solve_once(scip, lp, default_config, args.time_limit).time
              for _ in range(args.default_repeats)]
        med = float(np.median(ts))
        state["default_checks"].append(dict(
            after_n_configs=len(state["configs"]), tag=tag,
            median=med, times=ts, at=datetime.now().strftime("%H:%M:%S")))
        print(f"  [default @{tag}] median {med:.3f}s  (n={len(ts)}, "
              f"min {min(ts):.3f} max {max(ts):.3f})", flush=True)
        return med

    print("=" * 72)
    print(f"POOL PROBE  {tag}")
    print(f"  instance : {lp.name}")
    print(f"  pool     : {args.pool} ({len(entries)} configs)")
    print(f"  reference: {args.isac_results} member {args.isac_member}")
    print(f"  budget   : {len(entries) + 1} config solves + "
          f"{args.default_repeats * (1 + len(entries) // args.recheck_every + 1)} default solves")
    print(f"  KEPT     : config survived distillation into {args.distilled or '(none)'}, "
          f"i.e. the pipeline can actually select it")
    print("=" * 72, flush=True)

    if not state["default_checks"]:
        measure_default("start")

    t0 = time.perf_counter()
    for n, (cid, cfg) in enumerate(entries):
        if str(cid) in state["configs"]:
            continue
        if n and n % args.recheck_every == 0:
            measure_default(f"n={n}")
        r = solve_once(scip, lp, cfg, args.time_limit)
        state["configs"][str(cid)] = dict(
            cand_id=cid, time=r.time, nodes=r.nodes, status=r.status,
            objective=r.objective, gap=r.gap, has_solution=r.has_solution,
            in_distilled_100=cid in kept, order=n,
            measured_after_checks=len(state["default_checks"]) - 1)
        done = len(state["configs"])
        el = time.perf_counter() - t0
        rate = el / max(done, 1)
        # Live speedup against the latest checkpoint; the final report recomputes
        # it against the drift-interpolated local default.
        live_def = state["default_checks"][-1]["median"]
        print(f"  [{done:4d}/{len(entries)}] cand {cid:4d} "
              f"{r.time:7.3f}s {live_def / max(r.time, 1e-3):6.2f}x "
              f"nodes={r.nodes:<7d} {r.status:<10s} "
              f"{'KEPT' if cid in kept else '    '} "
              f"| {el / 60:5.1f}min, ~{rate * (len(entries) - done) / 60:5.1f}min left",
              flush=True)
        if done % 25 == 0 or n == len(entries) - 1:
            json.dump(state, open(out, "w", encoding="utf-8"), indent=1)

    measure_default("end")

    # Reference config LAST, same session
    if args.isac_results:
        isac = json.load(open(args.isac_results, encoding="utf-8"))
        ref_cfg = isac["meta"]["portfolio"][args.isac_member]
        r = solve_once(scip, lp, ref_cfg, args.time_limit)
        state["reference"] = dict(
            source=args.isac_results, member=args.isac_member, config=ref_cfg,
            time=r.time, nodes=r.nodes, status=r.status, objective=r.objective)
        print(f"\n  [reference] ISAC+ member {args.isac_member}: {r.time:.3f}s ({r.status})")

    checks = [c["median"] for c in state["default_checks"]]
    dref = float(np.median(checks))
    # Drift correction: the default is re-measured periodically, so estimate the
    # default in force at each config's sweep position by interpolating between
    # checkpoints. Using one global median instead would rescale every config
    # measured while the machine was fast/slow and can reorder the ranking.
    cx = [c["after_n_configs"] for c in state["default_checks"]]
    cy = checks
    local_default = (lambda pos: float(np.interp(pos, cx, cy))) if len(cx) > 1 \
        else (lambda pos: dref)
    for r_ in state["configs"].values():
        r_["local_default"] = local_default(r_.get("order", 0))
        r_["speedup"] = r_["local_default"] / max(r_["time"], 1e-3)
    if state.get("reference"):
        state["reference"]["local_default"] = cy[-1]
        state["reference"]["speedup"] = cy[-1] / max(state["reference"]["time"], 1e-3)

    state["meta"] = dict(
        family=args.family, variant=args.variant, seed=args.seed, lp=str(lp), tag=tag,
        pool=args.pool, n_configs=len(state["configs"]), time_limit=args.time_limit,
        default_median_overall=dref, default_checks_span=[min(checks), max(checks)],
        drift_ratio=max(checks) / max(min(checks), 1e-9),
        finished=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(state, open(out, "w", encoding="utf-8"), indent=1)

    # ── report ──
    rows = sorted(state["configs"].values(), key=lambda r: -r["speedup"])
    print("\n" + "=" * 72)
    print(f"RESULTS  {args.family}/{args.seed}")
    print(f"  default: {len(checks)} checkpoints, median {dref:.3f}s, "
          f"range {min(checks):.3f}-{max(checks):.3f}s, "
          f"drift {state['meta']['drift_ratio']:.2f}x")
    print(f"  speedups below are drift-corrected against the interpolated local default")
    print("=" * 72)
    print(f"\nTop 15 of {len(rows)} pool configs:")
    print(f"  {'rank':>4s} {'cand_id':>8s} {'time':>8s} {'speedup':>8s} {'nodes':>8s} "
          f"{'status':>10s} {'in top100':>10s}")
    for i, r_ in enumerate(rows[:15]):
        print(f"  {i + 1:4d} {r_['cand_id']:8d} {r_['time']:8.3f} {r_['speedup']:7.2f}x "
              f"{r_['nodes']:8d} {r_['status']:>10s} {str(r_['in_distilled_100']):>10s}")
    print(f"\nWorst 5:")
    for i, r_ in enumerate(rows[-5:], start=len(rows) - 4):
        print(f"  {i:4d} {r_['cand_id']:8d} {r_['time']:8.3f} {r_['speedup']:7.2f}x "
              f"{r_['nodes']:8d} {r_['status']:>10s} {str(r_['in_distilled_100']):>10s}")

    spa = np.array([r_["speedup"] for r_ in rows])
    print(f"\nDistribution of pool speedups:")
    for q in (100, 99, 95, 75, 50, 25, 0):
        print(f"  p{q:<3d} speedup {np.percentile(spa, q):6.2f}x")
    print(f"  configs beating default : "
          f"{int((spa > 1).sum())}/{len(spa)} ({100 * (spa > 1).mean():.1f}%)")
    print(f"  configs >=2x            : {int((spa >= 2).sum())}/{len(spa)}")
    print(f"  configs hitting the {args.time_limit:.0f}s limit: "
          f"{sum(1 for r_ in rows if r_['status'] == 'timelimit')}")

    ref = state.get("reference")
    if ref:
        better = int((spa > ref["speedup"]).sum())
        print(f"\nISAC+ member {args.isac_member}: {ref['time']:.3f}s -> "
              f"{ref['speedup']:.2f}x (vs local default {ref['local_default']:.3f}s)")
        print(f"  rank among the {len(spa)} pool configs: {better + 1}")
        print(f"  pool configs strictly faster: {better} ({100 * better / len(spa):.1f}%)")

    kept_rows = [r_ for r_ in rows if r_["in_distilled_100"]]
    if kept_rows:
        best_kept = kept_rows[0]
        print(f"\nDistillation check ({len(kept_rows)} of the distilled-100 are in this pool):")
        print(f"  best config overall   : cand {rows[0]['cand_id']} "
              f"{rows[0]['speedup']:.2f}x "
              f"{'[KEPT]' if rows[0]['in_distilled_100'] else '[DISCARDED]'}")
        print(f"  best SURVIVING config : cand {best_kept['cand_id']} "
              f"{best_kept['speedup']:.2f}x, pool rank {rows.index(best_kept) + 1}")
        lost = rows[0]["speedup"] - best_kept["speedup"]
        if lost > 0.01:
            print(f"  -> distillation cost {lost:.2f}x of speedup on this instance")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
