"""
Re-measure several arms against one shared default, in one session.

The arms' configuration choices are already fixed (read from their result files), so
nothing is decided here -- this only re-measures. The point is to remove every
confound that separates independent runs of each arm:

  - Each instance gets ONE default measurement, shared by all arms, so a score
    difference can no longer come from two arms having measured the default at
    different moments. On facility_location the same instances were measured at
    default ratios of 0.98x-1.50x across runs, which is the size of the effect.
  - default and every arm are solved ROUND-ROBIN rather than in blocks, so drift
    inside an instance's solves is shared instead of accumulating against whichever
    arm ran last.
  - --repeats measurements per arm, reported as the median, so one outlier solve
    cannot decide an instance.

Arms are named on the command line, each pointing at a result file that carries a
`predicted_config` per (family_id, seed). An arm may instead be written
`NAME=PATH::top1` to re-measure the EPM rank-1 nominee (the "no-LLM" arm) read from that
file's shortlist rather than its own choice. Any arm whose file is missing an instance is
dropped from that instance; only instances present in ALL arms are measured, so every
comparison is fully paired.

Usage:
    python scripts/head2head_final.py                      # the five default arms
    python scripts/head2head_final.py --limit 5            # smoke test
    python scripts/head2head_final.py \
        --arm llm=data/results/FINAL_llm_v3.json \
        --arm top1=data/results/FINAL_llm_v3.json::top1 \
        --arm isacplus=data/results/FINAL_isac_plus.json \
        --reference llm
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from al_param_tuning.solver import SCIPWrapper
from probe_pool_on_instance import solve_once

DEFAULT_ARMS = [
    "llm=data/results/FINAL_llm_v3.json",
    "top1=data/results/FINAL_llm_v3.json::top1",
    "isacplus=data/results/FINAL_isac_plus.json",
    "roar=data/results/FINAL_isac_roar.json",
    "smac=data/results/FINAL_isac_smac.json",
]


def sgm(x, s=1.0):
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.mean(np.log(x + s))) - s)


def score(t, dt):
    return float(np.clip((dt - t) / (dt + t + 1e-10), -1.0, 1.0))


def _epm_top1_config(r):
    """The EPM rank-1 (no-LLM) configuration from a shortlist record.

    The shortlist is stored as parallel lists `topk_cand_ids` and `top5_configs`
    aligned by EPM rank, with `topk_results` carrying each candidate's rank. We map
    the rank-1 cand_id back to its full configuration; rank-1 is index 0 on every
    record, so the lookup degrades gracefully to index 0.
    """
    ids = r.get("topk_cand_ids")
    cfgs = r.get("top5_configs")
    if not ids or not cfgs or len(ids) != len(cfgs):
        return None
    idx = 0
    tr = r.get("topk_results")
    if tr:
        rank1 = min(tr, key=lambda e: e.get("rank", 10**9))
        cid = rank1.get("cand_id")
        if cid in ids:
            idx = ids.index(cid)
    return cfgs[idx]


def load_arm(path, pick="chosen"):
    """Return {(family_id, seed): config} from any of our result formats.

    pick="chosen" reads each arm's own decision (`predicted_config`); pick="top1"
    reads the EPM rank-1 nominee instead -- the no-LLM arm that takes the top of the
    same shortlist without the verifier.
    """
    raw = json.load(open(path, encoding="utf-8"))
    rows = raw["results"] if isinstance(raw, dict) else raw
    out = {}
    for r in rows:
        cfg = _epm_top1_config(r) if pick == "top1" else r.get("predicted_config")
        if cfg:
            out[(r["family_id"], int(r["seed"]))] = cfg
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Re-measure several arms against one shared default, one session")
    ap.add_argument("--arm", action="append", default=None, metavar="NAME=PATH",
                    help="Arm to measure; repeatable. Defaults to llm / isacplus / "
                         "roar / smac")
    ap.add_argument("--reference", default="llm",
                    help="Arm that the pairwise comparisons are taken against")
    ap.add_argument("--instance-dir", default="data/results/instances")
    ap.add_argument("--variant", default="ecole")
    ap.add_argument("--param-config", default="config/param_config.json")
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--repeats", type=int, default=3, help="Repeats per arm per instance")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N instances")
    ap.add_argument("--output", default="data/results/FINAL_head2head_5arm.json")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    spec = args.arm if args.arm else DEFAULT_ARMS
    arms, paths = {}, {}
    for s in spec:
        if "=" not in s:
            sys.exit(f"ERROR: --arm expects NAME=PATH[::MODE], got {s!r}")
        name, rhs = s.split("=", 1)
        path, pick = rhs, "chosen"
        if "::" in rhs:
            path, pick = rhs.rsplit("::", 1)
            if pick not in ("chosen", "top1"):
                sys.exit(f"ERROR: unknown pick mode {pick!r} in {s!r}; "
                         f"use '::top1' for the EPM rank-1 (no-LLM) arm")
        if not Path(path).exists():
            sys.exit(f"ERROR: arm {name!r} points at {path}, which does not exist.\n"
                     f"Produce it first, e.g.\n"
                     f"  python scripts/isac_roar_baseline.py --load-model "
                     f"--test-seed-start 301 --test-n 100 --output {path}")
        loaded = load_arm(path, pick)
        if not loaded:
            sys.exit(f"ERROR: arm {name!r} (pick={pick}) extracted no configs from {path}")
        arms[name] = loaded
        paths[name] = rhs
    if args.reference not in arms:
        sys.exit(f"ERROR: --reference {args.reference!r} is not one of {list(arms)}")
    names = list(arms)

    # Fully paired: keep only instances every arm has a configuration for.
    keys = sorted(set.intersection(*(set(a) for a in arms.values())))
    for n in names:
        drop = len(arms[n]) - len(keys)
        if drop:
            print(f"note: arm {n} has {drop} instance(s) outside the shared set")
    if not keys:
        sys.exit("ERROR: the arms share no instances")
    if args.limit:
        keys = keys[:args.limit]

    inst_dir = Path(args.instance_dir)
    missing = [f"{f}_{args.variant}_{s}.lp" for f, s in keys
               if not (inst_dir / f"{f}_{args.variant}_{s}.lp").exists()]
    if missing:
        sys.exit(f"ERROR: {len(missing)} LP file(s) missing, e.g. {missing[:3]}")

    param_config = json.load(open(args.param_config, encoding="utf-8"))
    scip = SCIPWrapper(param_config)
    default_config = scip.get_default_config()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists() and not args.no_resume:
        try:
            prev = json.load(open(out, encoding="utf-8"))
            pm = prev.get("meta") or {}
            # per_instance keys are family/seed only, so resuming a file measured for a
            # different arm set would silently mix configurations across arms.
            if pm.get("arms") and pm["arms"] != paths:
                sys.exit(f"ERROR: {out} was measured for arms {pm['arms']} but this run "
                         f"wants {paths}. Its per-instance keys carry no arm identity, "
                         f"so resuming would mix them. Use a different --output, or "
                         f"--no-resume to discard it.")
            if pm.get("repeats") not in (None, args.repeats):
                sys.exit(f"ERROR: {out} used repeats={pm['repeats']}, this run wants "
                         f"{args.repeats}. Use a different --output or --no-resume.")
            done = dict(prev.get("per_instance", {}))
            print(f"resume: {len(done)} instances already measured")
        except SystemExit:
            raise
        except Exception as e:
            print(f"WARNING: could not read {out} ({e}); starting fresh")

    per_inst = (1 + len(names)) * args.repeats
    print("=" * 86)
    print(f"HEAD-TO-HEAD RE-MEASUREMENT  {len(keys)} instances x {len(names)} arms")
    for n in names:
        print(f"  arm {n:<10s} {paths[n]}" + ("   [reference]" if n == args.reference else ""))
    print(f"  per instance : {args.repeats} x default + {args.repeats} x each arm, "
          f"round-robin = {per_inst} solves")
    print(f"  budget       : {len(keys) * per_inst} solves "
          f"(minus {len(done) * per_inst} done)")
    print(f"  statistic    : median of {args.repeats}; one shared default per instance")
    print("=" * 86, flush=True)

    t0 = time.perf_counter()
    for n_i, k in enumerate(keys):
        kk = f"{k[0]}/{k[1]}"
        if kk in done:
            continue
        lp = inst_dir / f"{k[0]}_{args.variant}_{k[1]}.lp"
        cfgs = {"def": default_config}
        cfgs.update({n: arms[n][k] for n in names})
        order = ["def"] + names
        times, status = defaultdict(list), {}
        for _ in range(args.repeats):          # rep-major: drift is shared
            for a in order:
                r = solve_once(scip, lp, cfgs[a], args.time_limit)
                times[a].append(r.time)
                status[a] = r.status
        rec = {a: dict(times=times[a], median=float(np.median(times[a])),
                       mean=float(np.mean(times[a])),
                       spread=float(max(times[a]) / max(min(times[a]), 1e-9)),
                       status=status[a]) for a in order}
        rec["family_id"], rec["seed"] = k[0], k[1]
        dt = rec["def"]["median"]
        for n in names:
            rec[n]["score"] = score(rec[n]["median"], dt)
        done[kk] = rec

        el = time.perf_counter() - t0
        ref = rec[args.reference]["score"]
        best = max(names, key=lambda n: rec[n]["score"])
        print(f"  [{len(done):3d}/{len(keys)}] {kk:30s} def={dt:6.3f}s "
              + " ".join(f"{n}={rec[n]['median']:6.3f}s" for n in names)
              + f" | best={best:<9s} | {el/60:5.1f}min", flush=True)
        if len(done) % 10 == 0 or n_i == len(keys) - 1:
            json.dump(dict(meta=dict(arms=paths, reference=args.reference,
                                     repeats=args.repeats, time_limit=args.time_limit,
                                     n=len(done),
                                     at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                           per_instance=done), open(out, "w", encoding="utf-8"), indent=1)
    json.dump(dict(meta=dict(arms=paths, reference=args.reference, repeats=args.repeats,
                             time_limit=args.time_limit, n=len(done),
                             at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                   per_instance=done), open(out, "w", encoding="utf-8"), indent=1)

    # ── report ──
    from scipy import stats
    recs = [done[f"{f}/{s}"] for f, s in keys if f"{f}/{s}" in done]
    fams = sorted({r["family_id"] for r in recs})
    ref = args.reference

    def block(rs, label):
        print(f"\n{label}  (n={len(rs)})")
        print(f"  {'arm':<10s} {'mean score':>10s} {'SGM':>7s} {'win%':>6s} "
              f"{'total s':>9s} | vs {ref}: {'diff':>8s} {'W:L:T':>10s} {'p':>8s} "
              f"{'>20% W:L':>9s} {'p':>8s}")
        dsum = sum(r["def"]["median"] for r in rs)
        print(f"  {'default':<10s} {0.0:>10.4f} {1.0:>7.2f} {'--':>6s} {dsum:>9.1f}")
        for n in names:
            sc = np.array([r[n]["score"] for r in rs])
            sp = [r["def"]["median"] / max(r[n]["median"], 1e-3) for r in rs]
            tot = sum(r[n]["median"] for r in rs)
            line = (f"  {n:<10s} {sc.mean():>+10.4f} {sgm(sp):>7.2f} "
                    f"{100*(sc>0).mean():>5.1f}% {tot:>9.1f}")
            if n != ref:
                d = np.array([r[ref]["score"] - r[n]["score"] for r in rs])
                c = Counter("W" if x > 1e-3 else ("L" if x < -1e-3 else "T") for x in d)
                p = stats.wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else float("nan")
                fw = sum(1 for r in rs if r[ref]["median"] < 0.8 * r[n]["median"])
                fl = sum(1 for r in rs if r[ref]["median"] > 1.25 * r[n]["median"])
                pf = stats.binomtest(fw, fw + fl, 0.5).pvalue if fw + fl else float("nan")
                line += (f" | {d.mean():>+8.4f} "
                         f"{f'{c[chr(87)]}:{c[chr(76)]}:{c[chr(84)]}':>10s} {p:>8.4f} "
                         f"{f'{fw}:{fl}':>9s} {pf:>8.4f}")
            print(line)

    print("\n" + "=" * 86)
    print(f"RESULTS  median of {args.repeats}, shared default, reference arm = {ref}")
    print("=" * 86)
    block(recs, "ALL FAMILIES")
    for f in fams:
        block([r for r in recs if r["family_id"] == f], f)

    print("\nmeasurement noise (max/min across repeats):")
    for a in ["def"] + names:
        sp = np.array([r[a]["spread"] for r in recs])
        print(f"  {a:<10s} median {np.median(sp):.2f}x  p90 {np.percentile(sp,90):.2f}x  "
              f"max {sp.max():.2f}x   (>1.5x on {int((sp>1.5).sum())})")
    tl = [r for r in recs if any(r[a]["status"] == "timelimit" for a in ["def"] + names)]
    print(f"\ninstances touching the {args.time_limit:.0f}s limit: {len(tl)}")
    if tl and len(tl) < len(recs):
        block([r for r in recs if r not in tl], "EXCLUDING TIME-LIMIT INSTANCES")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
