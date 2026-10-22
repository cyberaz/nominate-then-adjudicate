"""Summarize the stored Ecole baseline measurements."""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(path):
    data = json.loads(path.read_text())
    rows = list(data["per_instance"].values())
    result = {"default": {"mean_score": 0.0, "sgm_speedup": 1.0}}
    for arm in data["meta"]["arms"]:
        speedups = [r["def"]["median"] / max(r[arm]["median"], 1e-3) for r in rows]
        result[arm] = {
            "mean_score": statistics.mean(r[arm]["score"] for r in rows),
            "sgm_speedup": math.exp(statistics.mean(math.log1p(x) for x in speedups)) - 1,
        }
    return {"instances": len(rows), "methods": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "ecole/baseline.json")
    args = parser.parse_args()
    print(json.dumps(summarize(args.input), indent=2))


if __name__ == "__main__":
    main()
