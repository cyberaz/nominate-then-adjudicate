"""Replay a saved nomination/selection and its final measured result; no inference."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / "examples/selection.json").read_text())
    baseline = json.loads((root / "results/ecole/baseline.json").read_text())
    rank = record["llm_choice"]
    assert 1 <= rank <= len(record["topk_cand_ids"])
    assert record["chosen_cand_id"] == record["topk_cand_ids"][rank - 1]
    assert record["predicted_config"] == record["top5_configs"][rank - 1]
    key = f"{record['family_id']}/{record['seed']}"
    timing = baseline["per_instance"][key]
    print(json.dumps({
        "mode": "recorded_selection",
        "instance": key,
        "nomination_width": len(record["topk_cand_ids"]),
        "selected_rank": rank,
        "selected_candidate_id": record["chosen_cand_id"],
        "final_default_median_seconds": timing["def"]["median"],
        "final_method_median_seconds": timing["llm"]["median"],
        "final_score": timing["llm"]["score"],
    }, indent=2))


if __name__ == "__main__":
    main()
