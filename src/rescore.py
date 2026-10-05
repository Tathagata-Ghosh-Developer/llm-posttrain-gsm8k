"""Re-score saved greedy completions with the current answer extractor (no model needed).

Used after fixing the extractor: greedy decoding is deterministic, so re-scoring the saved
completions gives the same greedy pass@1 as re-running the evaluation.
"""
import csv
import json
import os
import sys

from src.data import is_correct, truncate_completion

NAMES = ["base", "base_4shot", "sft", "dpo", "dpo_nll", "grpo"]


def main(results_dir):
    out = []
    for name in NAMES:
        path = os.path.join(results_dir, f"greedy_{name}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        n = len(rows)
        out.append({
            "name": name,
            "n_eval": n,
            "greedy_pass@1_logged": sum(r["correct"] for r in rows) / n,
            "greedy_pass@1_rescored": sum(is_correct(r["completion"], r["gold"]) for r in rows) / n,
            "format_rate_rescored": sum("####" in truncate_completion(r["completion"]) for r in rows) / n,
        })
    with open(os.path.join(results_dir, "rescored_greedy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for row in out:
        print(row)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
