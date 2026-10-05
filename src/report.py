"""Collect results/*.json into a summary table and draw the training curves."""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STAGES = ["base", "base_4shot", "sft", "dpo", "dpo_nll", "grpo"]


def read_csv(path):
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def moving_average(values, window):
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def table(results_dir):
    rows = []
    for stage in STAGES:
        path = os.path.join(results_dir, f"eval_{stage}.json")
        if os.path.exists(path):
            with open(path) as f:
                rows.append(json.load(f))
    if not rows:
        return
    cols = ["name", "params", "n_eval", "fewshot", "greedy_pass@1", "pass@1", "pass@4", "pass@8", "greedy_format_rate"]
    with open(os.path.join(results_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["| model | greedy pass@1 | pass@1 (T=0.7) | pass@4 | pass@8 | format rate |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {100 * r['greedy_pass@1']:.1f} | {100 * r['pass@1']:.1f} | "
                     f"{100 * r['pass@4']:.1f} | {100 * r['pass@8']:.1f} | {100 * r['greedy_format_rate']:.1f} |")
    with open(os.path.join(results_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def plots(results_dir):
    path = os.path.join(results_dir, "sft_log.csv")
    if os.path.exists(path):
        log = read_csv(path)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot([r["step"] for r in log], [r["loss"] for r in log], lw=1, alpha=0.4, label="per step")
        ax.plot([r["step"] for r in log], moving_average([r["loss"] for r in log], 10), lw=2, label="10-step mean")
        ax.set(xlabel="optimizer step", ylabel="token cross-entropy", title="SFT training loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "sft_loss.png"), dpi=150)

    for name, title in [("dpo", "DPO"), ("dpo_nll", "DPO + NLL")]:
        path = os.path.join(results_dir, f"{name}_log.csv")
        if not os.path.exists(path):
            continue
        log = read_csv(path)
        steps = [r["step"] for r in log]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
        a1.plot(steps, [r["chosen_reward"] for r in log], label="chosen")
        a1.plot(steps, [r["rejected_reward"] for r in log], label="rejected")
        a1.plot(steps, [r["margin"] for r in log], label="margin")
        a1.axhline(0, color="grey", lw=0.5)
        a1.set(xlabel="optimizer step", ylabel="implicit reward (beta * log-ratio)", title=f"{title} rewards")
        a1.legend()
        a2.plot(steps, [r["reward_acc"] for r in log], alpha=0.4, label="per batch")
        a2.plot(steps, moving_average([r["reward_acc"] for r in log], 10), lw=2, label="10-step mean")
        a2.set(xlabel="optimizer step", ylabel="reward accuracy", title=f"{title} reward accuracy (train batches)")
        a2.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"{name}_margins.png"), dpi=150)

    path = os.path.join(results_dir, "grpo_log.csv")
    if os.path.exists(path):
        log = read_csv(path)
        steps = [r["step"] for r in log]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
        a1.plot(steps, [r["reward"] for r in log], alpha=0.4, label="per step")
        a1.plot(steps, moving_average([r["reward"] for r in log], 10), lw=2, label="10-step mean")
        a1.set(xlabel="GRPO step", ylabel="mean reward (fraction correct)", title="GRPO training reward")
        a1.legend()
        a2.plot(steps, [r["kl"] for r in log])
        a2.set(xlabel="GRPO step", ylabel="KL to reference (k3)", title="GRPO KL")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "grpo_reward.png"), dpi=150)


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    table(results_dir)
    plots(results_dir)
