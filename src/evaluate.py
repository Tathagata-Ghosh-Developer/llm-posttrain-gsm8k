"""Greedy pass@1 and sampled pass@k on a fixed GSM8K test subset."""
import json
import os
import time

import numpy as np
import torch

from src.common import (count_params, generate, get_device, label, load_config, load_model,
                        load_tokenizer, parse_args, save_json, set_seed)
from src.data import extract_answer, format_prompt, is_correct, load_gsm8k, pass_at_k, truncate_completion


def main():
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(args.results, exist_ok=True)
    tok = load_tokenizer(args.model)
    model = load_model(args.model, torch.bfloat16, device)
    rows = load_gsm8k("test", cfg["n_eval"])
    prompts = [format_prompt(r["question"]) for r in rows]

    t0 = time.time()
    greedy_ids = generate(model, tok, prompts, temperature=0.0, max_new_tokens=cfg["max_new_tokens"],
                          batch_size=cfg["greedy_batch"])
    greedy = [tok.decode(ids, skip_special_tokens=True) for ids in greedy_ids]
    greedy_correct = [is_correct(text, r["gold"]) for text, r in zip(greedy, rows)]
    t_greedy = time.time() - t0

    n = cfg["n_samples"]
    t0 = time.time()
    sample_ids = generate(model, tok, prompts, n=n, temperature=cfg["temperature"],
                          max_new_tokens=cfg["max_new_tokens"], batch_size=cfg["sample_batch"])
    samples = [tok.decode(ids, skip_special_tokens=True) for ids in sample_ids]
    n_correct = [sum(is_correct(samples[i * n + j], r["gold"]) for j in range(n)) for i, r in enumerate(rows)]
    t_sample = time.time() - t0

    metrics = {
        "name": args.name,
        "model": label(args.model),
        "params": count_params(model),
        "n_eval": len(rows),
        "greedy_pass@1": float(np.mean(greedy_correct)),
        "greedy_format_rate": float(np.mean(["####" in truncate_completion(t) for t in greedy])),
        "greedy_mean_tokens": float(np.mean([len(ids) for ids in greedy_ids])),
        "n_samples": n,
        "temperature": cfg["temperature"],
        "max_new_tokens": cfg["max_new_tokens"],
        "seconds_greedy": round(t_greedy, 1),
        "seconds_sampling": round(t_sample, 1),
    }
    for k in cfg["ks"]:
        metrics[f"pass@{k}"] = float(np.mean([pass_at_k(n, c, k) for c in n_correct]))
    print(json.dumps(metrics, indent=2))
    save_json(metrics, os.path.join(args.results, f"eval_{args.name}.json"))
    with open(os.path.join(args.results, f"greedy_{args.name}.jsonl"), "w") as f:
        for r, text, ok, c in zip(rows, greedy, greedy_correct, n_correct):
            f.write(json.dumps({"question": r["question"], "gold": r["gold"], "completion": text,
                                "pred": extract_answer(text), "correct": ok, "samples_correct": c}) + "\n")


if __name__ == "__main__":
    main()
