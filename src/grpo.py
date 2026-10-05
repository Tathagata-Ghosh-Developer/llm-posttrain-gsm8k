"""GRPO: sample G answers per question, reward correct ones, update with group-normalised advantages.

No value network: the baseline for each answer is the mean reward of its own group.
"""
import os
import time

import torch

from src.common import (CsvLog, autocast, build_batch, count_params, generate, get_device, label, load_config,
                        load_model, load_tokenizer, parse_args, save_checkpoint, save_json, set_seed,
                        token_logprobs)
from src.data import format_prompt, is_correct, shuffled_train, truncate_completion
from src.losses import group_advantages, grpo_loss


@torch.no_grad()
def batched_logprobs(model, ids, attention_mask, micro):
    out = torch.zeros(ids.shape[0], ids.shape[1] - 1, device=ids.device)
    for i in range(0, ids.shape[0], micro):
        width = int(attention_mask[i : i + micro].sum(dim=1).max())
        with autocast(ids.device):
            out[i : i + micro, : width - 1] = token_logprobs(model, ids[i : i + micro, :width],
                                                             attention_mask[i : i + micro, :width])
    return out


def main():
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(args.results, exist_ok=True)
    tok = load_tokenizer(args.model)
    policy = load_model(args.model, torch.float32, device)
    policy.train()
    reference = load_model(args.model, torch.bfloat16, device).eval().requires_grad_(False)
    rows = shuffled_train(cfg["seed"])[cfg["question_offset"] :]

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup_steps"]))
    log = CsvLog(os.path.join(args.results, "grpo_log.csv"))
    Q, G, micro = cfg["questions_per_step"], cfg["group_size"], cfg["micro_batch"]
    budget = cfg["time_budget_min"] * 60
    t_start, step, last_step_time = time.time(), 0, 0.0

    while step < cfg["max_steps"] and (step + 1) * Q <= len(rows):
        if time.time() - t_start + last_step_time > budget:
            break
        t_step = time.time()
        batch_rows = rows[step * Q : (step + 1) * Q]
        prompts = [format_prompt(r["question"]) for r in batch_rows]
        completions = generate(policy, tok, prompts, n=G, temperature=cfg["temperature"],
                               max_new_tokens=cfg["max_new_tokens"], batch_size=Q)
        texts = [tok.decode(c, skip_special_tokens=True) for c in completions]
        rewards = torch.tensor([float(is_correct(t, batch_rows[i // G]["gold"])) for i, t in enumerate(texts)])
        rewards = rewards.view(Q, G)
        advantages = group_advantages(rewards).view(-1).to(device)

        prompt_ids = [tok(p)["input_ids"] for p in prompts]
        ids, attention_mask, mask = build_batch([prompt_ids[i // G] for i in range(Q * G)], completions,
                                                tok.pad_token_id, device)
        old_logp = batched_logprobs(policy, ids, attention_mask, micro)  # pi_old = policy that sampled
        ref_logp = batched_logprobs(reference, ids, attention_mask, micro)

        # several optimizer steps per sampled batch: after the first one the policy has moved,
        # so the ratio pi/pi_old differs from 1 and clipping matters
        stats = {"kl": 0.0, "clip_frac": 0.0}
        n_updates = 0
        perm = torch.randperm(Q * G).tolist()
        for mb in range(0, Q * G, cfg["minibatch"]):
            rows_mb = perm[mb : mb + cfg["minibatch"]]
            for c in range(0, len(rows_mb), micro):
                sub = rows_mb[c : c + micro]
                width = int(attention_mask[sub].sum(dim=1).max())
                with autocast(device):
                    logp = token_logprobs(policy, ids[sub, :width], attention_mask[sub, :width])
                per_seq, st = grpo_loss(logp, old_logp[sub, : width - 1], ref_logp[sub, : width - 1],
                                        advantages[sub], mask[sub, : width - 1], cfg["clip_eps"], cfg["kl_beta"])
                (per_seq.sum() / len(rows_mb)).backward()
                for k in stats:
                    stats[k] += st[k] * len(sub) / (Q * G)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            n_updates += 1

        step += 1
        last_step_time = time.time() - t_step
        std = rewards.std(dim=1)
        log.write({"step": step, "questions_seen": step * Q, "reward": rewards.mean().item(),
                   "solved_any": (rewards.max(dim=1).values > 0).float().mean().item(),
                   "informative_groups": (std > 0).float().mean().item(),
                   "format_rate": sum("####" in truncate_completion(t) for t in texts) / len(texts),
                   "mean_tokens": sum(len(c) for c in completions) / len(completions),
                   "kl": stats["kl"], "clip_frac": stats["clip_frac"], "updates": n_updates,
                   "step_seconds": round(last_step_time, 1), "elapsed_s": round(time.time() - t_start, 1)})
        print(f"step {step} reward {rewards.mean().item():.3f} kl {stats['kl']:.4f} "
              f"({last_step_time:.1f}s)", flush=True)

    save_json({"stage": "grpo", "init": label(args.model), "params": count_params(policy), "steps": step,
               "questions_seen": step * Q, "completions_sampled": step * Q * G,
               "train_seconds": round(time.time() - t_start, 1), "config": cfg},
              os.path.join(args.results, "grpo_summary.json"))
    save_checkpoint(policy, tok, args.out)


if __name__ == "__main__":
    main()
