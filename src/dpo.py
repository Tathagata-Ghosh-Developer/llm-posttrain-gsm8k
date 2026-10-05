"""DPO on preference pairs sampled from the SFT model itself (correct vs incorrect answers)."""
import json
import os
import random
import time

import torch

from src.common import (CsvLog, autocast, build_batch, count_params, generate, get_device, label, load_config,
                        load_model, load_tokenizer, parse_args, save_checkpoint, save_json, set_seed,
                        token_logprobs)
from src.data import format_prompt, is_correct, shuffled_train, truncate_completion
from src.losses import dpo_loss
from src.sft import warmup_cosine


def make_pairs(model, tok, rows, cfg):
    """Sample n completions per question; pair a correct one (chosen) with an incorrect one (rejected).

    Only well-formed completions (with a '#### x' line) are used, so a pair differs in
    correctness rather than in format. Duplicates are removed before pairing.
    """
    prompts = [format_prompt(r["question"]) for r in rows]
    n = cfg["samples_per_question"]
    ids = generate(model, tok, prompts, n=n, temperature=cfg["sample_temperature"],
                   max_new_tokens=cfg["max_new_tokens"], batch_size=cfg["gen_batch"])
    texts = [tok.decode(x, skip_special_tokens=True) for x in ids]
    pairs, n_correct, n_mixed = [], 0, 0
    for i, r in enumerate(rows):
        group = list(dict.fromkeys(truncate_completion(t) for t in texts[i * n : (i + 1) * n]))
        group = [t for t in group if "####" in t]
        good = [t for t in group if is_correct(t, r["gold"])]
        bad = [t for t in group if not is_correct(t, r["gold"])]
        n_correct += sum(is_correct(t, r["gold"]) for t in texts[i * n : (i + 1) * n])
        n_mixed += bool(good and bad)
        for chosen, rejected in list(zip(good, bad))[: cfg["pairs_per_question"]]:
            pairs.append({"prompt": prompts[i], "chosen": chosen, "rejected": rejected})
    stats = {"pair_questions": len(rows), "samples": len(texts), "sample_accuracy": n_correct / len(texts),
             "questions_with_pair": n_mixed, "pairs": len(pairs)}
    return pairs, stats


def tokenize_pair(tok, pair):
    eos = [tok.eos_token_id]
    return (tok(pair["prompt"])["input_ids"], tok(pair["chosen"])["input_ids"] + eos,
            tok(pair["rejected"])["input_ids"] + eos)


def sequence_logprobs(model, tok, batch, device):
    """Summed log-prob of chosen and rejected completions given the prompt, each (B,)."""
    prompts = [p for p, _, _ in batch]
    completions = [c for _, c, _ in batch] + [r for _, _, r in batch]
    ids, attention_mask, mask = build_batch(prompts + prompts, completions, tok.pad_token_id, device)
    with autocast(device):
        logp = token_logprobs(model, ids, attention_mask)
    seq = (logp * mask).sum(dim=1)
    return seq[: len(batch)], seq[len(batch) :]


def main():
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    tok = load_tokenizer(args.model)
    model = load_model(args.model, torch.float32, device)

    t0 = time.time()
    rows = shuffled_train(cfg["seed"])[: cfg["n_questions"]]
    pairs, pair_stats = make_pairs(model, tok, rows, cfg)
    pair_stats["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(pair_stats), flush=True)
    with open(os.path.join(args.out, "pairs.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    random.Random(cfg["seed"]).shuffle(pairs)
    n_val = max(1, int(cfg["val_fraction"] * len(pairs)))
    data = [tokenize_pair(tok, p) for p in pairs]
    val, train = data[:n_val], data[n_val:]

    # The reference policy is the SFT model, i.e. the starting weights: precompute its
    # log-probs once instead of keeping a second frozen copy on the GPU.
    t0 = time.time()
    micro = cfg["micro_batch"]

    @torch.no_grad()
    def all_logprobs(examples):
        out = [sequence_logprobs(model, tok, examples[i : i + micro], device) for i in range(0, len(examples), micro)]
        return torch.cat([c for c, _ in out]), torch.cat([r for _, r in out])

    ref_train, ref_val = all_logprobs(train), all_logprobs(val)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.0)
    batch = micro * cfg["grad_accum"]
    steps_per_epoch = len(train) // batch
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = max(1, int(cfg["warmup_ratio"] * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, total_steps, warmup))
    log = CsvLog(os.path.join(args.results, "dpo_log.csv"))
    gen = torch.Generator().manual_seed(cfg["seed"])
    beta, step = cfg["beta"], 0
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(len(train), generator=gen).tolist()
        for s in range(steps_per_epoch):
            idx = order[s * batch : (s + 1) * batch]
            losses, chosen_r, rejected_r = [], [], []
            for m in range(0, batch, micro):
                sub = idx[m : m + micro]
                pc, pr = sequence_logprobs(model, tok, [train[i] for i in sub], device)
                loss, rc, rr = dpo_loss(pc, pr, ref_train[0][sub], ref_train[1][sub], beta)
                (loss * len(sub) / batch).backward()
                losses.append(loss.item() * len(sub) / batch)
                chosen_r.append(rc)
                rejected_r.append(rr)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"]).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rc, rr = torch.cat(chosen_r), torch.cat(rejected_r)
            log.write({"step": step, "epoch": epoch, "loss": sum(losses), "reward_acc": (rc > rr).float().mean().item(),
                       "margin": (rc - rr).mean().item(), "chosen_reward": rc.mean().item(),
                       "rejected_reward": rr.mean().item(), "lr": sched.get_last_lr()[0], "grad_norm": grad_norm})
            if step % 10 == 0 or step == total_steps:
                print(f"step {step}/{total_steps} loss {sum(losses):.4f} margin {(rc - rr).mean().item():.3f}", flush=True)
    train_seconds = time.time() - t0

    model.eval()
    with torch.no_grad():
        pc, pr = all_logprobs(val)
        _, rc, rr = dpo_loss(pc, pr, ref_val[0], ref_val[1], beta)
    summary = {"stage": "dpo", "init": label(args.model), "params": count_params(model), **pair_stats,
               "train_pairs": len(train), "val_pairs": len(val), "optimizer_steps": step,
               "effective_batch_pairs": batch, "train_seconds": round(train_seconds, 1),
               "val_reward_acc": (rc > rr).float().mean().item(), "val_margin": (rc - rr).mean().item(),
               "val_chosen_reward": rc.mean().item(), "val_rejected_reward": rr.mean().item(), "config": cfg}
    print(json.dumps(summary, indent=2))
    save_json(summary, os.path.join(args.results, "dpo_summary.json"))
    save_checkpoint(model, tok, args.out)


if __name__ == "__main__":
    main()
