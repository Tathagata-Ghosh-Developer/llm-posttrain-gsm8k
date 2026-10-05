"""Supervised fine-tuning on GSM8K chain-of-thought solutions (loss on answer tokens only)."""
import math
import os
import time

import torch

from src.common import (CsvLog, autocast, count_params, get_device, label, load_config, load_model,
                        load_tokenizer, pad_right, parse_args, save_checkpoint, save_json, set_seed)
from src.data import format_prompt, load_gsm8k
from src.losses import sft_loss


def tokenize(tok, row, max_len):
    """prompt -> label -100 (no loss); ' reasoning\\n#### answer<eos>' -> its own token ids."""
    prompt = tok(format_prompt(row["question"]))["input_ids"]
    target = tok(" " + row["solution"])["input_ids"] + [tok.eos_token_id]
    ids = (prompt + target)[:max_len]
    labels = ([-100] * len(prompt) + target)[:max_len]
    return ids, labels


def collate(examples, pad_id, device):
    ids = pad_right([e[0] for e in examples], pad_id)
    labels = pad_right([e[1] for e in examples], -100)
    attention_mask = pad_right([[1] * len(e[0]) for e in examples], 0)
    return ids.to(device), labels.to(device), attention_mask.to(device)


def warmup_cosine(step, total, warmup):
    """LR multiplier: linear warmup, then cosine decay to zero."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(args.results, exist_ok=True)
    tok = load_tokenizer(args.model)
    rows = load_gsm8k("train", cfg["limit"])
    examples = [tokenize(tok, r, cfg["max_len"]) for r in rows]

    # fp32 master weights and AdamW state; forward/backward run in bf16 autocast
    model = load_model(args.model, torch.float32, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=cfg["weight_decay"])
    micro, batch = cfg["micro_batch"], cfg["micro_batch"] * cfg["grad_accum"]
    steps_per_epoch = len(examples) // batch
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = max(1, int(cfg["warmup_ratio"] * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, total_steps, warmup))
    log = CsvLog(os.path.join(args.results, "sft_log.csv"))
    gen = torch.Generator().manual_seed(cfg["seed"])

    step, t_start = 0, time.time()
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(len(examples), generator=gen).tolist()
        for s in range(steps_per_epoch):
            chunk = [examples[i] for i in order[s * batch : (s + 1) * batch]]
            # normalise by target tokens in the whole accumulated batch, not per micro-batch
            n_targets = sum(sum(l != -100 for l in e[1]) for e in chunk)
            loss_sum = 0.0
            for m in range(0, batch, micro):
                ids, labels, attention_mask = collate(chunk[m : m + micro], tok.pad_token_id, device)
                with autocast(device):
                    logits = model(input_ids=ids, attention_mask=attention_mask, use_cache=False).logits
                nll, _ = sft_loss(logits, labels)
                (nll / n_targets).backward()
                loss_sum += nll.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"]).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            log.write({"step": step, "epoch": epoch, "loss": loss_sum / n_targets, "lr": sched.get_last_lr()[0],
                       "grad_norm": grad_norm, "target_tokens": n_targets,
                       "elapsed_s": round(time.time() - t_start, 1)})
            if step % 10 == 0 or step == total_steps:
                print(f"step {step}/{total_steps} loss {loss_sum / n_targets:.4f}", flush=True)

    wall = time.time() - t_start
    save_json({"stage": "sft", "init": label(args.model), "params": count_params(model),
               "train_examples": len(examples), "examples_seen": step * batch, "optimizer_steps": step, "effective_batch": batch,
               "train_seconds": round(wall, 1), "config": cfg},
              os.path.join(args.results, "sft_summary.json"))
    save_checkpoint(model, tok, args.out)


if __name__ == "__main__":
    main()
