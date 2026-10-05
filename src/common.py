"""Shared helpers: config, seeding, model loading, log-probs, batched generation, logging."""
import argparse
import csv
import json
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import STOP_STRINGS


def parse_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True, help="hub id or local checkpoint directory")
    p.add_argument("--out", help="checkpoint output directory")
    p.add_argument("--name", help="run name used in result file names")
    p.add_argument("--results", default="results")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, key=value")
    return p.parse_args()


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(device):
    """bf16 autocast on GPU (weights may stay fp32); no-op on CPU."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"  # generation needs left padding; training batches are built by hand
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="sdpa")
    return model.to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters())  # tied embeddings are counted once


def save_checkpoint(model, tok, out_dir):
    model.to(torch.bfloat16).save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    open(os.path.join(out_dir, "DONE"), "w").close()  # lets the other GPU lane wait on this file


def pad_right(seqs, value):
    width = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (width - len(s)) for s in seqs])


def build_batch(prompt_ids, completion_ids, pad_id, device):
    """Right-padded prompt+completion batch.

    Returns input_ids, attention_mask (B, T) and completion_mask (B, T-1), which is
    aligned with token_logprobs(): entry t marks whether token t+1 is a completion token.
    """
    seqs = [p + c for p, c in zip(prompt_ids, completion_ids)]
    input_ids = pad_right(seqs, pad_id)
    attention_mask = pad_right([[1] * len(s) for s in seqs], 0)
    completion_mask = pad_right([[0] * len(p) + [1] * len(c) for p, c in zip(prompt_ids, completion_ids)], 0)
    return input_ids.to(device), attention_mask.to(device), completion_mask[:, 1:].float().to(device)


def token_logprobs(model, input_ids, attention_mask):
    """log p(x_{t+1} | x_<=t) for every position, shape (B, T-1)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    return log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def cut_after_eos(ids, eos_id):
    """Keep tokens up to and including the first EOS (generate pads finished rows with EOS)."""
    return ids[: ids.index(eos_id) + 1] if eos_id in ids else ids


@torch.no_grad()
def generate(model, tok, prompts, n=1, temperature=0.0, max_new_tokens=300, batch_size=64):
    """Batched generation with left padding and the KV cache.

    Returns completion token ids, len(prompts) * n lists, prompt-major order
    (the n samples of prompt i are at positions i*n .. i*n+n-1).
    temperature 0 means greedy; otherwise plain sampling (no top-k / top-p).
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    if temperature > 0:
        decoding = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0, num_return_sequences=n)
    else:
        decoding = dict(do_sample=False)
    completions = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True).to(device)
        with autocast(device):
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                stop_strings=STOP_STRINGS,
                tokenizer=tok,
                **decoding,
            )
        for row in out[:, enc["input_ids"].shape[1] :].tolist():
            completions.append(cut_after_eos(row, tok.eos_token_id))
    model.train(was_training)
    return completions


class CsvLog:
    """Append one dict per row to a CSV file (header taken from the first row)."""

    def __init__(self, path):
        self.path, self.writer, self.file = path, None, None

    def write(self, row):
        if self.writer is None:
            self.file = open(self.path, "w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow({k: float(f"{v:.6g}") if isinstance(v, float) else v for k, v in row.items()})
        self.file.flush()


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def label(path):
    """Short name for a model path, so result files never contain directory names."""
    return os.path.basename(os.path.normpath(path))
