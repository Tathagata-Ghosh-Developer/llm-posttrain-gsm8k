"""GSM8K loading, prompt format, answer extraction and pass@k."""
import random
import re

import numpy as np
from datasets import load_dataset

PROMPT = "Question: {question}\nAnswer:"
STOP_STRINGS = ["Question:"]  # a base model keeps writing new questions; stop there

_CALCULATOR = re.compile(r"<<[^>]*>>")  # GSM8K annotations such as <<48/2=24>>
_NUMBER = re.compile(r"-?[\d,]*\.?\d+")


def format_prompt(question):
    return PROMPT.format(question=question.strip())


def clean_solution(answer):
    """Drop calculator annotations and keep the final line as '#### <answer>'."""
    reasoning, final = answer.split("####")
    reasoning = _CALCULATOR.sub("", reasoning).strip()
    return f"{reasoning}\n#### {final.strip()}"


def normalize_number(text):
    """'1,234' -> '1234', '$18.00' -> '18', '0.50' -> '0.5'; None if not a number."""
    text = text.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        value = float(text)
    except ValueError:
        return None
    if value.is_integer():
        return str(int(value))
    return str(round(value, 6))


def gold_answer(answer):
    return normalize_number(answer.split("####")[-1])


def truncate_completion(text):
    """Cut a completion at the next 'Question:' and right after the '#### x' line."""
    text = text.split("Question:")[0]
    if "####" in text:
        head, tail = text.split("####", 1)
        text = head + "####" + tail.split("\n")[0]
    return text


def extract_answer(completion):
    """Model answer = last number in the truncated completion."""
    numbers = _NUMBER.findall(truncate_completion(completion))
    return normalize_number(numbers[-1]) if numbers else None


def is_correct(completion, gold):
    pred = extract_answer(completion)
    return pred is not None and pred == gold


def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c correct: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def load_gsm8k(split, limit=None):
    rows = [
        {
            "question": r["question"],
            "solution": clean_solution(r["answer"]),
            "gold": gold_answer(r["answer"]),
        }
        for r in load_dataset("openai/gsm8k", "main", split=split)
    ]
    return rows[:limit] if limit else rows


def shuffled_train(seed):
    """Training problems in one fixed random order, shared by the DPO and GRPO stages."""
    rows = load_gsm8k("train")
    random.Random(seed).shuffle(rows)
    return rows
