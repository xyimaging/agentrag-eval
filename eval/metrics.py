"""
Evaluation metrics: Exact Match and token-level F1.

Follows the SQuAD normalization protocol: lowercase, remove articles
and punctuation, collapse whitespace before comparison.
"""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Normalize: lowercase, strip articles/punctuation/extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens   = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_batch(predictions: list[str], ground_truths: list[str]) -> dict:
    """Compute mean EM and F1 over a list of prediction/ground-truth pairs."""
    assert len(predictions) == len(ground_truths)
    ems, f1s = [], []
    for pred, gt in zip(predictions, ground_truths):
        ems.append(exact_match(pred, gt))
        f1s.append(f1_score(pred, gt))
    return {
        "em":      sum(ems) / len(ems),
        "f1":      sum(f1s) / len(f1s),
        "em_list": ems,
        "f1_list": f1s,
        "n":       len(ems),
    }
