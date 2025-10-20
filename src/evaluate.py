from typing import List, Tuple
import re
import numpy as np


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize_text(pred) == _normalize_text(gold))


def f1_score(pred: str, gold: str) -> float:
    p_tokens = _normalize_text(pred).split()
    g_tokens = _normalize_text(gold).split()
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return 0.0
    common = {}
    for t in p_tokens:
        common[t] = min(p_tokens.count(t), g_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_metrics(preds: List[str], golds: List[str]) -> Tuple[float, float]:
    ems = [exact_match(p, g) for p, g in zip(preds, golds)]
    f1s = [f1_score(p, g) for p, g in zip(preds, golds)]
    return float(np.mean(ems)), float(np.mean(f1s))
