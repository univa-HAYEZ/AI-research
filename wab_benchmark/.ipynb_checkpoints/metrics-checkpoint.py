
from typing import List, Dict, Tuple
import numpy as np

def _tokenize_action(a: str) -> List[str]:
    # very simple tokenization for actions like "grasp(red_cube)" or "place(x,y)"
    a = a.lower().replace("->", " ").replace(",", " ").replace("(", " ").replace(")", " ")
    toks = [t.strip() for t in a.split() if t.strip()]
    return toks

def action_f1(gold_plan: List[str], pred_plan: List[str]) -> float:
    """Set-F1 over action tokens (order-invariant, partial credit)."""
    g = set(sum((_tokenize_action(a) for a in gold_plan), []))
    p = set(sum((_tokenize_action(a) for a in pred_plan), []))
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    precision = len(g & p) / (len(p) if p else 1)
    recall    = len(g & p) / (len(g) if g else 1)
    return 0.0 if (precision+recall)==0 else 2*precision*recall/(precision+recall)

def sequence_exact_match(gold_plan: List[str], pred_plan: List[str]) -> bool:
    return [s.strip().lower() for s in gold_plan] == [s.strip().lower() for s in pred_plan]

def success_rate(golds: List[List[str]], preds: List[List[str]]) -> float:
    ok = sum(sequence_exact_match(g, p) for g, p in zip(golds, preds))
    return ok / (len(golds) if golds else 1)

def plan_length_error(gold_plan: List[str], pred_plan: List[str]) -> int:
    return abs(len(gold_plan) - len(pred_plan))

def average_length_error(golds: List[List[str]], preds: List[List[str]]) -> float:
    if not golds:
        return 0.0
    err = [plan_length_error(g, p) for g, p in zip(golds, preds)]
    return float(np.mean(err))

def generalization_gap(seen_scores: List[float], unseen_scores: List[float]) -> float:
    """Compute gap = seen_mean - unseen_mean (positive means worse on unseen)."""
    if not seen_scores or not unseen_scores:
        return float('nan')
    return float(np.mean(seen_scores) - np.mean(unseen_scores))
