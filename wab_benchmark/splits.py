
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import random

def build_compositional_splits(episodes: List[Dict[str, Any]], *, seed: int = 42) -> Dict[str, List[str]]:
    """
    Create simple 'seen' vs 'unseen_composition' splits by holding out certain (object,target) pairs.
    Assumes actions like "place(obj,target)" exist in gold_plan strings.
    """
    random.seed(seed)
    # Extract pairs
    pairs = set()
    ep_pairs = {}
    for ep in episodes:
        ep_id = ep["episode_id"]
        found = set()
        for a in ep["gold_plan"]:
            a_low = a.lower()
            if a_low.startswith("place(") and "," in a_low:
                inside = a_low[a_low.find("(")+1:a_low.find(")")]
                obj, tgt = [x.strip() for x in inside.split(",")]
                found.add((obj, tgt))
        ep_pairs[ep_id] = list(found)
        pairs |= found
    pairs = list(pairs)
    random.shuffle(pairs)
    # Hold out 30% of pairs for unseen_composition
    k = max(1, int(0.3 * len(pairs)))
    held_out = set(pairs[:k])
    seen, unseen = [], []
    for ep in episodes:
        ep_id = ep["episode_id"]
        # If episode contains any held-out pair, send to unseen
        if any(p in held_out for p in ep_pairs.get(ep_id, [])):
            unseen.append(ep_id)
        else:
            seen.append(ep_id)
    return {"seen": seen, "unseen_composition": unseen}

def build_long_horizon_splits(episodes: List[Dict[str, Any]], *, threshold: int = 3) -> Dict[str, List[str]]:
    """Split by action sequence length."""
    short, long_ = [], []
    for ep in episodes:
        (long_ if len(ep["gold_plan"]) > threshold else short).append(ep["episode_id"])
    return {"short_horizon": short, "long_horizon": long_}

def merge_splits(*split_dicts: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out = {}
    for d in split_dicts:
        out.update(d)
    return out
