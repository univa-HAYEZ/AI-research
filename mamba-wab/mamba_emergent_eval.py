#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mamba_emergent_eval.py

Mamba-friendly planning evaluation:
- Parse gold DSL plans reliably
- Parse pred plans flexibly (DSL if possible, otherwise heuristic NL->IR)
- Compute semantic/structural metrics and a composite "Emergent Score"
"""

from __future__ import annotations
import argparse, json, re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import Counter
import pandas as pd

# -----------------------------
# 1) Normalization helpers
# -----------------------------
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)

# -----------------------------
# 2) Domain vocabulary
# -----------------------------
ACTION_ALIASES = {
    "pick": "pick", "pickup": "pick", "pick up": "pick",
    "grab": "pick", "take": "pick",
    "place on": "place_on", "put on": "place_on", "set on": "place_on",
    "place in": "put_in", "put in": "put_in", "insert": "put_in",
    "open": "open", "close": "close",
    "turn on": "turn_on", "switch on": "turn_on",
    "turn off": "turn_off", "switch off": "turn_off",
}
CANON_ACTIONS = set(ACTION_ALIASES.values())

OBJ_PAT = [
    r"(moka pot|mokapot|moka)",
    r"(white mug|yellow mug|mug|cup)",
    r"(book)",
    r"(bowl)",
    r"(object)",
]

TGT_PAT = [
    r"(stove)",
    r"(microwave)",
    r"(basket)",
    r"(plate)",
    r"(drawer)",
]

MOD_PAT = [
    r"(right of|right)",
    r"(left of|left)",
    r"(back compartment|back)",
]

def extract_3tuple_from_text(text: str) -> Tuple[str, str, str]:
    t = _norm(text)
    obj, tgt, mod = "object", "target", "none"
    for p in OBJ_PAT:
        m = re.search(p, t)
        if m: obj = m.group(1); break
    for p in TGT_PAT:
        m = re.search(p, t)
        if m: tgt = m.group(1); break
    for p in MOD_PAT:
        m = re.search(p, t)
        if m: mod = m.group(1); break
    return obj, tgt, mod

# -----------------------------
# 3) IR
# -----------------------------
@dataclass
class ActionFrame:
    action: str
    obj: Optional[str] = None
    tgt: Optional[str] = None
    mod: Optional[str] = None

    def key(self):
        return (self.action, self.obj, self.tgt, self.mod)

# -----------------------------
# 4) DSL parsing
# -----------------------------
DSL_RE = re.compile(r"([a-zA-Z_]+)\(([^)]*)\)")

def canon_action(a: str) -> Optional[str]:
    a = _norm(a)
    return ACTION_ALIASES.get(a)

def parse_dsl_plan(plan: str) -> List[ActionFrame]:
    frames = []
    for m in DSL_RE.finditer(_safe_str(plan)):
        act = canon_action(m.group(1))
        if not act: continue
        args = [x.strip() for x in m.group(2).split(",") if x.strip()]
        obj = args[0] if len(args) > 0 else None
        tgt = args[1] if len(args) > 1 else None
        frames.append(ActionFrame(act, obj, tgt))
    return frames

# -----------------------------
# 5) Flexible pred parsing
# -----------------------------
def parse_pred_flexible(pred: str, instruction: str = "") -> List[ActionFrame]:
    frames = parse_dsl_plan(pred)
    if frames:
        return frames
    t = _norm(pred)
    obj, tgt, mod = extract_3tuple_from_text(instruction + " " + t)
    frames = []
    if "open" in t: frames.append(ActionFrame("open", None, tgt))
    if "turn on" in t: frames.append(ActionFrame("turn_on", None, tgt))
    if any(x in t for x in ["pick", "grab", "take"]):
        frames.append(ActionFrame("pick", obj))
    if "put in" in t:
        frames.append(ActionFrame("put_in", obj, tgt))
    if "put on" in t:
        frames.append(ActionFrame("place_on", obj, tgt))
    if not frames:
        frames.append(ActionFrame("pick", obj))
    return frames

# -----------------------------
# 6) Metrics
# -----------------------------
def multiset_f1(a: List[str], b: List[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    tp = sum((ca & cb).values())
    fp = sum((cb - ca).values())
    fn = sum((ca - cb).values())
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return 2*p*r/(p+r) if p+r else 0

def lcs_len(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i+1][j+1] = dp[i][j]+1 if a[i]==b[j] else max(dp[i][j+1], dp[i+1][j])
    return dp[-1][-1]

# -----------------------------
# 7) Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--gold_col", required=True)
    ap.add_argument("--pred_col", required=True)
    ap.add_argument("--instruction_col", default="")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rows = []

    for i, r in df.iterrows():
        gold = parse_dsl_plan(r[args.gold_col])
        pred = parse_pred_flexible(r[args.pred_col], r.get(args.instruction_col, ""))

        gold_actions = [f.action for f in gold]
        pred_actions = [f.action for f in pred]

        action_f1 = multiset_f1(gold_actions, pred_actions)
        order = lcs_len(gold_actions, pred_actions) / max(len(gold_actions), 1)
        score = 0.6*action_f1 + 0.4*order

        rows.append({
            "row_id": i,
            "action_f1": action_f1,
            "order_lcs": order,
            "emergent_score": score
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

    with open(args.out_json, "w") as f:
        json.dump({
            "mean_emergent_score": float(out["emergent_score"].mean())
        }, f, indent=2)

if __name__ == "__main__":
    main()
