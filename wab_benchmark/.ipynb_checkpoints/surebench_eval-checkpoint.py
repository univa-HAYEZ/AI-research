#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUREBench: Seen–Unseen Evaluation Benchmark (prototype)
=======================================================

Run emergent-capability evaluation on a LIBERO-style instruction dataset.

Usage:
    python surebench_eval.py --csv path/to/libero_instructions_compressed.csv --agent rule
    python surebench_eval.py --csv path/to/libero_instructions_compressed.csv --agent naive
    # (optional; requires transformers + a local model or internet)
    python surebench_eval.py --csv path/to/libero_instructions_compressed.csv --agent lm --lm_name google/flan-t5-small

Notes:
- This prototype tries to be robust to CSV schema differences. It looks for likely
  column names such as 'instruction_en', 'instruction', 'instruction_ko', and
  'gold_action_sequence' / 'actions' / 'gold_plan'. If no gold plan is present,
  it will auto-derive a gold plan using simple rules from the instruction text.
- The LM agent is optional; if unavailable, use 'naive' or 'rule' which are fully offline.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import pandas as pd


# ---------------------------
# Utility: text normalization
# ---------------------------
def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------
# Parse object/target from instruction (very simple heuristics)
# ---------------------------
OBJ_LEX = [
    "white mug", "yellow mug", "mug", "cup", "white object", "item", "two cans", "can", "cream cheese",
]
TGT_LEX = [
    "stove", "plate", "plates", "black box", "box", "drawer", "bottom drawer", "basket", "caddy", "compartment",
    "desk", "right of the plate", "left of the plate",
]

def find_first_match(text: str, candidates: List[str]) -> str:
    for c in candidates:
        if c in text:
            return c
    return ""


# ---------------------------
# Action vocabulary & helpers
# ---------------------------
# We normalize actions as verb(args) strings.
def standardize_action(a: str) -> str:
    a = norm_text(a)
    # unify synonyms
    a = a.replace("put_on", "place_on").replace("put in", "put_in").replace("place in", "put_in")
    a = a.replace("store in", "put_in").replace("store_in", "put_in")
    a = a.replace("place on", "place_on")
    # remove spaces inside parentheses
    a = re.sub(r"\(\s*", "(", a)
    a = re.sub(r"\s*\)", ")", a)
    a = re.sub(r"\s*,\s*", ",", a)
    return a

def tokenize_action_seq(seq: List[str]) -> List[str]:
    return [standardize_action(s) for s in seq if s and str(s).strip() != ""]

def f1_score(pred: List[str], gold: List[str]) -> float:
    # Bag-of-actions F1 (order-agnostic); robust for prototypes
    cp = Counter(pred)
    cg = Counter(gold)
    tp = sum((cp & cg).values())
    fp = sum((cp - cg).values())
    fn = sum((cg - cp).values())
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return 2 * precision * recall / (precision + recall + 1e-9)

def avg_len_error(pred: List[str], gold: List[str]) -> float:
    return abs(len(pred) - len(gold))


# ---------------------------
# Gold plan derivation (fallback if not provided)
# ---------------------------
def derive_gold_plan_from_instruction(instr: str) -> List[str]:
    """
    Heuristic rules to make a reasonable gold plan for common LIBERO-style tasks.
    You can extend these rules as needed.
    """
    t = norm_text(instr)

    obj = ""
    # pick the longest matching object phrase
    for cand in sorted(OBJ_LEX, key=len, reverse=True):
        if cand in t:
            obj = cand.replace(" ", "_")
            break
    if not obj:
        # fallback generic 'item' or 'object'
        obj = "item" if "item" in t else "object"

    # STOVE / HEAT
    if ("heat" in t or "turn on the stove" in t or "turn_on the stove" in t) and "stove" in t:
        return [f"turn_on(stove)", f"place_on({obj},stove)"]

    # PUT/STORE IN BOX / DRAWER / BASKET / CADDY
    if ("put" in t or "store" in t or "place" in t) and "in the" in t:
        tgt = ""
        if "black box" in t or "box" in t:
            tgt = "black_box" if "black box" in t else "box"
        elif "bottom drawer" in t or "drawer" in t:
            tgt = "bottom_drawer" if "bottom drawer" in t else "drawer"
        elif "basket" in t:
            tgt = "basket"
        elif "caddy" in t or "compartment" in t:
            tgt = "caddy_back" if "back compartment" in t else "caddy"
        if tgt in ("drawer", "bottom_drawer", "box", "black_box"):
            return [f"open({tgt})", f"put_in({obj},{tgt})", f"close({tgt})"]
        if tgt:
            return [f"put_in({obj},{tgt})"]

    # PUT/PLACE ON PLATE / DESK SIDE
    if ("put" in t or "place" in t) and ("on the plate" in t or "on their nearest plates" in t or "on the plate" in t):
        if "both" in t or "two" in t:
            # approximate multi-object as two place actions
            return [f"place_on(cup,plate)", f"place_on(cup,plate)"]
        return [f"place_on({obj},plate)"]

    # COLOR MATCH RIGHT OF PLATE
    if "matching item" in t or "matching item to the right of the plate" in t or "matching_item" in t:
        return [f"place_on(white_mug,plate)", f"place_on(color_match,desk_right_of_plate)"]

    # LEFT/RIGHT/FRONT of something
    if "left of the cup" in t and "caddy" in t:
        return [f"pick(item_left_of(cup))", f"put_in(item,caddy_back)"]

    # fallback single pick/place
    if "on the" in t:
        tgt = find_first_match(t, ["stove", "plate", "desk"])
        tgt = tgt.replace(" ", "_") if tgt else "surface"
        return [f"pick({obj})", f"place_on({obj},{tgt})"]

    return [f"pick({obj})", f"place_on({obj},surface)"]


# ---------------------------
# Planner implementations
# ---------------------------
class BasePlanner:
    name = "base"
    def predict(self, instruction: str) -> List[str]:
        raise NotImplementedError

class NaivePlanner(BasePlanner):
    name = "naive"
    def predict(self, instruction: str) -> List[str]:
        t = norm_text(instruction)
        # extremely naive: map to a single-step or two-step ignoring open/close
        if "stove" in t:
            return ["place_on(object,stove)"]
        if "drawer" in t or "box" in t or "basket" in t or "caddy" in t:
            return ["put_in(object,container)"]
        if "plate" in t:
            return ["place_on(object,plate)"]
        return ["place_on(object,surface)"]

class RulePlanner(BasePlanner):
    name = "rule"
    def predict(self, instruction: str) -> List[str]:
        # mirror our gold-derivation rules (but not identical) to test sensitivity
        plan = derive_gold_plan_from_instruction(instruction)
        # drop one step occasionally to simulate imperfect rule behavior
        t = norm_text(instruction)
        if ("open(" in " ".join(plan) and "close(" in " ".join(plan)) and ("quick" in t or "nearest" in t):
            # simulate skipping close in some "quick" cases
            plan = [p for p in plan if not p.startswith("close(")]
        return plan

class LMPlanner(BasePlanner):
    name = "lm"
    def __init__(self, lm_name: str = "google/flan-t5-small", device: str = "cpu"):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except Exception as e:
            raise RuntimeError("Transformers not available. Install 'transformers' to use LMPlanner.") from e
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(lm_name)
        self.device = device
        self.model.to(device)

    def predict(self, instruction: str) -> List[str]:
        from transformers import AutoTokenizer  # for type hints
        prompt = (
            "Translate the instruction into a step-by-step action plan using the following verbs: "
            "turn_on(x); open(x); close(x); pick(x); place_on(x,y); put_in(x,y). "
            "Use lowercase tokens and arguments, and return actions separated by semicolons.\n"
            f"Instruction: {instruction}\n"
            "Action plan:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=64)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        # split by ';' and clean
        acts = [a.strip() for a in text.split(";") if a.strip()]
        return tokenize_action_seq(acts)


# ---------------------------
# Dataset loading
# ---------------------------
def load_rows(csv_path: str, lang: str = "en") -> List[Dict]:
    df = pd.read_csv(csv_path)
    # Try to find instruction column
    ins_cols = [c for c in df.columns if "instruction" in c.lower()]
    if lang == "en":
        pref = [c for c in ins_cols if "en" in c.lower()] or ins_cols
    else:
        pref = [c for c in ins_cols if "ko" in c.lower()] or ins_cols
    if not pref:
        raise ValueError("No instruction column found. Expected columns like 'instruction_en' or 'instruction'.")
    ins_col = pref[0]

    # Try to find gold plan column
    gold_cols = [c for c in df.columns if any(k in c.lower() for k in ["gold_action", "gold_plan", "actions"])]
    gold_col = gold_cols[0] if gold_cols else None

    # Optional scene/file columns
    scene_cols = [c for c in df.columns if "scene" in c.lower()]
    file_cols = [c for c in df.columns if "bddl" in c.lower() or "file" in c.lower()]

    rows = []
    for _, r in df.iterrows():
        instr = str(r[ins_col])
        scene = str(r[scene_cols[0]]) if scene_cols else ""
        bddl = str(r[file_cols[0]]) if file_cols else ""
        if gold_col and pd.notna(r[gold_col]):
            raw = str(r[gold_col])
            # allow formats: "a;b;c" or "['a', 'b']"
            if raw.strip().startswith("["):
                try:
                    arr = json.loads(raw)
                    gold = [str(x) for x in arr]
                except Exception:
                    gold = [s.strip() for s in re.split(r"[;,\u2192]+", raw) if s.strip()]
            else:
                gold = [s.strip() for s in re.split(r"[;,\u2192]+", raw) if s.strip()]
        else:
            gold = derive_gold_plan_from_instruction(instr)

        rows.append({
            "instruction": instr,
            "scene": scene,
            "bddl": bddl,
            "gold_plan": tokenize_action_seq(gold),
        })
    return rows


# ---------------------------
# Compositional key & split
# ---------------------------
OBJ_PATTERNS = [
    r"(white mug)", r"(yellow mug)", r"(white object)", r"(mug)", r"(cup)", r"(item)", r"(can|two cans)",
    r"(cream cheese)"
]
TGT_PATTERNS = [
    r"(stove)", r"(black box|box)", r"(bottom drawer|drawer)", r"(basket)", r"(plate|plates)",
    r"(caddy|back compartment|compartment)"
]

def extract_composition_key(instr: str) -> Tuple[str, str]:
    t = norm_text(instr)
    obj = ""
    tgt = ""
    for p in OBJ_PATTERNS:
        m = re.search(p, t)
        if m: obj = m.group(1); break
    for p in TGT_PATTERNS:
        m = re.search(p, t)
        if m: tgt = m.group(1); break
    obj = obj.replace(" ", "_") if obj else "object"
    tgt = tgt.replace(" ", "_") if tgt else "target"
    return obj, tgt

def build_seen_unseen_split(rows: List[Dict], unseen_ratio: float = 0.3, seed: int = 42):
    random.seed(seed)
    key2idxs = defaultdict(list)
    for i, r in enumerate(rows):
        key = extract_composition_key(r["instruction"])
        key2idxs[key].append(i)
    keys = list(key2idxs.keys())
    random.shuffle(keys)
    n_unseen = max(1, int(len(keys) * unseen_ratio))
    unseen_keys = set(keys[:n_unseen])
    seen_idxs, unseen_idxs = [], []
    for k, idxs in key2idxs.items():
        if k in unseen_keys:
            unseen_idxs.extend(idxs)
        else:
            seen_idxs.extend(idxs)
    return seen_idxs, unseen_idxs, key2idxs


# ---------------------------
# Evaluation loop
# ---------------------------
def evaluate(rows: List[Dict], idxs: List[int], planner: BasePlanner) -> Dict:
    succ, f1s, lens = [], [], []
    for i in idxs:
        instr = rows[i]["instruction"]
        gold = rows[i]["gold_plan"]
        pred = planner.predict(instr)
        pred = tokenize_action_seq(pred)
        # Success (strict)
        success = int(pred == gold)
        succ.append(success)
        # F1
        f1s.append(f1_score(pred, gold))
        # Length error
        lens.append(avg_len_error(pred, gold))
    if not succ:
        return {"success_rate": 0.0, "action_f1": 0.0, "avg_len_error": 0.0, "count": 0}
    return {
        "success_rate": sum(succ) / len(succ),
        "action_f1": sum(f1s) / len(f1s),
        "avg_len_error": sum(lens) / len(lens),
        "count": len(succ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to libero_instructions_compressed.csv")
    ap.add_argument("--agent", choices=["naive", "rule", "lm"], default="rule")
    ap.add_argument("--lm_name", default="google/flan-t5-small", help="HF model name if agent=lm")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lang", choices=["en", "ko"], default="en")
    ap.add_argument("--unseen_ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_rows(args.csv, lang=args.lang)
    seen_idxs, unseen_idxs, key2idxs = build_seen_unseen_split(rows, unseen_ratio=args.unseen_ratio, seed=args.seed)

    if args.agent == "naive":
        planner = NaivePlanner()
    elif args.agent == "rule":
        planner = RulePlanner()
    else:
        planner = LMPlanner(lm_name=args.lm_name, device=args.device)

    seen_stats = evaluate(rows, seen_idxs, planner)
    unseen_stats = evaluate(rows, unseen_idxs, planner)

    overall_stats = evaluate(rows, list(range(len(rows))), planner)

    gap_f1 = seen_stats["action_f1"] - unseen_stats["action_f1"]
    out = {
        "overall": overall_stats,
        "seen": seen_stats,
        "unseen": unseen_stats,
        "generalization_gap_action_f1": gap_f1,
        "meta": {
            "csv": args.csv,
            "agent": planner.name,
            "unseen_ratio": args.unseen_ratio,
            "seed": args.seed,
            "n_seen_keys": len(set(extract_composition_key(rows[i]['instruction']) for i in seen_idxs)),
            "n_unseen_keys": len(set(extract_composition_key(rows[i]['instruction']) for i in unseen_idxs)),
        }
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
