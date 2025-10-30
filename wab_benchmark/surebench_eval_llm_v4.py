#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUREBench (LLM-only) — Compositional 3-Tuple & Horizon Splits
==============================================================
- LLM-only evaluation with Seen-only few-shot prompting
- Robust action formatting via a repair() step
- Reports BOTH:
    1) Compositional split: SEEN vs UNSEEN by 3-tuple composition keys (object, target, modifier)
    2) Horizon split: SHORT vs LONG (by gold plan length threshold)
- Metrics: Success, Action F1 (bag-of-actions), Avg Len Error, Gaps
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import pandas as pd

# ---------------------------
# Text utils
# ---------------------------
def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def standardize_action(a: str) -> str:
    a = norm_text(a)
    a = a.replace("put_on", "place_on").replace("put in", "put_in").replace("place in", "put_in")
    a = a.replace("store in", "put_in").replace("store_in", "put_in")
    a = a.replace("place on", "place_on")
    a = re.sub(r"\(\s*", "(", a)
    a = re.sub(r"\s*\)", ")", a)
    a = re.sub(r"\s*,\s*", ",", a)
    return a

def tokenize_action_seq(seq) -> List[str]:
    if isinstance(seq, str):
        parts = [p.strip() for p in re.split(r"[;,\u2192]+", seq) if p.strip()]
    else:
        parts = [str(x).strip() for x in seq if str(x).strip()]
    return [standardize_action(p) for p in parts]

def f1_score(pred: List[str], gold: List[str]) -> float:
    cp, cg = Counter(pred), Counter(gold)
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

def norm_entity(x: str) -> str:
    x = (x or "").strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", x)

# ---------------------------
# Lexicons to parse object/target/modifier
# ---------------------------
OBJ_LEX = [
    "white mug", "yellow mug", "mug", "cup", "white object", "item", "two cans", "can", "cream cheese",
]

TGT_LEX = [
    "stove", "plate", "plates", "black box", "box", "drawer", "bottom drawer", "basket",
    "caddy", "compartment", "desk", "right of the plate", "left of the plate",
]

# Modifiers: quantity, spatial, container slot, state
MOD_LEX = [
    # quantity
    "two", "both", "single", "one",
    # spatial relations / placement
    "right of the plate", "left of the plate", "near the plate", "on the desk",
    "back compartment", "front compartment", "middle compartment",
    "top drawer", "bottom drawer",
    # state/intent
    "heated", "warm", "hot", "cool",
    # speed/constraint
    "quick", "nearest",
]

def find_first_match(text: str, candidates: List[str]) -> str:
    for c in candidates:
        if c in text:
            return c
    return ""

# ---------------------------
# Allowed verbs & repair
# ---------------------------
ALLOWED_VERBS = ["turn_on", "open", "close", "pick", "place_on", "put_in"]
ACTION_PAT = re.compile(
    r"(turn_on|open|close|pick|place_on|put_in)\s*\(\s*([a-z0-9_]+)?\s*(?:,\s*([a-z0-9_]+)\s*)?\)", re.I
)

def guess_obj_tgt_from_instruction(instr: str) -> Tuple[str, str]:
    t = norm_text(instr)
    obj = find_first_match(t, OBJ_LEX) or "object"
    tgt = find_first_match(t, TGT_LEX) or "target"
    return norm_entity(obj), norm_entity(tgt)

def repair_actions(raw_text: str, fallback_instr: str) -> List[str]:
    """
    1) Try to extract allowed verb calls via regex.
    2) If nothing valid, guess minimal plan from the instruction.
    """
    text = norm_text(raw_text)
    acts = []
    for m in ACTION_PAT.finditer(text):
        verb = m.group(1).lower()
        a1 = norm_entity(m.group(2) or "")
        a2 = norm_entity(m.group(3) or "")
        if verb not in ALLOWED_VERBS:
            continue
        if verb in ["place_on", "put_in"]:
            if not a1 or not a2: 
                continue
            acts.append(f"{verb}({a1},{a2})")
        else:
            if not a1: 
                continue
            acts.append(f"{verb}({a1})")
    if not acts:
        obj, tgt = guess_obj_tgt_from_instruction(fallback_instr)
        if "stove" in text:
            acts = [f"place_on({obj},stove)"]
        elif any(k in text for k in ["drawer","box","basket","caddy","compartment"]):
            container = "drawer" if "drawer" in text else ("box" if "box" in text else ("basket" if "basket" in text else "caddy"))
            acts = [f"open({container})", f"put_in({obj},{container})", f"close({container})"]
        elif "plate" in text:
            acts = [f"place_on({obj},plate)"]
        else:
            acts = [f"pick({obj})", f"place_on({obj},surface)"]
    return tokenize_action_seq(acts)

# ---------------------------
# Heuristic gold plan (fallback when no gold provided)
# ---------------------------
def derive_gold_plan_from_instruction(instr: str) -> List[str]:
    t = norm_text(instr)
    obj = ""
    for cand in sorted(OBJ_LEX, key=len, reverse=True):
        if cand in t:
            obj = cand.replace(" ", "_")
            break
    if not obj:
        obj = "item" if "item" in t else "object"
    if ("heat" in t or "turn on the stove" in t or "turn_on the stove" in t) and "stove" in t:
        return [f"turn_on(stove)", f"place_on({obj},stove)"]
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
    if ("put" in t or "place" in t) and ("on the plate" in t or "on their nearest plates" in t):
        if "both" in t or "two" in t:
            return [f"place_on(cup,plate)", f"place_on(cup,plate)"]
        return [f"place_on({obj},plate)"]
    if "matching item" in t or "matching_item" in t:
        return [f"place_on(white_mug,plate)", f"place_on(color_match,desk_right_of_plate)"]
    if "left of the cup" in t and "caddy" in t:
        return [f"pick(item_left_of(cup))", f"put_in(item,caddy_back)"]
    if "on the" in t:
        tgt = find_first_match(t, ["stove","plate","desk"]) or "surface"
        tgt = tgt.replace(" ", "_")
        return [f"pick({obj})", f"place_on({obj},{tgt})"]
    return [f"pick({obj})", f"place_on({obj},surface)"]

# ---------------------------
# Dataset I/O
# ---------------------------
def load_rows(csv_path: str, lang: str="en") -> List[Dict]:
    df = pd.read_csv(csv_path)
    ins_cols = [c for c in df.columns if "instruction" in c.lower()]
    pref = [c for c in ins_cols if lang in c.lower()] or ins_cols
    if not pref:
        raise ValueError("No instruction column like instruction_en/instruction_ko found.")
    ins_col = pref[0]
    gold_cols = [c for c in df.columns if any(k in c.lower() for k in ["gold_action","gold_plan","actions"])]
    gold_col = gold_cols[0] if gold_cols else None
    scene_cols = [c for c in df.columns if "scene" in c.lower()]
    file_cols  = [c for c in df.columns if "bddl" in c.lower() or "file" in c.lower()]
    rows = []
    for _, r in df.iterrows():
        instr = str(r[ins_col])
        scene = str(r[scene_cols[0]]) if scene_cols else ""
        bddl  = str(r[file_cols[0]])  if file_cols  else ""
        if gold_col and pd.notna(r[gold_col]):
            raw = str(r[gold_col]).strip()
            if raw.startswith("["):
                try:
                    arr = json.loads(raw); gold = [str(x) for x in arr]
                except Exception:
                    gold = [s.strip() for s in re.split(r"[;,\u2192]+", raw) if s.strip()]
            else:
                gold = [s.strip() for s in re.split(r"[;,\u2192]+", raw) if s.strip()]
        else:
            gold = derive_gold_plan_from_instruction(instr)
        rows.append({"instruction": instr, "scene": scene, "bddl": bddl, "gold_plan": tokenize_action_seq(gold)})
    return rows

# ---------------------------
# Compositional 3-tuple split (object, target, modifier)
# ---------------------------
OBJ_PATTERNS = [
    r"(white mug)", r"(yellow mug)", r"(white object)", r"(mug)", r"(cup)", r"(item)", r"(can|two cans)",
    r"(cream cheese)"
]
TGT_PATTERNS = [
    r"(stove)", r"(black box|box)", r"(bottom drawer|drawer)", r"(basket)", r"(plate|plates)",
    r"(caddy|back compartment|compartment|desk)"
]

def normalize_modifier_from_text(t: str) -> str:
    # quantity
    if "two" in t or "both" in t: return "qty_two"
    if "single" in t or "one " in t: return "qty_one"
    # spatial / placement
    if "right of the plate" in t or "desk_right_of_plate" in t: return "right_of_plate"
    if "left of the plate" in t: return "left_of_plate"
    if "near the plate" in t: return "near_plate"
    if "on the desk" in t or "desk " in t: return "on_desk"
    # container slots
    if "back compartment" in t: return "back_compartment"
    if "front compartment" in t: return "front_compartment"
    if "middle compartment" in t: return "middle_compartment"
    if "top drawer" in t: return "top_drawer"
    if "bottom drawer" in t: return "bottom_drawer"
    # state/intent
    if "heated" in t or "warm" in t or "hot" in t: return "heated"
    # speed/constraint
    if "quick" in t: return "quick"
    if "nearest" in t: return "nearest"
    return "none"

def extract_composition_key_3t(instr: str) -> Tuple[str,str,str]:
    t = norm_text(instr)
    # object
    obj = ""
    for p in OBJ_PATTERNS:
        m = re.search(p, t)
        if m: obj = m.group(1); break
    obj = norm_entity(obj) if obj else "object"
    # target
    tgt = ""
    for p in TGT_PATTERNS:
        m = re.search(p, t)
        if m: tgt = m.group(1); break
    tgt = norm_entity(tgt) if tgt else "target"
    # modifier
    mod = normalize_modifier_from_text(t)
    return (obj, tgt, mod)

def build_compositional_split_3t(rows: List[Dict], unseen_ratio: float=0.6, seed: int=42):
    random.seed(seed)
    key2idxs = defaultdict(list)
    for i, r in enumerate(rows):
        key = extract_composition_key_3t(r["instruction"])
        key2idxs[key].append(i)
    keys = list(key2idxs.keys()); random.shuffle(keys)
    n_unseen = max(1, int(len(keys) * unseen_ratio))
    unseen_keys = set(keys[:n_unseen])
    seen_idxs, unseen_idxs = [], []
    for k, idxs in key2idxs.items():
        (unseen_idxs if k in unseen_keys else seen_idxs).extend(idxs)
    return seen_idxs, unseen_idxs, key2idxs

# ---------------------------
# Horizon split
# ---------------------------
def build_horizon_split(rows: List[Dict], len_threshold: int=3):
    short_idxs, long_idxs = [], []
    for i, r in enumerate(rows):
        (long_idxs if len(r["gold_plan"]) >= len_threshold else short_idxs).append(i)
    return short_idxs, long_idxs

# ---------------------------
# LLM planner
# ---------------------------
class LMPlanner:
    name = "lm"
    def __init__(self, lm_name="google/flan-t5-large", device="cpu"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tok = AutoTokenizer.from_pretrained(lm_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(lm_name).to(device)
        self.device = device

    def predict(self, instruction: str, fewshot: str = "") -> List[str]:
        preamble = (
            "You are a planner that outputs ONLY action calls.\n"
            "Allowed verbs:\n"
            "- turn_on(x)\n- open(x)\n- close(x)\n- pick(x)\n- place_on(x,y)\n- put_in(x,y)\n\n"
            "STRICT RULES:\n"
            "1) Output MUST be a single line of semicolon-separated calls (e.g., pick(cup); open(drawer); put_in(cup,drawer)).\n"
            "2) No explanations, no extra words, no quotes, no newlines.\n"
            "3) Use lowercase tokens and underscores inside names.\n"
            "4) If target is a container (drawer/box), include open() before and close() after put_in().\n"
        )
        prompt = preamble
        if fewshot:
            prompt += "\n=== examples (seen only) ===\n" + fewshot + "\n=== end examples ===\n"
        prompt += f"\nInstruction: {instruction}\nAction plan:"
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, do_sample=False, temperature=0.0, max_new_tokens=128)
        text = self.tok.decode(out[0], skip_special_tokens=True).strip()
        actions = [a.strip() for a in text.split(";") if a.strip()]
        actions = tokenize_action_seq(actions)
        if not actions:
            actions = repair_actions(text, instruction)
        return actions

# ---------------------------
# Evaluation
# ---------------------------
def evaluate(rows: List[Dict], idxs: List[int], planner: LMPlanner, fewshot: str="") -> Dict:
    succ, f1s, lens = [], [], []
    for i in idxs:
        instr = rows[i]["instruction"]; gold = rows[i]["gold_plan"]
        pred = planner.predict(instr, fewshot=fewshot)
        pred = tokenize_action_seq(pred)
        succ.append(int(pred == gold))
        f1s.append(f1_score(pred, gold))
        lens.append(avg_len_error(pred, gold))
    if not succ:
        return {"success_rate": 0.0, "action_f1": 0.0, "avg_len_error": 0.0, "count": 0}
    return {
        "success_rate": sum(succ)/len(succ),
        "action_f1": sum(f1s)/len(f1s),
        "avg_len_error": sum(lens)/len(lens),
        "count": len(succ),
    }

def build_fewshot_from_seen(rows, seen_idxs, k=6, seed=7):
    rnd = random.Random(seed)
    if not seen_idxs: return ""
    picks = rnd.sample(seen_idxs, min(k, len(seen_idxs)))
    lines = []
    for i in picks:
        ins = rows[i]["instruction"]
        gold = "; ".join(rows[i]["gold_plan"])
        lines.append(f"Instruction: {ins}\nAction plan: {gold}")
    return "\n\n".join(lines)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--lm_name", default="google/flan-t5-large")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lang", choices=["en","ko"], default="en")
    ap.add_argument("--unseen_ratio", type=float, default=0.6, help="ratio on 3-tuple composition keys")
    ap.add_argument("--len_threshold", type=int, default=3, help=">= threshold → long-horizon")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fewshot_k", type=int, default=14)
    ap.add_argument("--fewshot_seed", type=int, default=7)
    args = ap.parse_args()

    rows = load_rows(args.csv, lang=args.lang)

    # 3-tuple Compositional split
    seen_idxs, unseen_idxs, key2idxs = build_compositional_split_3t(rows, unseen_ratio=args.unseen_ratio, seed=args.seed)
    # Horizon split
    short_idxs, long_idxs = build_horizon_split(rows, len_threshold=args.len_threshold)

    # Planner + Seen-only few-shot
    planner = LMPlanner(lm_name=args.lm_name, device=args.device)
    fewshot = build_fewshot_from_seen(rows, seen_idxs, k=args.fewshot_k, seed=args.fewshot_seed)

    # Evaluate splits
    overall_stats = evaluate(rows, list(range(len(rows))), planner, fewshot=fewshot)
    seen_stats    = evaluate(rows, seen_idxs, planner, fewshot=fewshot)
    unseen_stats  = evaluate(rows, unseen_idxs, planner, fewshot=fewshot)
    short_stats   = evaluate(rows, short_idxs, planner, fewshot=fewshot)
    long_stats    = evaluate(rows, long_idxs, planner, fewshot=fewshot)

    out = {
        "overall": overall_stats,
        "compositional_3tuple": {
            "seen": seen_stats,
            "unseen": unseen_stats,
            "gap_action_f1": seen_stats["action_f1"] - unseen_stats["action_f1"],
            "counts": {
                "n_seen_samples": len(seen_idxs),
                "n_unseen_samples": len(unseen_idxs),
                "n_seen_keys": len(set(extract_composition_key_3t(rows[i]['instruction']) for i in seen_idxs)),
                "n_unseen_keys": len(set(extract_composition_key_3t(rows[i]['instruction']) for i in unseen_idxs)),
            }
        },
        "horizon": {
            "short": short_stats,
            "long": long_stats,
            "gap_action_f1": short_stats["action_f1"] - long_stats["action_f1"],
            "counts": {
                "n_short": len(short_idxs),
                "n_long": len(long_idxs),
                "len_threshold": args.len_threshold
            }
        },
        "meta": {
            "csv": args.csv,
            "lm_name": args.lm_name,
            "lang": args.lang,
            "unseen_ratio": args.unseen_ratio,
            "fewshot_k": args.fewshot_k,
            "seed": args.seed,
            "composition_key": "object,target,modifier"
        }
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
