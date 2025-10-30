# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_individual_instructions.py

Few-shot 컨텍스트를 고정해 둔 상태에서, instruction을 '하나씩' 평가하여
각 instruction별 점수(success_rate(=strict), action_f1, avg_len_error 등)를
CSV로 저장합니다. (배치 평균이 아니라 '개별 데이터' 점수 분포를 얻기 위함)

Usage:
  python evaluate_individual_instructions.py \
      --csv libero_instructions_patched.csv \
      --agent lm \
      --lm_name google/flan-t5-large \
      --lang en \
      --fewshot_k 10 \
      --unseen_ratio 0.5 \
      --out_csv per_instruction_scores.csv \
      --out_json summary.json
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

# ---------------------------
# Action normalization & metrics
# ---------------------------
def standardize_action(a: str) -> str:
    a = (a or "").strip().lower()
    # verb synonyms
    a = a.replace("put on", "place_on").replace("put_on", "place_on").replace("put onto", "place_on")
    a = a.replace("put in", "put_in").replace("put_into", "put_in").replace("place in", "put_in").replace("place into", "put_in")
    a = a.replace("store in", "put_in").replace("store_in", "put_in")
    a = a.replace("turn on", "turn_on").replace("turn-on", "turn_on")
    # spaces in args
    a = re.sub(r"\(\s*", "(", a)
    a = re.sub(r"\s*\)", ")", a)
    a = re.sub(r"\s*,\s*", ",", a)
    return a

def tokenize_action_seq(seq_str_or_list) -> List[str]:
    if isinstance(seq_str_or_list, str):
        parts = [p.strip() for p in seq_str_or_list.split(";") if p.strip()]
    else:
        parts = [str(p).strip() for p in seq_str_or_list if str(p).strip()]
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

# ---------------------------
# Very simple gold fallback (optional, rarely used)
# ---------------------------
def derive_gold_plan_from_instruction(instr: str) -> List[str]:
    t = norm_text(instr)
    # crude fallback
    if "drawer" in t or "box" in t or "basket" in t or "caddy" in t:
        return ["open(drawer)", "pick(object)", "put_in(object,drawer)", "close(drawer)"]
    if "plate" in t:
        return ["pick(object)", "place_on(object,plate)"]
    if "stove" in t and ("turn on" in t or "turn_on" in t or "heat" in t):
        return ["turn_on(stove)", "place_on(object,stove)"]
    return ["pick(object)", "place_on(object,surface)"]

# ---------------------------
# Composition key & split
# ---------------------------
OBJ_PATTERNS = [
    r"(white mug)", r"(yellow mug)", r"(moka pot|moka)", r"(cup|mug)", r"(book)", r"(plate)", r"(pot|pan)",
    r"(cream cheese)", r"(item|object)"
]
TGT_PATTERNS = [
    r"(stove)", r"(plate|plates)", r"(drawer|bottom drawer)", r"(basket|caddy|compartment|shelf)",
    r"(sink|counter|table|desk|box|black box)"
]
MOD_PATTERNS = [
    r"(left of .*|right of .*|front of .*|back of .*|nearest .*|rear section|back compartment)"
]

def extract_composition_key(instr: str) -> Tuple[str, str, str]:
    t = norm_text(instr)
    def first(pats):
        for p in pats:
            m = re.search(p, t)
            if m: return m.group(1)
        return ""
    obj = first(OBJ_PATTERNS).replace(" ", "_") or "object"
    tgt = first(TGT_PATTERNS).replace(" ", "_") or "target"
    mod = first(MOD_PATTERNS).replace(" ", "_") or "none"
    return obj, tgt, mod

def build_seen_unseen_split(rows: List[Dict], unseen_ratio: float = 0.5, seed: int = 42):
    rnd = random.Random(seed)
    key2idxs = defaultdict(list)
    for i, r in enumerate(rows):
        key = extract_composition_key(r["instruction"])
        key2idxs[key].append(i)
    keys = list(key2idxs.keys())
    rnd.shuffle(keys)
    n_unseen = max(1, int(len(keys) * unseen_ratio))
    unseen_keys = set(keys[:n_unseen])
    seen_idxs, unseen_idxs = [], []
    for k, idxs in key2idxs.items():
        (unseen_idxs if k in unseen_keys else seen_idxs).extend(idxs)
    return seen_idxs, unseen_idxs, key2idxs

# ---------------------------
# Planners
# ---------------------------
class BasePlanner:
    name = "base"
    def predict(self, instruction: str) -> List[str]:
        raise NotImplementedError

class NaivePlanner(BasePlanner):
    name = "naive"
    def predict(self, instruction: str) -> List[str]:
        t = norm_text(instruction)
        if "drawer" in t or "box" in t or "basket" in t or "caddy" in t:
            return ["put_in(object,container)"]
        if "plate" in t:
            return ["place_on(object,plate)"]
        if "stove" in t:
            return ["place_on(object,stove)"]
        return ["place_on(object,surface)"]

class RulePlanner(BasePlanner):
    name = "rule"
    def predict(self, instruction: str) -> List[str]:
        return derive_gold_plan_from_instruction(instruction)

class LMPlanner(BasePlanner):
    name = "lm"
    def __init__(self, lm_name: str = "google/flan-t5-small", device: str = "cpu",
                 max_new_tokens: int = 96, temperature: float = 0.0, top_p: float = 1.0):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except Exception as e:
            raise RuntimeError("Transformers not installed. `pip install transformers`") from e
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(lm_name)
        self.model.to(device)
        self.device = device
        self.gen_kwargs = dict(max_new_tokens=max_new_tokens,
                               do_sample=(temperature > 0.0),
                               temperature=temperature,
                               top_p=top_p)

    def _format_prompt(self, instruction: str, fewshot_block: str = "") -> str:
        preface = (
            "Return ONLY a semicolon-separated plan using exactly these verbs:\n"
            "turn_on(x); open(x); close(x); pick(x); place_on(x,y); put_in(x,y)\n"
            "Use lowercase and parentheses, no extra words.\n"
        )
        if fewshot_block:
            preface += "\n=== examples ===\n" + fewshot_block + "\n=== end examples ===\n"
        prompt = (
            f"{preface}\n"
            f"Instruction: {instruction}\n"
            f"Action plan:"
        )
        return prompt

    def predict(self, instruction: str, fewshot_block: str = "") -> List[str]:
        prompt = self._format_prompt(instruction, fewshot_block=fewshot_block)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, **self.gen_kwargs)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        # If LM produced natural text, try a second-pass formatter
        acts = [a.strip() for a in text.split(";") if a.strip()]
        valid = [re.match(r'^(turn_on|open|close|pick|place_on|put_in)\([a-z0-9_]+(,[a-z0-9_]+)?\)$', a) for a in acts]
        if not acts or not all(valid):
            # try to reformat
            fix_prompt = (
                "Reformat the following text into valid semicolon-separated actions using only:\n"
                "turn_on(x); open(x); close(x); pick(x); place_on(x,y); put_in(x,y)\n"
                f"Text: {text}\n"
                "Action plan:"
            )
            inputs2 = self.tokenizer(fix_prompt, return_tensors="pt").to(self.device)
            outputs2 = self.model.generate(**inputs2, **self.gen_kwargs)
            text2 = self.tokenizer.decode(outputs2[0], skip_special_tokens=True).strip().lower()
            acts = [a.strip() for a in text2.split(";") if a.strip()]
        return tokenize_action_seq(acts)

# ---------------------------
# Few-shot builder
# ---------------------------
def build_fewshot_from_seen(rows: List[Dict], seen_idxs: List[int], k: int = 6, seed: int = 7) -> str:
    if not seen_idxs or k <= 0:
        return ""
    rnd = random.Random(seed)
    picks = rnd.sample(seen_idxs, min(k, len(seen_idxs)))
    lines = []
    for i in picks:
        ins = rows[i]["instruction"]
        gold = "; ".join(rows[i]["gold_plan"])
        lines.append(f"Instruction: {ins}\nAction plan: {gold}")
    return "\n\n".join(lines)

# ---------------------------
# Data loading
# ---------------------------
def load_rows(csv_path: str, lang: str = "en") -> List[Dict]:
    df = pd.read_csv(csv_path)
    # instruction column
    ins_cols = [c for c in df.columns if "instruction" in c.lower()]
    if lang == "en":
        pref = [c for c in ins_cols if "en" in c.lower()] or ins_cols
    else:
        pref = [c for c in ins_cols if "ko" in c.lower()] or ins_cols
    if not pref:
        raise ValueError("No instruction column found. Expected columns like instruction_en/instruction_ko/instruction.")
    ins_col = pref[0]

    # gold column
    gold_cols = [c for c in df.columns if any(k in c.lower() for k in ["gold_action", "gold_plan", "actions"])]
    gold_col = gold_cols[0] if gold_cols else None

    rows = []
    for _, r in df.iterrows():
        instr = str(r[ins_col])
        if gold_col and isinstance(r[gold_col], str) and r[gold_col].strip():
            gold = tokenize_action_seq(r[gold_col])
        else:
            gold = derive_gold_plan_from_instruction(instr)
        rows.append({"instruction": instr, "gold_plan": gold})
    return rows

# ---------------------------
# Per-instruction evaluation
# ---------------------------
def evaluate_one(instr: str, gold: List[str], planner, fewshot_block: str = "", len_threshold: int = 3):
    # predict
    if isinstance(planner, LMPlanner):
        pred = planner.predict(instr, fewshot_block=fewshot_block)
    else:
        pred = planner.predict(instr)
    pred = tokenize_action_seq(pred)
    # metrics
    f1 = f1_score(pred, gold)
    le = avg_len_error(pred, gold)
    success = int(pred == gold)
    horizon = "long" if len(gold) >= len_threshold else "short"
    return {
        "pred": pred, "action_f1": f1, "avg_len_error": le,
        "success": success, "horizon": horizon
    }

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--agent", choices=["naive", "rule", "lm"], default="lm")
    ap.add_argument("--lm_name", default="google/flan-t5-large")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lang", choices=["en", "ko"], default="en")
    ap.add_argument("--unseen_ratio", type=float, default=0.5)
    ap.add_argument("--fewshot_k", type=int, default=10)
    ap.add_argument("--fewshot_seed", type=int, default=7)
    ap.add_argument("--len_threshold", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--out_csv", default="per_instruction_scores.csv")
    ap.add_argument("--out_json", default="summary.json")
    args = ap.parse_args()

    rows = load_rows(args.csv, lang=args.lang)
    seen_idxs, unseen_idxs, key2idxs = build_seen_unseen_split(rows, unseen_ratio=args.unseen_ratio, seed=42)

    # planner
    if args.agent == "naive":
        planner = NaivePlanner()
    elif args.agent == "rule":
        planner = RulePlanner()
    else:
        planner = LMPlanner(lm_name=args.lm_name, device=args.device,
                            temperature=args.temperature, top_p=args.top_p)

    fewshot_block = build_fewshot_from_seen(rows, seen_idxs, k=args.fewshot_k, seed=args.fewshot_seed)

    per_rows = []
    for i, r in enumerate(rows):
        instr = r["instruction"]
        gold = r["gold_plan"]
        res = evaluate_one(instr, gold, planner, fewshot_block=fewshot_block, len_threshold=args.len_threshold)
        key = extract_composition_key(instr)
        split = "unseen" if any(i in unseen_idxs for i in [i]) else "seen" 
        per_rows.append({
            "idx": i,
            "instruction": instr,
            "gold_plan": "; ".join(gold),
            "pred_plan": "; ".join(res["pred"]),
            "action_f1": res["action_f1"],
            "avg_len_error": res["avg_len_error"],
            "success": res["success"],
            "horizon": res["horizon"],
            "composition_key": "|".join(key),
            "split": split
        })

    df = pd.DataFrame(per_rows)
    df.to_csv(args.out_csv, index=False)

    def mean(series):
        return float(series.mean()) if len(series) else 0.0

    overall = {
        "success_rate": mean(df["success"]),
        "action_f1": mean(df["action_f1"]),
        "avg_len_error": mean(df["avg_len_error"]),
        "count": int(len(df))
    }
    seen_df = df[df["split"] == "seen"]
    unseen_df = df[df["split"] == "unseen"]
    seen_stats = {"success_rate": mean(seen_df["success"]), "action_f1": mean(seen_df["action_f1"]),
                  "avg_len_error": mean(seen_df["avg_len_error"]), "count": int(len(seen_df))}
    unseen_stats = {"success_rate": mean(unseen_df["success"]), "action_f1": mean(unseen_df["action_f1"]),
                    "avg_len_error": mean(unseen_df["avg_len_error"]), "count": int(len(unseen_df))}
    gap_action_f1 = seen_stats["action_f1"] - unseen_stats["action_f1"]

    out = {
        "overall": overall,
        "seen": seen_stats,
        "unseen": unseen_stats,
        "generalization_gap_action_f1": gap_action_f1,
        "meta": {
            "csv": args.csv,
            "agent": getattr(planner, "name", "lm"),
            "lm_name": getattr(planner, "tokenizer", None).__class__.__name__ if isinstance(planner, LMPlanner) else None,
            "lang": args.lang,
            "unseen_ratio": args.unseen_ratio,
            "fewshot_k": args.fewshot_k,
            "seed": args.fewshot_seed,
            "len_threshold": args.len_threshold
        }
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[Saved] per-sample scores -> {args.out_csv}")

if __name__ == "__main__":
    main()
