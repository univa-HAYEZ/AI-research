"""
make_wab_csv.py
"""

from __future__ import annotations
import argparse, re
from typing import List
import pandas as pd

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

OBJ_LEX = [
    "moka pot", "book", "white mug", "yellow mug", "mug", "cup", "black bowl", "bowl",
    "alphabet soup", "bbq sauce", "tomato sauce", "cream cheese", "butter",
    "ketchup", "milk", "orange juice", "salad dressing", "chocolate pudding"
]
TGT_LEX = [
    "stove", "microwave", "basket", "plate", "drawer", "bottom drawer", "caddy", "box", "black box",
]

def find_obj(t: str) -> str:
    for cand in sorted(OBJ_LEX, key=len, reverse=True):
        if cand in t:
            return cand.replace(" ", "_")
    return "object"

def derive_gold_plan(instr: str) -> List[str]:
    t = norm(instr)
    obj = find_obj(t)

    if "stove" in t and ("turn on" in t or "turn_on" in t or "heat" in t):
        return [f"turn_on(stove)", f"place_on({obj},stove)"]

    if "microwave" in t and ("put" in t or "place" in t):
        return [f"open(microwave)", f"put_in({obj},microwave)", f"close(microwave)"]
    if "bottom drawer" in t or ("drawer" in t and "bottom" in t):
        return [f"open(bottom_drawer)", f"put_in({obj},bottom_drawer)", f"close(bottom_drawer)"]
    if "drawer" in t and ("put" in t or "place" in t):
        return [f"open(drawer)", f"put_in({obj},drawer)", f"close(drawer)"]
    if "black box" in t or ("box" in t and ("put" in t or "place" in t)):
        tgt = "black_box" if "black box" in t else "box"
        return [f"open({tgt})", f"put_in({obj},{tgt})", f"close({tgt})"]

    if "basket" in t and ("put" in t or "place" in t or "pick up" in t):
        if "both" in t or "and" in t:
            return ["put_in(item,basket)", "put_in(item,basket)"]
        return [f"put_in({obj},basket)"]

    if "plate" in t and ("put" in t or "place" in t):
        if "and" in t:
            return ["place_on(item,plate)", "place_on(item,plate)"]
        return [f"place_on({obj},plate)"]

    return [f"pick({obj})", f"place_on({obj},surface)"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--lang", choices=["en","ko"], default="en")
    ap.add_argument("--use_transformed", action="store_true", help="Use transformed_instruction if present")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

 
    if args.use_transformed and "transformed_instruction" in df.columns:
        ins = df["transformed_instruction"].fillna("")
    elif "instruction_en" in df.columns:
        ins = df["instruction_en"].fillna("")
    elif "instruction" in df.columns:
        ins = df["instruction"].fillna("")
    elif "raw_instruction" in df.columns:
        ins = df["raw_instruction"].fillna("")
    else:
        raise ValueError("No instruction column found (expected transformed_instruction / instruction_en / instruction / raw_instruction)")

    out = pd.DataFrame()
    out[f"instruction_{args.lang}"] = ins

    gold_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ("gold_plan","gold_action_sequence","actions","gold_actions"):
            gold_col = c
            break

    if gold_col:
        out["gold_plan"] = df[gold_col].astype(str)
    else:
        out["gold_plan"] = [ "; ".join(derive_gold_plan(x)) for x in ins.astype(str).tolist() ]

    out.to_csv(args.out_csv, index=False)
    print(f"[DONE] {args.out_csv} rows={len(out)}")

if __name__ == "__main__":
    main()
