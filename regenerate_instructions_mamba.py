#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regenerate_instructions_mamba.py
"""

from __future__ import annotations
import argparse, os, re, json, time
from typing import List, Dict, Tuple, Optional
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def pick_raw_column(df: pd.DataFrame) -> str:
    cands = ["raw_instruction", "instruction", "instruction_en"]
    for c in cands:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find raw instruction column. Tried {cands}. Found: {df.columns.tolist()}")

SYN = {
    "moka pot": ["moka pot", "mokapot", "moka"],
    "stove": ["stove", "burner", "hob"],
    "drawer": ["drawer"],
    "bottom drawer": ["bottom drawer", "lower drawer"],
    "microwave": ["microwave"],
    "basket": ["basket"],
    "plate": ["plate"],
    "caddy": ["caddy"],
    "book": ["book"],
    "mug": ["mug", "cup"],
}

def extract_keywords(raw: str) -> List[str]:
    t = raw.lower()
    keys = []
    # objects
    for k in ["moka pot", "book", "mug", "cup", "bowl", "butter", "cream cheese", "tomato sauce",
              "alphabet soup", "bbq sauce", "ketchup", "milk", "orange juice", "salad dressing", "chocolate pudding"]:
        if k in t:
            keys.append(k)
    # targets
    for k in ["stove", "microwave", "basket", "plate", "drawer", "bottom drawer", "caddy", "box"]:
        if k in t:
            keys.append(k)
    return list(dict.fromkeys(keys))  # unique preserve order

def keyword_ok(raw: str, transformed: str, min_hit_ratio: float = 0.6) -> Tuple[bool, str]:
    raw = raw or ""
    transformed = (transformed or "").lower()
    keys = extract_keywords(raw)
    if not keys:
        return True, "no_keywords"
    hits = 0
    for k in keys:
        vocab = SYN.get(k, [k])
        if any(v in transformed for v in vocab):
            hits += 1
    ratio = hits / max(1, len(keys))
    return (ratio >= min_hit_ratio), f"hit_ratio={ratio:.2f} keys={keys}"

def build_prompt(raw_instruction: str) -> str:
    return (
        "Rewrite the following robot instruction into ONE concise natural sentence.\n"
        "Constraints:\n"
        "- Keep the SAME objects and SAME target locations.\n"
        "- Do NOT add new goals.\n"
        "- Do NOT remove required objects.\n"
        "- Output ONLY the rewritten instruction.\n"
        f"Instruction: {raw_instruction}\n"
        "Rewritten:"
    )

@torch.no_grad()
def generate_one(model, tok, prompt: str, device: str, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float) -> str:
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
    )
    txt = tok.decode(out[0], skip_special_tokens=True)
    if "Rewritten:" in txt:
        txt = txt.split("Rewritten:")[-1]
    return norm(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model", required=True, help="HF model name/path for Mamba")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--n_variants", type=int, default=1, help="Generate N candidates per raw instruction")
    ap.add_argument("--max_retries", type=int, default=2, help="Retry if guardrail fails")
    ap.add_argument("--min_hit_ratio", type=float, default=0.6)
    ap.add_argument("--partial_save_every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    raw_col = pick_raw_column(df)

    if args.resume and os.path.exists(args.out_csv):
        out_df = pd.read_csv(args.out_csv)
        if len(out_df) == len(df) and "transformed_instruction" in out_df.columns:
            df = out_df
            print(f"[INFO] Resuming from {args.out_csv}")
        else:
            print(f"[WARN] Cannot resume cleanly (row mismatch). Will overwrite.")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if "cuda" in args.device else None)
    model.to(args.device)
    model.eval()

    if "transformed_instruction" not in df.columns:
        df["transformed_instruction"] = ""
    if "regen_status" not in df.columns:
        df["regen_status"] = ""
    if "regen_reason" not in df.columns:
        df["regen_reason"] = ""

    for v in range(1, args.n_variants + 1):
        col = f"transformed_instruction_v{v}"
        if col not in df.columns:
            df[col] = ""

    for i in range(len(df)):
        if isinstance(df.at[i, "transformed_instruction"], str) and df.at[i, "transformed_instruction"].strip():
            continue  # already done

        raw = str(df.at[i, raw_col])
        prompt = build_prompt(raw)

        best = ""
        best_reason = ""
        status = "fail"
        for v in range(1, args.n_variants + 1):
            cand = ""
            reason = ""
            ok = False
            for _ in range(args.max_retries + 1):
                cand = generate_one(
                    model, tok, prompt, args.device,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                ok, reason = keyword_ok(raw, cand, min_hit_ratio=args.min_hit_ratio)
                if ok:
                    break
            df.at[i, f"transformed_instruction_v{v}"] = cand
            if ok and not best:
                best = cand
                best_reason = reason
                status = "ok"

        df.at[i, "transformed_instruction"] = best if best else df.at[i, "transformed_instruction_v1"]
        df.at[i, "regen_status"] = status
        df.at[i, "regen_reason"] = best_reason if best_reason else "guardrail_failed"

        if args.partial_save_every and (i + 1) % args.partial_save_every == 0:
            df.to_csv(args.out_csv, index=False)
            print(f"[INFO] saved partial: {i+1}/{len(df)} -> {args.out_csv}")

    df.to_csv(args.out_csv, index=False)
    print(f"[DONE] saved: {args.out_csv}")

if __name__ == "__main__":
    main()
