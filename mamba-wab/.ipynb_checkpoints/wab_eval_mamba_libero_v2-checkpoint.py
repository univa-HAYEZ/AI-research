
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mamba-friendly step-wise evaluation for Libero instruction CSV (no gold plans).

Expected CSV columns:
- 원본
- 수정_한글
- 수정_영어

Outputs:
- per_episode.csv : per-row rollout stats
- summary.json    : aggregated stats (overall/seen/unseen pseudo split)

This script is robust to different MambaPlanner constructor styles:
- MambaPlanner(cfg)
- planner = MambaPlanner(); planner.init(cfg)
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


def _import_planner():
    last_err = None
    for modname in ("mamba_planner_fixed", "mamba_planner"):
        try:
            m = __import__(modname, fromlist=["MambaPlanner", "MambaPlannerConfig"])
            return m, modname
        except Exception as e:
            last_err = e
    raise ImportError(f"Could not import planner module (mamba_planner_fixed or mamba_planner). Last error: {last_err}")


ACTION_RE = re.compile(
    r"^(stop|STOP|turn_on\([a-z0-9_]+\)|open\([a-z0-9_]+\)|close\([a-z0-9_]+\)|pick\([a-z0-9_]+\)|place_on\([a-z0-9_]+,[a-z0-9_]+\)|put_in\([a-z0-9_]+,[a-z0-9_]+\))$"
)

def is_valid_action(a: str) -> bool:
    return bool(a) and bool(ACTION_RE.match(a.strip()))

def detect_repeat_stats(actions: List[str]) -> Tuple[float, int]:
    if not actions:
        return 0.0, 0
    repeats = 0
    max_run = 1
    run = 1
    for i in range(1, len(actions)):
        if actions[i] == actions[i - 1]:
            repeats += 1
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return repeats / max(1, len(actions) - 1), max_run

def detect_loop_flag(actions: List[str]) -> int:
    if len(actions) >= 6:
        last6 = actions[-6:]
        if last6[0:2] == last6[2:4] == last6[4:6]:
            return 1
    _, max_run = detect_repeat_stats(actions)
    if max_run >= 4:
        return 1
    return 0

def make_splits(n: int, unseen_ratio: float, seed: int) -> List[str]:
    import random
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    k = int(round(n * unseen_ratio))
    unseen = set(idxs[:k])
    return ["unseen" if i in unseen else "seen" for i in range(n)]

def agg(df: pd.DataFrame) -> Dict:
    if len(df) == 0:
        return {
            "count": 0,
            "avg_traj_len": 0.0,
            "stop_rate": 0.0,
            "invalid_rate": 0.0,
            "repeat_ratio": 0.0,
            "loop_rate": 0.0,
            "avg_seconds": 0.0,
        }
    return {
        "count": int(len(df)),
        "avg_traj_len": float(df["traj_len"].mean()),
        "stop_rate": float(df["stopped"].mean()),
        "invalid_rate": float(df["invalid_rate"].mean()),
        "repeat_ratio": float(df["repeat_ratio"].mean()),
        "loop_rate": float(df["loop_flag"].mean()),
        "avg_seconds": float(df["seconds"].mean()),
    }

def build_planner(args):
    mod, modname = _import_planner()
    MambaPlanner = getattr(mod, "MambaPlanner")
    MambaPlannerConfig = getattr(mod, "MambaPlannerConfig", None)
    if MambaPlannerConfig is None:
        raise ImportError(f"{modname} does not export MambaPlannerConfig")

    cfg = MambaPlannerConfig(
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        do_sample=False,
    )

    try:
        planner = MambaPlanner(cfg) 
        return planner
    except TypeError:
        planner = MambaPlanner()
        if not hasattr(planner, "init"):
            raise
        planner.init(cfg)
        return planner

def rollout_stepwise(planner, goal: str, history_k: int, max_steps: int) -> Tuple[List[str], bool, int]:
    pred_steps: List[str] = []
    stopped = False
    invalid_steps = 0

    for _ in range(max_steps):
        prev = pred_steps[-history_k:] if history_k > 0 else []
        a = planner.predict_next_action(goal=goal, prev_actions=prev, fewshot_steps="")

        if isinstance(a, str) and a.upper().startswith("STOP"):
            stopped = True
            break

        a = (a or "").strip().lower()
        if a == "stop":
            stopped = True
            break

        if not is_valid_action(a):
            invalid_steps += 1
        pred_steps.append(a)

    return pred_steps, stopped, invalid_steps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to libero_instructions_compressed.csv")
    ap.add_argument("--model", required=True, help="HF model name or local path for Mamba")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lang", choices=["en", "ko"], default="en")

    ap.add_argument("--mode", choices=["step", "full"], default="step", help="Ignored; step-wise only.")
    ap.add_argument("--unseen_ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--max_retries", type=int, default=1)

    ap.add_argument("--history_k", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=10)

    ap.add_argument("--partial_save_every", type=int, default=50)
    ap.add_argument("--out_json", default="summary.json")
    ap.add_argument("--out_per_episode", default="per_episode.csv")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = {"원본", "수정_한글", "수정_영어"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found columns={list(df.columns)}")

    goal_col = "수정_영어" if args.lang == "en" else "수정_한글"
    rows = df.to_dict(orient="records")
    splits = make_splits(len(rows), unseen_ratio=args.unseen_ratio, seed=args.seed)

    per_rows: List[Dict] = []
    done = set()
    if args.resume and os.path.exists(args.out_per_episode):
        prev = pd.read_csv(args.out_per_episode)
        if "idx" in prev.columns:
            done = set(int(x) for x in prev["idx"].tolist())
            per_rows = prev.to_dict(orient="records")
            print(f"[INFO] resume: already have {len(done)} rows")

    planner = build_planner(args)

    t_all = time.time()
    for i, r in enumerate(rows):
        if i in done:
            continue

        goal = str(r.get(goal_col, "")).strip()
        source = str(r.get("원본", "")).strip()

        t0 = time.time()
        pred_steps, stopped, invalid_steps = rollout_stepwise(
            planner=planner,
            goal=goal,
            history_k=args.history_k,
            max_steps=args.max_steps,
        )
        seconds = time.time() - t0

        repeat_ratio, max_run = detect_repeat_stats(pred_steps)
        loop_flag = detect_loop_flag(pred_steps)

        traj_len = len(pred_steps)
        invalid_rate = invalid_steps / max(1, args.max_steps)

        per_rows.append({
            "idx": i,
            "원본": source,
            "goal": goal,
            "pred_plan": "; ".join(pred_steps),
            "traj_len": traj_len,
            "stopped": int(stopped),
            "invalid_steps": int(invalid_steps),
            "invalid_rate": float(invalid_rate),
            "repeat_ratio": float(repeat_ratio),
            "max_repeat_run": int(max_run),
            "loop_flag": int(loop_flag),
            "split": splits[i],
            "seconds": float(seconds),
        })
        done.add(i)

        if args.partial_save_every and (len(per_rows) % args.partial_save_every == 0):
            pd.DataFrame(per_rows).to_csv(args.out_per_episode, index=False)
            print(f"[INFO] partial saved {len(per_rows)}/{len(rows)} -> {args.out_per_episode}")

    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(args.out_per_episode, index=False)

    summary = {
        "meta": {
            "csv": args.csv,
            "model": args.model,
            "device": args.device,
            "lang": args.lang,
            "history_k": args.history_k,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "unseen_ratio": args.unseen_ratio,
            "seed": args.seed,
            "seconds_total": float(time.time() - t_all),
        },
        "overall": agg(per_df),
        "seen": agg(per_df[per_df["split"] == "seen"]),
        "unseen": agg(per_df[per_df["split"] == "unseen"]),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"Wrote: {args.out_per_episode}")
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
