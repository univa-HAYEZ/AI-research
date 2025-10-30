
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_bddl_to_wab.py  (with --force_heuristic)
------------------------------------------------
Convert one or more BDDL files into a single WAB(SUREBench)-ready CSV.

New:
- --force_heuristic : Ignore any action_seq found in BDDL and derive a
  short, symbolic gold plan directly from the instruction. This helps
  align abstraction level with LLM planners and avoids very long,
  low-level motor sequences.

Columns: instruction_en, actions, object, target, modifier, src_file
"""
import argparse, csv, glob, os, re, sys
from typing import Tuple

def norm(s):
    import re as _re
    return _re.sub(r"\s+", " ", (s or "").strip().lower())

def norm_entity(x: str) -> str:
    import re as _re
    x = (x or "").strip().lower().replace(" ", "_")
    return _re.sub(r"[^a-z0-9_]", "", x)

def derive_from_instruction(instr: str):
    """
    Heuristic gold plan generator (symbolic, short).
    Returns: actions(list[str]), object, target, modifier
    """
    t = norm(instr)
    # object
    obj = "moka_pot" if "moka pot" in t else (
          "cup" if ("cup" in t or "mug" in t) else (
          "plate" if "plate" in t else "object"))
    # common targets / patterns
    if ("stove" in t) and ("turn on" in t or "turn_on" in t or "on it" in t or "put the moka pot on it" in t):
        return [f"turn_on(stove)", f"place_on({obj},stove)"], obj, "stove", "none"
    if "drawer" in t:
        tgt = "bottom_drawer" if "bottom drawer" in t else "drawer"
        return [f"open({tgt})", f"put_in({obj},{tgt})", f"close({tgt})"], obj, tgt, "none"
    if "plate" in t and ("put" in t or "place" in t):
        return [f"place_on({obj},plate)"], obj, "plate", "none"
    if "desk" in t:
        return [f"place_on({obj},desk)"], obj, "desk", "none"
    # fallback
    return [f"pick({obj})", f"place_on({obj},surface)"], obj, "surface", "none"

def parse_bddl(path: str, force_heuristic: bool) -> Tuple[str, list, str, str, str]:
    """
    Try to parse instruction/action_seq; if --force_heuristic, ignore actions
    and derive symbolic plan from instruction.
    Returns: instruction, actions(list[str]), object, target, modifier
    """
    txt = open(path, "r", encoding="utf-8").read()
    # instruction: "..." / language_instruction: '...'
    m_instr = re.search(r"(?:^|\n)\s*(instruction|language_instruction)\s*[:=]\s*['\"](.+?)['\"]", txt, re.I)
    instr = m_instr.group(2) if m_instr else None

    acts = None
    if not force_heuristic:
        # action_seq / actions: ["open(drawer)","put_in(mug,drawer)"]
        m_actions = re.search(r"(?:^|\n)\s*(action_seq|actions?)\s*[:=]\s*\[(.*?)\]", txt, re.I | re.S)
        if m_actions:
            raw = m_actions.group(2)
            parts = re.split(r",|\n", raw)
            acts = []
            for p in parts:
                p = p.strip(" \"'")
                if re.match(r"^(turn_on|open|close|pick|place_on|put_in)\s*\(", p, re.I):
                    acts.append(re.sub(r"\s+", "", p.lower()))
            if not acts:
                acts = None

    # Instruction fallback: derive from filename
    if not instr:
        base = os.path.basename(path).replace(".bddl", "")
        instr = base.replace("_", " ")
        instr = re.sub(r"^kitchen\s*scene\d*\s*", "", instr, flags=re.I)

    # If no actions OR user forced heuristic, derive symbolic plan
    if force_heuristic or not acts:
        acts, obj, tgt, mod = derive_from_instruction(instr)
    else:
        # Derive object/target from actions/instruction (rudimentary)
        joined = " ".join(acts).lower()
        if "moka_pot" in joined or "moka pot" in instr.lower():
            obj = "moka_pot"
        elif "cup" in joined or "mug" in joined:
            obj = "cup"
        elif "plate" in joined:
            obj = "plate"
        else:
            obj = "object"
        if "stove" in joined or "stove" in instr.lower():
            tgt = "stove"
        elif "drawer" in joined or "drawer" in instr.lower():
            tgt = "drawer"
        elif "plate" in joined or "plate" in instr.lower():
            tgt = "plate"
        elif "desk" in joined or "desk" in instr.lower():
            tgt = "desk"
        else:
            tgt = "target"
        mod = "none"

    # normalize
    acts = [re.sub(r"\s+", "", a.lower()) for a in acts]
    obj = norm_entity(obj); tgt = norm_entity(tgt)

    return instr, acts, obj, tgt, mod

def write_rows(rows: list, out_path: str, append: bool):
    header = ["instruction_en", "actions", "object", "target", "modifier", "src_file"]
    mode = "a" if (append and os.path.exists(out_path)) else "w"
    with open(out_path, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(header)
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="BDDL file or quoted glob, e.g., 'KITCHEN_SCENE*.bddl'")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--append", action="store_true", help="Append to CSV if exists")
    ap.add_argument("--force_heuristic", action="store_true",
                    help="Ignore action_seq in BDDL and derive short symbolic plan from instruction.")
    args = ap.parse_args()

    paths = glob.glob(args.input)
    if not paths:
        print(f"[WARN] No files matched: {args.input}")
        sys.exit(1)

    out_rows = []
    for p in sorted(paths):
        try:
            instr, acts, obj, tgt, mod = parse_bddl(p, force_heuristic=args.force_heuristic)
            out_rows.append([instr, "; ".join(acts), obj, tgt, mod, os.path.basename(p)])
            print(f"[OK] {os.path.basename(p)} -> {instr} | {acts}")
        except Exception as e:
            print(f"[ERR] Failed on {p}: {e}")

    write_rows(out_rows, args.out, append=args.append)
    print(f"[DONE] Wrote {len(out_rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
