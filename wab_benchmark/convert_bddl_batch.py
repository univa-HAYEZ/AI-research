
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_bddl_batch.py
---------------------
Batch helper that calls convert_bddl_to_wab.py over a folder or glob,
and merges everything into a single CSV.

Usage
  python convert_bddl_batch.py --glob "KITCHEN_SCENE*.bddl" --out kitchen_all.csv
  python convert_bddl_batch.py --dir ./bddl_dir --out kitchen_all.csv
"""
import argparse, glob, os, subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--glob", help="Quoted glob pattern, e.g., 'KITCHEN_SCENE*.bddl'")
    g.add_argument("--dir", help="Directory to scan for *.bddl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pattern = args.glob if args.glob else os.path.join(args.dir, "*.bddl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[WARN] No BDDL files found for pattern: {pattern}")
        sys.exit(1)

    # First call creates CSV, subsequent calls append
    for i, p in enumerate(paths):
        cmd = [
            sys.executable, "convert_bddl_to_wab.py",
            "--input", p,
            "--out", args.out
        ]
        if i > 0:
            cmd.append("--append")
        print("[RUN]", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr)
            sys.exit(res.returncode)

    print(f"[DONE] Merged {len(paths)} files into {args.out}")

if __name__ == "__main__":
    main()
