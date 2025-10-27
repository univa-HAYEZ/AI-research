
# WAB (Web Action Bridge) – Emergent Reasoning Benchmark Scaffold

This is a **minimal, runnable** scaffold to measure *emergent reasoning* with an
**O–I–A (Object–Instruction–Action)** format, compositional & long-horizon splits,
and an evaluation toolkit.

## What you get
- `schema.py`: O–I–A Episode dataclass with JSON I/O
- `splits.py`: simple **compositional** (held-out object–target pairs) and **long-horizon** splits
- `metrics.py`: success rate, action F1 (token-set), exact match, plan length error, generalization gap
- `evaluator.py`: per-split evaluation + generalization gap computation
- `demo_build_dataset.py`: generates a small toy dataset + splits

## Quickstart
```bash
pip install -U pandas numpy
python -m wab_benchmark.demo_build_dataset
```

This writes `sample_data/episodes.jsonl` and `sample_data/splits.json`.

Then run evaluation with dummy predictions (example below).

## Example: Evaluate
```python
import json
from pathlib import Path
from wab_benchmark import Evaluator

data_dir = Path("wab_benchmark/sample_data")
episodes = {}
with (data_dir/"episodes.jsonl").open() as f:
    for line in f:
        d = json.loads(line)
        episodes[d["episode_id"]] = d

with (data_dir/"splits.json").open() as f:
    splits = json.load(f)

# naive baseline: predict just ["grasp(obj)", "place(obj,tgt)"] without extra step
preds = {}
for eid, ep in episodes.items():
    obj, tgt = ep["meta"]["obj"], ep["meta"]["tgt"]
    preds[eid] = [f"grasp({obj})", f"place({obj},{tgt})"]

ev = Evaluator(episodes, splits)
report = ev.evaluate(preds)
print(report)
```
