
import json
from pathlib import Path
from wab_benchmark import Evaluator

data_dir = Path(__file__).resolve().parent / "sample_data"
episodes = {}
with (data_dir/"episodes.jsonl").open() as f:
    for line in f:
        d = json.loads(line)
        episodes[d["episode_id"]] = d

with (data_dir/"splits.json").open() as f:
    splits = json.load(f)

# naive baseline: ignore extra step "move(...)" when it exists
preds = {}
for eid, ep in episodes.items():
    obj, tgt = ep["meta"]["obj"], ep["meta"]["tgt"]
    preds[eid] = [f"grasp({obj})", f"place({obj},{tgt})"]

ev = Evaluator(episodes, splits)
report = ev.evaluate(preds)
print(json.dumps(report, indent=2))
