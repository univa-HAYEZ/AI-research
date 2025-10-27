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
# wab_benchmark/eval_agents.py
import json
from pathlib import Path
from typing import Dict, List
from wab_benchmark import Evaluator
from wab_benchmark.baselines import NaivePlanner, RulePlanner, LMPlanner

def load_episodes(data_dir: Path) -> Dict[str, dict]:
    episodes = {}
    with (data_dir/"episodes.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            episodes[d["episode_id"]] = d
    return episodes

def load_splits(data_dir: Path) -> Dict[str, List[str]]:
    with (data_dir/"splits.json").open() as f:
        return json.load(f)

def run(agent_name: str = "naive"):
    data_dir = Path(__file__).resolve().parent / "sample_data"
    episodes = load_episodes(data_dir)
    splits = load_splits(data_dir)

    if agent_name == "naive":
        agent = NaivePlanner()
    elif agent_name == "rule":
        agent = RulePlanner()
    elif agent_name == "lm":
        agent = LMPlanner()
    else:
        raise ValueError("agent_name must be one of: naive, rule, lm")

    preds = {}
    for eid, ep in episodes.items():
        preds[eid] = agent.predict(ep)

    ev = Evaluator(episodes, splits)
    report = ev.evaluate(preds)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="naive", choices=["naive","rule","lm"])
    args = p.parse_args()
    run(args.agent)

