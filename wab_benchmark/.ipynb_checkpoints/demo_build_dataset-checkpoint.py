
"""
Build a tiny toy WAB dataset with O–I–A format and example splits.
"""
import json, random, itertools
from pathlib import Path
from .schema import Episode
from .splits import build_compositional_splits, build_long_horizon_splits, merge_splits

def make_instruction(obj, tgt, variant=0):
    bases = [
        f"pick up the {obj} and place it on the {tgt}.",
        f"grasp the {obj}, then put it onto the {tgt}.",
        f"move the {obj} to the {tgt}.",
    ]
    return bases[variant % len(bases)]

def make_plan(obj, tgt, extra_steps=False):
    plan = [f"grasp({obj})", f"place({obj},{tgt})"]
    if extra_steps:
        plan.insert(1, f"move({obj},near_{tgt})")
    return plan

def build_toy_dataset(n_scenes=10, seed=7):
    random.seed(seed)
    objs  = ["red_cube", "blue_cube", "green_cup"]
    tgts  = ["blue_plate", "yellow_plate", "tray"]
    scenes = ["scene_%03d" % i for i in range(n_scenes)]
    episodes = []
    eid = 0
    for s in scenes:
        for obj, tgt in itertools.product(objs, tgts):
            for v in range(2):  # paraphrase variants
                extra = random.random() < 0.4
                ep = Episode(
                    episode_id=f"ep_{eid:05d}",
                    scene_id=s,
                    objects=[obj, tgt],
                    instruction=make_instruction(obj, tgt, variant=v),
                    observations={"image_path": f"images/{s}.png"},
                    gold_plan=make_plan(obj, tgt, extra_steps=extra),
                    meta={"obj": obj, "tgt": tgt, "extra_steps": extra},
                )
                episodes.append(ep)
                eid += 1
    return episodes

def save_dataset(episodes, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(ep.to_json()+"\n")

    # Splits
    eps = [json.loads(ep.to_json()) for ep in episodes]
    comp = build_compositional_splits(eps, seed=13)
    longh = build_long_horizon_splits(eps, threshold=2)
    splits = merge_splits(comp, longh)

    with (out_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    # Small readme
    with (out_dir / "README_DATASET.md").open("w", encoding="utf-8") as f:
        f.write("# Toy WAB Dataset\n")
        f.write("- O–I–A schema\n- compositional and horizon splits\n")

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "sample_data"
    episodes = build_toy_dataset(n_scenes=6)
    save_dataset(episodes, out)
    print("Saved toy dataset to:", out)
