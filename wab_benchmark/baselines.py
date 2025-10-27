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
# wab_benchmark/baselines.py
from typing import Dict, List
import re

class NaivePlanner:
    def predict(self, episode: dict) -> List[str]:
        obj = episode["meta"]["obj"]
        tgt = episode["meta"]["tgt"]
        return [f"grasp({obj})", f"place({obj},{tgt})"]

class RulePlanner:
    def predict(self, episode: dict) -> List[str]:
        obj = episode["meta"]["obj"]
        tgt = episode["meta"]["tgt"]
        instr = episode["instruction"].lower()
        need_move = any(k in instr for k in ["then", "near", "move", "put it onto"])
        plan = [f"grasp({obj})"]
        if need_move:
            plan.append(f"move({obj},near_{tgt})")
        plan.append(f"place({obj},{tgt})")
        return plan

class LMPlanner:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def predict(self, episode: dict) -> List[str]:
        instr = episode["instruction"]
        prompt = f"Translate the instruction into atomic actions.\nInstruction: {instr}\nActions:"
        ids = self.tok(prompt, return_tensors="pt").input_ids
        out = self.model.generate(ids, max_length=64)
        text = self.tok.decode(out[0], skip_special_tokens=True)
        actions = []
        for m in re.finditer(r"(grasp|move|place)\s*\(([^)]*)\)", text.lower()):
            actions.append(f"{m.group(1)}({m.group(2)})")
        if not actions:
            obj = episode["meta"]["obj"]
            tgt = episode["meta"]["tgt"]
            actions = [f"grasp({obj})", f"place({obj},{tgt})"]
        return actions

