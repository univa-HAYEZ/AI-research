#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mamba_planner_fixed.py
- step-wise generation FIXED
- pad_token issue FIXED
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re
import torch

ALLOWED_VERBS = ("turn_on", "open", "close", "pick", "place_on", "put_in")

_ACTION_ANY_RE = re.compile(
    r"(turn_on\([^)]+\)|open\([^)]+\)|close\([^)]+\)|pick\([^)]+\)|place_on\([^)]+\)|put_in\([^)]+\)|stop)",
    re.IGNORECASE,
)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def standardize_action(a: str) -> str:
    a = _norm(a)
    a = a.replace("put_on", "place_on").replace("place on", "place_on")
    a = a.replace("put in", "put_in").replace("place in", "put_in").replace("store in", "put_in")
    a = re.sub(r"\(\s*", "(", a)
    a = re.sub(r"\s*\)", ")", a)
    a = re.sub(r"\s*,\s*", ",", a)
    return a

@dataclass
class MambaPlannerConfig:
    model_name: str
    device: str = "cuda"
    max_new_tokens: int = 48
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    max_retries: int = 1
    batch_size: int = 16
    fp16: bool = True

class MambaPlanner:
    name = "mamba"

    def __init__(self, cfg: MambaPlannerConfig):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)

        # ✅ pad_token FIX
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16 if (cfg.fp16 and "cuda" in cfg.device) else None,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(cfg.device)
        self.model.eval()

    def build_step_prompt(self, goal: str, prev_actions: List[str]) -> str:
        hist = "; ".join(prev_actions[-5:]) if prev_actions else "(none)"
        return (
            "You are a robot action generator.\n"
            "Return EXACTLY ONE next action OR STOP.\n"
            "Allowed verbs: turn_on(x); open(x); close(x); pick(x); place_on(x,y); put_in(x,y).\n"
            "Rules:\n"
            "- Output only one line.\n"
            "- Use lowercase and snake_case arguments.\n"
            "- If the goal is already satisfied, output STOP.\n\n"
            f"GOAL: {goal}\n"
            f"PREVIOUS_ACTIONS: {hist}\n"
            "NEXT_ACTION:"
        )

    @torch.no_grad()
    def _generate_step_completion(self, prompt: str) -> str:
        cfg = self.cfg
        inputs = self.tokenizer(prompt, return_tensors="pt").to(cfg.device)
        in_len = inputs["input_ids"].shape[-1]

        out = self.model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature if cfg.do_sample else None,
            top_p=cfg.top_p if cfg.do_sample else None,
            repetition_penalty=cfg.repetition_penalty,
        )

        gen_ids = out[0][in_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def parse_one_action(self, text: str) -> str:
        if not text:
            return ""
        line = text.splitlines()[0]
        m = _ACTION_ANY_RE.search(line)
        if not m:
            return ""
        a = m.group(1).lower()
        return "STOP" if a == "stop" else standardize_action(a)

    def predict_next_action(self, goal: str, prev_actions: List[str], fewshot_steps: str = "") -> str:
        prompt = self.build_step_prompt(goal, prev_actions)
        completion = self._generate_step_completion(prompt)
        return self.parse_one_action(completion)
