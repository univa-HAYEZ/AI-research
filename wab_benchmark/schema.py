
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import json

@dataclass
class Episode:
    episode_id: str
    scene_id: str
    objects: List[str]
    instruction: str
    observations: Dict[str, Any]  # e.g., {"image_path": "..."} or {"state": {...}}
    gold_plan: List[str]          # e.g., ["grasp(red_cube)", "place(red_cube,blue_plate)"]
    meta: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "Episode":
        d = json.loads(s)
        # Basic validation
        for key in ["episode_id","scene_id","objects","instruction","observations","gold_plan"]:
            if key not in d:
                raise ValueError(f"Missing required field: {key}")
        if not isinstance(d["gold_plan"], list):
            raise ValueError("gold_plan must be a list of action strings")
        return Episode(**d)
