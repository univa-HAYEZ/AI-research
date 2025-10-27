
from typing import Dict, List, Any
from .metrics import action_f1, sequence_exact_match, success_rate, average_length_error, generalization_gap

class Evaluator:
    def __init__(self, episodes: Dict[str, Any], splits: Dict[str, List[str]]):
        self.episodes = episodes  # dict: id -> episode dict
        self.splits = splits

    def evaluate(self, predictions: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        predictions: dict of episode_id -> predicted plan (list[str])
        Returns a dict of metrics per split + overall.
        """
        report = {}
        # overall
        ids = [eid for eid in self.episodes.keys() if eid in predictions]
        golds = [self.episodes[i]["gold_plan"] for i in ids]
        preds = [predictions[i] for i in ids]
        report["overall"] = {
            "success_rate": success_rate(golds, preds),
            "action_f1": sum(action_f1(g,p) for g,p in zip(golds,preds)) / (len(golds) or 1),
            "avg_len_error": average_length_error(golds, preds),
        }

        # per split
        for name, split_ids in self.splits.items():
            ids = [i for i in split_ids if i in predictions]
            golds = [self.episodes[i]["gold_plan"] for i in ids]
            preds = [predictions[i] for i in ids]
            report[name] = {
                "success_rate": success_rate(golds, preds),
                "action_f1": sum(action_f1(g,p) for g,p in zip(golds,preds)) / (len(golds) or 1),
                "avg_len_error": average_length_error(golds, preds),
                "count": len(ids),
            }

        # generalization gap (seen vs unseen if present)
        if "seen" in self.splits and "unseen_composition" in self.splits:
            seen_ids   = [i for i in self.splits["seen"] if i in predictions]
            unseen_ids = [i for i in self.splits["unseen_composition"] if i in predictions]
            seen_f1    = [action_f1(self.episodes[i]["gold_plan"], predictions[i]) for i in seen_ids]
            unseen_f1  = [action_f1(self.episodes[i]["gold_plan"], predictions[i]) for i in unseen_ids]
            report["generalization_gap_action_f1"] = generalization_gap(seen_f1, unseen_f1)
        return report
