"""
FusionMechanism
- compute dynamic fusion weights (vector vs graph vs interaction)
- provides .fuse(query, vector_list, graph_list, k)
- supports linear baseline and dynamic weighting based on query features
"""

from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

# We'll reuse utilities for query analysis
from .utils import compute_entity_density, compute_syntactic_complexity, compute_semantic_ambiguity, tokenize

class FusionMechanism:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.weights = self.config.get("default_weights", {"vector": 0.6, "graph": 0.3, "interaction": 0.1})
        self.dynamic = self.config.get("dynamic", {"enabled": True})

    def update_weights(self, new_weights: Dict[str, float]):
        self.weights.update(new_weights)

    def _compute_dynamic_weights(self, query: str) -> Dict[str, float]:
        # analyze query
        ent_density = compute_entity_density(query)
        syntactic = compute_syntactic_complexity(query)
        ambiguity = compute_semantic_ambiguity(query)

        # default
        w = dict(self.weights)

        if self.dynamic.get("enabled", False):
            # heuristics:
            # if entity-dense -> increase graph weight
            if ent_density > self.dynamic.get("entity_density_threshold", 0.15):
                w["graph"] = min(0.8, w["graph"] + 0.3)
                w["vector"] = max(0.1, w["vector"] - 0.2)

            # if ambiguous -> boost vector (semantic)
            if ambiguity > self.dynamic.get("ambiguity_threshold", 0.6):
                w["vector"] = min(0.9, w["vector"] + 0.2)
                w["graph"] = max(0.0, w["graph"] - 0.1)

            # normalize
            total = sum(w.values())
            for k in w:
                w[k] = w[k] / total if total > 0 else 0.0
        return w

    def _interaction_score(self, vscore: float, gscore: float) -> float:
        # simple multiplicative interaction
        return float(vscore * gscore)

    def fuse(self, query: str, vector_list: List[Dict[str, Any]], graph_list: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
        # build score dicts
        vdict = {item["id"]: item["score"] for item in vector_list}
        gdict = {item["id"]: item["score"] for item in graph_list}

        # dynamic weights
        w = self._compute_dynamic_weights(query)

        # union of doc ids
        all_ids = set(vdict.keys()) | set(gdict.keys())

        fused_scores = {}
        for did in all_ids:
            v = vdict.get(did, 0.0)
            g = gdict.get(did, 0.0)
            inter = self._interaction_score(v, g)
            score = w.get("vector", 0.0) * v + w.get("graph", 0.0) * g + w.get("interaction", 0.0) * inter
            fused_scores[did] = score

        # rank
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        # produce list
        results = [{"id": did, "score": float(score)} for did, score in ranked]
        return results

