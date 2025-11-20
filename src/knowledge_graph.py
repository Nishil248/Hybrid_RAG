"""
KnowledgeGraph
- build a document-level knowledge graph using spaCy for entity extraction
- store entity->document mappings
- simple relation extraction using dependency windows
- pruning utilities
- query returns scoring per document (simple overlap-based scoring)
"""

import os
import json
from typing import List, Dict, Any, Tuple
import networkx as nx
import spacy
from collections import defaultdict, Counter

class KnowledgeGraph:
    def __init__(self,
                 spacy_model: str = "en_core_web_sm",
                 min_entity_freq: int = 2,
                 relation_window: int = 3,
                 pruning: Dict[str, Any] = None):
        self.nlp = spacy.load(spacy_model)
        self.min_entity_freq = min_entity_freq
        self.relation_window = relation_window
        self.graph = nx.Graph()
        self.entity_doc_map = defaultdict(set)
        self.doc_entities = {}
        self.pruning = pruning or {}

    def add_document(self, doc_id: str, text: str, metadata: Dict = None):
        doc = self.nlp(text)
        entities = [ent.text.strip().lower() for ent in doc.ents if len(ent.text.strip()) > 1]
        # filter by min freq is done during prune
        self.doc_entities[doc_id] = entities
        for ent in entities:
            self.entity_doc_map[ent].add(doc_id)

        # relation extraction (local co-occurrence)
        for i, ent1 in enumerate(entities):
            for j in range(i+1, min(i + 1 + self.relation_window, len(entities))):
                ent2 = entities[j]
                if ent1 == ent2:
                    continue
                self.graph.add_edge(ent1, ent2, weight=self.graph.get_edge_data(ent1, ent2, default={'weight':0}).get('weight',0) + 1)

    def prune(self, method: str = None):
        method = method or self.pruning.get("method", "degree")
        if method == "degree":
            deg_thresh = self.pruning.get("degree_threshold", 2)
            to_remove = [n for n, d in self.graph.degree() if d < deg_thresh]
            self.graph.remove_nodes_from(to_remove)
            for n in to_remove:
                if n in self.entity_doc_map:
                    del self.entity_doc_map[n]
        elif method == "centrality":
            k = self.pruning.get("centrality_top_k", 500)
            central = nx.degree_centrality(self.graph)
            top = sorted(central.items(), key=lambda x: x[1], reverse=True)[:k]
            keep = set([n for n, _ in top])
            remove = set(self.graph.nodes()) - keep
            self.graph.remove_nodes_from(remove)
            for n in remove:
                if n in self.entity_doc_map:
                    del self.entity_doc_map[n]
        else:
            # hybrid: combine both
            self.prune("degree")
            self.prune("centrality")

    def query(self, query_text: str, k: int = 10) -> List[Tuple[str, float]]:
        qdoc = self.nlp(query_text)
        q_entities = set([ent.text.strip().lower() for ent in qdoc.ents])
        doc_scores = Counter()

        # score documents by entity overlap and graph neighbor overlap
        for ent in q_entities:
            # direct documents
            for doc_id in self.entity_doc_map.get(ent, []):
                doc_scores[doc_id] += 1.0
            # neighbors
            if ent in self.graph:
                for neighbor in self.graph.neighbors(ent):
                    for doc_id in self.entity_doc_map.get(neighbor, []):
                        doc_scores[doc_id] += 0.5

        # produce top-k
        top = doc_scores.most_common(k)
        return [(doc_id, float(score)) for doc_id, score in top]

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        nx.write_gpickle(self.graph, os.path.join(path, "graph.gpickle"))
        with open(os.path.join(path, "entity_doc_map.json"), "w") as f:
            json.dump({k: list(v) for k, v in self.entity_doc_map.items()}, f)
        with open(os.path.join(path, "doc_entities.json"), "w") as f:
            json.dump(self.doc_entities, f)

    def load(self, path: str):
        self.graph = nx.read_gpickle(os.path.join(path, "graph.gpickle"))
        with open(os.path.join(path, "entity_doc_map.json"), "r") as f:
            data = json.load(f)
            self.entity_doc_map = {k: set(v) for k, v in data.items()}
        with open(os.path.join(path, "doc_entities.json"), "r") as f:
            self.doc_entities = json.load(f)
