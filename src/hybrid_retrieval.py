"""
Main Hybrid Retrieval System
Integrates vector DB and Knowledge Graph, provides indexing and retrieval APIs.
"""

import os
import json
import time
from typing import List, Dict, Any

from src.vector_db import OptimizedVectorDB
from src.knowledge_graph import KnowledgeGraph
from src.fusion import FusionMechanism


class HybridRetrievalSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_config = config.get("vector_config", {})
        self.graph_config = config.get("graph_config", {})
        self.fusion_config = config.get("fusion_config", {})
        self.vector_db = OptimizedVectorDB(**self.vector_config)
        self.kg = KnowledgeGraph(**self.graph_config)
        self.fusion_mechanism = FusionMechanism(self.fusion_config)

    def index_documents(self, documents: List[Dict[str, Any]], batch_size: int = None):
        if batch_size is None:
            batch_size = self.config.get("system", {}).get("index_batch_size", 256)
        self.vector_db.index_documents(documents, batch_size=batch_size)
        for doc in documents:
            self.kg.add_document(doc["id"], doc.get("text", ""), metadata=doc.get("meta"))

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        vector_results = self.vector_db.search(query, k=k)
        graph_results = self.kg.query(query, k=k)
        vector_list = [{"id": did, "score": s, "source": "vector"} for did, s in vector_results]
        graph_list = [{"id": did, "score": s, "source": "graph"} for did, s in graph_results]
        fused = self.fusion_mechanism.fuse(query, vector_list, graph_list, k=k)
        return fused

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f)
        self.vector_db.save(os.path.join(path, "vector_db"))
        self.kg.save(os.path.join(path, "kg"))

    @classmethod
    def from_pretrained(cls, path: str):
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        inst = cls(config)
        inst.vector_db.load(os.path.join(path, "vector_db"))
        inst.kg.load(os.path.join(path, "kg"))
        return inst
