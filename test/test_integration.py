"""
tests/test_integration.py
Simple integration smoke test: index tiny corpus, build KG, run hybrid retrieve.
"""

from src.hybrid_retrieval import HybridRetrievalSystem
from src.vector_db import OptimizedVectorDB
from src.knowledge_graph import KnowledgeGraph

def test_hybrid_end_to_end():
    docs = [
        {"id": "d1", "text": "Deep learning for natural language processing."},
        {"id": "d2", "text": "Graph neural networks for knowledge graphs."},
        {"id": "d3", "text": "Introduction to machine learning."}
    ]
    # minimal config
    config = {
        "vector_config": {"encoder_model": "sentence-transformers/all-MiniLM-L6-v2"},
        "graph_config": {"spacy_model": "en_core_web_sm"},
        "fusion_config": {"default_weights": {"vector": 0.7, "graph": 0.2, "interaction": 0.1}, "dynamic": {"enabled": False}}
    }
    system = HybridRetrievalSystem(config)
    system.index_documents(docs)
    res = system.retrieve("natural language processing", k=3)
    assert isinstance(res, list)
    assert len(res) > 0
