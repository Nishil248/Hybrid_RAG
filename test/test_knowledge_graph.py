"""
tests/test_knowledge_graph.py
Unit tests for KnowledgeGraph basic functions.
"""

from src.knowledge_graph import KnowledgeGraph

def test_add_and_query():
    kg = KnowledgeGraph(spacy_model="en_core_web_sm")
    docs = [
        {"id": "d1", "text": "Apple releases new iPhone in California."},
        {"id": "d2", "text": "Google opens an office in California."},
        {"id": "d3", "text": "Banana is a fruit grown in tropical regions."}
    ]
    for d in docs:
        kg.add_document(d["id"], d["text"])
    # simple query
    results = kg.query("California office", k=5)
    assert isinstance(results, list)
    # Expect at least one result referencing California docs
    assert any(r[0] in {"d1", "d2"} for r in results)
