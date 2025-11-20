"""
tests/test_fusion.py
Unit tests for fusion mechanism.
"""

from src.fusion import FusionMechanism

def test_fusion_score_monotonicity():
    fuse = FusionMechanism({"default_weights": {"vector": 0.6, "graph": 0.3, "interaction": 0.1}, "dynamic": {"enabled": False}})
    vector_list = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.1}]
    graph_list = [{"id": "a", "score": 0.2}, {"id": "c", "score": 0.8}]
    res = fuse.fuse("test query", vector_list, graph_list, k=5)
    # top result should be id 'a' since it has strong vector score
    assert res[0]["id"] == "a"
