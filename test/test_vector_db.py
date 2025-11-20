"""
tests/test_vector_db.py
Unit tests for OptimizedVectorDB (basic functionality).
"""

import tempfile
import json
import os
from src.vector_db import OptimizedVectorDB

def make_docs(n=10):
    return [{"id": f"doc{i}", "text": f"This is document number {i}. Machine learning and AI."} for i in range(n)]

def test_index_and_search(tmp_path):
    docs = make_docs(20)
    vdb = OptimizedVectorDB(encoder_model="sentence-transformers/all-MiniLM-L6-v2")
    vdb.index_documents(docs, batch_size=8)
    res = vdb.search("machine learning", k=5)
    assert isinstance(res, list)
    assert len(res) <= 5
    # test save/load
    outdir = tmp_path / "vdb"
    vdb.save(str(outdir))
    loaded = OptimizedVectorDB(encoder_model="sentence-transformers/all-MiniLM-L6-v2")
    loaded.load(str(outdir))
    res2 = loaded.search("machine learning", k=3)
    assert isinstance(res2, list)
