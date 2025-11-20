"""
OptimizedVectorDB
- sentence-transformers for embeddings
- optional PCA
- FAISS wrapper (CPU) supporting HNSW/Flat/IVF
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import faiss
import json

class OptimizedVectorDB:
    def __init__(self,
                 encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embed_dim: int = 384,
                 pca_dim: int = None,
                 faiss_index_type: str = "HNSW",
                 hnsw_m: int = 32,
                 ef_search: int = 128,
                 use_gpu: bool = False,
                 quantization: Dict[str, Any] = None):
        self.encoder = SentenceTransformer(encoder_model)
        self.embed_dim = embed_dim
        self.pca_dim = pca_dim
        self.index_type = faiss_index_type
        self.hnsw_m = hnsw_m
        self.ef_search = ef_search
        self.use_gpu = use_gpu
        self.quantization = quantization or {"enabled": False}
        self.docid_to_idx = {}
        self.idx_to_docid = {}
        self.embeddings = None
        self.index = None
        self.pca = None

    def _build_index(self, dim: int):
        if self.index_type.upper() == "HNSW":
            index = faiss.IndexHNSWFlat(dim, self.hnsw_m)
            index.hnsw.efSearch = self.ef_search
        elif self.index_type.upper() == "FLAT":
            index = faiss.IndexFlatIP(dim)
        else:
            nlist = 100
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        self.index = index

    def index_documents(self, documents: List[Dict[str, Any]], batch_size: int = 256):
        texts = [doc.get("text","") for doc in documents]
        ids = [doc["id"] for doc in documents]
        embs = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=batch_size)
        if self.pca_dim and self.pca_dim < embs.shape[1]:
            self.pca = PCA(n_components=self.pca_dim)
            embs = self.pca.fit_transform(embs)
            dim = self.pca_dim
        else:
            dim = embs.shape[1]
        self._build_index(dim)
        faiss.normalize_L2(embs)
        self.index.add(embs)
        for i, doc_id in enumerate(ids):
            self.docid_to_idx[doc_id] = i
            self.idx_to_docid[i] = doc_id
        self.embeddings = embs

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        q_emb = self.encoder.encode([query], convert_to_numpy=True)
        if self.pca is not None:
            q_emb = self.pca.transform(q_emb)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            doc_id = self.idx_to_docid.get(idx, str(idx))
            results.append((doc_id, float(score)))
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "mappings.json"), "w") as f:
            json.dump({"docid_to_idx": self.docid_to_idx, "idx_to_docid": self.idx_to_docid}, f)
        if self.pca is not None:
            import joblib
            joblib.dump(self.pca, os.path.join(path, "pca.joblib"))

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "mappings.json"), "r") as f:
            mappings = json.load(f)
            self.docid_to_idx = mappings.get("docid_to_idx", {})
            self.idx_to_docid = {int(k): v for k, v in mappings.get("idx_to_docid", {}).items()}
        pca_path = os.path.join(path, "pca.joblib")
        if os.path.exists(pca_path):
            import joblib
            self.pca = joblib.load(pca_path)
