"""
Utility functions for Hybrid RAG: metrics, tokenization, helpers.
(Adapted from the utils you previously provided.)
"""

import numpy as np
from typing import List, Dict, Set, Tuple
import re
from collections import Counter
import spacy

# Load spaCy model for analysis
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Please install spaCy model: python -m spacy download en_core_web_sm")
    nlp = None

def compute_entity_density(text: str) -> float:
    if nlp is None:
        return 0.0
    doc = nlp(text)
    entities = [ent for ent in doc.ents]
    words = [token for token in doc if not token.is_punct and not token.is_space]
    return len(entities) / max(1, len(words))

def compute_syntactic_complexity(text: str) -> float:
    if nlp is None:
        return 0.0
    doc = nlp(text)
    def get_depth(token, depth=0):
        if not list(token.children):
            return depth
        return max(get_depth(child, depth + 1) for child in token.children)
    depths = [get_depth(sent.root) for sent in doc.sents]
    return np.mean(depths) if depths else 0.0

def compute_semantic_ambiguity(text: str) -> float:
    if nlp is None:
        return 0.0
    doc = nlp(text)
    noun_chunks = list(doc.noun_chunks)
    nouns = [token for token in doc if token.pos_ == 'NOUN']
    return len(noun_chunks) / max(1, len(nouns))

def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score == 0:
        return {k: 0.0 for k in scores}
    return {k: v / max_score for k, v in scores.items()}

def compute_recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_at_k = set(retrieved[:k])
    return len(retrieved_at_k & relevant) / len(relevant)

def compute_precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_at_k = set(retrieved[:k])
    return len(retrieved_at_k & relevant) / k

def compute_mrr(retrieved: List[str], relevant: Set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def compute_ndcg_at_k(retrieved: List[str], relevance_scores: Dict[str, float], k: int) -> float:
    def dcg(scores: List[float]) -> float:
        return sum((2**score - 1) / np.log2(idx + 2) for idx, score in enumerate(scores))
    retrieved_scores = [relevance_scores.get(doc_id, 0.0) for doc_id in retrieved[:k]]
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    dcg_score = dcg(retrieved_scores)
    idcg_score = dcg(ideal_scores)
    if idcg_score == 0:
        return 0.0
    return dcg_score / idcg_score

def compute_map(retrieved_lists: List[List[str]], relevant_sets: List[Set[str]]) -> float:
    average_precisions = []
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        if not relevant:
            continue
        precisions = []
        relevant_count = 0
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                relevant_count += 1
                precision = relevant_count / rank
                precisions.append(precision)
        if precisions:
            average_precisions.append(np.mean(precisions))
        else:
            average_precisions.append(0.0)
    return np.mean(average_precisions) if average_precisions else 0.0

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return tokens

def compute_token_overlap(text1: str, text2: str) -> float:
    tokens1 = set(tokenize(text1))
    tokens2 = set(tokenize(text2))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    tokens = tokenize(text)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
                 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'should', 'could', 'may', 'might', 'must', 'can'}
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    counter = Counter(tokens)
    return [word for word, _ in counter.most_common(top_k)]

def load_json(filepath: str) -> Dict:
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Dict, filepath: str):
    import json, os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def batch_iterator(items: List, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"

def format_bytes(bytes_count: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f} PB"

class MetricsTracker:
    def __init__(self):
        self.queries = []
        self.retrieved_lists = []
        self.relevant_sets = []
        self.relevance_scores = []

    def add_query(self, query: str, retrieved: List[str], relevant: Set[str], relevance: Dict[str, float] = None):
        self.queries.append(query)
        self.retrieved_lists.append(retrieved)
        self.relevant_sets.append(relevant)
        self.relevance_scores.append(relevance or {})

    def compute_metrics(self, k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        metrics = {}
        for k in k_values:
            recalls = [compute_recall_at_k(ret, rel, k) for ret, rel in zip(self.retrieved_lists, self.relevant_sets)]
            metrics[f'recall@{k}'] = np.mean(recalls)
        for k in k_values:
            precisions = [compute_precision_at_k(ret, rel, k) for ret, rel in zip(self.retrieved_lists, self.relevant_sets)]
            metrics[f'precision@{k}'] = np.mean(precisions)
        mrrs = [compute_mrr(ret, rel) for ret, rel in zip(self.retrieved_lists, self.relevant_sets)]
        metrics['mrr'] = np.mean(mrrs)
        metrics['map'] = compute_map(self.retrieved_lists, self.relevant_sets)
        if any(self.relevance_scores):
            for k in k_values:
                ndcgs = []
                for ret, rel_scores in zip(self.retrieved_lists, self.relevance_scores):
                    if rel_scores:
                        ndcgs.append(compute_ndcg_at_k(ret, rel_scores, k))
                if ndcgs:
                    metrics[f'ndcg@{k}'] = np.mean(ndcgs)
        return metrics

    def print_summary(self):
        metrics = self.compute_metrics()
        print("\n" + "="*50)
        print("RETRIEVAL METRICS SUMMARY")
        print("="*50)
        print(f"Number of queries: {len(self.queries)}")
        print("\nMetrics:")
        for metric, value in sorted(metrics.items()):
            print(f"  {metric:.<20} {value:.4f}")
        print("="*50 + "\n")
