"""
Evaluation script (save as scripts/evaluate.py)
This is adapted from the evaluate.py you shared; minimal adjustments to module imports.
"""

import argparse
import yaml
import json
from pathlib import Path
import time
import numpy as np
from typing import List, Dict
from tqdm import tqdm

from src.hybrid_retrieval import HybridRetrievalSystem
from src.utils import MetricsTracker, format_time, save_json

def load_test_data(data_path: str) -> tuple:
    queries_file = Path(data_path) / "test_queries.json"
    relevance_file = Path(data_path) / "relevance.json"
    with open(queries_file, 'r') as f:
        queries = json.load(f)
    with open(relevance_file, 'r') as f:
        relevance = json.load(f)
    return queries, relevance

def evaluate_retrieval(system: HybridRetrievalSystem, queries: List[Dict], relevance: Dict, k_values: List[int] = [5, 10, 20]) -> Dict:
    tracker = MetricsTracker()
    latencies = []
    for query_data in tqdm(queries):
        query_id = query_data['id']
        query_text = query_data['text']
        start_time = time.time()
        results = system.retrieve(query_text, k=max(k_values))
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
        retrieved = [r['id'] for r in results]
        query_relevance = relevance.get(query_id, {})
        relevant_docs = set(doc_id for doc_id, score in query_relevance.items() if score > 0)
        relevance_scores = {doc_id: float(score) for doc_id, score in query_relevance.items()}
        tracker.add_query(query=query_text, retrieved=retrieved, relevant=relevant_docs, relevance=relevance_scores)
    metrics = tracker.compute_metrics(k_values=k_values)
    metrics['avg_latency_ms'] = np.mean(latencies)
    metrics['median_latency_ms'] = np.median(latencies)
    metrics['p95_latency_ms'] = np.percentile(latencies, 95)
    metrics['p99_latency_ms'] = np.percentile(latencies, 99)
    return metrics

def evaluate_baseline(baseline_name: str, documents: List[Dict], queries: List[Dict], relevance: Dict, k_values: List[int] = [5, 10, 20]) -> Dict:
    print(f"\nEvaluating baseline: {baseline_name}")
    if baseline_name == 'bm25':
        from rank_bm25 import BM25Okapi
        corpus = [doc['text'].lower().split() for doc in documents]
        bm25 = BM25Okapi(corpus)
        doc_ids = [doc['id'] for doc in documents]
        tracker = MetricsTracker()
        latencies = []
        for query_data in tqdm(queries):
            query_text = query_data['text']
            start_time = time.time()
            tokenized_query = query_text.lower().split()
            scores = bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:max(k_values)]
            retrieved = [doc_ids[idx] for idx in top_indices]
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            query_relevance = relevance.get(query_data['id'], {})
            relevant_docs = set(doc_id for doc_id, score in query_relevance.items() if score > 0)
            relevance_scores = {doc_id: float(score) for doc_id, score in query_relevance.items()}
            tracker.add_query(query_text, retrieved, relevant_docs, relevance_scores)
        metrics = tracker.compute_metrics(k_values=k_values)
        metrics['avg_latency_ms'] = np.mean(latencies)
        return metrics
    elif baseline_name == 'dense':
        from src.vector_db import OptimizedVectorDB
        config = {"encoder_model": "sentence-transformers/all-MiniLM-L6-v2"}
        vector_db = OptimizedVectorDB(**config)
        vector_db.index_documents(documents)
        tracker = MetricsTracker()
        latencies = []
        for query_data in tqdm(queries):
            query_text = query_data['text']
            start_time = time.time()
            results = vector_db.search(query_text, k=max(k_values))
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            retrieved = [doc_id for doc_id, _ in results]
            query_relevance = relevance.get(query_data['id'], {})
            relevant_docs = set(doc_id for doc_id, score in query_relevance.items() if score > 0)
            relevance_scores = {doc_id: float(score) for doc_id, score in query_relevance.items()}
            tracker.add_query(query_text, retrieved, relevant_docs, relevance_scores)
        metrics = tracker.compute_metrics(k_values=k_values)
        metrics['avg_latency_ms'] = np.mean(latencies)
        return metrics
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

def run_ablation_study(system: HybridRetrievalSystem, queries: List[Dict], relevance: Dict) -> Dict:
    fusion_configs = [
        {'vector': 1.0, 'graph': 0.0, 'interaction': 0.0},
        {'vector': 0.0, 'graph': 1.0, 'interaction': 0.0},
        {'vector': 0.5, 'graph': 0.5, 'interaction': 0.0},
        {'vector': 0.5, 'graph': 0.4, 'interaction': 0.1},
    ]
    results = {}
    for i, config in enumerate(fusion_configs):
        system.fusion_mechanism.update_weights(config)
        tracker = MetricsTracker()
        for query_data in tqdm(queries, desc=f"Config {i+1}"):
            query_text = query_data['text']
            results_list = system.retrieve(query_text, k=10)
            retrieved = [r['id'] for r in results_list]
            query_relevance = relevance.get(query_data['id'], {})
            relevant_docs = set(doc_id for doc_id, score in query_relevance.items() if score > 0)
            relevance_scores = {doc_id: float(score) for doc_id, score in query_relevance.items()}
            tracker.add_query(query_text, retrieved, relevant_docs, relevance_scores)
        metrics = tracker.compute_metrics(k_values=[10])
        results[f"config_{i+1}"] = {'weights': config, 'metrics': metrics}
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Hybrid Retrieval System')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--system_path', type=str, default=None, help='Path to saved system (optional)')
    parser.add_argument('--config', type=str, default='config/default_config.yaml', help='Path to configuration file')
    parser.add_argument('--baselines', nargs='+', default=['bm25', 'dense'], help='Baselines to evaluate')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--output', type=str, default='results/evaluation.json', help='Output file for results')
    args = parser.parse_args()

    queries, relevance = load_test_data(args.data_path)
    docs_file = Path(args.data_path) / "documents.json"
    with open(docs_file, 'r') as f:
        documents = json.load(f)

    if args.system_path:
        system = HybridRetrievalSystem.from_pretrained(args.system_path)
    else:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        system = HybridRetrievalSystem(config)
        system.index_documents(documents)

    results = {}
    start_time = time.time()
    results['hybrid'] = evaluate_retrieval(system, queries, relevance)
    results['hybrid']['total_time'] = time.time() - start_time

    for baseline in args.baselines:
        try:
            start_time = time.time()
            results[baseline] = evaluate_baseline(baseline, documents, queries, relevance)
            results[baseline]['total_time'] = time.time() - start_time
        except Exception as e:
            print(f"Error evaluating {baseline}: {e}")

    if args.ablation:
        results['ablation'] = run_ablation_study(system, queries, relevance)

    save_json(results, args.output)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
