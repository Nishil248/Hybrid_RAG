"""
scripts/benchmark.py
Run simple throughput and latency benchmarks on the vector DB and hybrid system.
"""

import argparse
import time
import json
from pathlib import Path
from src.vector_db import OptimizedVectorDB
from src.hybrid_retrieval import HybridRetrievalSystem
import yaml
import numpy as np

def load_queries(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def benchmark_vector_db(vector_db: OptimizedVectorDB, queries, rounds=100):
    latencies = []
    for i, q in enumerate(queries[:rounds]):
        start = time.time()
        vector_db.search(q["text"], k=10)
        latencies.append((time.time() - start) * 1000)
    return {
        "avg_ms": float(np.mean(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }

def benchmark_hybrid(system: HybridRetrievalSystem, queries, rounds=100):
    latencies = []
    for i, q in enumerate(queries[:rounds]):
        start = time.time()
        system.retrieve(q["text"], k=10)
        latencies.append((time.time() - start) * 1000)
    import numpy as np
    return {"avg_ms": float(np.mean(latencies)), "p95_ms": float(np.percentile(latencies, 95))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--vector-index", type=str, default=None)
    parser.add_argument("--system-config", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args()

    queries = load_queries(Path(args.queries))
    results = {}

    if args.vector_index:
        print("Loading vector DB ...")
        vdb = OptimizedVectorDB()
        vdb.load(args.vector_index)
        results["vector_db"] = benchmark_vector_db(vdb, queries, rounds=args.rounds)
        print("Vector DB benchmark:", results["vector_db"])

    if args.system_config:
        print("Loading hybrid system ...")
        with open(args.system_config, "r") as f:
            cfg = yaml.safe_load(f)
        system = HybridRetrievalSystem(cfg)
        # assume models already indexed/added
        results["hybrid"] = benchmark_hybrid(system, queries, rounds=args.rounds)
        print("Hybrid benchmark:", results["hybrid"])

    print("Benchmark finished.")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
