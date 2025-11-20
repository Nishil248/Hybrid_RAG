"""
scripts/build_knowledge_graph.py
Builds a knowledge graph from documents and persists it.
"""

import argparse
import json
from pathlib import Path
from src.knowledge_graph import KnowledgeGraph

def load_documents(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to documents.json")
    parser.add_argument("--output", type=str, default="data/knowledge_graphs/default_kg")
    parser.add_argument("--spacy-model", type=str, default="en_core_web_sm")
    parser.add_argument("--min-entity-freq", type=int, default=2)
    args = parser.parse_args()

    docs = load_documents(Path(args.data))
    kg = KnowledgeGraph(spacy_model=args.spacy_model, min_entity_freq=args.min_entity_freq)
    print(f"Indexing {len(docs)} documents into KG...")
    for doc in docs:
        kg.add_document(doc["id"], doc["text"], metadata=doc.get("meta"))
    print("Pruning graph...")
    kg.prune()
    print(f"Saving KG to {args.output} ...")
    kg.save(args.output)
    print("Done.")

if __name__ == "__main__":
    main()
