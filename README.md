
# Hybrid Retrieval System for Resource-Constrained RAG

A lightweight hybrid retrieval framework that combines vector similarity search with graph-based relational reasoning for Retrieval-Augmented Generation (RAG) systems.

## Overview

This implementation combines:
- **Vector Database**: Semantic similarity using Sentence-BERT embeddings with HNSW indexing
- **Knowledge Graph**: Structured entity relationships with efficient traversal
- **Adaptive Fusion**: Query-dependent weight adjustment for optimal results

## Key Features

- 23% improvement in Recall@10 over dense retrieval baselines
- 18% enhancement in generation quality (BERTScore)
- Optimized for resource-constrained environments (4.5GB memory, 78ms latency)
- 67% memory reduction through dimensionality reduction and quantization

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python scripts/download_models.py
```

## Requirements

- Python 3.8+
- 8GB RAM minimum (4.5GB for system operation)
- CUDA-compatible GPU (optional, for faster encoding)

## Quick Start

```python
from hybrid_retrieval import HybridRetrievalSystem

# Initialize system
config = {
    'vector_config': {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'embedding_dim': 256,
        'index_type': 'hnsw'
    },
    'kg_config': {
        'pruning_strategy': 'hybrid',
        'edge_retention': 0.65
    },
    'fusion_config': {
        'base_vector_weight': 0.5,
        'base_graph_weight': 0.4,
        'interaction_weight': 0.1
    }
}

system = HybridRetrievalSystem(config)

# Index documents
documents = [
    {"id": "doc1", "text": "Your document text here..."},
    {"id": "doc2", "text": "Another document..."}
]
system.index_documents(documents)

# Retrieve relevant passages
query = "What are transformer architectures?"
results = system.retrieve(query, k=10)

for result in results:
    print(f"Document: {result['id']}, Score: {result['score']:.3f}")
```

## Project Structure

```
hybrid-rag/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── default_config.yaml
│   └── experiments/
├── src/
│   ├── __init__.py
│   ├── hybrid_retrieval.py      # Main system class
│   ├── vector_db.py              # Vector database component
│   ├── knowledge_graph.py        # Knowledge graph component
│   ├── fusion.py                 # Adaptive fusion mechanism
│   ├── preprocessing.py          # Text preprocessing utilities
│   └── utils.py                  # Helper functions
├── scripts/
│   ├── download_models.py        # Download pretrained models
│   ├── build_knowledge_graph.py  # KG construction pipeline
│   ├── evaluate.py               # Evaluation script
│   └── benchmark.py              # Performance benchmarking
├── data/
│   ├── raw/                      # Raw datasets
│   ├── processed/                # Preprocessed data
│   └── knowledge_graphs/         # Constructed KGs
├── experiments/
│   ├── scirag/                   # SciRAG experiments
│   ├── techdoc/                  # TechDoc experiments
│   └── biokb/                    # BioKB experiments
├── tests/
│   ├── test_vector_db.py
│   ├── test_knowledge_graph.py
│   ├── test_fusion.py
│   └── test_integration.py
└── notebooks/
    ├── demo.ipynb                # Interactive demo
    ├── analysis.ipynb            # Results analysis
    └── visualization.ipynb       # Performance visualization
```

## Detailed Usage

### 1. Data Preparation

```python
from src.preprocessing import DocumentPreprocessor

preprocessor = DocumentPreprocessor()
documents = preprocessor.load_and_process("data/raw/corpus.json")
```

### 2. Knowledge Graph Construction

```python
from src.knowledge_graph import KnowledgeGraphBuilder

kg_builder = KnowledgeGraphBuilder(
    entity_recognition='spacy',
    relation_extraction='pattern',
    pruning_threshold=0.2
)

kg = kg_builder.build_from_documents(documents)
kg.save("data/knowledge_graphs/corpus_kg.pkl")
```

### 3. Vector Database Indexing

```python
from src.vector_db import OptimizedVectorDB

vector_db = OptimizedVectorDB(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    embedding_dim=256,
    use_pca=True,
    index_type='hnsw',
    hnsw_params={'M': 16, 'ef_construction': 200}
)

vector_db.index_documents(documents)
vector_db.save("data/indexes/vector_index.faiss")
```

### 4. Hybrid Retrieval

```python
from src.hybrid_retrieval import HybridRetrievalSystem

system = HybridRetrievalSystem.from_pretrained(
    vector_db_path="data/indexes/vector_index.faiss",
    kg_path="data/knowledge_graphs/corpus_kg.pkl",
    config_path="config/default_config.yaml"
)

results = system.retrieve(
    query="How do attention mechanisms work in transformers?",
    k=10,
    return_scores=True
)
```

### 5. Evaluation

```python
from scripts.evaluate import evaluate_retrieval

metrics = evaluate_retrieval(
    system=system,
    test_queries="data/processed/test_queries.json",
    relevance_judgments="data/processed/relevance.json"
)

print(f"Recall@10: {metrics['recall@10']:.3f}")
print(f"NDCG@10: {metrics['ndcg@10']:.3f}")
print(f"MRR: {metrics['mrr']:.3f}")
```

## Configuration

Edit `config/default_config.yaml`:

```yaml
vector_config:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  embedding_dim: 256
  use_pca: true
  pca_variance: 0.94
  index_type: "hnsw"
  hnsw_params:
    M: 16
    ef_construction: 200
    ef_search: 50
  quantization: "product"
  
kg_config:
  entity_recognition:
    method: "spacy"
    model: "en_core_web_sm"
  relation_extraction:
    method: "hybrid"
    confidence_threshold: 0.7
  pruning:
    strategy: "hybrid"
    edge_retention: 0.65
    centrality_threshold: 0.2
    
fusion_config:
  method: "adaptive"
  base_weights:
    vector: 0.5
    graph: 0.4
    interaction: 0.1
  adaptive_features:
    - entity_density
    - syntactic_complexity
    - semantic_ambiguity
```

## Benchmarking

Run performance benchmarks:

```bash
# Full evaluation on all datasets
python scripts/benchmark.py --datasets scirag techdoc biokb

# Memory profiling
python scripts/benchmark.py --profile-memory

# Latency analysis
python scripts/benchmark.py --profile-latency --queries 1000
```

## Optimization Tips

### For Lower Memory Usage:
```python
config['vector_config']['embedding_dim'] = 128  # Further reduce dimensions
config['vector_config']['quantization'] = 'scalar'  # More aggressive quantization
config['kg_config']['pruning']['edge_retention'] = 0.5  # More pruning
```

### For Better Accuracy:
```python
config['vector_config']['embedding_dim'] = 384  # Higher dimensions
config['kg_config']['pruning']['edge_retention'] = 0.8  # Less pruning
config['fusion_config']['base_weights']['graph'] = 0.5  # More graph weight
```

### For Lower Latency:
```python
config['vector_config']['hnsw_params']['ef_search'] = 25  # Faster search
config['kg_config']['max_traversal_depth'] = 2  # Limit graph traversal
```

## Reproducing Paper Results

```bash
# Run all experiments from the paper
python scripts/run_experiments.py --config experiments/paper_config.yaml

# Generate result tables
python scripts/generate_tables.py --output experiments/results/

# Create visualizations
python scripts/visualize_results.py --input experiments/results/
```

## API Reference

### HybridRetrievalSystem

```python
class HybridRetrievalSystem:
    def __init__(self, config: dict)
    def index_documents(self, documents: List[dict])
    def retrieve(self, query: str, k: int = 10) -> List[dict]
    def save(self, path: str)
    @classmethod
    def from_pretrained(cls, vector_db_path: str, kg_path: str, config_path: str)
```

### OptimizedVectorDB

```python
class OptimizedVectorDB:
    def __init__(self, model_name: str, embedding_dim: int, **kwargs)
    def encode(self, texts: List[str]) -> np.ndarray
    def index_documents(self, documents: List[dict])
    def search(self, query: str, k: int) -> List[Tuple[str, float]]
```

### KnowledgeGraph

```python
class KnowledgeGraph:
    def __init__(self, entities: List[str], relations: List[Tuple])
    def add_entity(self, entity: str, attributes: dict)
    def add_relation(self, head: str, relation: str, tail: str)
    def traverse(self, start_entities: List[str], max_depth: int) -> List[str]
    def prune(self, strategy: str, threshold: float)
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_vector_db.py -v

# Run with coverage
pytest --cov=src tests/
```

## Performance Metrics

Expected performance on standard hardware (64GB RAM, V100 GPU):

| Metric | Value |
|--------|-------|
| Recall@10 | 0.912 |
| NDCG@10 | 0.825 |
| BERTScore | 0.847 |
| Memory Usage | 4.5 GB |
| Query Latency | 78 ms |
| Throughput | 23.1 QPS |

## Troubleshooting

### Out of Memory
- Reduce `embedding_dim` to 128
- Enable more aggressive quantization
- Increase `edge_retention` threshold for graph pruning

### Slow Queries
- Reduce `ef_search` parameter
- Limit `max_traversal_depth` in KG
- Use smaller batch sizes for encoding

### Low Accuracy
- Increase `embedding_dim` to 384 or 512
- Reduce pruning (higher `edge_retention`)
- Tune fusion weights for your domain

## Citation

If you use this code in your research, please cite:

```bibtex
@article{patel2024hybrid,
  title={Evaluating Hybrid Retrieval Strategies: Combining Vector Databases and Knowledge Graphs for Resource-Constrained RAG Systems},
  author={Patel, Nishil and Patel, Dhruv and Vekaria, Rushikesh},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: 12102080601089@adit.ac.in

## Acknowledgments

- Sentence-Transformers team for embedding models
- FAISS team for efficient similarity search
- spaCy team for NLP components
- Research supported by A D Patel Institute of Technology