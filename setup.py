from setuptools import setup, find_packages

setup(
    name="hybrid_rag",
    version="0.1.0",
    description="Hybrid retrieval system: dense vectors + knowledge graph + adaptive fusion",
    packages=find_packages(exclude=("tests", "notebooks", "data", "scripts")),
    include_package_data=True,
    install_requires=[
        "numpy",
        "tqdm",
        "pyyaml",
        "sentence-transformers",
        "transformers",
        "faiss-cpu",
        "rank-bm25",
        "spacy",
        "networkx",
    ],
    entry_points={
        "console_scripts": [
            "hybrid-eval = scripts.evaluate:main",
            "hybrid-build = scripts.build_knowledge_graph:main",
        ],
    },
)
