"""
scripts/download_models.py
Download and cache heavy models (sentence-transformers, spaCy models).
"""

import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import subprocess
import sys

def download_sentence_transformer(model_name: str):
    print(f"Downloading SentenceTransformer: {model_name}")
    SentenceTransformer(model_name)  # will download & cache

def download_spacy_model(model_name: str):
    print(f"Downloading spaCy model: {model_name}")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--st-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--spacy-model", type=str, default="en_core_web_sm")
    args = parser.parse_args()

    download_sentence_transformer(args.st_model)
    download_spacy_model(args.spacy_model)
    print("All models downloaded.")

if __name__ == "__main__":
    main()
