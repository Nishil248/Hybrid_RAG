"""
Text preprocessing utilities for Hybrid RAG.
Contains lightweight cleaning and chunking helpers.
"""

import re
from typing import List

def clean_text(text: str) -> str:
    text = text.replace("\\n", " ").strip()
    text = re.sub(r"\\s+", " ", text)
    return text

def chunk_text(text: str, max_tokens: int = 256) -> List[str]:
    # Very simple whitespace-based chunking (replace with tokenizer-based chunking if needed)
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i:i+max_tokens]))
    return chunks
