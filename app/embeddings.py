"""
Lightweight embeddings using scikit-learn TF-IDF vectorizer.
Zero model download, ~5MB RAM — works on Render free tier.
"""
import logging
import pickle
import os
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

# We use a simple hash-based embedding that needs no downloads
def _hash_embed(text: str, dim: int = 384) -> List[float]:
    """
    Deterministic embedding via character n-gram hashing.
    No model download needed. Good enough for small document sets.
    """
    import hashlib
    vec = np.zeros(dim, dtype=np.float32)
    # sliding window of word n-grams
    words = text.lower().split()
    ngrams = []
    for n in range(1, 4):  # unigrams, bigrams, trigrams
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i:i+n]))
    for ng in ngrams:
        h = int(hashlib.md5(ng.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    logger.debug("Generating hash embeddings for %d texts", len(texts))
    return [_hash_embed(t) for t in texts]


def generate_single_embedding(text: str) -> List[float]:
    return _hash_embed(text)