import logging
import hashlib
import numpy as np
from typing import List

logger = logging.getLogger(__name__)


def _hash_embed(text: str, dim: int = 384) -> List[float]:
    vec = [0.0] * dim
    words = text.lower().split()
    ngrams = []
    for n in range(1, 4):
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i:i+n]))
    for ng in ngrams:
        h = int(hashlib.md5(ng.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(x*x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    return [_hash_embed(t) for t in texts]


def generate_single_embedding(text: str) -> List[float]:
    return _hash_embed(text)