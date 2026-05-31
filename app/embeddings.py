"""
Embeddings module — wraps sentence-transformers for generating text embeddings.
"""

import logging
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model (singleton)."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded successfully.")
    return model


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []

    model = get_embedding_model()
    logger.debug("Generating embeddings for %d text(s).", len(texts))
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


def generate_single_embedding(text: str) -> List[float]:
    """
    Generate an embedding for a single text string.

    Args:
        text: The string to embed.

    Returns:
        Embedding vector as a list of floats.
    """
    return generate_embeddings([text])[0]
