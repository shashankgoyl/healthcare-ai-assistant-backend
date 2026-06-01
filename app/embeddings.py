import logging
import os
from functools import lru_cache
from typing import List

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-MiniLM-L3-v2")
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Model loaded.")
    return model

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

def generate_single_embedding(text: str) -> List[float]:
    return generate_embeddings([text])[0]