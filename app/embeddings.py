import os, httpx, logging
from functools import lru_cache
from typing import List

logger = logging.getLogger(__name__)

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    from app.config import GROQ_API_KEY
    response = httpx.post(
        "https://api.groq.com/openai/v1/embeddings",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": "nomic-embed-text-v1_5", "input": texts},
        timeout=30.0
    )
    data = response.json()
    return [d["embedding"] for d in data["data"]]

def generate_single_embedding(text: str) -> List[float]:
    return generate_embeddings([text])[0]