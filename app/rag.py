"""
RAG (Retrieval-Augmented Generation) pipeline.

Handles:
- Document ingestion (reading, chunking, embedding, storing)
- Semantic retrieval of relevant chunks for a query
- Confidence scoring based on retrieval distances
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MED_THRESHOLD,
    DATA_DIR,
    SIMILARITY_THRESHOLD,
    TOP_K_RESULTS,
)
from app.embeddings import generate_embeddings, generate_single_embedding

logger = logging.getLogger(__name__)

# ── ChromaDB client (module-level singleton) ─────────────────────────────────

os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

_chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_collection() -> chromadb.Collection:
    """Return (or create) the ChromaDB collection."""
    return _chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Document chunking  ─

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split *text* into overlapping chunks by word count.

    Args:
        text:       The full document text.
        chunk_size: Target number of words per chunk.
        overlap:    Number of words to overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ── Ingestion  ─────────

def ingest_documents(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Read all .txt and .pdf files from *data_dir*, chunk them, embed them,
    and upsert them into ChromaDB.

    Args:
        data_dir: Path to folder containing documents.
                  Defaults to the configured DATA_DIR.

    Returns:
        Summary dict with counts.
    """
    source_dir = Path(data_dir) if data_dir else DATA_DIR
    if not source_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {source_dir}")

    collection = get_collection()
    files_processed = 0
    chunks_stored = 0
    errors: List[str] = []

    supported_ext = {".txt", ".md", ".pdf"}

    for file_path in sorted(source_dir.iterdir()):
        if file_path.suffix.lower() not in supported_ext:
            logger.debug("Skipping unsupported file: %s", file_path.name)
            continue

        logger.info("Ingesting file: %s", file_path.name)
        try:
            text = _read_file(file_path)
            if not text.strip():
                logger.warning("Empty file skipped: %s", file_path.name)
                continue

            chunks = chunk_text(text)
            logger.info("  → %d chunks generated", len(chunks))

            embeddings = generate_embeddings(chunks)

            ids = [f"{file_path.stem}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "document": file_path.name,
                    "chunk_index": i,
                    "source": str(file_path),
                }
                for i in range(len(chunks))
            ]

            # Upsert in batches of 100 to stay within limits
            batch_size = 100
            for b_start in range(0, len(chunks), batch_size):
                b_end = b_start + batch_size
                collection.upsert(
                    ids=ids[b_start:b_end],
                    embeddings=embeddings[b_start:b_end],
                    documents=chunks[b_start:b_end],
                    metadatas=metadatas[b_start:b_end],
                )

            files_processed += 1
            chunks_stored += len(chunks)
            logger.info("  → Stored %d chunks for %s", len(chunks), file_path.name)

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to ingest %s: %s", file_path.name, exc)
            errors.append(f"{file_path.name}: {exc}")

    return {
        "files_processed": files_processed,
        "chunks_stored": chunks_stored,
        "errors": errors,
        "collection_count": collection.count(),
    }


def _read_file(file_path: Path) -> str:
    """Read a text or PDF file and return its content as a string."""
    if file_path.suffix.lower() == ".pdf":
        try:
            import pdfplumber  # optional dependency
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except ImportError:
            logger.warning("pdfplumber not installed; skipping PDF %s", file_path.name)
            return ""
    else:
        return file_path.read_text(encoding="utf-8", errors="ignore")


# ── Retrieval  ─────────

def retrieve_relevant_chunks(
    query: str,
    top_k: int = TOP_K_RESULTS,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Retrieve the *top_k* most relevant chunks for *query*.

    Args:
        query: The user's question.
        top_k: Number of chunks to retrieve.

    Returns:
        Tuple of (list of chunk dicts, average similarity score 0-1).

    Each chunk dict has keys: ``document``, ``chunk``, ``distance``, ``chunk_index``.
    """
    collection = get_collection()
    if collection.count() == 0:
        logger.warning("Vector store is empty. Run /ingest first.")
        return [], 0.0

    query_embedding = generate_single_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks: List[Dict[str, Any]] = []
    distances = results["distances"][0] if results["distances"] else []

    for i, doc in enumerate(results["documents"][0]):
        distance = distances[i] if i < len(distances) else 1.0
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score 0-1
        similarity = max(0.0, 1.0 - distance)

        if similarity < SIMILARITY_THRESHOLD:
            logger.debug("Chunk below threshold (sim=%.3f), skipping.", similarity)
            continue

        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        chunks.append(
            {
                "document": meta.get("document", "unknown"),
                "chunk": doc,
                "similarity": round(similarity, 4),
                "chunk_index": meta.get("chunk_index", i),
            }
        )

    avg_similarity = (
        sum(c["similarity"] for c in chunks) / len(chunks) if chunks else 0.0
    )

    logger.info(
        "Retrieved %d relevant chunks for query (avg_sim=%.3f).",
        len(chunks),
        avg_similarity,
    )
    return chunks, avg_similarity


def score_confidence(avg_similarity: float) -> str:
    """Map an average similarity score to a human-readable confidence label."""
    if avg_similarity >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if avg_similarity >= CONFIDENCE_MED_THRESHOLD:
        return "medium"
    return "low"


def get_collection_stats() -> Dict[str, Any]:
    """Return basic statistics about the vector store collection."""
    collection = get_collection()
    return {
        "collection_name": CHROMA_COLLECTION_NAME,
        "total_chunks": collection.count(),
    }
