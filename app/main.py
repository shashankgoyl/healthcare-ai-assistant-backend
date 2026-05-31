"""
Healthcare AI Assistant — FastAPI Application

Endpoints:
  POST /ingest   — Ingest documents into the vector store
  POST /ask      — Answer a healthcare question via RAG + Agent
  GET  /health   — Health check
  GET  /stats    — Vector store statistics
"""

import logging
import logging.config
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import ALLOWED_ORIGINS, LOG_FORMAT, LOG_LEVEL
from app.agent import process_question
from app.rag import ingest_documents, get_collection_stats

# ── Logging setup  ──────

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ── Lifespan  ───────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Healthcare AI Assistant starting up…")
    yield
    logger.info("Healthcare AI Assistant shutting down.")


# ── App factory  ────────

app = FastAPI(
    title="Healthcare AI Assistant",
    description=(
        "A RAG-powered AI assistant that answers healthcare questions "
        "from a curated knowledge base of clinical and operational documents."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas  ───

class IngestRequest(BaseModel):
    data_dir: Optional[str] = Field(
        default=None,
        description="Path to the folder containing documents. Defaults to ./data.",
    )


class IngestResponse(BaseModel):
    status: str
    files_processed: int
    chunks_stored: int
    collection_count: int
    errors: List[str]
    message: str


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The healthcare question to answer.",
        example="Can a patient request a medication refill through telehealth?",
    )


class SourceReference(BaseModel):
    document: str
    chunk: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
    confidence: str
    intent: str
    tool_used: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    collection_stats: Dict[str, Any]


# ── Endpoints  ──────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns service health status and vector store statistics."""
    try:
        stats = get_collection_stats()
        return HealthResponse(
            status="ok",
            version="1.0.0",
            collection_stats=stats,
        )
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest healthcare documents into the vector store.

    Reads documents from the specified folder (or default ./data),
    splits them into chunks, generates embeddings, and stores them in ChromaDB.
    """
    logger.info("POST /ingest — data_dir=%s", request.data_dir)
    try:
        result = ingest_documents(data_dir=request.data_dir)
        msg = (
            f"Successfully ingested {result['files_processed']} file(s) "
            f"into {result['chunks_stored']} chunks."
        )
        if result["errors"]:
            msg += f" {len(result['errors'])} file(s) had errors."
        logger.info(msg)
        return IngestResponse(
            status="success",
            files_processed=result["files_processed"],
            chunks_stored=result["chunks_stored"],
            collection_count=result["collection_count"],
            errors=result["errors"],
            message=msg,
        )
    except FileNotFoundError as exc:
        logger.warning("Ingest failed — directory not found: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Ingest error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@app.post("/ask", response_model=AskResponse, tags=["QA"])
async def ask(request: AskRequest):
    """
    Answer a healthcare question using the RAG pipeline.

    The agent will:
    1. Detect whether the question is about appointments or general healthcare knowledge.
    2. Route to the appropriate tool (appointment mock or RAG pipeline).
    3. Return a grounded answer with source references and confidence score.
    """
    logger.info("POST /ask — question: %s", request.question[:120])
    t0 = time.perf_counter()

    try:
        result = await process_question(request.question)
    except RuntimeError as exc:
        logger.error("LLM error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error in /ask: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process question: {exc}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "POST /ask — completed in %.1f ms | confidence=%s | intent=%s",
        elapsed_ms,
        result.get("confidence"),
        result.get("intent"),
    )

    return AskResponse(
        answer=result["answer"],
        sources=[SourceReference(**s) for s in result.get("sources", [])],
        confidence=result.get("confidence", "low"),
        intent=result.get("intent", "rag"),
        tool_used=result.get("tool_used", "rag_pipeline"),
        processing_time_ms=round(elapsed_ms, 1),
    )


@app.get("/stats", tags=["System"])
async def stats():
    """Returns vector store statistics."""
    try:
        return get_collection_stats()
    except Exception as exc:
        logger.error("Stats error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "Healthcare AI Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
