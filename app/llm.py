"""
LLM integration module.

Uses the Groq API (OpenAI-compatible) to generate answers grounded in
retrieved context.  The system prompt enforces:
  - Answers only from the provided context
  - No hallucination / guessing
  - Professional, clear language
  - Refusal to provide direct medical diagnoses
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)

# ── System prompt  ──────

SYSTEM_PROMPT = """You are a knowledgeable and professional healthcare information assistant.

Your role is to answer questions using ONLY the context passages provided to you.

STRICT RULES you must always follow:
1. Base your answer EXCLUSIVELY on the provided context. Do not use any outside knowledge.
2. If the context does not contain the information needed to answer the question, respond exactly with:
   "I could not find this information in the provided documents."
3. Never guess, speculate, or extrapolate beyond what is explicitly stated in the context.
4. Do not provide direct medical diagnoses, prescribe treatments, or give personalised medical advice.
5. When appropriate, advise the user to consult a qualified healthcare professional.
6. Keep responses clear, concise, and professional.
7. If the question is partially answered by the context, share what you found and note the limitation.
8. Cite document names when referencing specific policies or guidelines.

You are here to help users understand healthcare policies and guidelines — not to replace a doctor."""


def build_user_prompt(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Construct the user-facing prompt by injecting retrieved context.

    Args:
        question:      The user's question.
        context_chunks: List of retrieved chunk dicts (document, chunk, similarity).

    Returns:
        Formatted prompt string.
    """
    if not context_chunks:
        context_text = "[No relevant context found in the knowledge base.]"
    else:
        context_parts = []
        for i, chunk in enumerate(context_chunks, start=1):
            context_parts.append(
                f"[Source {i} — {chunk['document']}]\n{chunk['chunk']}"
            )
        context_text = "\n\n---\n\n".join(context_parts)

    return (
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER (based solely on the context above):"
    )


async def generate_answer(
    question: str,
    context_chunks: List[Dict[str, Any]],
) -> str:
    """
    Call the Groq LLM API and return the generated answer.

    Args:
        question:      The user's question.
        context_chunks: Retrieved context chunks from the RAG pipeline.

    Returns:
        The model's text response.

    Raises:
        RuntimeError: If the API call fails.
    """
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set.")
        raise RuntimeError(
            "LLM API key is not configured. Please set GROQ_API_KEY in your .env file."
        )

    user_prompt = build_user_prompt(question, context_chunks)

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    logger.info("Calling Groq LLM (model=%s)…", LLM_MODEL)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{GROQ_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        )

    if response.status_code != 200:
        logger.error(
            "Groq API error %d: %s", response.status_code, response.text
        )
        raise RuntimeError(
            f"LLM API returned status {response.status_code}: {response.text}"
        )

    data = response.json()
    answer: str = data["choices"][0]["message"]["content"].strip()
    logger.info("LLM response received (%d chars).", len(answer))
    return answer


def generate_answer_sync(
    question: str,
    context_chunks: List[Dict[str, Any]],
) -> str:
    """
    Synchronous wrapper for generate_answer (for use in non-async contexts).
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running inside an existing event loop (e.g., Jupyter, tests)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, generate_answer(question, context_chunks)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                generate_answer(question, context_chunks)
            )
    except RuntimeError:
        return asyncio.run(generate_answer(question, context_chunks))
