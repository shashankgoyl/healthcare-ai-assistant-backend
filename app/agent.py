"""
Agent module — implements a simple intent-based router.

Workflow:
  1. Classify the user's question intent.
  2. Route to the appropriate tool:
       - appointment_tool  →  check_available_slots()
       - rag_tool          →  RAG pipeline (LLM + vector search)
  3. Return a structured result.
"""

import logging
import random
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from app.config import TOP_K_RESULTS
from app.llm import generate_answer
from app.rag import retrieve_relevant_chunks, score_confidence

logger = logging.getLogger(__name__)

#   Intent detection   

APPOINTMENT_KEYWORDS = {
    "appointment",
    "book",
    "booking",
    "schedule",
    "slot",
    "available",
    "availability",
    "reserve",
    "reschedule",
    "cancel appointment",
    "visit date",
    "open slot",
    "next available",
    "cardiology",
    "orthopedics",
    "dermatology",
    "neurology",
    "pediatrics",
    "ob/gyn",
    "oncology",
    "mental health",
    "primary care",
    "radiology",
}


def detect_intent(question: str) -> str:
    """
    Classify the question as 'appointment' or 'rag'.

    Args:
        question: The raw user question.

    Returns:
        Intent label: ``"appointment"`` or ``"rag"``.
    """
    lower = question.lower()
    if any(kw in lower for kw in APPOINTMENT_KEYWORDS):
        return "appointment"
    return "rag"


#   Mock appointment tool                 

DEPARTMENTS = {
    "cardiology": ["Mon", "Wed", "Fri"],
    "orthopedics": ["Tue", "Thu"],
    "dermatology": ["Mon", "Tue", "Wed", "Thu"],
    "neurology": ["Wed", "Fri"],
    "pediatrics": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    "ob/gyn": ["Mon", "Tue", "Wed", "Thu", "Fri"],
    "mental health": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    "primary care": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    "oncology": ["Mon", "Tue", "Wed", "Thu"],
    "radiology": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
}

TIME_SLOTS = ["9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM", "4:00 PM"]


def _extract_department(question: str) -> str:
    """Best-effort extraction of department name from question."""
    lower = question.lower()
    for dept in DEPARTMENTS:
        if dept in lower:
            return dept
    return "primary care"


def check_available_slots(department: str, requested_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Mock tool: returns simulated available appointment slots.

    Args:
        department:      Department name (e.g. "cardiology").
        requested_date:  ISO date string or natural language date (optional).

    Returns:
        Dict with department, date range checked, and available slots.
    """
    dept_lower = department.lower()
    available_days = DEPARTMENTS.get(dept_lower, ["Mon", "Tue", "Wed", "Thu", "Fri"])

    # Generate slots for the next 7 days
    slots = []
    today = date.today()
    for offset in range(1, 8):
        check_date = today + timedelta(days=offset)
        day_abbr = check_date.strftime("%a")
        if day_abbr in available_days:
            # Randomly pick 2-3 available time slots
            day_slots = random.sample(TIME_SLOTS, k=min(3, len(TIME_SLOTS)))
            slots.append(
                {
                    "date": check_date.strftime("%A, %B %d %Y"),
                    "day": day_abbr,
                    "times": sorted(day_slots),
                }
            )

    return {
        "department": department.title(),
        "search_window": f"{today + timedelta(days=1)} to {today + timedelta(days=7)}",
        "available_slots": slots[:3],  # Return at most 3 dates
        "note": (
            "These are mock slots for demonstration. "
            "Please call scheduling or use the patient portal to confirm real availability."
        ),
    }


def _format_slots_response(department: str, slot_data: Dict[str, Any]) -> str:
    """Format the slot data into a readable response string."""
    slots = slot_data.get("available_slots", [])
    if not slots:
        return (
            f"I checked mock availability for **{department.title()}** "
            "but found no open slots in the next 7 days. "
            "Please call our scheduling department directly."
        )

    lines = [
        f"I checked mock appointment availability for **{slot_data['department']}**.",
        "",
        "Available slots (next 7 days):",
    ]
    for s in slots:
        times_str = ", ".join(s["times"])
        lines.append(f"  • {s['date']}: {times_str}")

    lines += [
        "",
        f"_{slot_data['note']}_",
        "",
        "To confirm or book, please use the **patient portal** or call our **scheduling department**.",
    ]
    return "\n".join(lines)


#   Main agent entry point                 

async def process_question(question: str) -> Dict[str, Any]:
    """
    Route a question to the appropriate tool and return a structured response.

    Args:
        question: The user's question.

    Returns:
        Dict with keys: answer, sources, confidence, intent, tool_used.
    """
    intent = detect_intent(question)
    logger.info("Detected intent: %s for question: %s", intent, question[:80])

    #   Appointment routing                
    if intent == "appointment":
        department = _extract_department(question)
        logger.info("Routing to appointment tool for department: %s", department)

        slot_data = check_available_slots(department)
        answer = _format_slots_response(department, slot_data)

        return {
            "answer": answer,
            "sources": [],
            "confidence": "high",
            "intent": "appointment",
            "tool_used": "check_available_slots",
            "slot_data": slot_data,
        }

    #   RAG routing    
    logger.info("Routing to RAG pipeline.")
    chunks, avg_similarity = retrieve_relevant_chunks(question, top_k=TOP_K_RESULTS)

    if not chunks:
        return {
            "answer": "I could not find this information in the provided documents.",
            "sources": [],
            "confidence": "low",
            "intent": "rag",
            "tool_used": "rag_pipeline",
        }

    answer = await generate_answer(question, chunks)
    confidence = score_confidence(avg_similarity)

    sources = [
        {"document": c["document"], "chunk": c["chunk"][:300] + "…" if len(c["chunk"]) > 300 else c["chunk"]}
        for c in chunks
    ]
    # Deduplicate sources by document name while preserving order
    seen: set = set()
    unique_sources = []
    for s in sources:
        key = (s["document"], s["chunk"][:60])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {
        "answer": answer,
        "sources": unique_sources,
        "confidence": confidence,
        "intent": "rag",
        "tool_used": "rag_pipeline",
    }
