"""
Unit and integration tests for the Healthcare AI Assistant backend.

Run with:
    pytest tests/ -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.rag import chunk_text, score_confidence
from app.agent import detect_intent, check_available_slots, _extract_department


#   FastAPI test client  

client = TestClient(app)


#   RAG unit tests     

class TestChunkText:
    def test_basic_chunking(self):
        text = " ".join([f"word{i}" for i in range(1000)])
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(c, str)
            assert len(c) > 0

    def test_short_text_single_chunk(self):
        text = "Hello world this is a short text."
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=100)
        assert chunks == []

    def test_overlap_less_than_chunk_size(self):
        text = " ".join([f"w{i}" for i in range(200)])
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        # Should not raise; just verify output type
        assert all(isinstance(c, str) for c in chunks)


class TestScoreConfidence:
    def test_high_confidence(self):
        assert score_confidence(0.9) == "high"

    def test_medium_confidence(self):
        assert score_confidence(0.6) == "medium"

    def test_low_confidence(self):
        assert score_confidence(0.2) == "low"

    def test_boundary_high(self):
        assert score_confidence(0.75) == "high"

    def test_boundary_medium(self):
        assert score_confidence(0.50) == "medium"


#   Agent unit tests    

class TestDetectIntent:
    def test_appointment_keywords(self):
        assert detect_intent("I want to book a cardiology appointment") == "appointment"
        assert detect_intent("Can I schedule a visit on Monday?") == "appointment"
        assert detect_intent("What slots are available?") == "appointment"

    def test_rag_keywords(self):
        assert detect_intent("What is HIPAA?") == "rag"
        assert detect_intent("How do I refill my medication?") == "rag"
        assert detect_intent("What are the telehealth requirements?") == "rag"

    def test_case_insensitive(self):
        assert detect_intent("BOOK AN APPOINTMENT") == "appointment"


class TestExtractDepartment:
    def test_known_department(self):
        assert _extract_department("I need a cardiology appointment") == "cardiology"
        assert _extract_department("Orthopedics visit please") == "orthopedics"

    def test_fallback_to_primary_care(self):
        assert _extract_department("I need an appointment") == "primary care"


class TestCheckAvailableSlots:
    def test_known_department(self):
        result = check_available_slots("cardiology")
        assert result["department"] == "Cardiology"
        assert "available_slots" in result
        assert isinstance(result["available_slots"], list)

    def test_unknown_department_fallback(self):
        result = check_available_slots("unknowndept")
        assert "available_slots" in result

    def test_slot_structure(self):
        result = check_available_slots("pediatrics")
        for slot in result["available_slots"]:
            assert "date" in slot
            assert "times" in slot
            assert isinstance(slot["times"], list)


#   API endpoint tests   

class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "collection_stats" in data


class TestAskEndpoint:
    @patch("app.main.process_question")
    def test_ask_returns_answer(self, mock_process):
        mock_process.return_value = {
            "answer": "Test answer",
            "sources": [{"document": "test.txt", "chunk": "some chunk text"}],
            "confidence": "high",
            "intent": "rag",
            "tool_used": "rag_pipeline",
        }
        # Make it a coroutine
        mock_process.side_effect = None
        mock_process.return_value = mock_process.return_value

        async def async_return(*args, **kwargs):
            return {
                "answer": "Test answer",
                "sources": [{"document": "test.txt", "chunk": "some chunk text"}],
                "confidence": "high",
                "intent": "rag",
                "tool_used": "rag_pipeline",
            }
        mock_process.side_effect = async_return

        response = client.post("/ask", json={"question": "What is HIPAA?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data

    def test_ask_rejects_empty_question(self):
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422  # Validation error

    def test_ask_rejects_too_short_question(self):
        response = client.post("/ask", json={"question": "Hi"})
        assert response.status_code == 422


class TestIngestEndpoint:
    def test_ingest_missing_dir(self):
        response = client.post(
            "/ingest", json={"data_dir": "/nonexistent/path/to/data"}
        )
        assert response.status_code == 404


class TestStatsEndpoint:
    def test_stats_returns_200(self):
        response = client.get("/stats")
        assert response.status_code == 200
