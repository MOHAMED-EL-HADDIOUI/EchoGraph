from __future__ import annotations

from fastapi import HTTPException, Request

from backend.config import settings
from backend.database import get_db
from backend.graph.manager import GraphManager
from backend.services.extraction import CompleteJson, make_openai_complete
from backend.services.query import CompleteText, make_openai_answer

__all__ = [
    "get_answer_complete",
    "get_db",
    "get_extraction_complete",
    "get_graph_manager",
]


def get_graph_manager(request: Request) -> GraphManager:
    return request.app.state.graph


def get_extraction_complete() -> CompleteJson:
    """LLM hook for ingestion. Override in tests with a fake; 503 without a key."""
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    return make_openai_complete(settings.OPENAI_MODEL, settings.OPENAI_API_KEY)


def get_answer_complete() -> CompleteText | None:
    """LLM hook for query answers. Returns None (retrieval-only) without a key."""
    if not settings.OPENAI_API_KEY:
        return None
    return make_openai_answer(settings.OPENAI_MODEL, settings.OPENAI_API_KEY)
