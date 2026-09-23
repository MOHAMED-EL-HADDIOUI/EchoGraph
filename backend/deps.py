from __future__ import annotations

from fastapi import HTTPException, Request

from backend.config import settings
from backend.database import get_db
from backend.graph.manager import GraphManager
from backend.services.extraction import CompleteJson, make_openai_complete

__all__ = ["get_db", "get_extraction_complete", "get_graph_manager"]


def get_graph_manager(request: Request) -> GraphManager:
    return request.app.state.graph


def get_extraction_complete() -> CompleteJson:
    """LLM hook for ingestion. Override in tests with a fake; 503 without a key."""
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    return make_openai_complete(settings.OPENAI_MODEL, settings.OPENAI_API_KEY)
