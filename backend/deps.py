from __future__ import annotations

import logging
import secrets

from fastapi import HTTPException, Request

from backend.config import settings
from backend.database import get_db
from backend.graph.manager import GraphManager
from backend.services.extraction import (
    CompleteJson,
    EmbedTexts,
    make_openai_complete,
    make_openai_embeddings,
)
from backend.services.query import CompleteText, make_openai_answer
from backend.services.transcription import TranscribePath, make_transcriber

logger = logging.getLogger(__name__)

__all__ = [
    "get_answer_complete",
    "get_db",
    "get_embedder",
    "get_extraction_complete",
    "get_graph_manager",
    "get_transcriber",
    "require_api_key",
]


async def require_api_key(request: Request) -> None:
    """Enforce X-API-Key when API_KEY is configured; open server otherwise."""
    if not settings.API_KEY:
        return
    provided = request.headers.get("x-api-key", "")
    if not provided or not secrets.compare_digest(provided, settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_graph_manager(request: Request) -> GraphManager:
    return request.app.state.graph


def get_extraction_complete() -> CompleteJson | None:
    """LLM hook for ingestion. Override in tests with a fake.

    Returns None without a key: the sync route turns that into 503, while
    the background path enqueues regardless (the worker fails the job if
    still keyless).
    """
    if not settings.OPENAI_API_KEY:
        return None
    return make_openai_complete(settings.OPENAI_EXTRACTION_MODEL, settings.OPENAI_API_KEY)


def get_answer_complete() -> CompleteText | None:
    """LLM hook for query answers. Returns None (retrieval-only) without a key."""
    if not settings.OPENAI_API_KEY:
        return None
    return make_openai_answer(settings.OPENAI_ANSWER_MODEL, settings.OPENAI_API_KEY)


def get_embedder() -> EmbedTexts | None:
    """Embedding hook for dedup. None unless enabled AND keyed (flag-gated)."""
    if not settings.ENABLE_EMBEDDING_DEDUP or not settings.OPENAI_API_KEY:
        return None
    return make_openai_embeddings(settings.EMBEDDING_MODEL, settings.OPENAI_API_KEY)


def get_transcriber() -> TranscribePath:
    """Audio hook for transcription. Override in tests; 503 without a key (api mode)."""
    if settings.WHISPER_MODE == "local":
        return make_transcriber("local", settings.OPENAI_API_KEY)
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    return make_transcriber("api", settings.OPENAI_API_KEY)
