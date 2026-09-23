from __future__ import annotations

import hashlib
import json
import logging
import time
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

from fastapi import Request
from fastapi.responses import Response

# Correlation ID for the current request; "-" outside request scope.
request_id: ContextVar[str] = ContextVar("echograph_request_id", default="-")


def new_request_id() -> str:
    return uuid4().hex[:12]


async def request_id_middleware(request: Request, call_next) -> Response:
    rid = request.headers.get("x-request-id") or new_request_id()
    request_id.set(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


def fingerprint(content: str) -> str:
    """Short content hash for logs. Never log raw transcripts (see below)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Structured log line. Rule: lengths and hashes only — raw source text
    (transcripts, chunk bodies, prompts with user data) must never be logged."""
    payload = {"request_id": request_id.get(), **fields}
    logger.log(level, "%s %s", event, json.dumps(payload, default=str))


class Timer:
    """Sync/async latency helper: `elapsed_ms()` since creation."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0
