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
    timer = Timer()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    logging.getLogger(__name__).info(
        "http.request %s",
        json.dumps(
            {
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(timer.elapsed_ms(), 1),
            }
        ),
    )
    return response


def fingerprint(content: str) -> str:
    """Short content hash for logs. Never log raw transcripts (see below)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def fingerprint_content(text: str) -> str:
    """SHA-256 content fingerprint (full hex). Identity without leakage."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# USD per 1M tokens (prompt, completion). Only used when the provider
# reports usage; unknown models yield None (no invented pricing).
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def estimate_cost_usd(model: str, usage: dict | None) -> float | None:
    """Estimated call cost from reported token usage. None when unknown."""
    if not usage or model not in _MODEL_PRICES:
        return None
    prompt_price, completion_price = _MODEL_PRICES[model]
    try:
        return round(
            usage.get("prompt_tokens", 0) / 1_000_000 * prompt_price
            + usage.get("completion_tokens", 0) / 1_000_000 * completion_price,
            6,
        )
    except (TypeError, AttributeError):
        return None


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
