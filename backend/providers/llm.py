from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from backend import obs
from backend.exceptions import ProviderError

logger = logging.getLogger(__name__)


class ExtractionLLM(Protocol):
    """Structured extraction contract: strict JSON, never free-form text."""

    async def extract(self, system: str, user: str) -> dict: ...


class AnswerLLM(Protocol):
    """Grounded answering contract: plain text over supplied context."""

    async def answer(self, system: str, user: str) -> str: ...


def _usage_of(resp: object) -> dict | None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    dump = getattr(usage, "model_dump", None)
    return dump() if callable(dump) else dict(usage)


class OpenAIExtractionProvider:
    """JSON-mode chat completions for extraction."""

    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise ProviderError("OPENAI_API_KEY not configured")
        self.model = model
        self.api_key = api_key

    async def extract(self, system: str, user: str) -> dict:
        from openai import AsyncOpenAI

        timer = obs.Timer()
        try:
            client = AsyncOpenAI(api_key=self.api_key)
            resp = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:
            raise ProviderError(f"extraction call failed: {exc}") from exc
        obs.log_event(
            logger,
            "llm.complete",
            model=self.model,
            latency_ms=round(timer.elapsed_ms(), 1),
            usage=_usage_of(resp),
        )
        try:
            return json.loads(resp.choices[0].message.content or "{}")
        except ValueError as exc:
            raise ProviderError(f"extraction returned non-JSON: {exc}") from exc

    def as_complete(self) -> Callable[[str, str], Awaitable[dict]]:
        async def complete(system: str, user: str) -> dict:
            return await self.extract(system, user)

        return complete


class OpenAIAnswerProvider:
    """Plain-text chat completions for grounded answers."""

    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise ProviderError("OPENAI_API_KEY not configured")
        self.model = model
        self.api_key = api_key

    async def answer(self, system: str, user: str) -> str:
        from openai import AsyncOpenAI

        timer = obs.Timer()
        try:
            client = AsyncOpenAI(api_key=self.api_key)
            resp = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
            )
        except Exception as exc:
            raise ProviderError(f"answer call failed: {exc}") from exc
        obs.log_event(
            logger,
            "llm.answer",
            model=self.model,
            latency_ms=round(timer.elapsed_ms(), 1),
            usage=_usage_of(resp),
        )
        return resp.choices[0].message.content or ""

    def as_answer(self) -> Callable[[str, str], Awaitable[str]]:
        async def complete(system: str, user: str) -> str:
            return await self.answer(system, user)

        return complete


class FakeExtractionProvider:
    """Deterministic extraction fake. Tests MUST use this, never a live key."""

    def __init__(self, payloads: list[dict] | dict) -> None:
        self._payloads = [payloads] if isinstance(payloads, dict) else list(payloads)
        self.calls: list[tuple[str, str]] = []

    async def extract(self, system: str, user: str) -> dict:
        self.calls.append((system, user))
        if not self._payloads:
            return {"nodes": [], "edges": []}
        if len(self._payloads) > 1:
            return self._payloads.pop(0)
        return self._payloads[0]

    def as_complete(self) -> Callable[[str, str], Awaitable[dict]]:
        async def complete(system: str, user: str) -> dict:
            return await self.extract(system, user)

        return complete


class FakeAnswerProvider:
    """Deterministic answer fake."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[str, str]] = []

    async def answer(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.text

    def as_answer(self) -> Callable[[str, str], Awaitable[str]]:
        async def complete(system: str, user: str) -> str:
            return await self.answer(system, user)

        return complete
