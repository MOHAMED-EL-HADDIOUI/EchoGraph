from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from backend import obs
from backend.exceptions import ProviderError

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Text → vectors contract."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise ProviderError("OPENAI_API_KEY not configured")
        self.model = model
        self.api_key = api_key

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import AsyncOpenAI

        timer = obs.Timer()
        try:
            client = AsyncOpenAI(api_key=self.api_key)
            resp = await client.embeddings.create(model=self.model, input=texts)
        except Exception as exc:
            raise ProviderError(f"embedding call failed: {exc}") from exc
        obs.log_event(
            logger,
            "llm.embed",
            model=self.model,
            latency_ms=round(timer.elapsed_ms(), 1),
            texts=len(texts),
        )
        return [list(d.embedding) for d in resp.data]

    def as_embed(self) -> Callable[[list[str]], Awaitable[list[list[float]]]]:
        async def embed(texts: list[str]) -> list[list[float]]:
            return await self.embed(texts)

        return embed


class FakeEmbeddingProvider:
    """Deterministic embedding fake backed by a caller-supplied mapping."""

    def __init__(self, fn: Callable[[list[str]], list[list[float]]]) -> None:
        self._fn = fn
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return self._fn(texts)

    def as_embed(self) -> Callable[[list[str]], Awaitable[list[list[float]]]]:
        async def embed(texts: list[str]) -> list[list[float]]:
            return await self.embed(texts)

        return embed
