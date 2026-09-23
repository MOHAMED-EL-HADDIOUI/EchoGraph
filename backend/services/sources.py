from __future__ import annotations

import datetime as dt
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.models.ingestion import SourceType


class NormalizedDocument(BaseModel):
    """Single pipeline input shape. Email, Slack, meetings, and transcripts
    all converge here so extraction never changes per source."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: SourceType = SourceType.MANUAL
    title: str = ""
    text: str = ""
    author: str | None = None
    timestamp: dt.datetime | None = None
    metadata: dict = Field(default_factory=dict)


class SourceAdapter(Protocol):
    """Future connectors (Slack, Gmail, Teams, …) implement this."""

    async def normalize(self, raw: dict) -> NormalizedDocument: ...


class ManualAdapter:
    """Adapter for direct API submissions."""

    async def normalize(self, raw: dict) -> NormalizedDocument:
        return NormalizedDocument(
            source_type=raw.get("source_type", SourceType.MANUAL),
            title=raw.get("title", ""),
            text=raw.get("content", ""),
            author=raw.get("author"),
            timestamp=raw.get("timestamp"),
            metadata=raw.get("metadata", {}),
        )


class AudioAdapter:
    """Adapter for transcribed audio."""

    async def normalize(self, raw: dict) -> NormalizedDocument:
        return NormalizedDocument(
            source_type=SourceType.AUDIO,
            title=raw.get("filename", "audio"),
            text=raw.get("transcript", ""),
            timestamp=raw.get("timestamp"),
            metadata={
                "filename": raw.get("filename"),
                "provider": raw.get("provider", ""),
                "model": raw.get("model", ""),
                "duration_s": raw.get("duration_s"),
                "language": raw.get("language"),
                "transcript_sha": raw.get("transcript_sha", ""),
            },
        )
