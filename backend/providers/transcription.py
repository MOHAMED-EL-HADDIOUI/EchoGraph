from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from pydantic import BaseModel

from backend import obs
from backend.exceptions import TranscriptionError

logger = logging.getLogger(__name__)


class TranscriptResult(BaseModel):
    """Structured transcription outcome with provider metadata."""

    text: str = ""
    provider: str = ""
    model: str = ""
    duration_s: float | None = None
    language: str | None = None
    audio_bytes: int = 0
    transcript_sha: str = ""


class TranscriptionUnavailable(TranscriptionError):
    """Raised when the configured transcription backend cannot run."""


class WhisperAPITranscriber:
    """OpenAI Whisper API transcriber."""

    provider = "whisper-api"

    def __init__(self, api_key: str, model: str = "whisper-1") -> None:
        if not api_key:
            raise TranscriptionError("OPENAI_API_KEY not configured")
        self.api_key = api_key
        self.model = model

    async def transcribe_file(self, path: str) -> TranscriptResult:
        from openai import AsyncOpenAI

        timer = obs.Timer()
        try:
            client = AsyncOpenAI(api_key=self.api_key)
            data = await asyncio.to_thread(Path(path).read_bytes)
            resp = await client.audio.transcriptions.create(
                model=self.model,
                file=(Path(path).name, data),
                response_format="verbose_json",
            )
        except Exception as exc:
            raise TranscriptionError(f"transcription call failed: {exc}") from exc
        text = getattr(resp, "text", "") or ""
        result = TranscriptResult(
            text=text,
            provider=self.provider,
            model=self.model,
            duration_s=getattr(resp, "duration", None),
            language=getattr(resp, "language", None),
            audio_bytes=os.path.getsize(path),
            transcript_sha=obs.fingerprint(text),
        )
        obs.log_event(
            logger,
            "transcription.finish",
            mode="api",
            audio_bytes=result.audio_bytes,
            transcript_chars=len(text),
            latency_ms=round(timer.elapsed_ms(), 1),
        )
        return result


class FasterWhisperTranscriber:
    """Local faster-whisper transcriber (optional dependency)."""

    provider = "faster-whisper"

    def __init__(self, model: str = "base") -> None:
        self.model = model

    async def transcribe_file(self, path: str) -> TranscriptResult:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionUnavailable(
                "faster-whisper not installed (pip install .[transcription])"
            ) from exc

        timer = obs.Timer()

        def _run() -> TranscriptResult:
            whisper = WhisperModel(self.model)
            segments, info = whisper.transcribe(path)
            text = " ".join(s.text for s in segments).strip()
            return TranscriptResult(
                text=text,
                provider=self.provider,
                model=self.model,
                duration_s=getattr(info, "duration", None),
                language=getattr(info, "language", None),
                audio_bytes=os.path.getsize(path),
                transcript_sha=obs.fingerprint(text),
            )

        result = await asyncio.to_thread(_run)
        obs.log_event(
            logger,
            "transcription.finish",
            mode="local",
            audio_bytes=result.audio_bytes,
            transcript_chars=len(result.text),
            latency_ms=round(timer.elapsed_ms(), 1),
        )
        return result


def make_transcriber(mode: str, api_key: str, model: str = "whisper-1") -> object:
    """Factory used by deps: api mode needs a key, local mode needs the package."""
    if mode == "local":
        return FasterWhisperTranscriber()
    return WhisperAPITranscriber(api_key=api_key, model=model)


# Re-exported field list so API responses stay in sync with the model.
TRANSCRIPT_META_FIELDS = ("provider", "model", "duration_s", "language", "transcript_sha")
