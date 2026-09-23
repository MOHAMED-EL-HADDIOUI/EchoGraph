from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable

from backend import obs

logger = logging.getLogger(__name__)

# Async callable taking an audio file path and returning transcript text.
TranscribePath = Callable[[str], Awaitable[str]]


class TranscriptionUnavailable(Exception):
    """Raised when the configured transcription backend cannot run."""


async def transcribe_api(path: str, api_key: str, model: str = "whisper-1") -> str:
    from pathlib import Path

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    data = await asyncio.to_thread(Path(path).read_bytes)
    resp = await client.audio.transcriptions.create(model=model, file=(Path(path).name, data))
    return resp.text or ""


async def transcribe_local(path: str, model: str = "base") -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionUnavailable(
            "faster-whisper not installed (pip install .[transcription])"
        ) from exc

    def _run() -> str:
        whisper = WhisperModel(model)
        segments, _ = whisper.transcribe(path)
        return " ".join(s.text for s in segments).strip()

    return await asyncio.to_thread(_run)


def make_transcriber(mode: str, api_key: str) -> TranscribePath:
    async def transcribe(path: str) -> str:
        timer = obs.Timer()
        text = (
            await transcribe_local(path) if mode == "local" else await transcribe_api(path, api_key)
        )
        obs.log_event(
            logger,
            "transcription.finish",
            mode=mode,
            audio_bytes=os.path.getsize(path),
            transcript_chars=len(text),
            latency_ms=round(timer.elapsed_ms(), 1),
        )
        return text

    return transcribe
