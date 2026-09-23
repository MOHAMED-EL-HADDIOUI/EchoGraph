from __future__ import annotations

from collections.abc import Awaitable, Callable

from backend.exceptions import TranscriptionError
from backend.providers.transcription import (
    FasterWhisperTranscriber,
    TranscriptionUnavailable,
    TranscriptResult,
    WhisperAPITranscriber,
)

# Async callable taking an audio file path and returning transcript text.
TranscribePath = Callable[[str], Awaitable[str]]

__all__ = [
    "TranscribePath",
    "TranscriptResult",
    "TranscriptionError",
    "TranscriptionUnavailable",
    "make_transcriber",
    "transcribe_api",
    "transcribe_local",
]


async def transcribe_api(path: str, api_key: str, model: str = "whisper-1") -> str:
    return (await WhisperAPITranscriber(api_key=api_key, model=model).transcribe_file(path)).text


async def transcribe_local(path: str, model: str = "base") -> str:
    return (await FasterWhisperTranscriber(model=model).transcribe_file(path)).text


def make_transcriber(mode: str, api_key: str, model: str = "whisper-1") -> TranscribePath:
    async def transcribe(path: str) -> str:
        if mode == "local":
            return await transcribe_local(path)
        return await transcribe_api(path, api_key, model)

    return transcribe
