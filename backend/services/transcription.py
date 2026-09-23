from __future__ import annotations

from collections.abc import Awaitable, Callable

from backend.exceptions import TranscriptionError
from backend.providers.transcription import (
    TranscriptionUnavailable,
    TranscriptResult,
)

# Async callable taking an audio file path and returning a structured result.
TranscribePath = Callable[[str], Awaitable[TranscriptResult]]

__all__ = [
    "TranscribePath",
    "TranscriptResult",
    "TranscriptionError",
    "TranscriptionUnavailable",
    "make_transcriber",
]


def make_transcriber(mode: str, api_key: str, model: str = "whisper-1") -> TranscribePath:
    from backend.providers.transcription import (
        FasterWhisperTranscriber,
        WhisperAPITranscriber,
    )

    async def transcribe(path: str) -> TranscriptResult:
        if mode == "local":
            return await FasterWhisperTranscriber().transcribe_file(path)
        return await WhisperAPITranscriber(api_key=api_key, model=model).transcribe_file(path)

    return transcribe
