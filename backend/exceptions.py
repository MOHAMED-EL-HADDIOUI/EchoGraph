from __future__ import annotations


class EchoGraphError(Exception):
    """Base class for all EchoGraph domain errors."""


class ConfigurationError(EchoGraphError):
    """Invalid or missing configuration."""


class GraphError(EchoGraphError):
    """Graph backend failure."""


class GraphValidationError(GraphError):
    """Graph mutation rejected by schema validation."""


class IngestionError(EchoGraphError):
    """Ingestion lifecycle failure."""


class ExtractionError(EchoGraphError):
    """Extraction pipeline failure (not bad LLM output — that is recorded)."""


class ProviderError(EchoGraphError):
    """External provider (LLM, embeddings, transcription) failure."""


class TranscriptionError(ProviderError):
    """Transcription backend failure."""


class AuthenticationError(EchoGraphError):
    """Authentication failure (services raise; routes map to 401)."""
