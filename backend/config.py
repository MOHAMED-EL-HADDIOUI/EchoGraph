from __future__ import annotations

import json
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_ENV: str = "development"

    # LLM (separate models per task; OPENAI_MODEL kept as legacy fallback)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EXTRACTION_MODEL: str = "gpt-4o"
    OPENAI_ANSWER_MODEL: str = "gpt-4o"
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("OPENAI_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
    )
    LLM_TIMEOUT_SECONDS: float = Field(
        default=60.0, validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS", "EXTRACTION_TIMEOUT_S")
    )
    MAX_EXTRACTION_ERRORS: int = 20

    # Prompts (bumped deliberately; recorded in processing metadata + eval reports)
    EXTRACTION_PROMPT_VERSION: str = "v1"
    ANSWER_PROMPT_VERSION: str = "v1"

    # Chunking
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 200
    MAX_CHUNKS: int = 10

    # Graph
    GRAPH_BACKEND: Literal["neo4j", "networkx"] = "networkx"
    NETWORKX_PATH: str = Field(
        default="echograph_data.json",
        validation_alias=AliasChoices("NETWORKX_PATH", "GRAPH_DATA_PATH"),
    )
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = Field(
        default="neo4j", validation_alias=AliasChoices("NEO4J_USERNAME", "NEO4J_USER")
    )
    NEO4J_PASSWORD: str = "password"
    GRAPH_DATA_PATH: str = "echograph_data.json"  # legacy alias of NETWORKX_PATH

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./echograph.db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Transcription
    WHISPER_MODE: Literal["api", "local"] = Field(
        default="api", validation_alias=AliasChoices("TRANSCRIPTION_PROVIDER", "WHISPER_MODE")
    )
    WHISPER_MODEL: str = "whisper-1"

    # Server
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]
    # Empty API_KEY = open server (dev). Set it in any shared deployment.
    API_KEY: str = ""
    MAX_TEXT_CHARS: int = Field(
        default=50_000, validation_alias=AliasChoices("MAX_TEXT_CHARS", "MAX_INGESTION_CHARS")
    )
    MAX_INGESTION_CHARS: int = 50_000  # legacy alias of MAX_TEXT_CHARS
    MAX_AUDIO_BYTES: int = 26_214_400
    # When true, POST /ingestion/{id}/process enqueues a Celery task (202)
    # instead of extracting inline. Requires a running worker + Redis.
    USE_BACKGROUND_JOBS: bool = False

    # Embeddings
    ENABLE_EMBEDDING_DEDUP: bool = False
    EMBEDDING_DEDUP_THRESHOLD: float = 0.85
    ENABLE_EXTRACTION_CACHE: bool = False

    # Experimental stale-decision detection (off unless explicitly enabled)
    ENABLE_STALE_DECISION_DETECTION: bool = False
    STALE_AFTER_DAYS: int = 90

    # Result caps
    MAX_EVIDENCE_ITEMS: int = 200

    @property
    def cors_origins_list(self) -> list[str]:
        if isinstance(self.CORS_ORIGINS, str):
            return json.loads(self.CORS_ORIGINS)
        return self.CORS_ORIGINS


settings = Settings()
