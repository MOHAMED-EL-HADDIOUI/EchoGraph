from __future__ import annotations

import json
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EXTRACTION_TIMEOUT_S: float = 60.0
    ENABLE_EMBEDDING_DEDUP: bool = False
    EMBEDDING_DEDUP_THRESHOLD: float = 0.85

    # Graph
    GRAPH_BACKEND: Literal["neo4j", "networkx"] = "networkx"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    GRAPH_DATA_PATH: str = "echograph_data.json"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./echograph.db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Transcription
    WHISPER_MODE: Literal["api", "local"] = "api"

    # Server
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]
    # Empty API_KEY = open server (dev). Set it in any shared deployment.
    API_KEY: str = ""
    MAX_INGESTION_CHARS: int = 50_000

    @property
    def cors_origins_list(self) -> list[str]:
        if isinstance(self.CORS_ORIGINS, str):
            return json.loads(self.CORS_ORIGINS)
        return self.CORS_ORIGINS


settings = Settings()
