from __future__ import annotations

import datetime as dt
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.models.knowledge_graph import GraphDiff


class SourceType(str, Enum):
    MEETING = "MEETING"
    SLACK = "SLACK"
    EMAIL = "EMAIL"
    DOCUMENT = "DOCUMENT"
    CODE_REVIEW = "CODE_REVIEW"
    MANUAL = "MANUAL"
    AUDIO = "AUDIO"


class IngestionStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class IngestionRequest(BaseModel):
    """Inbound ingestion request."""

    source_type: SourceType
    content: str | None = None
    file_path: str | None = None
    title: str | None = None
    metadata: dict = Field(default_factory=dict)


class IngestionResult(BaseModel):
    """Outcome of a single ingestion job."""

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    status: IngestionStatus = IngestionStatus.COMPLETED
    nodes_created: int = 0
    edges_created: int = 0
    errors: list[str] = Field(default_factory=list)


class IngestionJobRead(BaseModel):
    """API view of a persisted ingestion job."""

    job_id: str
    source_type: str
    status: str
    nodes_created: int = 0
    edges_created: int = 0
    title: str = ""
    error: str = ""
    attempts: int = 0
    content_sha: str = ""
    metadata: dict = Field(default_factory=dict)
    created_at: dt.datetime | None = None
    completed_at: dt.datetime | None = None
    # Set only by the process endpoint; 0 elsewhere.
    notifications_created: int = 0
    # Set only by the process endpoint: what the run changed.
    diff: GraphDiff | None = None


class Document(BaseModel):
    """Normalised intermediate representation used by connectors."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: SourceType
    title: str = ""
    content: str = ""
    chunks: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
