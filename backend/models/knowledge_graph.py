from __future__ import annotations

import datetime as dt
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

# ── Node & Edge type enums ──────────────────────────────────────


class NodeType(str, Enum):
    DECISION = "DECISION"
    RATIONALE = "RATIONALE"
    QUESTION = "QUESTION"
    PERSON = "PERSON"
    TOPIC = "TOPIC"
    DOCUMENT = "DOCUMENT"
    ACTION_ITEM = "ACTION_ITEM"


class EdgeType(str, Enum):
    DECIDED_BY = "DECIDED_BY"
    RELATES_TO = "RELATES_TO"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDES = "SUPERSEDES"
    OWNS = "OWNS"
    BLOCKED_BY = "BLOCKED_BY"
    REFERENCES = "REFERENCES"
    DERIVED_FROM = "DERIVED_FROM"
    ANSWERS = "ANSWERS"


# ── Core graph models ───────────────────────────────────────────


class KnowledgeNode(BaseModel):
    """A single entity in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: NodeType
    title: str
    content: str = ""
    source_ref: str | None = None
    source_type: str | None = None
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)
    embedding: list[float] | None = None


class KnowledgeEdge(BaseModel):
    """A directed relationship between two nodes."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = Field(default=1.0, ge=0.0)
    label: str | None = None
    evidence: str | None = None
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    metadata: dict = Field(default_factory=dict)


class GraphData(BaseModel):
    """Collection of nodes and edges for transport / rendering."""

    nodes: list[KnowledgeNode] = Field(default_factory=list)
    edges: list[KnowledgeEdge] = Field(default_factory=list)


class GraphQuery(BaseModel):
    """Natural-language graph query request."""

    query: str
    filters: dict | None = None
    limit: int = 50


class GraphQueryResult(BaseModel):
    """Result of a natural-language graph query."""

    nodes: list[KnowledgeNode] = Field(default_factory=list)
    edges: list[KnowledgeEdge] = Field(default_factory=list)
    answer: str | None = None
    confidence: float = 0.0
