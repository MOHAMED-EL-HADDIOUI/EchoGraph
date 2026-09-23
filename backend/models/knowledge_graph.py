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
    TEAM = "TEAM"
    ORGANIZATION = "ORGANIZATION"
    PROJECT = "PROJECT"
    MEETING = "MEETING"
    MESSAGE = "MESSAGE"
    EVENT = "EVENT"
    TOPIC = "TOPIC"
    DOCUMENT = "DOCUMENT"
    ACTION_ITEM = "ACTION_ITEM"


class EdgeType(str, Enum):
    DECIDED_BY = "DECIDED_BY"
    DECIDES = "DECIDES"
    RELATES_TO = "RELATES_TO"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDES = "SUPERSEDES"
    OWNS = "OWNS"
    ASSIGNED_TO = "ASSIGNED_TO"
    PART_OF = "PART_OF"
    MENTIONS = "MENTIONS"
    DISCUSSES = "DISCUSSES"
    BLOCKED_BY = "BLOCKED_BY"
    REFERENCES = "REFERENCES"
    DERIVED_FROM = "DERIVED_FROM"
    ANSWERS = "ANSWERS"


# ── Core graph models ───────────────────────────────────────────


class Evidence(BaseModel):
    """Provenance for one extracted fact. Never fabricated: empty quote means
    the extractor could not tie the fact to a source span."""

    ingestion_id: str = ""
    source_type: str = ""
    source_id: str = ""
    quote: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)


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
    evidence: list[Evidence] = Field(default_factory=list)
    # Temporal lifecycle: valid_from/valid_until bound the claim when known;
    # observed_at records when extraction saw it; state tracks decision
    # evolution (active / superseded / uncertain). Never inferred from
    # ingestion order alone.
    valid_from: dt.datetime | None = None
    valid_until: dt.datetime | None = None
    observed_at: dt.datetime | None = None
    state: str = "active"


class KnowledgeEdge(BaseModel):
    """A directed relationship between two nodes."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = Field(default=1.0, ge=0.0)
    label: str | None = None
    evidence: str | None = None
    ingestion_id: str | None = None
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    metadata: dict = Field(default_factory=dict)


class GraphData(BaseModel):
    """Collection of nodes and edges for transport / rendering."""

    nodes: list[KnowledgeNode] = Field(default_factory=list)
    edges: list[KnowledgeEdge] = Field(default_factory=list)


class GraphDiff(BaseModel):
    """What one processing run changed (or would change, for dry runs)."""

    nodes_added: int = 0
    nodes_merged: int = 0
    edges_added: int = 0
    notifications_created: int = 0
    contradictions: int = 0
    missing_owners: int = 0


class DryRunResult(BaseModel):
    """Proposed mutations from a dry run. Nothing was persisted."""

    job_id: str
    diff: GraphDiff = Field(default_factory=GraphDiff)
    proposed_nodes: list[KnowledgeNode] = Field(default_factory=list)
    proposed_edges: list[KnowledgeEdge] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class Verdict(str, Enum):
    SUPPORTED = "supported"  # answer/context comes from extracted facts
    INFERRED = "inferred"  # reserved: derived, not directly extracted
    UNCERTAIN = "uncertain"  # not enough evidence to answer
    CONTRADICTORY = "contradictory"  # evidence conflicts


class EvidenceItem(BaseModel):
    """One citable provenance entry attached to a query result."""

    kind: str  # "node" or "edge"
    ref_id: str
    label: str
    ingestion_id: str = ""
    quote: str = ""
    confidence: float = 0.0


class GraphQuery(BaseModel):
    """Natural-language graph query request."""

    query: str
    filters: dict | None = None
    limit: int = 50
    include_evidence: bool = False


class GraphQueryResult(BaseModel):
    """Result of a natural-language graph query."""

    nodes: list[KnowledgeNode] = Field(default_factory=list)
    edges: list[KnowledgeEdge] = Field(default_factory=list)
    answer: str | None = None
    confidence: float = 0.0
    verdict: Verdict = Verdict.UNCERTAIN
    citations: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
