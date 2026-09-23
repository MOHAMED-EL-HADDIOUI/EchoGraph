from __future__ import annotations

from backend.models.knowledge_graph import EdgeType, NodeType

# ── Node schema ─────────────────────────────────────────────────

NODE_SCHEMA: dict[NodeType, dict] = {
    NodeType.DECISION: {"label": "Decision", "color": "#3B82F6", "icon": "⚡"},
    NodeType.RATIONALE: {"label": "Rationale", "color": "#8B5CF6", "icon": "💡"},
    NodeType.QUESTION: {"label": "Question", "color": "#F59E0B", "icon": "❓"},
    NodeType.PERSON: {"label": "Person", "color": "#10B981", "icon": "👤"},
    NodeType.TOPIC: {"label": "Topic", "color": "#EC4899", "icon": "📌"},
    NodeType.DOCUMENT: {"label": "Document", "color": "#6B7280", "icon": "📄"},
    NodeType.ACTION_ITEM: {"label": "Action Item", "color": "#EF4444", "icon": "✅"},
    NodeType.TEAM: {"label": "Team", "color": "#14B8A6", "icon": "👥"},
    NodeType.ORGANIZATION: {"label": "Organization", "color": "#6366F1", "icon": "🏢"},
    NodeType.PROJECT: {"label": "Project", "color": "#0EA5E9", "icon": "📁"},
    NodeType.MEETING: {"label": "Meeting", "color": "#F472B6", "icon": "📅"},
    NodeType.MESSAGE: {"label": "Message", "color": "#94A3B8", "icon": "💬"},
    NodeType.EVENT: {"label": "Event", "color": "#A3E635", "icon": "📍"},
}

EDGE_SCHEMA: dict[EdgeType, dict] = {
    EdgeType.DECIDED_BY: {"label": "decided by", "color": "#60A5FA", "directed": True},
    EdgeType.DECIDES: {"label": "decides", "color": "#60A5FA", "directed": True},
    EdgeType.RELATES_TO: {"label": "relates to", "color": "#A78BFA", "directed": False},
    EdgeType.CONTRADICTS: {"label": "contradicts", "color": "#F87171", "directed": False},
    EdgeType.SUPERSEDES: {"label": "supersedes", "color": "#FBBF24", "directed": True},
    EdgeType.OWNS: {"label": "owns", "color": "#34D399", "directed": True},
    EdgeType.ASSIGNED_TO: {"label": "assigned to", "color": "#34D399", "directed": True},
    EdgeType.PART_OF: {"label": "part of", "color": "#FACC15", "directed": True},
    EdgeType.MENTIONS: {"label": "mentions", "color": "#CBD5E1", "directed": True},
    EdgeType.DISCUSSES: {"label": "discusses", "color": "#67E8F9", "directed": True},
    EdgeType.BLOCKED_BY: {"label": "blocked by", "color": "#FB923C", "directed": True},
    EdgeType.REFERENCES: {"label": "references", "color": "#9CA3AF", "directed": True},
    EdgeType.DERIVED_FROM: {"label": "derived from", "color": "#C084FC", "directed": True},
    EdgeType.ANSWERS: {"label": "answers", "color": "#2DD4BF", "directed": True},
}

# Valid relationship triples: (source_type, edge_type, target_type)
VALID_EDGES: set[tuple[NodeType, EdgeType, NodeType]] = {
    (NodeType.DECISION, EdgeType.DECIDED_BY, NodeType.PERSON),
    (NodeType.DECISION, EdgeType.RELATES_TO, NodeType.TOPIC),
    (NodeType.DECISION, EdgeType.CONTRADICTS, NodeType.DECISION),
    (NodeType.DECISION, EdgeType.SUPERSEDES, NodeType.DECISION),
    (NodeType.DECISION, EdgeType.REFERENCES, NodeType.DOCUMENT),
    (NodeType.RATIONALE, EdgeType.DERIVED_FROM, NodeType.DECISION),
    (NodeType.RATIONALE, EdgeType.REFERENCES, NodeType.DOCUMENT),
    (NodeType.QUESTION, EdgeType.RELATES_TO, NodeType.TOPIC),
    (NodeType.QUESTION, EdgeType.BLOCKED_BY, NodeType.DECISION),
    (NodeType.QUESTION, EdgeType.ANSWERS, NodeType.QUESTION),
    (NodeType.PERSON, EdgeType.OWNS, NodeType.ACTION_ITEM),
    (NodeType.PERSON, EdgeType.OWNS, NodeType.DECISION),
    (NodeType.ACTION_ITEM, EdgeType.RELATES_TO, NodeType.DECISION),
    (NodeType.ACTION_ITEM, EdgeType.BLOCKED_BY, NodeType.QUESTION),
    (NodeType.TOPIC, EdgeType.RELATES_TO, NodeType.TOPIC),
    (NodeType.DOCUMENT, EdgeType.REFERENCES, NodeType.DOCUMENT),
    # Organizational structure (additive; existing triples unchanged)
    (NodeType.PERSON, EdgeType.DECIDES, NodeType.DECISION),
    (NodeType.ACTION_ITEM, EdgeType.ASSIGNED_TO, NodeType.PERSON),
    (NodeType.PERSON, EdgeType.PART_OF, NodeType.TEAM),
    (NodeType.TEAM, EdgeType.PART_OF, NodeType.ORGANIZATION),
    (NodeType.PROJECT, EdgeType.PART_OF, NodeType.ORGANIZATION),
    (NodeType.MEETING, EdgeType.DISCUSSES, NodeType.TOPIC),
    (NodeType.PERSON, EdgeType.DISCUSSES, NodeType.TOPIC),
    (NodeType.MESSAGE, EdgeType.MENTIONS, NodeType.PERSON),
    (NodeType.MESSAGE, EdgeType.MENTIONS, NodeType.TOPIC),
    (NodeType.MEETING, EdgeType.REFERENCES, NodeType.DOCUMENT),
}

NEO4J_CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:KnowledgeNode) REQUIRE n.id IS UNIQUE",
]


def validate_node(node_type: NodeType) -> bool:
    return node_type in NODE_SCHEMA


def validate_edge(source_type: NodeType, edge_type: EdgeType, target_type: NodeType) -> bool:
    """Return True if the triple is in VALID_EDGES or if we allow open-schema."""
    # Open-schema: always allow RELATES_TO between any types
    if edge_type == EdgeType.RELATES_TO:
        return True
    return (source_type, edge_type, target_type) in VALID_EDGES
