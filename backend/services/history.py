from __future__ import annotations

from backend.graph.manager import GraphManager
from backend.models.knowledge_graph import (
    DecisionHistory,
    EdgeType,
    HistoryEvent,
    KnowledgeNode,
)

# Relations that carry temporal/evolution meaning, both traversal directions.
_HISTORY_EDGES = {EdgeType.SUPERSEDES, EdgeType.CONTRADICTS}


def _relation(edge_type: EdgeType, outgoing: bool) -> str:
    if edge_type == EdgeType.SUPERSEDES:
        return "supersedes" if outgoing else "superseded_by"
    return "contradicts" if outgoing else "contradicted_by"


async def get_decision_history(
    graph: GraphManager, node_id: str, depth: int = 3
) -> DecisionHistory | None:
    """Bounded BFS over SUPERSEDES/CONTRADICTS edges in both directions.

    Answers "how did this decision evolve?" deterministically from explicit
    edges — never inferred from ingestion order. Returns None for unknown nodes.
    """
    root = await graph.get_node(node_id)
    if root is None:
        return None
    events: list[HistoryEvent] = []
    visited = {node_id}
    frontier: list[tuple[KnowledgeNode, int]] = [(root, 0)]
    while frontier:
        current, level = frontier.pop(0)
        if level >= depth:
            continue
        try:
            edges = await graph.get_edges(current.id)
        except Exception:  # noqa: BLE001 — unreadable node ends this branch
            edges = []
        for edge in sorted(edges, key=lambda e: e.id):
            if edge.edge_type not in _HISTORY_EDGES:
                continue
            outgoing = edge.source_id == current.id
            other_id = edge.target_id if outgoing else edge.source_id
            if other_id in visited:
                continue
            visited.add(other_id)
            try:
                other = await graph.get_node(other_id)
            except Exception:  # noqa: BLE001 — deleted nodes are skipped
                other = None
            if other is None:
                continue
            events.append(
                HistoryEvent(
                    relation=_relation(edge.edge_type, outgoing),
                    node=other,
                    edge_id=edge.id,
                    quote=edge.evidence or "",
                    ingestion_id=edge.ingestion_id or "",
                )
            )
            frontier.append((other, level + 1))
    return DecisionHistory(node=root, events=events)
