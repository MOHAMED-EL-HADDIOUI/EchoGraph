from __future__ import annotations

import json
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import NotificationRow
from backend.graph.manager import GraphManager
from backend.models.knowledge_graph import EdgeType, KnowledgeEdge, KnowledgeNode, NodeType
from backend.models.notifications import (
    NotificationExplanation,
    NotificationPriority,
    NotificationRead,
    NotificationType,
)
from backend.repositories import NotificationRepository

# Node types that should have an owner; edge types that confer ownership.
OWNERLESS_TYPES = {NodeType.ACTION_ITEM, NodeType.DECISION}
OWNERSHIP_EDGES = {EdgeType.OWNS, EdgeType.DECIDED_BY, EdgeType.ASSIGNED_TO, EdgeType.DECIDES}


async def find_duplicate(
    db: AsyncSession,
    ntype: NotificationType,
    node_ids: set[str],
    ingestion_id: str = "",
) -> NotificationRow | None:
    """Return an unread notification covering the same finding, if any.

    Sames means: same type, same affected-node set, and same ingestion
    (both empty counts as same). Read notifications never suppress.
    """
    return await NotificationRepository(db).find_duplicate(ntype.value, node_ids, ingestion_id)


async def evaluate_notifications(
    graph: GraphManager,
    nodes: list[KnowledgeNode],
    edges: list[KnowledgeEdge],
    ingestion_id: str,
    db: AsyncSession,
) -> list[NotificationRow]:
    """Build (unsaved) notification rows for freshly extracted nodes/edges."""
    created: list[NotificationRow] = []

    def queue(
        ntype: NotificationType,
        priority: NotificationPriority,
        title: str,
        message: str,
        node_ids: list[str],
        edge_ids: list[str],
    ) -> None:
        created.append(
            NotificationRow(
                id=str(uuid4()),
                type=ntype.value,
                priority=priority.value,
                severity=priority.value,
                title=title,
                message=message,
                node_ids_json=json.dumps(sorted(set(node_ids))),
                ingestion_id=ingestion_id,
                evidence_ids_json=json.dumps(sorted(set(edge_ids))),
            )
        )

    for edge in edges:
        if edge.edge_type != EdgeType.CONTRADICTS:
            continue
        node_ids = {edge.source_id, edge.target_id}
        if await find_duplicate(db, NotificationType.CONTRADICTION, node_ids, ingestion_id):
            continue
        src = await graph.get_node(edge.source_id)
        tgt = await graph.get_node(edge.target_id)
        names = " ↔ ".join(n.title for n in (src, tgt) if n is not None) or edge.id
        quote = edge.evidence or ""
        quotes = [e.quote for n in (src, tgt) if n is not None for e in n.evidence if e.quote]
        message = quote or " ".join(quotes)[:500]
        queue(
            NotificationType.CONTRADICTION,
            NotificationPriority.HIGH,
            f"Contradiction: {names}",
            message,
            [edge.source_id, edge.target_id],
            [edge.id],
        )

    for node in nodes:
        if node.type not in OWNERLESS_TYPES:
            continue
        node_edges = await graph.get_edges(node.id)
        owned = any(
            e.edge_type in OWNERSHIP_EDGES and (e.target_id == node.id or e.source_id == node.id)
            for e in node_edges
        )
        if owned:
            continue
        if await find_duplicate(db, NotificationType.MISSING_OWNER, {node.id}, ingestion_id):
            continue
        queue(
            NotificationType.MISSING_OWNER,
            NotificationPriority.MEDIUM,
            f"Missing owner: {node.title}",
            f"{node.type.value} has no ownership edge (OWNS / DECIDED_BY / ASSIGNED_TO / DECIDES).",
            [node.id],
            [node.id],
        )

    for row in created:
        db.add(row)
    if created:
        await db.commit()
        for row in created:
            await db.refresh(row)
    return created


async def generate_notifications(
    graph: GraphManager,
    db: AsyncSession,
    nodes: list[KnowledgeNode],
    edges: list[KnowledgeEdge],
    ingestion_id: str = "",
) -> list[NotificationRow]:
    """Create notifications for freshly extracted nodes/edges.

    Deterministic heuristics only (no LLM): contradictions and missing owners.
    Every notification carries severity, affected node IDs, the ingestion ID,
    and the evidence IDs it was raised from. Skips findings already covered
    by an unread notification. Commits.
    """
    created = await evaluate_notifications(graph, nodes, edges, ingestion_id, db)
    return await NotificationRepository(db).save_all(created)


def to_read(row: NotificationRow) -> NotificationRead:
    """ORM row → API model. Single conversion point used by routes and explain."""
    return NotificationRead(
        id=row.id,
        type=row.type,
        priority=row.priority,
        severity=row.severity,
        title=row.title,
        message=row.message,
        related_node_ids=json.loads(row.node_ids_json or "[]"),
        ingestion_id=row.ingestion_id,
        evidence_ids=json.loads(row.evidence_ids_json or "[]"),
        is_read=row.is_read,
        created_at=row.created_at,
        read_at=row.read_at,
    )


async def explain_notification(
    db: AsyncSession, graph: GraphManager, notification_id: str
) -> NotificationExplanation | None:
    """Deterministic answer to 'why was this notification created?'."""
    row = await db.get(NotificationRow, notification_id)
    if row is None:
        return None
    node_titles: list[str] = []
    for nid in json.loads(row.node_ids_json or "[]"):
        node = await graph.get_node(nid)
        node_titles.append(node.title if node else f"<deleted:{nid[:8]}>")
    edge_labels: list[str] = []
    for eid in json.loads(row.evidence_ids_json or "[]"):
        found = None
        for nid in json.loads(row.node_ids_json or "[]"):
            for edge in await graph.get_edges(nid):
                if edge.id == eid:
                    found = edge
                    break
            if found:
                break
        if found:
            src = await graph.get_node(found.source_id)
            tgt = await graph.get_node(found.target_id)
            edge_labels.append(
                f"{src.title if src else '?'} --{found.edge_type.value}--> "
                f"{tgt.title if tgt else '?'}"
            )
        else:
            edge_labels.append(f"<deleted:{eid[:8]}>")
    return NotificationExplanation(
        notification=to_read(row),
        node_titles=node_titles,
        edge_labels=edge_labels,
    )
