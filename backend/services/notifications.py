from __future__ import annotations

import json
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import NotificationRow
from backend.graph.manager import GraphManager
from backend.models.knowledge_graph import EdgeType, KnowledgeEdge, KnowledgeNode, NodeType
from backend.models.notifications import NotificationPriority, NotificationType

# Node types that should have an owner; edge types that confer ownership.
OWNERLESS_TYPES = {NodeType.ACTION_ITEM, NodeType.DECISION}
OWNERSHIP_EDGES = {EdgeType.OWNS, EdgeType.DECIDED_BY}


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
    result = await db.execute(
        select(NotificationRow).where(NotificationRow.type == ntype.value).limit(200)
    )
    for row in result.scalars().all():
        if row.is_read:
            continue
        if set(json.loads(row.node_ids_json or "[]")) != node_ids:
            continue
        if (row.ingestion_id or "") != (ingestion_id or ""):
            continue
        return row
    return None


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
            f"{node.type.value} has no ownership edge (OWNS / DECIDED_BY).",
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
