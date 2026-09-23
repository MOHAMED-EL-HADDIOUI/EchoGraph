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


async def _already_reported(db: AsyncSession, ntype: NotificationType, node_ids: set[str]) -> bool:
    result = await db.execute(
        select(NotificationRow).where(NotificationRow.type == ntype.value).limit(200)
    )
    for row in result.scalars().all():
        if set(json.loads(row.node_ids_json or "[]")) == node_ids and not row.is_read:
            return True
    return False


async def generate_notifications(
    graph: GraphManager,
    db: AsyncSession,
    nodes: list[KnowledgeNode],
    edges: list[KnowledgeEdge],
) -> list[NotificationRow]:
    """Create notifications for freshly extracted nodes/edges.

    Deterministic heuristics only (no LLM): contradictions and missing owners.
    Skips anything already reported by an unread notification. Commits.
    """
    created: list[NotificationRow] = []

    def queue(
        ntype: NotificationType,
        priority: NotificationPriority,
        title: str,
        message: str,
        node_ids: list[str],
    ) -> None:
        created.append(
            NotificationRow(
                id=str(uuid4()),
                type=ntype.value,
                priority=priority.value,
                title=title,
                message=message,
                node_ids_json=json.dumps(sorted(set(node_ids))),
            )
        )

    for edge in edges:
        if edge.edge_type != EdgeType.CONTRADICTS:
            continue
        node_ids = {edge.source_id, edge.target_id}
        if await _already_reported(db, NotificationType.CONTRADICTION, node_ids):
            continue
        src = await graph.get_node(edge.source_id)
        tgt = await graph.get_node(edge.target_id)
        names = " ↔ ".join(n.title for n in (src, tgt) if n is not None) or edge.id
        queue(
            NotificationType.CONTRADICTION,
            NotificationPriority.HIGH,
            f"Contradiction: {names}",
            edge.evidence or "",
            [edge.source_id, edge.target_id],
        )

    for node in nodes:
        if node.type not in OWNERLESS_TYPES:
            continue
        node_edges = await graph.get_edges(node.id)
        owned = any(
            e.edge_type in OWNERSHIP_EDGES and (e.target_id == node.id or e.source_id == node.id)
            for e in node_edges
        )
        if owned or await _already_reported(db, NotificationType.MISSING_OWNER, {node.id}):
            continue
        queue(
            NotificationType.MISSING_OWNER,
            NotificationPriority.MEDIUM,
            f"Missing owner: {node.title}",
            f"{node.type.value} has no ownership edge (OWNS / DECIDED_BY).",
            [node.id],
        )

    for row in created:
        db.add(row)
    if created:
        await db.commit()
        for row in created:
            await db.refresh(row)
    return created
