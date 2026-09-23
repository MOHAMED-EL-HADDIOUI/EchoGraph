from __future__ import annotations

import datetime as dt
import json
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import NotificationRow, get_db
from backend.deps import get_graph_manager
from backend.graph.manager import GraphManager
from backend.models.notifications import (
    NotificationCreate,
    NotificationExplanation,
    NotificationRead,
)
from backend.repositories import NotificationRepository
from backend.services.notifications import explain_notification, to_read

router = APIRouter(prefix="/notifications", tags=["notifications"])


def _row_to_read(row: NotificationRow) -> NotificationRead:
    return to_read(row)


@router.post(
    "",
    response_model=NotificationRead,
    summary="Create a notification manually",
    description="Auto-generated notifications normally come from extraction; "
    "this endpoint covers manual/system callers.",
)
async def create_notification(
    payload: NotificationCreate, db: AsyncSession = Depends(get_db)
) -> NotificationRead:
    row = NotificationRow(
        id=str(uuid4()),
        type=payload.type.value,
        priority=payload.priority.value,
        severity=payload.priority.value,
        title=payload.title,
        message=payload.message,
        node_ids_json=json.dumps(payload.related_node_ids),
    )
    return _row_to_read(await NotificationRepository(db).save(row))


@router.get("", response_model=list[NotificationRead], summary="List notifications")
async def list_notifications(
    unread_only: bool = False,
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[NotificationRead]:
    rows = await NotificationRepository(db).list(
        unread_only=unread_only, limit=limit, offset=offset
    )
    return [_row_to_read(r) for r in rows]


@router.patch(
    "/{notification_id}/read",
    response_model=NotificationRead,
    summary="Mark a notification as read",
)
async def mark_read(notification_id: str, db: AsyncSession = Depends(get_db)) -> NotificationRead:
    repo = NotificationRepository(db)
    row = await repo.get(notification_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Notification not found")
    row.is_read = True
    row.read_at = dt.datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return _row_to_read(row)


@router.post(
    "/{notification_id}/read",
    response_model=NotificationRead,
    summary="Mark a notification as read (POST alias)",
)
async def mark_read_post(
    notification_id: str, db: AsyncSession = Depends(get_db)
) -> NotificationRead:
    return await mark_read(notification_id, db)


@router.get(
    "/{notification_id}/explain",
    response_model=NotificationExplanation,
    summary="Why was this notification created?",
    description="Deterministic provenance: related node titles and evidence edge "
    "labels resolved from current graph state. Deleted entities are marked.",
)
async def explain(
    notification_id: str,
    db: AsyncSession = Depends(get_db),
    graph: GraphManager = Depends(get_graph_manager),
) -> NotificationExplanation:
    explanation = await explain_notification(db, graph, notification_id)
    if explanation is None:
        raise HTTPException(status_code=404, detail="Notification not found")
    return explanation
