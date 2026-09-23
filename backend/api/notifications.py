from __future__ import annotations

import datetime as dt
import json
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import NotificationRow, get_db
from backend.models.notifications import NotificationCreate, NotificationRead

router = APIRouter(prefix="/notifications", tags=["notifications"])


def _row_to_read(row: NotificationRow) -> NotificationRead:
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
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _row_to_read(row)


@router.get("", response_model=list[NotificationRead], summary="List notifications")
async def list_notifications(
    unread_only: bool = False,
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[NotificationRead]:
    stmt = (
        select(NotificationRow)
        .order_by(NotificationRow.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if unread_only:
        stmt = stmt.where(NotificationRow.is_read.is_(False))
    result = await db.execute(stmt)
    return [_row_to_read(r) for r in result.scalars().all()]


@router.patch(
    "/{notification_id}/read",
    response_model=NotificationRead,
    summary="Mark a notification as read",
)
async def mark_read(notification_id: str, db: AsyncSession = Depends(get_db)) -> NotificationRead:
    row = await db.get(NotificationRow, notification_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Notification not found")
    row.is_read = True
    row.read_at = dt.datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return _row_to_read(row)
