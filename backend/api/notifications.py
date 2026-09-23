from __future__ import annotations

import json
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import NotificationRow, get_db
from backend.models.notifications import NotificationCreate

router = APIRouter(prefix="/notifications", tags=["notifications"])


def _row_to_dict(row: NotificationRow) -> dict:
    return {
        "id": row.id,
        "type": row.type,
        "priority": row.priority,
        "title": row.title,
        "message": row.message,
        "related_node_ids": json.loads(row.node_ids_json or "[]"),
        "is_read": row.is_read,
        "created_at": row.created_at,
    }


@router.post("", response_model=dict)
async def create_notification(
    payload: NotificationCreate, db: AsyncSession = Depends(get_db)
) -> dict:
    row = NotificationRow(
        id=str(uuid4()),
        type=payload.type.value,
        priority=payload.priority.value,
        title=payload.title,
        message=payload.message,
        node_ids_json=json.dumps(payload.related_node_ids),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _row_to_dict(row)


@router.get("", response_model=list[dict])
async def list_notifications(
    unread_only: bool = False, db: AsyncSession = Depends(get_db)
) -> list[dict]:
    stmt = select(NotificationRow).order_by(NotificationRow.created_at.desc()).limit(100)
    if unread_only:
        stmt = stmt.where(NotificationRow.is_read.is_(False))
    result = await db.execute(stmt)
    return [_row_to_dict(r) for r in result.scalars().all()]


@router.patch("/{notification_id}/read", response_model=dict)
async def mark_read(notification_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    row = await db.get(NotificationRow, notification_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Notification not found")
    row.is_read = True
    await db.commit()
    await db.refresh(row)
    return _row_to_dict(row)
