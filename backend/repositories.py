from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import IngestionJobRow, NotificationRow

# Repository abstractions: services and routes persist through these, never
# with inline SQL outside this module.


class IngestionJobRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get(self, job_id: str) -> IngestionJobRow | None:
        return await self.db.get(IngestionJobRow, job_id)

    async def list(self, limit: int = 100, offset: int = 0) -> list[IngestionJobRow]:
        result = await self.db.execute(
            select(IngestionJobRow)
            .order_by(IngestionJobRow.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def save(self, row: IngestionJobRow) -> IngestionJobRow:
        self.db.add(row)
        await self.db.commit()
        await self.db.refresh(row)
        return row

    async def commit(self) -> None:
        await self.db.commit()


class NotificationRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get(self, notification_id: str) -> NotificationRow | None:
        return await self.db.get(NotificationRow, notification_id)

    async def list(
        self, unread_only: bool = False, limit: int = 100, offset: int = 0
    ) -> list[NotificationRow]:
        stmt = (
            select(NotificationRow)
            .order_by(NotificationRow.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if unread_only:
            stmt = stmt.where(NotificationRow.is_read.is_(False))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def by_type(self, ntype: str, limit: int = 200) -> list[NotificationRow]:
        result = await self.db.execute(
            select(NotificationRow).where(NotificationRow.type == ntype).limit(limit)
        )
        return list(result.scalars().all())

    async def find_duplicate(
        self, ntype: str, node_ids: set[str], ingestion_id: str = ""
    ) -> NotificationRow | None:
        for row in await self.by_type(ntype):
            if row.is_read:
                continue
            if set(json.loads(row.node_ids_json or "[]")) != node_ids:
                continue
            if (row.ingestion_id or "") != (ingestion_id or ""):
                continue
            return row
        return None

    async def save_all(self, rows: list[NotificationRow]) -> list[NotificationRow]:
        for row in rows:
            self.db.add(row)
        if rows:
            await self.db.commit()
            for row in rows:
                await self.db.refresh(row)
        return rows

    async def save(self, row: NotificationRow) -> NotificationRow:
        return (await self.save_all([row]))[0]
