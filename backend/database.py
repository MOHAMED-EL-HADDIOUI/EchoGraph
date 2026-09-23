from __future__ import annotations

import datetime as dt
import json
from collections.abc import AsyncGenerator

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from backend.config import settings


class Base(DeclarativeBase):
    pass


# ── ORM models ──────────────────────────────────────────────────


class IngestionJobRow(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_type: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    nodes_created: Mapped[int] = mapped_column(Integer, default=0)
    edges_created: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(500), default="")
    content: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class NotificationRow(Base):
    __tablename__ = "notifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    type: Mapped[str] = mapped_column(String(30))
    priority: Mapped[str] = mapped_column(String(10), default="MEDIUM")
    title: Mapped[str] = mapped_column(String(500))
    message: Mapped[str] = mapped_column(Text, default="")
    node_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    @property
    def related_node_ids(self) -> list[str]:
        return json.loads(self.node_ids_json)


# ── Engine / session ────────────────────────────────────────────

engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async DB session."""
    async with async_session() as session:
        yield session
