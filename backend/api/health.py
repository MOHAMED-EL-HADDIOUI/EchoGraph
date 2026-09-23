from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import get_db
from backend.deps import get_graph_manager
from backend.graph.manager import GraphManager

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe (always public)")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def _check_database(db: AsyncSession) -> tuple[str, str]:
    try:
        await db.execute(text("SELECT 1"))
        return "ok", ""
    except Exception as exc:
        return "down", str(exc)[:200]


async def _check_graph(graph: GraphManager) -> tuple[str, str]:
    try:
        healthy = await graph.health_check()
        return ("ok", "") if healthy else ("down", "backend reported unhealthy")
    except Exception as exc:
        return "down", str(exc)[:200]


async def _check_redis() -> tuple[str, str]:
    if not settings.USE_BACKGROUND_JOBS:
        return "skipped", "background jobs disabled"
    try:
        import redis.asyncio as aioredis
    except ImportError:
        return "down", "redis package not installed"
    try:
        client = aioredis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        await client.ping()
        await client.aclose()
        return "ok", ""
    except Exception as exc:
        return "down", str(exc)[:200]


@router.get("/ready", summary="Readiness probe with dependency breakdown")
async def ready(
    db: AsyncSession = Depends(get_db),
    graph: GraphManager = Depends(get_graph_manager),
) -> dict:
    """Dependency check for orchestrators. 503 with per-dependency detail."""
    database, db_detail = await _check_database(db)
    graph_status, graph_detail = await _check_graph(graph)
    redis_status, redis_detail = await _check_redis()
    dependencies = {
        "database": {"status": database, "detail": db_detail},
        "graph": {"status": graph_status, "detail": graph_detail},
        "redis": {"status": redis_status, "detail": redis_detail},
    }
    required = [database, graph_status] + ([redis_status] if settings.USE_BACKGROUND_JOBS else [])
    if all(s == "ok" for s in required):
        return {"status": "ready", "dependencies": dependencies}
    raise HTTPException(
        status_code=503, detail={"status": "not_ready", "dependencies": dependencies}
    )
