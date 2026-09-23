from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.deps import get_graph_manager
from backend.graph.manager import GraphManager

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(
    db: AsyncSession = Depends(get_db),
    graph: GraphManager = Depends(get_graph_manager),
) -> dict[str, str]:
    """Dependency check for orchestrators: DB reachable + graph readable."""
    try:
        await db.execute(text("SELECT 1"))
        await graph.get_statistics()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"not ready: {exc}") from exc
    return {"status": "ready"}
