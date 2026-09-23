from __future__ import annotations

import datetime as dt
import json
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import IngestionJobRow, get_db
from backend.deps import get_extraction_complete, get_graph_manager
from backend.graph.manager import GraphManager
from backend.models.ingestion import IngestionRequest, IngestionStatus
from backend.services.extraction import CompleteJson, run_extraction

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


def _row_to_dict(row: IngestionJobRow) -> dict:
    return {
        "job_id": row.id,
        "source_type": row.source_type,
        "status": row.status,
        "nodes_created": row.nodes_created,
        "edges_created": row.edges_created,
        "title": row.title,
        "error": row.error,
        "metadata": json.loads(row.metadata_json or "{}"),
        "created_at": row.created_at,
        "completed_at": row.completed_at,
    }


@router.post("", response_model=dict)
async def submit_ingestion(req: IngestionRequest, db: AsyncSession = Depends(get_db)) -> dict:
    row = IngestionJobRow(
        id=str(uuid4()),
        source_type=req.source_type.value,
        status=IngestionStatus.PENDING.value,
        title=req.title or "",
        content=req.content or "",
        metadata_json=json.dumps(req.metadata),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _row_to_dict(row)


@router.get("/{job_id}", response_model=dict)
async def read_job(job_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    row = await db.get(IngestionJobRow, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _row_to_dict(row)


@router.get("", response_model=list[dict])
async def list_jobs(db: AsyncSession = Depends(get_db)) -> list[dict]:
    result = await db.execute(
        select(IngestionJobRow).order_by(IngestionJobRow.created_at.desc()).limit(100)
    )
    return [_row_to_dict(r) for r in result.scalars().all()]


@router.post("/{job_id}/process", response_model=dict)
async def process_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    graph: GraphManager = Depends(get_graph_manager),
    complete_json: CompleteJson = Depends(get_extraction_complete),
) -> dict:
    """Run LLM extraction for a job synchronously (no worker queue yet)."""
    row = await db.get(IngestionJobRow, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if row.status not in (IngestionStatus.PENDING.value, IngestionStatus.FAILED.value):
        raise HTTPException(status_code=409, detail=f"Job is {row.status}, not processable")
    if not row.content:
        raise HTTPException(status_code=422, detail="Job has no content to process")
    row.status = IngestionStatus.PROCESSING.value
    await db.commit()
    try:
        nodes, edges, errors = await run_extraction(
            graph, row.content, complete_json, source_type=row.source_type
        )
    except Exception as exc:  # noqa: BLE001 — any extraction failure becomes FAILED
        row.status = IngestionStatus.FAILED.value
        row.error = str(exc)[:2000]
        await db.commit()
        await db.refresh(row)
        return _row_to_dict(row)
    row.status = IngestionStatus.COMPLETED.value
    row.nodes_created = nodes
    row.edges_created = edges
    row.error = "\n".join(errors)[:2000]
    row.completed_at = dt.datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return _row_to_dict(row)


@router.post("/{job_id}/complete", response_model=dict)
async def mark_job_complete(
    job_id: str,
    nodes_created: int = 0,
    edges_created: int = 0,
    db: AsyncSession = Depends(get_db),
) -> dict:
    row = await db.get(IngestionJobRow, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    row.status = IngestionStatus.COMPLETED.value
    row.nodes_created = nodes_created
    row.edges_created = edges_created
    row.completed_at = dt.datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return _row_to_dict(row)
