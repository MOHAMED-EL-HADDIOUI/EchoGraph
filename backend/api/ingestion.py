from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import IngestionJobRow, get_db
from backend.deps import (
    get_embedder,
    get_extraction_complete,
    get_graph_manager,
    get_transcriber,
)
from backend.graph.manager import GraphManager
from backend.models.ingestion import IngestionRequest, IngestionStatus, SourceType
from backend.services.extraction import CompleteJson, EmbedTexts, run_extraction
from backend.services.notifications import generate_notifications
from backend.services.transcription import TranscribePath, TranscriptionUnavailable

router = APIRouter(prefix="/ingestion", tags=["ingestion"])

# Matches the OpenAI audio limit; larger files are rejected, not truncated.
MAX_AUDIO_BYTES = 25 * 1024 * 1024


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
    if req.content and len(req.content) > settings.MAX_INGESTION_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Content exceeds {settings.MAX_INGESTION_CHARS} chars",
        )
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


@router.post("/transcribe", response_model=dict)
async def transcribe_upload(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    transcribe: TranscribePath = Depends(get_transcriber),
) -> dict:
    """Transcribe an audio file into a PENDING AUDIO job (process it next)."""
    suffix = Path(file.filename or "audio").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    try:
        if os.path.getsize(path) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio file exceeds 25 MB")
        try:
            transcript = await transcribe(path)
        except TranscriptionUnavailable as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
    finally:
        os.unlink(path)
    row = IngestionJobRow(
        id=str(uuid4()),
        source_type=SourceType.AUDIO.value,
        status=IngestionStatus.PENDING.value,
        title=file.filename or "audio",
        content=transcript,
        metadata_json=json.dumps({"filename": file.filename}),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _row_to_dict(row)


@router.post("/{job_id}/process", response_model=dict)
async def process_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    graph: GraphManager = Depends(get_graph_manager),
    complete_json: CompleteJson = Depends(get_extraction_complete),
    embed_texts: EmbedTexts | None = Depends(get_embedder),
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
        extracted = await run_extraction(
            graph,
            row.content,
            complete_json,
            source_type=row.source_type,
            embed_texts=embed_texts,
        )
    except Exception as exc:  # noqa: BLE001 — any extraction failure becomes FAILED
        row.status = IngestionStatus.FAILED.value
        row.error = str(exc)[:2000]
        await db.commit()
        await db.refresh(row)
        return _row_to_dict(row)
    row.status = IngestionStatus.COMPLETED.value
    row.nodes_created = extracted.nodes_created
    row.edges_created = extracted.edges_created
    row.error = "\n".join(extracted.errors)[:2000]
    row.completed_at = dt.datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    notifications = await generate_notifications(graph, db, extracted.nodes, extracted.edges)
    body = _row_to_dict(row)
    body["notifications_created"] = len(notifications)
    return body


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
