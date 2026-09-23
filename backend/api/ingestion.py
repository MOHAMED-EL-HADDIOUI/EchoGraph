from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend import obs
from backend.config import settings
from backend.database import IngestionJobRow, get_db
from backend.deps import (
    get_embedder,
    get_extraction_complete,
    get_graph_manager,
    get_transcriber,
)
from backend.graph.manager import GraphManager
from backend.models.ingestion import (
    IngestionJobRead,
    IngestionRequest,
    IngestionStatus,
    SourceType,
)
from backend.services.extraction import CompleteJson, EmbedTexts, run_extraction
from backend.services.notifications import generate_notifications
from backend.services.transcription import TranscribePath, TranscriptionUnavailable

router = APIRouter(prefix="/ingestion", tags=["ingestion"])

logger = logging.getLogger(__name__)

# Matches the OpenAI audio limit; larger files are rejected, not truncated.
MAX_AUDIO_BYTES = 25 * 1024 * 1024


def _row_to_read(row: IngestionJobRow) -> IngestionJobRead:
    return IngestionJobRead(
        job_id=row.id,
        source_type=row.source_type,
        status=row.status,
        nodes_created=row.nodes_created,
        edges_created=row.edges_created,
        title=row.title,
        error=row.error,
        attempts=row.attempts,
        metadata=json.loads(row.metadata_json or "{}"),
        created_at=row.created_at,
        completed_at=row.completed_at,
    )


@router.post(
    "",
    response_model=IngestionJobRead,
    summary="Submit content for extraction",
    description="Stores the content as a PENDING job. Run extraction next via "
    "POST /ingestion/{job_id}/process. Rejects payloads over MAX_INGESTION_CHARS (413).",
)
async def submit_ingestion(
    req: IngestionRequest, db: AsyncSession = Depends(get_db)
) -> IngestionJobRead:
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
    return _row_to_read(row)


@router.get("/{job_id}", response_model=IngestionJobRead, summary="Get one ingestion job")
async def read_job(job_id: str, db: AsyncSession = Depends(get_db)) -> IngestionJobRead:
    row = await db.get(IngestionJobRow, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _row_to_read(row)


@router.get("", response_model=list[IngestionJobRead], summary="List ingestion jobs")
async def list_jobs(
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[IngestionJobRead]:
    result = await db.execute(
        select(IngestionJobRow)
        .order_by(IngestionJobRow.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return [_row_to_read(r) for r in result.scalars().all()]


@router.post(
    "/transcribe",
    response_model=IngestionJobRead,
    summary="Transcribe audio into a PENDING job",
    description="Upload audio (25 MB cap). Returns a PENDING AUDIO job holding the "
    "transcript; process it next via POST /ingestion/{job_id}/process.",
)
async def transcribe_upload(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    transcribe: TranscribePath = Depends(get_transcriber),
) -> IngestionJobRead:
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
    return _row_to_read(row)


@router.post(
    "/{job_id}/process",
    response_model=IngestionJobRead,
    summary="Extract a knowledge graph from a job",
    description="Runs LLM extraction synchronously. Idempotent: COMPLETED jobs return "
    "their stored outcome; PENDING/FAILED jobs run; PROCESSING jobs get 409.",
)
async def process_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    graph: GraphManager = Depends(get_graph_manager),
    complete_json: CompleteJson = Depends(get_extraction_complete),
    embed_texts: EmbedTexts | None = Depends(get_embedder),
) -> IngestionJobRead:
    """Run LLM extraction for a job synchronously (no worker queue yet).

    Idempotent: re-processing a COMPLETED job returns its stored outcome
    without touching the graph; only PENDING/FAILED jobs run extraction.
    """
    row = await db.get(IngestionJobRow, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if row.status == IngestionStatus.COMPLETED.value:
        return _row_to_read(row)
    if row.status == IngestionStatus.PROCESSING.value:
        raise HTTPException(status_code=409, detail="Job is already processing")
    if row.status != IngestionStatus.PENDING.value and row.status != IngestionStatus.FAILED.value:
        raise HTTPException(status_code=409, detail=f"Job is {row.status}, not processable")
    if not row.content:
        raise HTTPException(status_code=422, detail="Job has no content to process")
    row.status = IngestionStatus.PROCESSING.value
    row.attempts += 1
    await db.commit()
    timer = obs.Timer()
    obs.log_event(
        logger,
        "ingestion.process_start",
        job_id=row.id,
        attempt=row.attempts,
        content_chars=len(row.content or ""),
        content_sha=obs.fingerprint(row.content or ""),
    )
    try:
        extracted = await run_extraction(
            graph,
            row.content,
            complete_json,
            source_type=row.source_type,
            embed_texts=embed_texts,
            ingestion_id=row.id,
        )
    except Exception as exc:  # noqa: BLE001 — any extraction failure becomes FAILED
        row.status = IngestionStatus.FAILED.value
        row.error = str(exc)[:2000]
        await db.commit()
        await db.refresh(row)
        return _row_to_read(row)
    row.status = IngestionStatus.COMPLETED.value
    row.nodes_created = extracted.nodes_created
    row.edges_created = extracted.edges_created
    row.error = "\n".join(extracted.errors)[:2000]
    row.completed_at = dt.datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    notifications = await generate_notifications(
        graph, db, extracted.nodes, extracted.edges, ingestion_id=row.id
    )
    obs.log_event(
        logger,
        "ingestion.process_finish",
        job_id=row.id,
        status=row.status,
        nodes=extracted.nodes_created,
        edges=extracted.edges_created,
        notifications=len(notifications),
        latency_ms=round(timer.elapsed_ms(), 1),
    )
    body = _row_to_read(row)
    body.notifications_created = len(notifications)
    return body


@router.post(
    "/{job_id}/complete",
    response_model=IngestionJobRead,
    summary="Manually mark a job complete (worker hook)",
)
async def mark_job_complete(
    job_id: str,
    nodes_created: int = 0,
    edges_created: int = 0,
    db: AsyncSession = Depends(get_db),
) -> IngestionJobRead:
    row = await db.get(IngestionJobRow, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    row.status = IngestionStatus.COMPLETED.value
    row.nodes_created = nodes_created
    row.edges_created = edges_created
    row.completed_at = dt.datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return _row_to_read(row)
