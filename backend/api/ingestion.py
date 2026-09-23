from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Response, UploadFile
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
)
from backend.models.knowledge_graph import DryRunResult
from backend.repositories import IngestionJobRepository
from backend.services.extraction import CompleteJson, EmbedTexts
from backend.services.processing import dry_run_job, process_job_core
from backend.services.sources import AudioAdapter, ManualAdapter
from backend.services.transcription import TranscribePath, TranscriptionUnavailable

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


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
        content_sha=row.content_sha,
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
    if req.content and len(req.content) > settings.MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Content exceeds {settings.MAX_TEXT_CHARS} chars",
        )
    doc = await ManualAdapter().normalize(
        {
            "source_type": req.source_type,
            "title": req.title or "",
            "content": req.content or "",
            "metadata": req.metadata,
        }
    )
    row = IngestionJobRow(
        id=str(uuid4()),
        source_type=doc.source_type.value,
        status=IngestionStatus.PENDING.value,
        title=doc.title,
        content=doc.text,
        content_sha=obs.fingerprint(doc.text),
        metadata_json=json.dumps(doc.metadata),
    )
    return _row_to_read(await IngestionJobRepository(db).save(row))


@router.get("/{job_id}", response_model=IngestionJobRead, summary="Get one ingestion job")
async def read_job(job_id: str, db: AsyncSession = Depends(get_db)) -> IngestionJobRead:
    row = await IngestionJobRepository(db).get(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _row_to_read(row)


@router.get("", response_model=list[IngestionJobRead], summary="List ingestion jobs")
async def list_jobs(
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[IngestionJobRead]:
    rows = await IngestionJobRepository(db).list(limit=limit, offset=offset)
    return [_row_to_read(r) for r in rows]


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
        if os.path.getsize(path) > settings.MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio file exceeds 25 MB")
        try:
            transcript = await transcribe(path)
        except TranscriptionUnavailable as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
    finally:
        os.unlink(path)
    doc = await AudioAdapter().normalize(
        {
            "filename": file.filename or "audio",
            "transcript": transcript.text,
            "provider": transcript.provider,
            "model": transcript.model,
            "duration_s": transcript.duration_s,
            "language": transcript.language,
            "transcript_sha": transcript.transcript_sha or obs.fingerprint(transcript.text),
        }
    )
    row = IngestionJobRow(
        id=str(uuid4()),
        source_type=doc.source_type.value,
        status=IngestionStatus.PENDING.value,
        title=doc.title,
        content=doc.text,
        content_sha=obs.fingerprint(doc.text),
        metadata_json=json.dumps(doc.metadata),
    )
    return _row_to_read(await IngestionJobRepository(db).save(row))


@router.post(
    "/{job_id}/process",
    response_model=IngestionJobRead,
    summary="Extract a knowledge graph from a job",
    description="Inline extraction by default. With USE_BACKGROUND_JOBS, enqueues a "
    "Celery task and returns 202. Idempotent: COMPLETED jobs return their stored "
    "outcome; PENDING/FAILED jobs run; PROCESSING jobs get 409.",
)
async def process_job(
    job_id: str,
    response: Response,
    db: AsyncSession = Depends(get_db),
    graph: GraphManager = Depends(get_graph_manager),
    complete_json: CompleteJson | None = Depends(get_extraction_complete),
    embed_texts: EmbedTexts | None = Depends(get_embedder),
) -> IngestionJobRead:
    """Run LLM extraction for a job synchronously (no worker queue yet).

    Idempotent: re-processing a COMPLETED job returns its stored outcome
    without touching the graph; only PENDING/FAILED jobs run extraction.
    """
    row = await IngestionJobRepository(db).get(job_id)
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
    if complete_json is None and not settings.USE_BACKGROUND_JOBS:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    if settings.USE_BACKGROUND_JOBS:
        try:
            from backend.worker import enqueue_process
        except ImportError as exc:
            raise HTTPException(status_code=501, detail="Background workers not installed") from exc
        enqueue_process(row.id)
        row.status = IngestionStatus.PROCESSING.value
        row.attempts += 1
        await db.commit()
        await db.refresh(row)
        response.status_code = 202
        return _row_to_read(row)
    row, notification_count, diff = await process_job_core(
        db, graph, row, complete_json, embed_texts
    )
    body = _row_to_read(row)
    body.notifications_created = notification_count
    body.diff = diff
    return body


@router.post(
    "/{job_id}/dry-run",
    response_model=DryRunResult,
    summary="Preview extraction without mutating anything",
    description="Runs extraction against an ephemeral copy of the graph and reports "
    "proposed nodes/edges plus a diff. The job, graph, and notifications are untouched.",
)
async def dry_run_process(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    graph: GraphManager = Depends(get_graph_manager),
    complete_json: CompleteJson | None = Depends(get_extraction_complete),
    embed_texts: EmbedTexts | None = Depends(get_embedder),
) -> DryRunResult:
    row = await IngestionJobRepository(db).get(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not row.content:
        raise HTTPException(status_code=422, detail="Job has no content to process")
    if complete_json is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    return await dry_run_job(db, graph, row, complete_json, embed_texts)


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
    row = await IngestionJobRepository(db).get(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    row.status = IngestionStatus.COMPLETED.value
    row.nodes_created = nodes_created
    row.edges_created = edges_created
    row.completed_at = dt.datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return _row_to_read(row)
