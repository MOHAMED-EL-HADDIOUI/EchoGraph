from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from backend import obs
from backend.database import IngestionJobRow
from backend.graph.manager import GraphManager
from backend.models.ingestion import IngestionStatus
from backend.services.extraction import CompleteJson, EmbedTexts, run_extraction
from backend.services.notifications import generate_notifications

logger = logging.getLogger(__name__)


async def process_job_core(
    db: AsyncSession,
    graph: GraphManager,
    row: IngestionJobRow,
    complete_json: CompleteJson,
    embed_texts: EmbedTexts | None = None,
) -> tuple[IngestionJobRow, int]:
    """Run extraction for a PENDING/FAILED row to completion.

    Shared by the API route (inline) and the Celery task (background).
    Caller owns state guards (404/COMPLETED/PROCESSING/empty-content).
    Returns the refreshed row plus the notification count.
    """
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
        return row, 0
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
    return row, len(notifications)
