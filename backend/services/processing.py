from __future__ import annotations

import datetime as dt
import json
import logging
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from backend import obs
from backend.config import settings
from backend.database import IngestionJobRow
from backend.graph.manager import GraphManager
from backend.models.ingestion import IngestionStatus
from backend.models.knowledge_graph import DryRunResult, GraphDiff
from backend.models.notifications import NotificationType
from backend.services.extraction import CompleteJson, EmbedTexts, run_extraction
from backend.services.notifications import evaluate_notifications, generate_notifications

logger = logging.getLogger(__name__)

MAX_STORED_RUNS = 10


def _record_run(row: IngestionJobRow, entry: dict) -> None:
    try:
        meta = json.loads(row.metadata_json or "{}")
    except ValueError:
        meta = {}
    runs = meta.get("runs", [])
    runs.append(entry)
    meta["runs"] = runs[-MAX_STORED_RUNS:]
    row.metadata_json = json.dumps(meta)


async def process_job_core(
    db: AsyncSession,
    graph: GraphManager,
    row: IngestionJobRow,
    complete_json: CompleteJson,
    embed_texts: EmbedTexts | None = None,
) -> tuple[IngestionJobRow, int, GraphDiff]:
    """Run extraction for a PENDING/FAILED row to completion.

    Shared by the API route (inline) and the Celery task (background).
    Caller owns state guards (404/COMPLETED/PROCESSING/empty-content).
    Returns the refreshed row, the notification count, and the run diff.
    """
    run_id = uuid4().hex[:12]
    started_at = dt.datetime.utcnow()
    row.status = IngestionStatus.PROCESSING.value
    row.attempts += 1
    await db.commit()
    timer = obs.Timer()
    obs.log_event(
        logger,
        "ingestion.process_start",
        job_id=row.id,
        run_id=run_id,
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
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001 — any extraction failure becomes FAILED
        row.status = IngestionStatus.FAILED.value
        row.error = str(exc)[:2000]
        _record_run(
            row,
            {
                "run_id": run_id,
                "attempt": row.attempts,
                "provider": "llm",
                "model": settings.OPENAI_EXTRACTION_MODEL,
                "prompt_version": settings.EXTRACTION_PROMPT_VERSION,
                "started_at": started_at.isoformat(),
                "completed_at": dt.datetime.utcnow().isoformat(),
                "status": row.status,
                "errors": [row.error],
            },
        )
        await db.commit()
        await db.refresh(row)
        return row, 0, GraphDiff()
    row.status = IngestionStatus.COMPLETED.value
    row.nodes_created = extracted.nodes_created
    row.edges_created = extracted.edges_created
    row.error = "\n".join(extracted.errors)[:2000]
    row.completed_at = dt.datetime.utcnow()
    _record_run(
        row,
        {
            "run_id": run_id,
            "attempt": row.attempts,
            "provider": "llm",
            "model": settings.OPENAI_EXTRACTION_MODEL,
            "prompt_version": settings.EXTRACTION_PROMPT_VERSION,
            "started_at": started_at.isoformat(),
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "status": row.status,
            "errors": extracted.errors,
        },
    )
    await db.commit()
    await db.refresh(row)
    notifications = await generate_notifications(
        graph, db, extracted.nodes, extracted.edges, ingestion_id=row.id
    )
    diff = build_diff(
        extracted.nodes_created,
        extracted.nodes_merged,
        extracted.edges_created,
        notifications,
        extracted.edges,
    )
    obs.log_event(
        logger,
        "ingestion.process_finish",
        job_id=row.id,
        run_id=run_id,
        status=row.status,
        nodes=extracted.nodes_created,
        edges=extracted.edges_created,
        notifications=len(notifications),
        latency_ms=round(timer.elapsed_ms(), 1),
    )
    return row, len(notifications), diff


def build_diff(
    extracted_nodes_created: int,
    extracted_nodes_merged: int,
    extracted_edges_created: int,
    notifications: list,
    edges: list,
) -> GraphDiff:
    from backend.models.knowledge_graph import EdgeType

    return GraphDiff(
        nodes_added=extracted_nodes_created,
        nodes_merged=extracted_nodes_merged,
        edges_added=extracted_edges_created,
        notifications_created=len(notifications),
        contradictions=sum(1 for e in edges if e.edge_type == EdgeType.CONTRADICTS),
        missing_owners=sum(
            1 for n in notifications if n.type == NotificationType.MISSING_OWNER.value
        ),
    )


async def dry_run_job(
    db: AsyncSession,
    graph: GraphManager,
    row: IngestionJobRow,
    complete_json: CompleteJson,
    embed_texts: EmbedTexts | None = None,
) -> DryRunResult:
    """Propose mutations without persisting anything.

    Extraction runs against an ephemeral in-memory copy seeded from the real
    graph, so validation and merging behave realistically while the real
    graph, the job row, and notifications stay untouched.
    """
    import tempfile

    from backend.graph.networkx_backend import NetworkXGraphManager

    timer = obs.Timer()
    with tempfile.TemporaryDirectory() as tmp:
        shadow = NetworkXGraphManager(data_path=f"{tmp}/dryrun.json")
        seed = await graph.get_full_graph(limit=2000)
        for node in seed.nodes:
            await shadow.add_node(node)
        for edge in seed.edges:
            try:
                await shadow.add_edge(edge)
            except Exception as exc:  # noqa: BLE001 — seed best-effort (DiGraph collapse)
                logger.debug("dry-run seed skipped edge: %s", exc)
                continue
        extracted = await run_extraction(
            shadow,
            row.content,
            complete_json,
            source_type=row.source_type,
            embed_texts=embed_texts,
            ingestion_id=row.id,
            run_id=f"dryrun-{row.id[:8]}",
        )
    proposed = await evaluate_notifications(graph, extracted.nodes, extracted.edges, row.id, db)
    diff = build_diff(
        extracted.nodes_created,
        extracted.nodes_merged,
        extracted.edges_created,
        proposed,
        extracted.edges,
    )
    obs.log_event(
        logger,
        "ingestion.dry_run",
        job_id=row.id,
        latency_ms=round(timer.elapsed_ms(), 1),
    )
    return DryRunResult(
        job_id=row.id,
        diff=diff,
        proposed_nodes=extracted.nodes,
        proposed_edges=extracted.edges,
        errors=extracted.errors,
    )
