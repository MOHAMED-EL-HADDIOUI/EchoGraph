from __future__ import annotations

import asyncio
import logging
from typing import Any

from backend import obs
from backend.config import settings

logger = logging.getLogger(__name__)

TASK_NAME = "echograph.process_ingestion_job"

_celery_app: Any | None = None


def __getattr__(name: str) -> Any:
    # Lets `celery -A backend.worker:celery_app` resolve without importing
    # celery at module load (broker deps stay optional for dev installs).
    if name == "celery_app":
        return get_celery_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_celery_app() -> Any:
    """Build (once) the Celery app. Imported lazily: broker deps are optional.

    Raises ImportError when celery/redis aren't installed.
    """
    global _celery_app
    if _celery_app is None:
        from celery import Celery

        app = Celery("echograph", broker=settings.REDIS_URL, backend=settings.REDIS_URL)
        app.conf.update(
            task_acks_late=True,
            worker_prefetch_multiplier=1,
            task_time_limit=600,
            task_default_queue="echograph",
        )
        app.task(name=TASK_NAME, bind=False)(process_ingestion_job)
        _celery_app = app
    return _celery_app


def enqueue_process(job_id: str) -> Any:
    """Enqueue a job for background processing. Raises ImportError without celery."""
    return get_celery_app().send_task(TASK_NAME, args=[job_id])


def process_ingestion_job(job_id: str) -> dict:
    """Celery task body (sync): run one job to completion with real services."""
    return asyncio.run(_run_job(job_id))


async def _run_job(job_id: str) -> dict:
    from backend.database import IngestionJobRow, async_session
    from backend.graph.manager import create_graph_manager
    from backend.models.ingestion import IngestionStatus
    from backend.services.extraction import make_openai_complete, make_openai_embeddings
    from backend.services.processing import process_job_core

    timer = obs.Timer()
    graph = create_graph_manager(settings.GRAPH_BACKEND)
    try:
        async with async_session() as session:
            row = await session.get(IngestionJobRow, job_id)
            if row is None:
                return {"ok": False, "error": "job not found"}
            if row.status == IngestionStatus.COMPLETED.value:
                return {"ok": True, "skipped": True, "status": row.status}
            if not row.content:
                row.status = IngestionStatus.FAILED.value
                row.error = "Job has no content to process"
                await session.commit()
                return {"ok": False, "status": row.status}
            if not settings.OPENAI_API_KEY:
                row.status = IngestionStatus.FAILED.value
                row.error = "OPENAI_API_KEY not configured"
                await session.commit()
                return {"ok": False, "status": row.status}
            complete = make_openai_complete(settings.OPENAI_MODEL, settings.OPENAI_API_KEY)
            embedder = None
            if settings.ENABLE_EMBEDDING_DEDUP:
                embedder = make_openai_embeddings(settings.EMBEDDING_MODEL, settings.OPENAI_API_KEY)
            row, notification_count = await process_job_core(
                session, graph, row, complete, embedder
            )
            obs.log_event(
                logger,
                "worker.job_done",
                job_id=job_id,
                status=row.status,
                notifications=notification_count,
                latency_ms=round(timer.elapsed_ms(), 1),
            )
            return {
                "ok": row.status == IngestionStatus.COMPLETED.value,
                "status": row.status,
                "nodes_created": row.nodes_created,
                "edges_created": row.edges_created,
                "notifications_created": notification_count,
            }
    finally:
        close = getattr(graph, "close", None)
        if callable(close):
            await close()
