from __future__ import annotations

import sys

from backend.app import app
from backend.deps import get_extraction_complete
from backend.services.processing import process_job_core


async def empty_complete(system: str, user: str) -> dict:
    return {"nodes": [], "edges": []}


async def test_process_core_success_and_failure(client, test_db, tmp_graph):
    from backend.database import IngestionJobRow

    r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "x"})
    job_id = r.json()["job_id"]

    async with test_db() as session:
        row = await session.get(IngestionJobRow, job_id)
        row, count, diff = await process_job_core(session, tmp_graph, row, empty_complete)
        assert row.status == "COMPLETED"
        assert row.attempts == 1
        assert count == 0
        assert diff.nodes_added == 0

    async def boom_complete(system: str, user: str) -> dict:
        raise RuntimeError("nope")

    async with test_db() as session:
        row = await session.get(IngestionJobRow, job_id)
        row.status = "PENDING"
        await session.commit()
        row, _, _ = await process_job_core(session, tmp_graph, row, boom_complete)
        assert row.status == "FAILED"
        assert "nope" in row.error


async def test_background_enqueue_returns_202(client, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("backend.api.ingestion.settings.USE_BACKGROUND_JOBS", True)
    monkeypatch.setattr("backend.worker.enqueue_process", lambda jid: calls.append(jid))

    r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "x"})
    job_id = r.json()["job_id"]
    r = await client.post(f"/ingestion/{job_id}/process")
    assert r.status_code == 202
    assert calls == [job_id]
    body = (await client.get(f"/ingestion/{job_id}")).json()
    assert body["status"] == "PROCESSING"
    assert body["attempts"] == 1


async def test_worker_run_job_success_and_missing_key(client, test_db, tmp_graph, monkeypatch):
    from backend.worker import _run_job

    # Hermetic: never depend on an ambient OPENAI_API_KEY.
    monkeypatch.setattr("backend.config.settings.OPENAI_API_KEY", "test-key")
    # Point the worker's own session/graph factories at the test doubles.
    monkeypatch.setattr("backend.database.async_session", test_db)
    monkeypatch.setattr(
        "backend.graph.manager.create_graph_manager", lambda backend="networkx": tmp_graph
    )

    async def good_complete(system: str, user: str) -> dict:
        return {
            "nodes": [{"type": "DECISION", "title": "Ship v1", "content": "", "confidence": 1.0}],
            "edges": [],
        }

    monkeypatch.setattr(
        "backend.services.extraction.make_openai_complete",
        lambda model, key: good_complete,
    )
    r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "x"})
    result = await _run_job(r.json()["job_id"])
    assert result["ok"] is True
    assert result["status"] == "COMPLETED"
    assert result["nodes_created"] == 1

    monkeypatch.setattr("backend.deps.settings.OPENAI_API_KEY", "")
    monkeypatch.setattr("backend.worker.settings.OPENAI_API_KEY", "")
    r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "x"})
    result = await _run_job(r.json()["job_id"])
    assert result == {"ok": False, "status": "FAILED"}
    body = (await client.get(f"/ingestion/{r.json()['job_id']}")).json()
    assert "OPENAI_API_KEY" in body["error"]


async def test_background_without_celery_is_501(client, monkeypatch):
    monkeypatch.setattr("backend.api.ingestion.settings.USE_BACKGROUND_JOBS", True)
    monkeypatch.setitem(sys.modules, "backend.worker", None)
    # Extraction dep must not fire before the enqueue path is reached.
    app.dependency_overrides[get_extraction_complete] = lambda: empty_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "x"})
        r = await client.post(f"/ingestion/{r.json()['job_id']}/process")
        assert r.status_code == 501
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)
