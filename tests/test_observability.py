from __future__ import annotations

import logging

SECRET = "ZEBRA-UNIQUE-TRANSCRIPT-42"


async def test_request_id_echoed_and_propagated(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert len(r.headers["x-request-id"]) == 12

    r = await client.get("/health", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"


async def test_pipeline_logs_contain_no_transcript(client, caplog):
    from backend.app import app
    from backend.deps import get_extraction_complete

    async def empty_complete(system: str, user: str) -> dict:
        return {"nodes": [], "edges": []}

    app.dependency_overrides[get_extraction_complete] = lambda: empty_complete
    try:
        with caplog.at_level(logging.INFO):
            r = await client.post("/ingestion", json={"source_type": "MEETING", "content": SECRET})
            await client.post(f"/ingestion/{r.json()['job_id']}/process")
            await client.post("/graph/query", json={"query": "anything"})
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)
    assert SECRET not in caplog.text
    assert "ingestion.process_start" in caplog.text
    assert "ingestion.process_finish" in caplog.text
    assert "extraction.finish" in caplog.text
    assert "query.finish" in caplog.text
    assert "request_id" in caplog.text
