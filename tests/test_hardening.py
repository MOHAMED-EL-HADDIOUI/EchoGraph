from __future__ import annotations

import asyncio

from backend.services.extraction import run_extraction


async def test_auth_open_by_default(client):
    r = await client.get("/graph/statistics")
    assert r.status_code == 200


async def test_auth_enforced_when_key_set(client, monkeypatch):
    monkeypatch.setattr("backend.deps.settings.API_KEY", "secret")

    r = await client.get("/graph/statistics")
    assert r.status_code == 401

    r = await client.get("/graph/statistics", headers={"x-api-key": "wrong"})
    assert r.status_code == 401

    r = await client.get("/graph/statistics", headers={"x-api-key": "secret"})
    assert r.status_code == 200

    # Health stays public for load balancers.
    r = await client.get("/health")
    assert r.status_code == 200


async def test_oversize_content_rejected(client, monkeypatch):
    # settings is a shared singleton; patching it once covers the route.
    monkeypatch.setattr("backend.api.ingestion.settings.MAX_INGESTION_CHARS", 10)
    r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "x" * 11})
    assert r.status_code == 413


async def test_ready_reports_dependencies(client):
    r = await client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}


async def test_slow_llm_chunk_times_out(tmp_graph, monkeypatch):
    monkeypatch.setattr("backend.services.extraction.settings.EXTRACTION_TIMEOUT_S", 0.05)

    async def slow_complete(system: str, user: str) -> dict:
        await asyncio.sleep(5)
        return {"nodes": [], "edges": []}

    result = await run_extraction(tmp_graph, "hello", slow_complete)
    assert result.nodes_created == 0
    assert any("timed out" in e for e in result.errors)
