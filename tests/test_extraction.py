from __future__ import annotations

from backend.app import app
from backend.deps import get_extraction_complete
from backend.services.extraction import chunk_text


async def fake_complete(system: str, user: str) -> dict:
    return {
        "nodes": [
            {"type": "DECISION", "title": "Ship v1", "content": "Team agreed", "confidence": 0.9},
            {"type": "PERSON", "title": "Ada", "content": "", "confidence": 1.0},
        ],
        "edges": [
            {
                "source_title": "Ship v1",
                "target_title": "Ada",
                "edge_type": "DECIDED_BY",
                "evidence": "Ada decided",
            },
            {
                "source_title": "Ship v1",
                "target_title": "Ada",
                "edge_type": "OWNS",
                "evidence": "",
            },
        ],
    }


def test_chunk_text_splits_on_blank_lines():
    paras = ["a" * 100, "b" * 100, "c" * 100]
    chunks = chunk_text("\n\n".join(paras), max_chars=150)
    assert len(chunks) == 3
    assert chunk_text("") == []


async def test_process_job_end_to_end(client):
    app.dependency_overrides[get_extraction_complete] = lambda: fake_complete
    try:
        r = await client.post(
            "/ingestion", json={"source_type": "MEETING", "content": "Ada decided to ship v1"}
        )
        job_id = r.json()["job_id"]
        assert r.json()["status"] == "PENDING"

        r = await client.post(f"/ingestion/{job_id}/process")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "COMPLETED"
        assert body["nodes_created"] == 2
        assert body["edges_created"] == 1  # OWNS triple invalid, skipped
        assert body["error"] != ""

        r = await client.get("/graph/search", params={"q": "ship"})
        assert any(n["title"] == "Ship v1" for n in r.json())
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_process_job_dedupes_on_reprocess(client):
    app.dependency_overrides[get_extraction_complete] = lambda: fake_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "x"})
        job_id = r.json()["job_id"]
        await client.post(f"/ingestion/{job_id}/process")

        r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "x"})
        job2 = r.json()["job_id"]
        r = await client.post(f"/ingestion/{job2}/process")
        assert r.json()["nodes_created"] == 0  # merged, not duplicated
        assert r.json()["status"] == "COMPLETED"
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_process_job_conflicts_and_validation(client):
    r = await client.post("/ingestion", json={"source_type": "SLACK", "content": ""})
    job_id = r.json()["job_id"]
    r = await client.post(f"/ingestion/{job_id}/process")
    assert r.status_code == 422  # no content


async def test_process_job_requires_api_key_without_override(client, monkeypatch):
    monkeypatch.setattr("backend.deps.settings.OPENAI_API_KEY", "")
    r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "hi"})
    job_id = r.json()["job_id"]
    r = await client.post(f"/ingestion/{job_id}/process")
    assert r.status_code == 503


async def test_query_endpoint(client):
    await client.post("/graph/nodes", json={"type": "PERSON", "title": "Ada Lovelace"})
    await client.post("/graph/nodes", json={"type": "TOPIC", "title": "Unrelated"})
    r = await client.post("/graph/query", json={"query": "ada"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert any(n["title"] == "Ada Lovelace" for n in body["nodes"])
    assert body["answer"] is None
    assert body["confidence"] > 0

    r = await client.post("/graph/query", json={"query": "ada", "filters": {"node_type": "TOPIC"}})
    assert all(n["type"] == "TOPIC" for n in r.json()["nodes"])


async def test_list_nodes_pagination(client):
    for i in range(3):
        await client.post("/graph/nodes", json={"type": "TOPIC", "title": f"T{i}"})
    r = await client.get("/graph/nodes", params={"limit": 2, "offset": 1})
    assert r.status_code == 200
    assert len(r.json()) == 2
