from __future__ import annotations

from backend.app import app
from backend.deps import get_answer_complete, get_extraction_complete


async def quoted_complete(system: str, user: str) -> dict:
    return {
        "nodes": [
            {
                "type": "DECISION",
                "title": "Ship v1",
                "content": "Team agreed",
                "confidence": 0.9,
                "quote": "we will ship v1",
            },
            {"type": "PERSON", "title": "Ada", "content": "", "confidence": 1.0, "quote": ""},
        ],
        "edges": [
            {
                "source_title": "Ship v1",
                "target_title": "Ada",
                "edge_type": "DECIDED_BY",
                "evidence": "Ada decided",
            },
        ],
    }


async def test_extraction_attaches_provenance(client):
    app.dependency_overrides[get_extraction_complete] = lambda: quoted_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "MEETING", "content": "x"})
        job_id = r.json()["job_id"]
        body = (await client.post(f"/ingestion/{job_id}/process")).json()
        assert body["status"] == "COMPLETED"

        r = await client.get("/graph/search", params={"q": "ship"})
        node = next(n for n in r.json() if n["title"] == "Ship v1")
        assert node["source_ref"] == job_id
        assert node["evidence"][0]["ingestion_id"] == job_id
        assert node["evidence"][0]["quote"] == "we will ship v1"

        r = await client.get(f"/graph/nodes/{node['id']}/edges")
        edge = r.json()[0]
        assert edge["ingestion_id"] == job_id
        assert edge["evidence"] == "Ada decided"
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_merge_appends_evidence_without_duplicating_nodes(client):
    app.dependency_overrides[get_extraction_complete] = lambda: quoted_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "MEETING", "content": "x"})
        await client.post(f"/ingestion/{r.json()['job_id']}/process")
        r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "x"})
        body = (await client.post(f"/ingestion/{r.json()['job_id']}/process")).json()
        assert body["nodes_created"] == 0  # merged into existing nodes

        r = await client.get("/graph/search", params={"q": "ship"})
        node = next(n for n in r.json() if n["title"] == "Ship v1")
        assert len(node["evidence"]) == 2  # one entry per ingestion
        assert len({e["ingestion_id"] for e in node["evidence"]}) == 2
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_empty_title_nodes_are_rejected(client):
    async def bad_complete(system: str, user: str) -> dict:
        return {
            "nodes": [{"type": "DECISION", "title": "   ", "content": "x", "confidence": 0.9}],
            "edges": [],
        }

    app.dependency_overrides[get_extraction_complete] = lambda: bad_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "MEETING", "content": "x"})
        body = (await client.post(f"/ingestion/{r.json()['job_id']}/process")).json()
        assert body["status"] == "COMPLETED"
        assert body["nodes_created"] == 0
        assert any("empty title" in e for e in body["error"].split("\n"))
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_query_verdicts_and_evidence(client):
    from backend.app import app
    from backend.deps import get_answer_complete

    app.dependency_overrides[get_answer_complete] = lambda: None
    try:
        d1 = (
            await client.post("/graph/nodes", json={"type": "DECISION", "title": "Ship v1"})
        ).json()
        d2 = (
            await client.post("/graph/nodes", json={"type": "DECISION", "title": "Kill v1"})
        ).json()
        await client.post(
            "/graph/edges",
            json={"source_id": d1["id"], "target_id": d2["id"], "edge_type": "CONTRADICTS"},
        )

        r = await client.post("/graph/query", json={"query": "zzz-no-such-thing"})
        assert r.json()["verdict"] == "uncertain"
        assert r.json()["answer"] is None

        # "v1" matches both decisions, so the CONTRADICTS edge is in-context.
        r = await client.post("/graph/query", json={"query": "v1", "include_evidence": True})
        body = r.json()
        assert body["verdict"] == "contradictory"
        kinds = {e["kind"] for e in body["evidence"]}
        assert {"node", "edge"} <= kinds
        assert all("ref_id" in e and "ingestion_id" in e for e in body["evidence"])

        r = await client.post("/graph/query", json={"query": "ship"})
        assert r.json()["evidence"] == []
    finally:
        app.dependency_overrides.pop(get_answer_complete, None)


async def test_query_citations_resolve_to_node_ids(client):
    node = (await client.post("/graph/nodes", json={"type": "DECISION", "title": "Ship v1"})).json()

    async def citing_answer(system: str, user: str) -> str:
        return "The team will do it, see [Ship v1] and [No Such Node]."

    app.dependency_overrides[get_answer_complete] = lambda: citing_answer
    try:
        r = await client.post("/graph/query", json={"query": "ship"})
        body = r.json()
        assert body["citations"] == [node["id"]]  # unknown titles ignored
    finally:
        app.dependency_overrides.pop(get_answer_complete, None)


async def test_query_never_calls_llm_without_context(client):
    async def explosive_answer(system: str, user: str) -> str:
        raise AssertionError("LLM must not be consulted with empty context")

    app.dependency_overrides[get_answer_complete] = lambda: explosive_answer
    try:
        r = await client.post("/graph/query", json={"query": "zzz-no-such-thing"})
        assert r.json()["answer"] is None
        assert r.json()["verdict"] == "uncertain"
    finally:
        app.dependency_overrides.pop(get_answer_complete, None)


async def test_notification_metadata_and_read_timestamp(client):
    r = await client.post(
        "/notifications",
        json={"type": "NEW_INSIGHT", "title": "Hi", "related_node_ids": ["n1"]},
    )
    body = r.json()
    assert body["severity"] == body["priority"] == "MEDIUM"
    assert body["ingestion_id"] == ""
    assert body["evidence_ids"] == []
    assert body["read_at"] is None

    r = await client.patch(f"/notifications/{body['id']}/read")
    assert r.json()["is_read"] is True
    assert r.json()["read_at"] is not None


async def test_rival_action_items_each_reported(client):
    async def rival_complete(system: str, user: str) -> dict:
        return {
            "nodes": [
                {"type": "ACTION_ITEM", "title": "Cut scope", "content": "", "confidence": 0.9},
                {"type": "ACTION_ITEM", "title": "Add scope", "content": "", "confidence": 0.9},
            ],
            "edges": [],
        }

    app.dependency_overrides[get_extraction_complete] = lambda: rival_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "MEETING", "content": "x"})
        body = (await client.post(f"/ingestion/{r.json()['job_id']}/process")).json()
        assert body["notifications_created"] == 2
        owners = [
            n for n in (await client.get("/notifications")).json() if n["type"] == "MISSING_OWNER"
        ]
        assert {n["title"] for n in owners} == {
            "Missing owner: Cut scope",
            "Missing owner: Add scope",
        }
        assert all(n["severity"] == "MEDIUM" for n in owners)
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_idempotent_reprocess_returns_stored_outcome(client):
    app.dependency_overrides[get_extraction_complete] = lambda: quoted_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "MEETING", "content": "x"})
        job_id = r.json()["job_id"]
        first = (await client.post(f"/ingestion/{job_id}/process")).json()
        stats_before = (await client.get("/graph/statistics")).json()

        second = (await client.post(f"/ingestion/{job_id}/process")).json()
        assert second["status"] == "COMPLETED"
        assert second["nodes_created"] == first["nodes_created"]
        assert second["edges_created"] == first["edges_created"]
        assert second["attempts"] == first["attempts"] == 1
        assert second["notifications_created"] == 0
        assert (await client.get("/graph/statistics")).json() == stats_before
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_failed_job_retries_and_preserves_content(client):
    calls = 0

    async def flaky_complete(system: str, user: str) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("LLM exploded")
        return {"nodes": [], "edges": []}

    app.dependency_overrides[get_extraction_complete] = lambda: flaky_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "remember me"})
        job_id = r.json()["job_id"]
        failed = (await client.post(f"/ingestion/{job_id}/process")).json()
        assert failed["status"] == "FAILED"
        assert failed["attempts"] == 1
        assert "exploded" in failed["error"]

        retried = (await client.post(f"/ingestion/{job_id}/process")).json()
        assert retried["status"] == "COMPLETED"
        assert retried["attempts"] == 2
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_processing_job_conflicts(client, test_db):
    r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "x"})
    job_id = r.json()["job_id"]
    async with test_db() as session:
        from backend.database import IngestionJobRow

        row = await session.get(IngestionJobRow, job_id)
        row.status = "PROCESSING"
        await session.commit()
    r = await client.post(f"/ingestion/{job_id}/process")
    assert r.status_code == 409
