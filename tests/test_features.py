from __future__ import annotations

import json

from backend.app import app
from backend.database import get_db
from backend.deps import get_answer_complete, get_extraction_complete


async def simple_complete(system: str, user: str) -> dict:
    return {
        "nodes": [
            {"type": "DECISION", "title": "Ship v1", "content": "", "confidence": 1.0},
        ],
        "edges": [],
    }


async def test_update_node_route(client):
    node = (await client.post("/graph/nodes", json={"type": "TOPIC", "title": "Old"})).json()
    r = await client.put(
        f"/graph/nodes/{node['id']}",
        json={"type": "TOPIC", "title": "New", "content": "updated"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["title"] == "New"
    assert r.json()["id"] == node["id"]
    assert (
        await client.put("/graph/nodes/missing", json={"type": "TOPIC", "title": "x"})
    ).status_code == 404


async def test_neighbors_and_remove_edge_routes(client):
    a = (await client.post("/graph/nodes", json={"type": "PERSON", "title": "Ada"})).json()
    b = (await client.post("/graph/nodes", json={"type": "TOPIC", "title": "T"})).json()
    edge = (
        await client.post(
            "/graph/edges",
            json={"source_id": a["id"], "target_id": b["id"], "edge_type": "RELATES_TO"},
        )
    ).json()
    neighbors = (await client.get(f"/graph/nodes/{a['id']}/neighbors")).json()
    assert {n["id"] for n in neighbors["nodes"]} == {a["id"], b["id"]}
    assert (await client.get("/graph/nodes/missing/neighbors")).status_code == 404

    r = await client.delete(f"/graph/edges/{edge['id']}")
    assert r.json() == {"deleted": True}
    assert (await client.delete(f"/graph/edges/{edge['id']}")).status_code == 404
    assert (await client.get(f"/graph/nodes/{a['id']}/edges")).json() == []


async def test_export_route_includes_provenance(client):
    app.dependency_overrides[get_extraction_complete] = lambda: simple_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "MEETING", "content": "x"})
        await client.post(f"/ingestion/{r.json()['job_id']}/process")
        body = (await client.get("/graph/export")).json()
        assert len(body["nodes"]) == 1
        assert body["nodes"][0]["evidence"][0]["ingestion_id"] == r.json()["job_id"]
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_lineage_endpoint(client):
    app.dependency_overrides[get_extraction_complete] = lambda: simple_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "MEETING", "content": "x"})
        job_id = r.json()["job_id"]
        await client.post(f"/ingestion/{job_id}/process")
        node = (await client.get("/graph/search", params={"q": "ship"})).json()[0]
        body = (await client.get(f"/graph/nodes/{node['id']}/lineage")).json()
        assert body["origin"]["ingestion_id"] == job_id
        assert body["origin"]["prompt_version"] == "v1"
        assert (await client.get("/graph/nodes/missing/lineage")).status_code == 404
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_get_query_route_and_filters(client):
    await client.post("/graph/nodes", json={"type": "PERSON", "title": "Ada"})
    r = await client.get("/graph/query", params={"q": "ada"})
    assert r.status_code == 200
    assert any(n["title"] == "Ada" for n in r.json()["nodes"])

    r = await client.get("/graph/query", params={"q": "ada", "node_type": "TOPIC"})
    assert r.json()["nodes"] == []

    r = await client.get("/graph/query", params={"q": "ada", "explain": "true"})
    debug = r.json()["debug"]
    assert debug["matched_nodes"] >= 1
    assert debug["ranking"][0]["reasons"]

    r = await client.post("/graph/query", json={"query": "ada"})
    assert r.json()["debug"] is None


async def test_query_ingestion_id_filter(client):
    app.dependency_overrides[get_extraction_complete] = lambda: simple_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "MEETING", "content": "x"})
        job1 = r.json()["job_id"]
        await client.post(f"/ingestion/{job1}/process")
        r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "y"})
        job2 = r.json()["job_id"]
        await client.post(f"/ingestion/{job2}/process")

        r = await client.post(
            "/graph/query", json={"query": "ship", "filters": {"ingestion_id": job1}}
        )
        got = {(n["title"], n["source_ref"]) for n in r.json()["nodes"]}
        assert got and all(ref == job1 for _, ref in got)
        assert len(got) == 1
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_structured_answer_parsed_with_citations(client):
    await client.post("/graph/nodes", json={"type": "DECISION", "title": "Ship v1"})

    async def json_answer(system: str, user: str) -> str:
        return json.dumps(
            {
                "answer": "We ship, see [Ship v1].",
                "citations": ["Ship v1"],
                "uncertainties": ["Date unknown."],
                "conflicts": [],
            }
        )

    app.dependency_overrides[get_answer_complete] = lambda: json_answer
    try:
        body = (await client.post("/graph/query", json={"query": "ship"})).json()
        assert body["answer"] == "We ship, see [Ship v1]."
        assert len(body["citations"]) == 1
        assert body["uncertainties"] == ["Date unknown."]
        assert body["verdict"] == "supported"
    finally:
        app.dependency_overrides.pop(get_answer_complete, None)


async def test_answer_provider_failure_abstains(client):
    await client.post("/graph/nodes", json={"type": "DECISION", "title": "Ship v1"})

    async def boom_answer(system: str, user: str) -> str:
        raise RuntimeError("provider down")

    app.dependency_overrides[get_answer_complete] = lambda: boom_answer
    try:
        body = (await client.post("/graph/query", json={"query": "ship"})).json()
        assert body["answer"] == "No grounded evidence was found."
        assert body["verdict"] == "uncertain"
        assert any("provider" in u for u in body["uncertainties"])
    finally:
        app.dependency_overrides.pop(get_answer_complete, None)


async def test_dry_run_changes_nothing(client):
    app.dependency_overrides[get_extraction_complete] = lambda: simple_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "MEETING", "content": "x"})
        job_id = r.json()["job_id"]
        stats_before = (await client.get("/graph/statistics")).json()
        body = (await client.post(f"/ingestion/{job_id}/dry-run")).json()
        assert body["diff"]["nodes_added"] == 1
        assert len(body["proposed_nodes"]) == 1
        assert (await client.get(f"/ingestion/{job_id}")).json()["status"] == "PENDING"
        assert (await client.get("/graph/statistics")).json() == stats_before
        assert (await client.get("/notifications")).json() == []
        assert (await client.post("/ingestion/missing/dry-run")).status_code == 404
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_process_response_carries_diff_and_run_metadata(client):
    app.dependency_overrides[get_extraction_complete] = lambda: simple_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "MEETING", "content": "x"})
        body = (await client.post(f"/ingestion/{r.json()['job_id']}/process")).json()
        assert body["diff"]["nodes_added"] == 1
        runs = body["metadata"]["runs"]
        assert len(runs) == 1
        assert runs[0]["prompt_version"] == "v1"
        assert runs[0]["status"] == "COMPLETED"
        assert runs[0]["run_id"]
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_notification_explain_and_post_read_alias(client):
    r = await client.post("/notifications", json={"type": "NEW_INSIGHT", "title": "Hi"})
    nid = r.json()["id"]
    body = (await client.get(f"/notifications/{nid}/explain")).json()
    assert body["notification"]["id"] == nid
    assert body["node_titles"] == []  # unknown node id, nothing to resolve
    assert (await client.get("/notifications/missing/explain")).status_code == 404

    r = await client.post(f"/notifications/{nid}/read")
    assert r.json()["is_read"] is True
    assert r.json()["read_at"] is not None


async def test_statistics_new_keys(client):
    await client.post("/graph/nodes", json={"type": "ACTION_ITEM", "title": "Lonely"})
    stats = (await client.get("/graph/statistics")).json()
    assert stats["orphan_nodes"] == 1
    assert stats["ownerless_actions"] == 1
    assert stats["contradictions"] == 0


async def test_ready_redis_down_when_workers_enabled(client, monkeypatch):
    monkeypatch.setattr("backend.api.health.settings.USE_BACKGROUND_JOBS", True)
    monkeypatch.setattr("backend.api.health.settings.REDIS_URL", "redis://localhost:1/0")
    r = await client.get("/ready")
    assert r.status_code == 503
    assert r.json()["detail"]["dependencies"]["redis"]["status"] == "down"


async def test_ready_database_down(client):
    class BoomSession:
        async def execute(self, *args, **kwargs):
            raise RuntimeError("db gone")

    async def boom_db():
        yield BoomSession()

    from backend.app import app as fastapi_app

    fastapi_app.dependency_overrides[get_db] = boom_db
    try:
        r = await client.get("/ready")
        assert r.status_code == 503
        assert r.json()["detail"]["dependencies"]["database"]["status"] == "down"
    finally:
        fastapi_app.dependency_overrides.pop(get_db, None)


async def test_embedding_failure_degrades_gracefully(tmp_graph):
    from backend.services.extraction import run_extraction

    async def fail_embed(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embeddings down")

    async def one_node(system: str, user: str) -> dict:
        return {
            "nodes": [{"type": "TOPIC", "title": "T", "content": "", "confidence": 1.0}],
            "edges": [],
        }

    result = await run_extraction(tmp_graph, "x", one_node, embed_texts=fail_embed)
    assert result.nodes_created == 1
    assert any("embedding dedup skipped" in e for e in result.errors)


async def test_unsupported_quote_rejected(tmp_graph):
    from backend.services.extraction import run_extraction

    async def bad_quote(system: str, user: str) -> dict:
        return {
            "nodes": [
                {
                    "type": "DECISION",
                    "title": "Ship v1",
                    "content": "",
                    "confidence": 0.9,
                    "quote": "we will launch nukes",
                }
            ],
            "edges": [],
        }

    result = await run_extraction(tmp_graph, "we will ship v1", bad_quote)
    assert result.nodes_created == 1
    assert result.nodes[0].evidence[0].quote == ""
    assert any("unsupported quote" in e for e in result.errors)


async def test_alembic_downgrade_roundtrip(tmp_path, monkeypatch):
    import asyncio
    import logging

    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    from alembic import command

    url = f"sqlite:///{tmp_path}/mig.db"
    monkeypatch.setattr(
        "backend.config.settings.DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/mig.db"
    )
    cfg = Config("alembic.ini")

    def columns():
        eng = create_engine(url)
        try:
            return [c["name"] for c in inspect(eng).get_columns("ingestion_jobs")]
        finally:
            eng.dispose()

    def reenable_logging():
        # Alembic's fileConfig disables existing loggers process-wide;
        # restore them so later tests (caplog) keep working.
        for name in list(logging.root.manager.loggerDict):
            logging.getLogger(name).disabled = False

    try:
        # Alembic uses asyncio.run internally: execute off the test loop.
        await asyncio.to_thread(command.upgrade, cfg, "head")
        assert "content_sha" in columns()
        await asyncio.to_thread(command.downgrade, cfg, "-1")
        assert "content_sha" not in columns()
        await asyncio.to_thread(command.upgrade, cfg, "head")
        assert "content_sha" in columns()
    finally:
        reenable_logging()
