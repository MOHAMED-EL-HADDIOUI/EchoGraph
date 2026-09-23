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


async def test_stale_decision_gated_and_detected(tmp_graph, test_db, monkeypatch):
    import datetime as dt

    from backend.models.knowledge_graph import KnowledgeNode, NodeType
    from backend.services.notifications import generate_notifications

    aged = await tmp_graph.add_node(
        KnowledgeNode(
            type=NodeType.DECISION,
            title="Old call",
            # Naive datetimes: aging math is calendar-day based by convention.
            created_at=dt.datetime(2020, 1, 1),  # noqa: DTZ001
            observed_at=dt.datetime(2020, 1, 1),  # noqa: DTZ001
        )
    )
    async with test_db() as session:
        # Off by default: only the missing-owner rule fires.
        created = await generate_notifications(tmp_graph, session, [aged], [], ingestion_id="i1")
        assert {r.type for r in created} == {"MISSING_OWNER"}

        monkeypatch.setattr(
            "backend.services.notifications.settings.ENABLE_STALE_DECISION_DETECTION", True
        )
        created = await generate_notifications(tmp_graph, session, [aged], [], ingestion_id="i1")
        # Missing-owner was already reported unread above; only the stale rule is new.
        assert {r.type for r in created} == {"STALE_DECISION"}
        stale = created[0]
        assert stale.priority == "LOW"
        assert stale.ingestion_id == "i1"

        # Duplicates suppressed while unread.
        again = await generate_notifications(tmp_graph, session, [aged], [], ingestion_id="i1")
        assert again == []


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


async def test_decision_history_timeline(client):
    old = (
        await client.post("/graph/nodes", json={"type": "DECISION", "title": "Use SQLite"})
    ).json()
    new = (
        await client.post("/graph/nodes", json={"type": "DECISION", "title": "Use Postgres"})
    ).json()
    rival = (
        await client.post("/graph/nodes", json={"type": "DECISION", "title": "Stay on MySQL"})
    ).json()
    await client.post(
        "/graph/edges",
        json={"source_id": new["id"], "target_id": old["id"], "edge_type": "SUPERSEDES"},
    )
    await client.post(
        "/graph/edges",
        json={"source_id": rival["id"], "target_id": new["id"], "edge_type": "CONTRADICTS"},
    )

    body = (await client.get(f"/graph/nodes/{new['id']}/history")).json()
    assert body["node"]["title"] == "Use Postgres"
    relations = {e["relation"] for e in body["events"]}
    assert relations == {"supersedes", "contradicted_by"}
    assert {e["node"]["title"] for e in body["events"]} == {"Use SQLite", "Stay on MySQL"}

    body = (await client.get(f"/graph/nodes/{old['id']}/history")).json()
    # Transitive: old was superseded by new, which rival contradicts.
    assert {e["relation"] for e in body["events"]} == {"superseded_by", "contradicted_by"}

    lonely = (
        await client.post("/graph/nodes", json={"type": "DECISION", "title": "Lonely"})
    ).json()
    assert (await client.get(f"/graph/nodes/{lonely['id']}/history")).json()["events"] == []
    assert (await client.get("/graph/nodes/missing/history")).status_code == 404


async def test_unresolved_questions_endpoint(client):
    open_q = (
        await client.post("/graph/nodes", json={"type": "QUESTION", "title": "When launch?"})
    ).json()
    answered_q = (
        await client.post("/graph/nodes", json={"type": "QUESTION", "title": "Who pays?"})
    ).json()
    answer = (
        await client.post("/graph/nodes", json={"type": "QUESTION", "title": "We pay"})
    ).json()
    await client.post(
        "/graph/edges",
        json={"source_id": answer["id"], "target_id": answered_q["id"], "edge_type": "ANSWERS"},
    )
    # QUESTION-ANSWERS->QUESTION is valid; an answer node is still a question type.
    titles = {n["title"] for n in (await client.get("/graph/questions/unresolved")).json()}
    assert "When launch?" in titles
    assert "Who pays?" not in titles
    assert open_q["id"] in {
        n["id"] for n in (await client.get("/graph/questions/unresolved")).json()
    }


async def test_enqueue_failure_maps_to_503(client, monkeypatch):
    monkeypatch.setattr("backend.api.ingestion.settings.USE_BACKGROUND_JOBS", True)

    def dead_enqueue(job_id: str):
        raise ConnectionError("redis gone")

    monkeypatch.setattr("backend.worker.enqueue_process", dead_enqueue)
    r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "x"})
    r = await client.post(f"/ingestion/{r.json()['job_id']}/process")
    assert r.status_code == 503
    assert "worker unavailable" in r.json()["detail"]


async def test_transcription_provider_failure_maps_to_502(client):
    from backend.app import app
    from backend.deps import get_transcriber
    from backend.exceptions import TranscriptionError

    async def failing_transcribe(path: str):
        raise TranscriptionError("asr exploded")

    app.dependency_overrides[get_transcriber] = lambda: failing_transcribe
    try:
        r = await client.post(
            "/ingestion/transcribe",
            files={"file": ("a.mp3", b"data", "audio/mpeg")},
        )
        assert r.status_code == 502
    finally:
        app.dependency_overrides.pop(get_transcriber, None)


async def test_unknown_edge_type_rejected(tmp_graph):
    from backend.services.extraction import run_extraction

    async def bad_edge_type(system: str, user: str) -> dict:
        return {
            "nodes": [{"type": "PERSON", "title": "Ada", "content": "", "confidence": 1.0}],
            "edges": [
                {
                    "source_title": "Ada",
                    "target_title": "Ada",
                    "edge_type": "MARRIED_TO",
                    "evidence": "",
                }
            ],
        }

    result = await run_extraction(tmp_graph, "Ada and Ada", bad_edge_type)
    assert result.nodes_created == 1
    assert result.edges_created == 0
    assert any("invalid edges record #0" in e for e in result.errors)


async def test_hybrid_falls_back_when_embeddings_fail(tmp_graph):
    from backend.models.knowledge_graph import KnowledgeNode, NodeType
    from backend.services.retrieval import HybridRetriever

    await tmp_graph.add_node(KnowledgeNode(type=NodeType.TOPIC, title="Search"))

    async def dead_embed(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embeddings down")

    retriever = HybridRetriever(embed_texts=dead_embed)
    ranked = await retriever.search(tmp_graph, "search")
    assert [r.node.title for r in ranked] == ["Search"]
    assert "keyword_match" in ranked[0].reasons


async def test_neo4j_health_check_false_without_server(monkeypatch):
    from backend.graph.neo4j_backend import Neo4jGraphManager

    manager = Neo4jGraphManager.__new__(Neo4jGraphManager)

    class DeadSession:
        async def __aenter__(self):
            raise RuntimeError("no server")

        async def __aexit__(self, *args):
            return False

    class DeadDriver:
        def session(self):
            return DeadSession()

        async def close(self):
            pass

    monkeypatch.setattr(manager, "_driver", DeadDriver(), raising=False)
    assert await manager.health_check() is False
