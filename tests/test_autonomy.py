from __future__ import annotations

from backend.app import app
from backend.deps import get_answer_complete, get_extraction_complete, get_transcriber


async def contradiction_complete(system: str, user: str) -> dict:
    return {
        "nodes": [
            {"type": "DECISION", "title": "Ship v1", "content": "", "confidence": 0.9},
            {"type": "DECISION", "title": "Kill v1", "content": "", "confidence": 0.8},
            {"type": "ACTION_ITEM", "title": "Write postmortem", "content": "", "confidence": 1.0},
        ],
        "edges": [
            {
                "source_title": "Ship v1",
                "target_title": "Kill v1",
                "edge_type": "CONTRADICTS",
                "evidence": "we will ship; we will not ship",
            },
        ],
    }


async def fake_answer(system: str, user: str) -> str:
    assert "Ship v1" in user  # context reached the prompt
    return "Ship v1 contradicts Kill v1."


async def test_process_creates_contradiction_and_owner_notifications(client):
    app.dependency_overrides[get_extraction_complete] = lambda: contradiction_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "MEETING", "content": "x"})
        r = await client.post(f"/ingestion/{r.json()['job_id']}/process")
        assert r.status_code == 200, r.text
        assert r.json()["notifications_created"] == 4  # 1 contradiction + 3 missing owners

        r = await client.get("/notifications")
        by_type = {}
        for n in r.json():
            by_type.setdefault(n["type"], []).append(n)
        assert len(by_type["CONTRADICTION"]) == 1
        assert by_type["CONTRADICTION"][0]["priority"] == "HIGH"
        assert len(by_type["CONTRADICTION"][0]["related_node_ids"]) == 2
        assert len(by_type["MISSING_OWNER"]) == 3  # 2 decisions + 1 action item
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_notifications_not_duplicated_on_reprocess(client):
    app.dependency_overrides[get_extraction_complete] = lambda: contradiction_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "MEETING", "content": "x"})
        first = r.json()["job_id"]
        await client.post(f"/ingestion/{first}/process")

        # Same ingestion reprocessed: idempotent no-op, zero new notifications.
        r = await client.post(f"/ingestion/{first}/process")
        assert r.json()["status"] == "COMPLETED"
        assert r.json()["notifications_created"] == 0

        r = await client.get("/notifications")
        assert len(r.json()) == 4  # 1 contradiction + 3 missing-owner, no dupes
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_query_answer_uses_llm_when_configured(client):
    await client.post("/graph/nodes", json={"type": "DECISION", "title": "Ship v1"})
    app.dependency_overrides[get_answer_complete] = lambda: fake_answer
    try:
        r = await client.post("/graph/query", json={"query": "ship"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["answer"] == "Ship v1 contradicts Kill v1."
        assert body["confidence"] >= 0.75
    finally:
        app.dependency_overrides.pop(get_answer_complete, None)


async def test_query_without_llm_is_retrieval_only(client, monkeypatch):
    monkeypatch.setattr("backend.deps.settings.OPENAI_API_KEY", "")
    await client.post("/graph/nodes", json={"type": "DECISION", "title": "Ship v1"})
    r = await client.post("/graph/query", json={"query": "ship"})
    assert r.json()["answer"] is None


async def fake_transcribe(path: str):
    from backend.providers.transcription import TranscriptResult

    assert path.endswith(".mp3")
    return TranscriptResult(
        text="Ada decided to ship v1.",
        provider="fake",
        model="fake",
        transcript_sha="abc123",
    )


async def test_transcribe_creates_audio_job(client):
    async def echo_complete(system: str, user: str) -> dict:
        assert "Ada decided" in user  # transcript persisted as job content
        return {"nodes": [], "edges": []}

    app.dependency_overrides[get_transcriber] = lambda: fake_transcribe
    app.dependency_overrides[get_extraction_complete] = lambda: echo_complete
    try:
        r = await client.post(
            "/ingestion/transcribe",
            files={"file": ("meeting.mp3", b"fake-audio", "audio/mpeg")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source_type"] == "AUDIO"
        assert body["status"] == "PENDING"
        assert body["title"] == "meeting.mp3"
        assert body["metadata"]["provider"] == "fake"
        assert body["metadata"]["transcript_sha"] == "abc123"

        r = await client.post(f"/ingestion/{body['job_id']}/process")
        assert r.json()["status"] == "COMPLETED"
    finally:
        app.dependency_overrides.pop(get_transcriber, None)
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_transcribe_full_loop_to_graph(client):
    """Audio -> transcript job -> process -> node in graph."""
    app.dependency_overrides[get_transcriber] = lambda: fake_transcribe
    app.dependency_overrides[get_extraction_complete] = lambda: contradiction_complete
    try:
        r = await client.post(
            "/ingestion/transcribe",
            files={"file": ("standup.mp3", b"fake-audio", "audio/mpeg")},
        )
        job_id = r.json()["job_id"]
        r = await client.post(f"/ingestion/{job_id}/process")
        assert r.json()["status"] == "COMPLETED"
        assert r.json()["nodes_created"] == 3
    finally:
        app.dependency_overrides.pop(get_transcriber, None)
        app.dependency_overrides.pop(get_extraction_complete, None)


async def test_transcribe_requires_api_key_without_override(client, monkeypatch):
    monkeypatch.setattr("backend.deps.settings.OPENAI_API_KEY", "")
    r = await client.post(
        "/ingestion/transcribe",
        files={"file": ("meeting.mp3", b"fake-audio", "audio/mpeg")},
    )
    assert r.status_code == 503
