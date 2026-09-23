from __future__ import annotations


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_create_and_read_node(client):
    r = await client.post("/graph/nodes", json={"type": "DECISION", "title": "Use Neo4j in prod"})
    assert r.status_code == 200, r.text
    node_id = r.json()["id"]

    r = await client.get(f"/graph/nodes/{node_id}")
    assert r.status_code == 200
    assert r.json()["title"] == "Use Neo4j in prod"


async def test_invalid_edge_triple_rejected(client):
    q = await client.post("/graph/nodes", json={"type": "QUESTION", "title": "Q"})
    t = await client.post("/graph/nodes", json={"type": "TOPIC", "title": "T"})
    qid, tid = q.json()["id"], t.json()["id"]

    # OWNS between QUESTION->TOPIC is not in VALID_EDGES
    r = await client.post(
        "/graph/edges",
        json={"source_id": qid, "target_id": tid, "edge_type": "OWNS"},
    )
    assert r.status_code == 400

    # RELATES_TO is open-schema, allowed between any types
    r = await client.post(
        "/graph/edges",
        json={"source_id": qid, "target_id": tid, "edge_type": "RELATES_TO"},
    )
    assert r.status_code == 200, r.text


async def test_search_and_statistics(client):
    await client.post("/graph/nodes", json={"type": "PERSON", "title": "Ada Lovelace"})
    r = await client.get("/graph/search", params={"q": "ada"})
    assert r.status_code == 200
    assert any("Ada" in n["title"] for n in r.json())

    r = await client.get("/graph/statistics")
    assert r.status_code == 200
    assert r.json()["total_nodes"] >= 1


async def test_ingestion_roundtrip(client):
    r = await client.post(
        "/ingestion", json={"source_type": "SLACK", "content": "hello", "title": "t"}
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    r = await client.get(f"/ingestion/{job_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "PENDING"


async def test_notification_roundtrip(client):
    r = await client.post(
        "/notifications",
        json={"type": "NEW_INSIGHT", "title": "Something interesting"},
    )
    assert r.status_code == 200, r.text
    nid = r.json()["id"]

    r = await client.patch(f"/notifications/{nid}/read")
    assert r.status_code == 200
    assert r.json()["is_read"] is True
