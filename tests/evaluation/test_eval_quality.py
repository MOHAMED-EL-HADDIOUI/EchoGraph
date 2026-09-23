from __future__ import annotations

from backend.models.knowledge_graph import EdgeType
from backend.services.extraction import run_extraction

# Frozen quality set: hand-checked transcript -> recorded LLM payload ->
# expected graph. Prompt tweaks must keep these green. Single-chunk inputs
# so the fake can ignore its arguments.

MEETING_TRANSCRIPT = (
    "Ada decided to ship v1 next Friday. "
    "Bob owns the rollout, which is blocked by the security review."
)


async def meeting_complete(system: str, user: str) -> dict:
    assert "Ada decided" in user
    return {
        "nodes": [
            {
                "type": "DECISION",
                "title": "Ship v1",
                "content": "Ship next Friday",
                "confidence": 0.9,
            },
            {"type": "PERSON", "title": "Ada", "content": "", "confidence": 1.0},
            {"type": "PERSON", "title": "Bob", "content": "", "confidence": 1.0},
            {
                "type": "ACTION_ITEM",
                "title": "Rollout",
                "content": "Roll out v1",
                "confidence": 0.8,
            },
            {
                "type": "QUESTION",
                "title": "Security review",
                "content": "Pending review",
                "confidence": 0.7,
            },
        ],
        "edges": [
            {
                "source_title": "Ship v1",
                "target_title": "Ada",
                "edge_type": "DECIDED_BY",
                "evidence": "",
            },
            {"source_title": "Bob", "target_title": "Rollout", "edge_type": "OWNS", "evidence": ""},
            {
                "source_title": "Rollout",
                "target_title": "Security review",
                "edge_type": "BLOCKED_BY",
                "evidence": "",
            },
        ],
    }


async def test_meeting_extraction_quality(tmp_graph):
    result = await run_extraction(tmp_graph, MEETING_TRANSCRIPT, meeting_complete)
    assert result.errors == []
    assert result.nodes_created == 5
    assert result.edges_created == 3
    titles = {n.title for n in result.nodes}
    assert {"Ship v1", "Ada", "Bob", "Rollout", "Security review"} <= titles
    assert {(e.edge_type) for e in result.edges} == {
        EdgeType.DECIDED_BY,
        EdgeType.OWNS,
        EdgeType.BLOCKED_BY,
    }


DISPUTE_TRANSCRIPT = "The team agreed to ship v1. Later, Ada reversed the call and killed v1."


async def dispute_complete(system: str, user: str) -> dict:
    assert "reversed" in user
    return {
        "nodes": [
            {"type": "DECISION", "title": "Ship v1", "content": "", "confidence": 0.9},
            {
                "type": "DECISION",
                "title": "Kill v1",
                "content": "Ada reversed the call",
                "confidence": 0.85,
            },
        ],
        "edges": [
            {
                "source_title": "Kill v1",
                "target_title": "Ship v1",
                "edge_type": "SUPERSEDES",
                "evidence": "reversed the call",
            },
        ],
    }


async def test_dispute_extraction_quality(tmp_graph):
    result = await run_extraction(tmp_graph, DISPUTE_TRANSCRIPT, dispute_complete)
    assert result.errors == []
    assert result.nodes_created == 2
    assert result.edges_created == 1
    assert result.edges[0].edge_type == EdgeType.SUPERSEDES


async def paraphrase_embed(texts: list[str]) -> list[list[float]]:
    # "ship version one" ~= "Ship v1"; everything else is far away.
    vecs = []
    for t in texts:
        if "version one" in t.lower() or t.startswith("Ship v1"):
            vecs.append([1.0, 0.0])
        else:
            vecs.append([0.0, 1.0])
    return vecs


async def paraphrase_complete(system: str, user: str) -> dict:
    return {
        "nodes": [
            {"type": "DECISION", "title": "Ship v1", "content": "", "confidence": 0.9},
            {"type": "DECISION", "title": "Ship version one", "content": "", "confidence": 0.8},
            {"type": "TOPIC", "title": "Launch", "content": "", "confidence": 0.7},
        ],
        "edges": [],
    }


async def test_embedding_dedup_merges_paraphrases(tmp_graph, monkeypatch):
    monkeypatch.setattr("backend.services.extraction.settings.EMBEDDING_DEDUP_THRESHOLD", 0.9)
    result = await run_extraction(
        tmp_graph, "ship it", paraphrase_complete, embed_texts=paraphrase_embed
    )
    assert result.errors == []
    assert result.nodes_created == 2  # paraphrase merged, Launch added
    titles = sorted(n.title for n in result.nodes)
    assert titles == ["Launch", "Ship v1"]
