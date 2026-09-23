from __future__ import annotations

import random
import string

from backend import obs
from backend.graph.networkx_backend import NetworkXGraphManager
from backend.models.knowledge_graph import (
    EdgeType,
    Evidence,
    KnowledgeEdge,
    KnowledgeNode,
    NodeType,
)
from backend.services.deduplication import normalize_title, quotes_match
from backend.services.extraction import chunk_text, merge_evidence
from backend.services.graph_ops import add_edge_validated


def _rand_title(rng: random.Random) -> str:
    words = [
        "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(1, 8)))
        for _ in range(3)
    ]
    sep = rng.choice([" ", "  ", "\t", " \n "])
    core = sep.join(words)
    return rng.choice([core, core.upper(), core.lower(), f"  {core}  "])


async def test_normalize_title_is_idempotent_and_insensitive():
    rng = random.Random(42)
    for _ in range(200):
        a = _rand_title(rng)
        _rand_title(rng)  # extra draw: distribution must not matter
        assert normalize_title(normalize_title(a)) == normalize_title(a)
        assert normalize_title("  Atlas   Migration ") == normalize_title("ATLAS migration")
        assert normalize_title(a) != "" or a.strip() == ""
    assert normalize_title("x") == "x"


async def test_fingerprint_is_deterministic():
    assert obs.fingerprint("hello") == obs.fingerprint("hello")
    assert obs.fingerprint("hello") != obs.fingerprint("world")
    assert len(obs.fingerprint_content("hello")) == 64


async def test_merge_is_idempotent(tmp_path):
    graph = NetworkXGraphManager(data_path=str(tmp_path / "g.json"))
    node = await graph.add_node(KnowledgeNode(type=NodeType.DECISION, title="Ship v1"))
    merged = await graph.merge_node(
        node.id, KnowledgeNode(type=NodeType.DECISION, title="SHIP  v1")
    )
    assert merged.id == node.id
    assert (await graph.get_statistics())["total_nodes"] == 1
    ev = Evidence(ingestion_id="i1", quote="q")
    assert merge_evidence(merge_evidence([], ev), ev) == [ev]


async def test_invalid_edge_leaves_graph_unchanged(tmp_path):
    graph = NetworkXGraphManager(data_path=str(tmp_path / "g.json"))
    before = await graph.get_statistics()
    person = await graph.add_node(KnowledgeNode(type=NodeType.PERSON, title="Ada"))
    topic = await graph.add_node(KnowledgeNode(type=NodeType.TOPIC, title="T"))
    from backend.services.graph_ops import InvalidEdgeError

    try:
        await add_edge_validated(
            graph, KnowledgeEdge(source_id=person.id, target_id=topic.id, edge_type=EdgeType.OWNS)
        )
        raise AssertionError("should have raised")
    except InvalidEdgeError:
        pass
    after = await graph.get_statistics()
    assert after["total_edges"] == before["total_edges"]
    assert after["total_nodes"] == before["total_nodes"] + 2


async def test_chunk_overlap_keeps_boundary_text():
    paras = ["alpha one", "beta two", "gamma three"]
    chunks = chunk_text("\n\n".join(paras), max_chars=12, overlap=9)
    assert len(chunks) == 3
    assert "beta two" in chunks[1]
    assert chunks[1].startswith("alpha one"[-9:])
    plain = chunk_text("\n\n".join(paras), max_chars=12)
    assert len(plain) == 3


async def test_quotes_match_is_case_insensitive_substring():
    assert quotes_match("Ship V1", "we will ship v1 today")
    assert not quotes_match("nukes", "we will ship v1 today")
    assert not quotes_match("", "anything")
    assert not quotes_match("   ", "anything")
