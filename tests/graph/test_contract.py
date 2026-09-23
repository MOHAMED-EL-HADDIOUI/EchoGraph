from __future__ import annotations

import pytest

from backend.graph.manager import GraphManager
from backend.graph.networkx_backend import NetworkXGraphManager
from backend.models.knowledge_graph import EdgeType, KnowledgeEdge, KnowledgeNode, NodeType
from backend.services.graph_ops import InvalidEdgeError, NodeNotFoundError, add_edge_validated


async def _make_backend(name: str, tmp_path) -> GraphManager | None:
    if name == "networkx":
        return NetworkXGraphManager(data_path=str(tmp_path / "contract.json"))
    try:
        from backend.graph.neo4j_backend import Neo4jGraphManager
    except ImportError:
        pytest.skip("neo4j driver not installed")
    manager = Neo4jGraphManager()
    try:
        if not await manager.health_check():
            await manager.close()
            pytest.skip("neo4j not reachable")
    except Exception:  # noqa: BLE001 — any connection failure means skip
        pytest.skip("neo4j not reachable")
    await manager.init_constraints()
    await manager.clear()
    return manager


@pytest.fixture(params=["networkx", "neo4j"])
async def backend(request, tmp_path):
    manager = await _make_backend(request.param, tmp_path)
    yield manager
    if manager is not None:
        try:
            await manager.clear()
        except Exception:  # noqa: BLE001, S110 — best-effort test cleanup
            pass
        close = getattr(manager, "close", None)
        if callable(close):
            await close()


async def _node(ntype: NodeType, title: str, **kw) -> KnowledgeNode:
    return KnowledgeNode(type=ntype, title=title, **kw)


async def test_contract_node_crud(backend):
    node = await backend.add_node(await _node(NodeType.PERSON, "Ada"))
    assert (await backend.get_node(node.id)).title == "Ada"
    assert await backend.get_node("missing") is None

    updated = await backend.update_node(
        node.id, KnowledgeNode(type=NodeType.PERSON, title="Ada L.", content="math")
    )
    assert updated.title == "Ada L."
    assert updated.content == "math"
    assert updated.id == node.id

    assert await backend.delete_node(node.id) is True
    assert await backend.delete_node(node.id) is False
    with pytest.raises(ValueError):
        await backend.update_node(node.id, await _node(NodeType.PERSON, "x"))


async def test_contract_merge_keeps_identity_and_evidence(backend):
    from backend.models.knowledge_graph import Evidence

    node = await backend.add_node(await _node(NodeType.DECISION, "Ship v1"))
    merged = await backend.merge_node(
        node.id,
        KnowledgeNode(
            type=NodeType.DECISION,
            title="Ship V1",
            evidence=[Evidence(ingestion_id="i2", quote="ship it")],
        ),
    )
    assert len(merged.evidence) >= 1
    same = await backend.merge_node(node.id, await _node(NodeType.DECISION, "Ship v1"))
    assert same.id == node.id


async def test_contract_edge_validation_leaves_graph_unchanged(backend):
    stats_before = await backend.get_statistics()
    with pytest.raises(NodeNotFoundError):
        await add_edge_validated(
            backend,
            KnowledgeEdge(source_id="a", target_id="b", edge_type=EdgeType.OWNS),
        )
    assert await backend.get_statistics() == stats_before

    person = await backend.add_node(await _node(NodeType.PERSON, "Ada"))
    topic = await backend.add_node(await _node(NodeType.TOPIC, "Search"))
    with pytest.raises(InvalidEdgeError):
        await add_edge_validated(
            backend,
            KnowledgeEdge(source_id=person.id, target_id=topic.id, edge_type=EdgeType.OWNS),
        )
    assert (await backend.get_statistics())["total_edges"] == 0

    edge = await add_edge_validated(
        backend,
        KnowledgeEdge(source_id=person.id, target_id=topic.id, edge_type=EdgeType.RELATES_TO),
    )
    assert len(await backend.get_edges(person.id)) == 1
    assert await backend.remove_edge(edge.id) is True
    assert await backend.remove_edge(edge.id) is False
    assert (await backend.get_statistics())["total_edges"] == 0


async def test_contract_search_traversal_statistics(backend):
    ada = await backend.add_node(await _node(NodeType.PERSON, "Ada Lovelace"))
    ship = await backend.add_node(await _node(NodeType.DECISION, "Ship v1"))
    await add_edge_validated(
        backend,
        KnowledgeEdge(source_id=ship.id, target_id=ada.id, edge_type=EdgeType.DECIDED_BY),
    )
    assert [n.title for n in await backend.search_nodes("lovelace")] == ["Ada Lovelace"]
    assert await backend.search_nodes("lovelace", node_type=NodeType.TOPIC) == []

    neighbors = await backend.get_neighbors(ship.id)
    assert {n.id for n in neighbors.nodes} == {ship.id, ada.id}
    assert len(neighbors.edges) == 1

    sub = await backend.get_subgraph(ship.id, depth=1)
    assert {n.id for n in sub.nodes} == {ship.id, ada.id}

    stats = await backend.get_statistics()
    assert stats["total_nodes"] == 2
    assert stats["total_edges"] == 1
    assert stats["node_types"]["PERSON"] == 1
    assert stats["contradictions"] == 0
    assert stats["orphan_nodes"] == 0

    listed = await backend.get_all_nodes(node_type=NodeType.PERSON)
    assert [n.title for n in listed] == ["Ada Lovelace"]
    full = await backend.get_full_graph()
    assert len(full.nodes) == 2 and len(full.edges) == 1


async def test_contract_health_and_clear(backend):
    assert await backend.health_check() is True
    await backend.add_node(await _node(NodeType.TOPIC, "T"))
    await backend.clear()
    stats = await backend.get_statistics()
    assert stats["total_nodes"] == 0
    assert stats["total_edges"] == 0
