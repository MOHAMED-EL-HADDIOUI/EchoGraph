from __future__ import annotations

import json
import logging
from typing import Any

from backend.config import settings
from backend.graph.manager import GraphManager
from backend.graph.schema import NEO4J_CONSTRAINTS
from backend.models.knowledge_graph import (
    GraphData,
    KnowledgeEdge,
    KnowledgeNode,
    NodeType,
)

logger = logging.getLogger(__name__)


class Neo4jGraphManager(GraphManager):
    """Production graph backend using Neo4j."""

    def __init__(self) -> None:
        from neo4j import AsyncGraphDatabase

        self._driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        )

    async def close(self) -> None:
        await self._driver.close()

    async def init_constraints(self) -> None:
        async with self._driver.session() as session:
            for stmt in NEO4J_CONSTRAINTS:
                await session.run(stmt)

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _record_to_node(record: dict) -> KnowledgeNode:
        # Keep embedding: find_similar_nodes depends on reading it back.
        return KnowledgeNode(**Neo4jGraphManager._deserialize_props(dict(record)))

    @staticmethod
    def _record_to_edge(record: dict) -> KnowledgeEdge:
        return KnowledgeEdge(**Neo4jGraphManager._deserialize_props(dict(record)))

    @staticmethod
    def _serialize_props(data: dict) -> dict:
        """Flatten non-primitive props: Neo4j rejects nested maps/lists of maps."""
        if isinstance(data.get("metadata"), dict):
            data["metadata"] = json.dumps(data["metadata"])
        if isinstance(data.get("evidence"), list):
            data["evidence"] = json.dumps(data["evidence"], default=str)
        return data

    @staticmethod
    def _deserialize_props(data: dict) -> dict:
        if isinstance(data.get("metadata"), str):
            try:
                data["metadata"] = json.loads(data["metadata"])
            except ValueError:
                data["metadata"] = {}
        if isinstance(data.get("evidence"), str):
            try:
                data["evidence"] = json.loads(data["evidence"])
            except ValueError:
                data["evidence"] = []
        return data

    # ── mutations ───────────────────────────────────────────────

    async def add_node(self, node: KnowledgeNode) -> KnowledgeNode:
        # NOTE: Cypher 5 removed CREATE (n:Label $props); use SET.
        query = """
        CREATE (n:KnowledgeNode)
        SET n = $props
        RETURN n
        """
        async with self._driver.session() as session:
            result = await session.run(
                query, props=self._serialize_props(node.model_dump(mode="json"))
            )
            await result.consume()
        return node

    async def add_edge(self, edge: KnowledgeEdge) -> KnowledgeEdge:
        query = """
        MATCH (a:KnowledgeNode {id: $src}), (b:KnowledgeNode {id: $tgt})
        CREATE (a)-[r:EDGE]->(b)
        SET r = $props
        RETURN r
        """
        async with self._driver.session() as session:
            result = await session.run(
                query,
                src=edge.source_id,
                tgt=edge.target_id,
                props=self._serialize_props(edge.model_dump(mode="json")),
            )
            await result.consume()
        return edge

    async def merge_node(self, existing_id: str, new_data: KnowledgeNode) -> KnowledgeNode:
        query = """
        MATCH (n:KnowledgeNode {id: $nid})
        SET n += $props
        RETURN n
        """
        props = self._serialize_props(
            {
                k: v
                for k, v in new_data.model_dump(mode="json").items()
                if v and k not in ("id", "created_at")
            }
        )
        async with self._driver.session() as session:
            result = await session.run(query, nid=existing_id, props=props)
            record = await result.single()
            if record:
                return self._record_to_node(dict(record["n"]))
        raise ValueError(f"Node {existing_id} not found")

    async def delete_node(self, node_id: str) -> bool:
        query = "MATCH (n:KnowledgeNode {id: $nid}) DETACH DELETE n RETURN count(n) as c"
        async with self._driver.session() as session:
            result = await session.run(query, nid=node_id)
            record = await result.single()
            return bool(record and record["c"] > 0)

    # ── queries ─────────────────────────────────────────────────

    async def get_node(self, node_id: str) -> KnowledgeNode | None:
        query = "MATCH (n:KnowledgeNode {id: $nid}) RETURN n"
        async with self._driver.session() as session:
            result = await session.run(query, nid=node_id)
            record = await result.single()
            if record:
                return self._record_to_node(dict(record["n"]))
        return None

    async def get_edges(self, node_id: str) -> list[KnowledgeEdge]:
        query = """
        MATCH (a:KnowledgeNode)-[r:EDGE]-(b:KnowledgeNode)
        WHERE a.id = $nid
        RETURN r
        """
        edges: list[KnowledgeEdge] = []
        async with self._driver.session() as session:
            result = await session.run(query, nid=node_id)
            async for record in result:
                edges.append(self._record_to_edge(dict(record["r"])))
        return edges

    async def search_nodes(
        self, query: str, node_type: NodeType | None = None, limit: int = 20
    ) -> list[KnowledgeNode]:
        type_clause = "AND n.type = $ntype" if node_type else ""
        cypher = f"""
        MATCH (n:KnowledgeNode)
        WHERE (toLower(n.title) CONTAINS toLower($q)
               OR toLower(n.content) CONTAINS toLower($q))
        {type_clause}
        RETURN n LIMIT $lim
        """
        params: dict[str, Any] = {"q": query, "lim": limit}
        if node_type:
            params["ntype"] = node_type.value
        nodes: list[KnowledgeNode] = []
        async with self._driver.session() as session:
            result = await session.run(cypher, **params)
            async for record in result:
                nodes.append(self._record_to_node(dict(record["n"])))
        return nodes

    async def get_subgraph(self, center_id: str, depth: int = 2) -> GraphData:
        # NOTE: depth is interpolated, not bound: Cypher has no bind parameter
        # for variable-length hop bounds. Safe because the API validates
        # depth as int in 1..5 before it reaches here.
        node_query = f"""
        MATCH (center:KnowledgeNode {{id: $cid}})
        CALL apoc.path.subgraphNodes(center, {{maxLevel: {depth}}})
        YIELD node RETURN node
        """
        # Fallback without APOC
        fallback_query = f"""
        MATCH path = (center:KnowledgeNode {{id: $cid}})-[*1..{depth}]-(n:KnowledgeNode)
        WITH collect(DISTINCT n) + [center] as allNodes
        UNWIND allNodes as node
        RETURN DISTINCT node
        """
        nodes: list[KnowledgeNode] = []
        node_ids: set[str] = set()
        async with self._driver.session() as session:
            try:
                result = await session.run(node_query, cid=center_id)
            except Exception:
                result = await session.run(fallback_query, cid=center_id)
            async for record in result:
                n = self._record_to_node(dict(record["node"]))
                nodes.append(n)
                node_ids.add(n.id)

        # Get edges between collected nodes
        edges: list[KnowledgeEdge] = []
        if node_ids:
            edge_query = """
            MATCH (a:KnowledgeNode)-[r:EDGE]->(b:KnowledgeNode)
            WHERE a.id IN $ids AND b.id IN $ids
            RETURN r
            """
            async with self._driver.session() as session:
                result = await session.run(edge_query, ids=list(node_ids))
                async for record in result:
                    edges.append(self._record_to_edge(dict(record["r"])))
        return GraphData(nodes=nodes, edges=edges)

    async def get_full_graph(self, limit: int = 500) -> GraphData:
        nodes: list[KnowledgeNode] = []
        async with self._driver.session() as session:
            result = await session.run("MATCH (n:KnowledgeNode) RETURN n LIMIT $lim", lim=limit)
            async for record in result:
                nodes.append(self._record_to_node(dict(record["n"])))

        node_ids = [n.id for n in nodes]
        edges: list[KnowledgeEdge] = []
        if node_ids:
            async with self._driver.session() as session:
                result = await session.run(
                    "MATCH (a)-[r:EDGE]->(b) WHERE a.id IN $ids AND b.id IN $ids RETURN r",
                    ids=node_ids,
                )
                async for record in result:
                    edges.append(self._record_to_edge(dict(record["r"])))
        return GraphData(nodes=nodes, edges=edges)

    async def find_similar_nodes(
        self, embedding: list[float], threshold: float = 0.75, limit: int = 10
    ) -> list[tuple[KnowledgeNode, float]]:
        # Pure-Python cosine over stored embeddings (no GDS/vector-index dependency).
        all_nodes = await self.get_all_nodes()
        import numpy as np

        query_vec = np.array(embedding)
        results: list[tuple[KnowledgeNode, float]] = []
        for node in all_nodes:
            if not node.embedding:
                continue
            node_vec = np.array(node.embedding)
            denom = np.linalg.norm(query_vec) * np.linalg.norm(node_vec)
            if denom == 0:
                continue
            sim = float(np.dot(query_vec, node_vec) / denom)
            if sim >= threshold:
                results.append((node, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def get_all_nodes(self, node_type: NodeType | None = None) -> list[KnowledgeNode]:
        if node_type:
            query = "MATCH (n:KnowledgeNode {type: $t}) RETURN n"
            params = {"t": node_type.value}
        else:
            query = "MATCH (n:KnowledgeNode) RETURN n"
            params = {}
        nodes: list[KnowledgeNode] = []
        async with self._driver.session() as session:
            result = await session.run(query, **params)
            async for record in result:
                nodes.append(self._record_to_node(dict(record["n"])))
        return nodes

    async def get_statistics(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        async with self._driver.session() as session:
            r = await session.run("MATCH (n:KnowledgeNode) RETURN count(n) as c")
            rec = await r.single()
            stats["total_nodes"] = rec["c"] if rec else 0

            r = await session.run("MATCH ()-[r:EDGE]->() RETURN count(r) as c")
            rec = await r.single()
            stats["total_edges"] = rec["c"] if rec else 0

            r = await session.run("MATCH (n:KnowledgeNode) RETURN n.type as t, count(*) as c")
            stats["node_types"] = {rec["t"]: rec["c"] async for rec in r}
        return stats
