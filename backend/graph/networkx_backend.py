from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from backend.config import settings
from backend.graph.manager import GraphManager
from backend.models.knowledge_graph import (
    EdgeType,
    GraphData,
    KnowledgeEdge,
    KnowledgeNode,
    NodeType,
)

logger = logging.getLogger(__name__)


class NetworkXGraphManager(GraphManager):
    """In-memory graph backend using NetworkX with JSON persistence."""

    def __init__(self, data_path: str | None = None) -> None:
        self._graph = nx.DiGraph()
        self._lock = asyncio.Lock()
        self._data_path = Path(data_path or settings.NETWORKX_PATH)
        self._load()

    # ── persistence ─────────────────────────────────────────────

    def _load(self) -> None:
        if not self._data_path.exists():
            return
        try:
            raw = json.loads(self._data_path.read_text(encoding="utf-8"))
            for n in raw.get("nodes", []):
                node = KnowledgeNode(**n)
                self._graph.add_node(node.id, **node.model_dump(mode="json"))
            for e in raw.get("edges", []):
                edge = KnowledgeEdge(**e)
                self._graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    key=edge.id,
                    **edge.model_dump(mode="json"),
                )
            logger.info(
                "Loaded %d nodes, %d edges from %s",
                self._graph.number_of_nodes(),
                self._graph.number_of_edges(),
                self._data_path,
            )
        except Exception:
            logger.exception("Failed to load graph data")

    def _save(self) -> None:
        nodes = [self._graph.nodes[n] for n in self._graph.nodes]
        edges = [self._graph.edges[u, v] for u, v in self._graph.edges]
        self._data_path.write_text(
            json.dumps({"nodes": nodes, "edges": edges}, default=str, indent=2),
            encoding="utf-8",
        )

    # ── mutations ───────────────────────────────────────────────

    async def add_node(self, node: KnowledgeNode) -> KnowledgeNode:
        async with self._lock:
            self._graph.add_node(node.id, **node.model_dump(mode="json"))
            self._save()
        return node

    async def add_edge(self, edge: KnowledgeEdge) -> KnowledgeEdge:
        async with self._lock:
            # ensure endpoints exist
            for nid in (edge.source_id, edge.target_id):
                if nid not in self._graph:
                    self._graph.add_node(nid)
            self._graph.add_edge(
                edge.source_id,
                edge.target_id,
                **edge.model_dump(mode="json"),
            )
            self._save()
        return edge

    async def merge_node(self, existing_id: str, new_data: KnowledgeNode) -> KnowledgeNode:
        async with self._lock:
            if existing_id not in self._graph:
                raise ValueError(f"Node {existing_id} not found")
            old = dict(self._graph.nodes[existing_id])
            # merge: prefer non-empty new fields
            merged = {**old}
            for key, val in new_data.model_dump(mode="json").items():
                if val and key not in ("id", "created_at"):
                    merged[key] = val
            merged["updated_at"] = dt.datetime.utcnow().isoformat()
            merged["confidence"] = max(old.get("confidence", 0), new_data.confidence)
            self._graph.nodes[existing_id].update(merged)
            self._save()
        return KnowledgeNode(**merged)

    async def delete_node(self, node_id: str) -> bool:
        async with self._lock:
            if node_id not in self._graph:
                return False
            self._graph.remove_node(node_id)
            self._save()
        return True

    async def update_node(self, node_id: str, patch: KnowledgeNode) -> KnowledgeNode:
        async with self._lock:
            if node_id not in self._graph:
                raise ValueError(f"Node {node_id} not found")
            old = dict(self._graph.nodes[node_id])
            data = patch.model_dump(mode="json")
            data["id"] = node_id
            data["created_at"] = old.get("created_at")
            data["updated_at"] = dt.datetime.utcnow().isoformat()
            self._graph.nodes[node_id].clear()
            self._graph.nodes[node_id].update(data)
            self._save()
        return KnowledgeNode(**self._graph.nodes[node_id])

    async def remove_edge(self, edge_id: str) -> bool:
        async with self._lock:
            for u, v, data in list(self._graph.edges(data=True)):
                if data.get("id") == edge_id:
                    self._graph.remove_edge(u, v)
                    self._save()
                    return True
        return False

    # ── queries ─────────────────────────────────────────────────

    async def get_node(self, node_id: str) -> KnowledgeNode | None:
        if node_id in self._graph:
            return KnowledgeNode(**self._graph.nodes[node_id])
        return None

    async def get_edges(self, node_id: str) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        for u, v, data in self._graph.edges(data=True):
            if u == node_id or v == node_id:
                try:
                    edges.append(KnowledgeEdge(**data))
                except Exception:
                    pass
        return edges

    async def search_nodes(
        self, query: str, node_type: NodeType | None = None, limit: int = 20
    ) -> list[KnowledgeNode]:
        q = query.lower()
        results: list[KnowledgeNode] = []
        for _, data in self._graph.nodes(data=True):
            if node_type and data.get("type") != node_type.value:
                continue
            haystacks = [
                str(data.get("title", "")).lower(),
                str(data.get("content", "")).lower(),
            ]
            metadata = data.get("metadata")
            if isinstance(metadata, dict):
                haystacks.extend(str(v).lower() for v in metadata.values())
            if any(q in hay for hay in haystacks):
                try:
                    results.append(KnowledgeNode(**data))
                except Exception:
                    pass
            if len(results) >= limit:
                break
        return results

    async def get_subgraph(self, center_id: str, depth: int = 2) -> GraphData:
        if center_id not in self._graph:
            return GraphData()
        # BFS to collect nodes within depth
        visited: set[str] = set()
        queue = [(center_id, 0)]
        while queue:
            nid, d = queue.pop(0)
            if nid in visited or d > depth:
                continue
            visited.add(nid)
            for neighbor in list(self._graph.successors(nid)) + list(self._graph.predecessors(nid)):
                if neighbor not in visited:
                    queue.append((neighbor, d + 1))

        nodes = []
        edges = []
        for nid in visited:
            if nid in self._graph.nodes:
                try:
                    nodes.append(KnowledgeNode(**self._graph.nodes[nid]))
                except Exception:
                    pass
        for u, v, data in self._graph.edges(data=True):
            if u in visited and v in visited:
                try:
                    edges.append(KnowledgeEdge(**data))
                except Exception:
                    pass
        return GraphData(nodes=nodes, edges=edges)

    async def get_full_graph(self, limit: int = 500) -> GraphData:
        nodes: list[KnowledgeNode] = []
        for _, data in list(self._graph.nodes(data=True))[:limit]:
            try:
                nodes.append(KnowledgeNode(**data))
            except Exception:
                pass
        edges: list[KnowledgeEdge] = []
        node_ids = {n.id for n in nodes}
        for u, v, data in self._graph.edges(data=True):
            if u in node_ids and v in node_ids:
                try:
                    edges.append(KnowledgeEdge(**data))
                except Exception:
                    pass
        return GraphData(nodes=nodes, edges=edges)

    async def find_similar_nodes(
        self, embedding: list[float], threshold: float = 0.75, limit: int = 10
    ) -> list[tuple[KnowledgeNode, float]]:
        query_vec = np.array(embedding)
        results: list[tuple[KnowledgeNode, float]] = []
        for _, data in self._graph.nodes(data=True):
            emb = data.get("embedding")
            if not emb:
                continue
            node_vec = np.array(emb)
            denom = np.linalg.norm(query_vec) * np.linalg.norm(node_vec)
            if denom == 0:
                continue
            sim = float(np.dot(query_vec, node_vec) / denom)
            if sim >= threshold:
                try:
                    results.append((KnowledgeNode(**data), sim))
                except Exception:
                    pass
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def get_all_nodes(self, node_type: NodeType | None = None) -> list[KnowledgeNode]:
        nodes: list[KnowledgeNode] = []
        for _, data in self._graph.nodes(data=True):
            if node_type and data.get("type") != node_type.value:
                continue
            try:
                nodes.append(KnowledgeNode(**data))
            except Exception:
                pass
        return nodes

    async def get_statistics(self) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            t = data.get("type", "UNKNOWN")
            type_counts[t] = type_counts.get(t, 0) + 1
        edge_type_counts: dict[str, int] = {}
        contradictions = 0
        for _, _, data in self._graph.edges(data=True):
            t = data.get("edge_type", "UNKNOWN")
            edge_type_counts[t] = edge_type_counts.get(t, 0) + 1
            if t == EdgeType.CONTRADICTS.value:
                contradictions += 1
        connected: set[str] = set()
        for u, v in self._graph.edges():
            connected.add(u)
            connected.add(v)
        orphan_nodes = sum(1 for n in self._graph.nodes if n not in connected)
        ownerless_actions = 0
        ownership = {
            EdgeType.OWNS.value,
            EdgeType.DECIDED_BY.value,
            EdgeType.ASSIGNED_TO.value,
            EdgeType.DECIDES.value,
        }
        for nid in self._graph.nodes:
            ntype = self._graph.nodes[nid].get("type")
            if ntype not in (NodeType.ACTION_ITEM.value, NodeType.DECISION.value):
                continue
            owned = any(
                d.get("edge_type") in ownership
                for _, _, d in list(self._graph.in_edges(nid, data=True))
                + list(self._graph.out_edges(nid, data=True))
            )
            if not owned:
                ownerless_actions += 1
        return {
            "total_nodes": self._graph.number_of_nodes(),
            "total_edges": self._graph.number_of_edges(),
            "node_types": type_counts,
            "edge_types": edge_type_counts,
            "orphan_nodes": orphan_nodes,
            "ownerless_actions": ownerless_actions,
            "contradictions": contradictions,
        }

    async def clear(self) -> None:
        async with self._lock:
            self._graph.clear()
            self._save()

    async def health_check(self) -> bool:
        return True
