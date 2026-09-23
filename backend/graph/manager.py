from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.models.knowledge_graph import (
    GraphData,
    KnowledgeEdge,
    KnowledgeNode,
    NodeType,
)


class GraphManager(ABC):
    """Abstract interface for knowledge graph storage backends."""

    @abstractmethod
    async def add_node(self, node: KnowledgeNode) -> KnowledgeNode: ...

    @abstractmethod
    async def add_edge(self, edge: KnowledgeEdge) -> KnowledgeEdge: ...

    @abstractmethod
    async def get_node(self, node_id: str) -> KnowledgeNode | None: ...

    @abstractmethod
    async def get_edges(self, node_id: str) -> list[KnowledgeEdge]: ...

    @abstractmethod
    async def merge_node(self, existing_id: str, new_data: KnowledgeNode) -> KnowledgeNode: ...

    @abstractmethod
    async def delete_node(self, node_id: str) -> bool: ...

    @abstractmethod
    async def search_nodes(
        self, query: str, node_type: NodeType | None = None, limit: int = 20
    ) -> list[KnowledgeNode]: ...

    @abstractmethod
    async def get_subgraph(self, center_id: str, depth: int = 2) -> GraphData: ...

    @abstractmethod
    async def get_full_graph(self, limit: int = 500) -> GraphData: ...

    @abstractmethod
    async def find_similar_nodes(
        self, embedding: list[float], threshold: float = 0.75, limit: int = 10
    ) -> list[tuple[KnowledgeNode, float]]: ...

    @abstractmethod
    async def get_all_nodes(self, node_type: NodeType | None = None) -> list[KnowledgeNode]: ...

    @abstractmethod
    async def get_statistics(self) -> dict[str, Any]: ...


def create_graph_manager(backend: str = "networkx") -> GraphManager:
    """Factory: return the requested backend implementation."""
    if backend == "neo4j":
        from backend.graph.neo4j_backend import Neo4jGraphManager

        return Neo4jGraphManager()
    from backend.graph.networkx_backend import NetworkXGraphManager

    return NetworkXGraphManager()
