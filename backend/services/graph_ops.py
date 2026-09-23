from __future__ import annotations

from backend.graph.manager import GraphManager
from backend.graph.schema import validate_edge
from backend.models.knowledge_graph import KnowledgeEdge


class NodeNotFoundError(ValueError):
    """Raised when an edge endpoint does not exist."""


class InvalidEdgeError(ValueError):
    """Raised when the (source, edge, target) triple violates VALID_EDGES."""


async def add_edge_validated(graph: GraphManager, edge: KnowledgeEdge) -> KnowledgeEdge:
    """Add an edge after checking endpoints exist and the triple is valid.

    Single choke point for edge writes — routes and background workers
    must use this instead of calling ``graph.add_edge`` directly.
    """
    source = await graph.get_node(edge.source_id)
    target = await graph.get_node(edge.target_id)
    if source is None or target is None:
        raise NodeNotFoundError("Source or target node not found")
    if not validate_edge(source.type, edge.edge_type, target.type):
        raise InvalidEdgeError(
            f"Invalid edge: {source.type.value} -{edge.edge_type.value}-> {target.type.value}"
        )
    return await graph.add_edge(edge)
