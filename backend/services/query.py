from __future__ import annotations

from backend.graph.manager import GraphManager
from backend.models.knowledge_graph import GraphQuery, GraphQueryResult, NodeType


async def answer_query(graph: GraphManager, query: GraphQuery) -> GraphQueryResult:
    """Keyword query over the graph (v0: no LLM rerank/answer yet).

    Honors an optional ``{"node_type": "<NAME>"}`` filter. Confidence is a
    simple hit heuristic until the LLM answer step lands.
    """
    node_type: NodeType | None = None
    if query.filters and isinstance(query.filters.get("node_type"), str):
        try:
            node_type = NodeType(query.filters["node_type"])
        except ValueError:
            node_type = None

    nodes = await graph.search_nodes(query.query, node_type=node_type, limit=query.limit)
    node_ids = {n.id for n in nodes}
    edges = []
    seen: set[str] = set()
    for node in nodes:
        for edge in await graph.get_edges(node.id):
            if edge.id in seen:
                continue
            seen.add(edge.id)
            if edge.source_id in node_ids and edge.target_id in node_ids:
                edges.append(edge)
    confidence = min(0.9, 0.3 + 0.1 * len(nodes)) if nodes else 0.0
    return GraphQueryResult(nodes=nodes, edges=edges, answer=None, confidence=confidence)
