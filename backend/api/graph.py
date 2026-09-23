from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.deps import get_graph_manager
from backend.graph.manager import GraphManager
from backend.models.knowledge_graph import (
    GraphData,
    GraphQuery,
    GraphQueryResult,
    KnowledgeEdge,
    KnowledgeNode,
    NodeType,
)
from backend.services.graph_ops import (
    InvalidEdgeError,
    NodeNotFoundError,
    add_edge_validated,
)
from backend.services.query import answer_query

router = APIRouter(prefix="/graph", tags=["graph"])


@router.post("/nodes", response_model=KnowledgeNode)
async def create_node(
    node: KnowledgeNode, graph: GraphManager = Depends(get_graph_manager)
) -> KnowledgeNode:
    return await graph.add_node(node)


@router.get("/nodes/{node_id}", response_model=KnowledgeNode)
async def read_node(
    node_id: str, graph: GraphManager = Depends(get_graph_manager)
) -> KnowledgeNode:
    node = await graph.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@router.delete("/nodes/{node_id}")
async def remove_node(
    node_id: str, graph: GraphManager = Depends(get_graph_manager)
) -> dict[str, bool]:
    deleted = await graph.delete_node(node_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Node not found")
    return {"deleted": True}


@router.post("/nodes/{node_id}/merge", response_model=KnowledgeNode)
async def merge_into_node(
    node_id: str, new_data: KnowledgeNode, graph: GraphManager = Depends(get_graph_manager)
) -> KnowledgeNode:
    try:
        return await graph.merge_node(node_id, new_data)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/nodes", response_model=list[KnowledgeNode])
async def list_nodes(
    node_type: NodeType | None = None,
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    graph: GraphManager = Depends(get_graph_manager),
) -> list[KnowledgeNode]:
    nodes = await graph.get_all_nodes(node_type=node_type)
    return nodes[offset : offset + limit]


@router.get("/search", response_model=list[KnowledgeNode])
async def search(
    q: str,
    node_type: NodeType | None = None,
    limit: int = Query(default=20, le=100),
    graph: GraphManager = Depends(get_graph_manager),
) -> list[KnowledgeNode]:
    return await graph.search_nodes(q, node_type=node_type, limit=limit)


@router.post("/query", response_model=GraphQueryResult)
async def query_graph(
    query: GraphQuery, graph: GraphManager = Depends(get_graph_manager)
) -> GraphQueryResult:
    return await answer_query(graph, query)


@router.post("/edges", response_model=KnowledgeEdge)
async def create_edge(
    edge: KnowledgeEdge, graph: GraphManager = Depends(get_graph_manager)
) -> KnowledgeEdge:
    try:
        return await add_edge_validated(graph, edge)
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidEdgeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/nodes/{node_id}/edges", response_model=list[KnowledgeEdge])
async def read_edges(
    node_id: str, graph: GraphManager = Depends(get_graph_manager)
) -> list[KnowledgeEdge]:
    node = await graph.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return await graph.get_edges(node_id)


@router.get("/nodes/{node_id}/subgraph", response_model=GraphData)
async def read_subgraph(
    node_id: str,
    depth: int = Query(default=2, ge=1, le=5),
    graph: GraphManager = Depends(get_graph_manager),
) -> GraphData:
    node = await graph.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return await graph.get_subgraph(node_id, depth=depth)


@router.get("", response_model=GraphData)
async def read_full_graph(
    limit: int = Query(default=500, le=2000),
    graph: GraphManager = Depends(get_graph_manager),
) -> GraphData:
    return await graph.get_full_graph(limit=limit)


@router.get("/statistics", response_model=dict)
async def read_statistics(
    graph: GraphManager = Depends(get_graph_manager),
) -> dict:
    return await graph.get_statistics()
