from __future__ import annotations

from collections.abc import Awaitable, Callable

from backend.graph.manager import GraphManager
from backend.models.knowledge_graph import (
    GraphQuery,
    GraphQueryResult,
    KnowledgeEdge,
    KnowledgeNode,
    NodeType,
)

# Async callable taking (system_prompt, user_content) and returning plain text.
CompleteText = Callable[[str, str], Awaitable[str]]

ANSWER_SYSTEM = """\
You answer questions using ONLY the knowledge-graph context below.
Each line is a node or an edge. If the context does not contain the answer, \
say so in one sentence. Keep the answer under 100 words.
"""


def make_openai_answer(model: str, api_key: str) -> CompleteText:
    """Build the production CompleteText backed by OpenAI chat completions."""

    async def complete(system: str, user: str) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    return complete


def format_context(nodes: list[KnowledgeNode], edges: list[KnowledgeEdge]) -> str:
    lines = [
        f"{n.type.value} {n.title!r}: {n.content} [confidence {n.confidence}]" for n in nodes[:10]
    ]
    names = {n.id: n.title for n in nodes}
    for e in edges[:30]:
        src = names.get(e.source_id, e.source_id)
        tgt = names.get(e.target_id, e.target_id)
        ev = f" ({e.evidence})" if e.evidence else ""
        lines.append(f"{src} --{e.edge_type.value}--> {tgt}{ev}")
    return "\n".join(lines)


async def answer_query(
    graph: GraphManager,
    query: GraphQuery,
    answer_text: CompleteText | None = None,
) -> GraphQueryResult:
    """Keyword retrieval over the graph, plus an optional LLM answer step.

    Honors an optional ``{"node_type": "<NAME>"}`` filter. Without
    ``answer_text`` the result is retrieval-only (``answer`` is None).
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

    answer: str | None = None
    if answer_text is not None and nodes:
        ctx = format_context(nodes, edges)
        answer = await answer_text(ANSWER_SYSTEM, f"Question: {query.query}\n\nContext:\n{ctx}")
        confidence = max(confidence, 0.75)
    return GraphQueryResult(nodes=nodes, edges=edges, answer=answer, confidence=confidence)
