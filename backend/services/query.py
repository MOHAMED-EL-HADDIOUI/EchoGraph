from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from backend import obs
from backend.graph.manager import GraphManager
from backend.models.knowledge_graph import (
    EdgeType,
    EvidenceItem,
    GraphQuery,
    GraphQueryResult,
    KnowledgeEdge,
    KnowledgeNode,
    NodeType,
    Verdict,
)

logger = logging.getLogger(__name__)

# Async callable taking (system_prompt, user_content) and returning plain text.
CompleteText = Callable[[str, str], Awaitable[str]]

ANSWER_SYSTEM = """\
You answer questions using ONLY the knowledge-graph context below.
Each line is a node or an edge. If the context does not contain the answer, \
say so in one sentence and do not guess.
Cite every factual claim with the exact node title in brackets, e.g. [Ship v1].
Keep the answer under 100 words.
"""


def make_openai_answer(model: str, api_key: str) -> CompleteText:
    """Build the production CompleteText backed by OpenAI chat completions."""

    async def complete(system: str, user: str) -> str:
        from openai import AsyncOpenAI

        timer = obs.Timer()
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        usage = getattr(resp, "usage", None)
        obs.log_event(
            logger,
            "llm.answer",
            model=model,
            latency_ms=round(timer.elapsed_ms(), 1),
            usage=usage.model_dump() if usage is not None else None,
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
    Verdict distinguishes supported / uncertain / contradictory contexts;
    ``citations`` resolves ``[Title]`` references in the answer to node IDs;
    ``evidence`` lists citable provenance when ``include_evidence`` is set.
    """
    node_type: NodeType | None = None
    if query.filters and isinstance(query.filters.get("node_type"), str):
        try:
            node_type = NodeType(query.filters["node_type"])
        except ValueError:
            node_type = None

    timer = obs.Timer()
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

    if not nodes:
        verdict = Verdict.UNCERTAIN
    elif any(e.edge_type == EdgeType.CONTRADICTS for e in edges):
        verdict = Verdict.CONTRADICTORY
    else:
        verdict = Verdict.SUPPORTED

    answer: str | None = None
    citations: list[str] = []
    if answer_text is not None and nodes:
        ctx = format_context(nodes, edges)
        answer = await answer_text(ANSWER_SYSTEM, f"Question: {query.query}\n\nContext:\n{ctx}")
        citations = resolve_citations(answer, nodes)
        confidence = max(confidence, 0.75)

    evidence: list[EvidenceItem] = []
    if query.include_evidence:
        evidence = build_evidence(nodes, edges)
    obs.log_event(
        logger,
        "query.finish",
        query_chars=len(query.query),
        nodes=len(nodes),
        edges=len(edges),
        verdict=verdict.value,
        answered=answer is not None,
        latency_ms=round(timer.elapsed_ms(), 1),
    )
    return GraphQueryResult(
        nodes=nodes,
        edges=edges,
        answer=answer,
        confidence=confidence,
        verdict=verdict,
        citations=citations,
        evidence=evidence,
    )


def resolve_citations(answer: str, nodes: list[KnowledgeNode]) -> list[str]:
    """Map ``[Title]`` references in an answer to node IDs (case-insensitive)."""
    import re

    mentioned = {m.group(1).strip().lower() for m in re.finditer(r"\[([^\]]+)\]", answer)}
    return [n.id for n in nodes if n.title.lower() in mentioned]


def build_evidence(nodes: list[KnowledgeNode], edges: list[KnowledgeEdge]) -> list[EvidenceItem]:
    """Flatten node/edge provenance into citable evidence items."""
    items: list[EvidenceItem] = []
    conf_by_id = {n.id: n.confidence for n in nodes}
    for n in nodes:
        first = n.evidence[0] if n.evidence else None
        items.append(
            EvidenceItem(
                kind="node",
                ref_id=n.id,
                label=f"{n.type.value}: {n.title}",
                ingestion_id=(first.ingestion_id if first else "") or n.source_ref or "",
                quote=(first.quote if first else "") or n.content,
                confidence=n.confidence,
            )
        )
    names = {n.id: n.title for n in nodes}
    for e in edges:
        src = names.get(e.source_id, e.source_id)
        tgt = names.get(e.target_id, e.target_id)
        endpoint_conf = max(conf_by_id.get(e.source_id, 0.0), conf_by_id.get(e.target_id, 0.0))
        items.append(
            EvidenceItem(
                kind="edge",
                ref_id=e.id,
                label=f"{src} --{e.edge_type.value}--> {tgt}",
                ingestion_id=e.ingestion_id or "",
                quote=e.evidence or "",
                confidence=endpoint_conf,
            )
        )
    return items
