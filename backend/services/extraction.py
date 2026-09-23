from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field, ValidationError

from backend.config import settings
from backend.graph.manager import GraphManager
from backend.models.knowledge_graph import EdgeType, KnowledgeEdge, KnowledgeNode, NodeType
from backend.services.graph_ops import InvalidEdgeError, NodeNotFoundError, add_edge_validated

logger = logging.getLogger(__name__)

# Async callable taking (system_prompt, user_content) and returning parsed JSON.
CompleteJson = Callable[[str, str], Awaitable[dict]]

# Async callable taking a list of texts and returning one vector per text.
EmbedTexts = Callable[[list[str]], Awaitable[list[list[float]]]]

SYSTEM_PROMPT = """\
You extract a knowledge graph from organizational text (meetings, chat, email).
Return JSON only, exactly: {"nodes": [...], "edges": [...]}.
Node: {"type": one of DECISION RATIONALE QUESTION PERSON TOPIC DOCUMENT ACTION_ITEM,
"title": short label, "content": one-sentence detail, "confidence": 0.0-1.0}.
Edge: {"source_title": exact title of a node above, "target_title": exact title of a \
node above, "edge_type": one of DECIDED_BY RELATES_TO CONTRADICTS SUPERSEDES OWNS \
BLOCKED_BY REFERENCES DERIVED_FROM ANSWERS, "evidence": short quote or ""}.
Prefer few high-confidence nodes over many guesses. Every edge endpoint must match a \
node title exactly.
"""

MAX_ERRORS = 20


class ExtractedNode(BaseModel):
    type: NodeType
    title: str
    content: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ExtractedEdge(BaseModel):
    source_title: str
    target_title: str
    edge_type: EdgeType
    evidence: str = ""


class ExtractedGraph(BaseModel):
    nodes: list[ExtractedNode] = Field(default_factory=list)
    edges: list[ExtractedEdge] = Field(default_factory=list)


def chunk_text(content: str, max_chars: int = 2000) -> list[str]:
    """Split on blank lines, packing paragraphs into chunks of ~max_chars."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for para in (p.strip() for p in content.split("\n\n")):
        if not para:
            continue
        if size + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def make_openai_complete(model: str, api_key: str) -> CompleteJson:
    """Build the production CompleteJson backed by OpenAI chat completions."""

    async def complete(system: str, user: str) -> dict:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content or "{}")

    return complete


def make_openai_embeddings(model: str, api_key: str) -> EmbedTexts:
    """Build the production EmbedTexts backed by OpenAI embeddings."""

    async def embed(texts: list[str]) -> list[list[float]]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        resp = await client.embeddings.create(model=model, input=texts)
        return [list(d.embedding) for d in resp.data]

    return embed


class ExtractionResult(BaseModel):
    """Outcome of a single extraction run, including created objects."""

    nodes_created: int = 0
    edges_created: int = 0
    errors: list[str] = Field(default_factory=list)
    nodes: list[KnowledgeNode] = Field(default_factory=list)
    edges: list[KnowledgeEdge] = Field(default_factory=list)


async def run_extraction(
    graph: GraphManager,
    content: str,
    complete_json: CompleteJson,
    source_type: str | None = None,
    embed_texts: EmbedTexts | None = None,
) -> ExtractionResult:
    """Extract nodes/edges from content into the graph.

    Never raises on bad LLM output: problems are collected into errors
    (capped at MAX_ERRORS). When ``embed_texts`` is given, new nodes are
    embedded and deduplicated by cosine similarity before exact-title matching
    falls through to ``add_node``.
    """
    result = ExtractionResult()

    def record(msg: str) -> None:
        if len(result.errors) < MAX_ERRORS:
            result.errors.append(msg)
        logger.warning("extraction: %s", msg)

    nodes_created = 0
    edges_created = 0
    created_nodes: list[KnowledgeNode] = []
    created_edges: list[KnowledgeEdge] = []
    by_title: dict[tuple[str, str], KnowledgeNode] = {}
    for existing in await graph.get_all_nodes():
        by_title.setdefault((existing.type.value, existing.title.lower()), existing)

    for chunk in chunk_text(content):
        try:
            raw = await asyncio.wait_for(
                complete_json(SYSTEM_PROMPT, chunk),
                timeout=settings.EXTRACTION_TIMEOUT_S,
            )
            extracted = ExtractedGraph.model_validate(raw)
        except TimeoutError:
            record(f"chunk timed out after {settings.EXTRACTION_TIMEOUT_S}s, skipped")
            continue
        except ValidationError as exc:
            record(f"invalid extraction payload: {exc}")
            continue

        for item in extracted.nodes:
            key = (item.type.value, item.title.lower())
            if key in by_title:
                try:
                    by_title[key] = await graph.merge_node(
                        by_title[key].id,
                        KnowledgeNode(
                            type=item.type,
                            title=item.title,
                            content=item.content or by_title[key].content,
                            source_type=source_type,
                            confidence=item.confidence,
                        ),
                    )
                except ValueError as exc:
                    record(f"merge failed for {item.title!r}: {exc}")
                continue
            if embed_texts is not None:
                vec: list[float] | None = None
                try:
                    vec = (await embed_texts([f"{item.title}\n{item.content}"]))[0]
                    similar = await graph.find_similar_nodes(
                        vec,
                        threshold=settings.EMBEDDING_DEDUP_THRESHOLD,
                        limit=1,
                    )
                except Exception as exc:  # noqa: BLE001 — embedding is best-effort
                    record(f"embedding dedup skipped for {item.title!r}: {exc}")
                    similar = []
                if similar:
                    match, _score = similar[0]
                    try:
                        merged = await graph.merge_node(
                            match.id,
                            KnowledgeNode(
                                type=item.type,
                                title=item.title,
                                content=item.content or match.content,
                                source_type=source_type,
                                confidence=item.confidence,
                            ),
                        )
                        by_title[key] = merged
                        by_title.setdefault((merged.type.value, merged.title.lower()), merged)
                    except ValueError as exc:
                        record(f"merge failed for {item.title!r}: {exc}")
                    continue
                node = await graph.add_node(
                    KnowledgeNode(
                        type=item.type,
                        title=item.title,
                        content=item.content,
                        source_type=source_type,
                        confidence=item.confidence,
                        embedding=vec,
                    )
                )
                by_title[key] = node
                nodes_created += 1
                created_nodes.append(node)
                continue
            node = await graph.add_node(
                KnowledgeNode(
                    type=item.type,
                    title=item.title,
                    content=item.content,
                    source_type=source_type,
                    confidence=item.confidence,
                )
            )
            by_title[key] = node
            nodes_created += 1
            created_nodes.append(node)

        for item in extracted.edges:
            src = next(
                (n for (t, ti), n in by_title.items() if ti == item.source_title.lower()), None
            )
            tgt = next(
                (n for (t, ti), n in by_title.items() if ti == item.target_title.lower()), None
            )
            if src is None or tgt is None:
                record(
                    f"edge references unknown node: {item.source_title!r} -> {item.target_title!r}"
                )
                continue
            try:
                created = await add_edge_validated(
                    graph,
                    KnowledgeEdge(
                        source_id=src.id,
                        target_id=tgt.id,
                        edge_type=item.edge_type,
                        evidence=item.evidence or None,
                        created_at=dt.datetime.utcnow(),
                    ),
                )
                edges_created += 1
                created_edges.append(created)
            except (InvalidEdgeError, NodeNotFoundError) as exc:
                record(f"edge skipped: {exc}")

    result.nodes_created = nodes_created
    result.edges_created = edges_created
    result.nodes = created_nodes
    result.edges = created_edges
    return result
