from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from backend.graph.manager import GraphManager
from backend.models.knowledge_graph import KnowledgeNode, NodeType


class RankedNode(BaseModel):
    node: KnowledgeNode
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class RankingEntry(BaseModel):
    node_id: str
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class RetrievalDebug(BaseModel):
    matched_nodes: int = 0
    expanded_nodes: int = 0
    edges_considered: int = 0
    ranking: list[RankingEntry] = Field(default_factory=list)


class Retriever(Protocol):
    """Seed-node retrieval contract. Graph expansion happens in query service."""

    async def search(
        self,
        graph: GraphManager,
        query: str,
        node_type: NodeType | None = None,
        limit: int = 50,
    ) -> list[RankedNode]: ...


class KeywordRetriever:
    """Deterministic keyword retrieval over titles, content, and metadata."""

    async def search(
        self,
        graph: GraphManager,
        query: str,
        node_type: NodeType | None = None,
        limit: int = 50,
    ) -> list[RankedNode]:
        nodes = await graph.search_nodes(query, node_type=node_type, limit=limit)
        q = query.lower()
        ranked = []
        for node in nodes:
            reasons = ["keyword_match"]
            score = 0.5
            if q in node.title.lower():
                score += 0.4
                reasons.append("title_match")
            if node.type.value in ("DECISION", "ACTION_ITEM", "QUESTION"):
                score += 0.1
                reasons.append("actionable_type")
            ranked.append(RankedNode(node=node, score=round(score, 3), reasons=reasons))
        ranked.sort(key=lambda r: (-r.score, r.node.title))
        return ranked


class EmbeddingRetriever:
    """Semantic retrieval over stored node embeddings (optional)."""

    def __init__(self, embed_texts, threshold: float = 0.5) -> None:
        self.embed_texts = embed_texts
        self.threshold = threshold

    async def search(
        self,
        graph: GraphManager,
        query: str,
        node_type: NodeType | None = None,
        limit: int = 50,
    ) -> list[RankedNode]:
        vectors = await self.embed_texts([query])
        if not vectors:
            return []
        hits = await graph.find_similar_nodes(vectors[0], threshold=self.threshold, limit=limit)
        ranked = []
        for node, score in hits:
            if node_type is not None and node.type != node_type:
                continue
            ranked.append(RankedNode(node=node, score=round(score, 3), reasons=["semantic_match"]))
        return ranked


class HybridRetriever:
    """Keyword ∪ semantic union with graph-relevance bonuses.

    Weights are configurable; without an embedder it degrades gracefully
    to keyword retrieval with graph bonuses.
    """

    def __init__(
        self,
        embed_texts=None,
        keyword_weight: float = 0.6,
        semantic_weight: float = 0.4,
        threshold: float = 0.5,
    ) -> None:
        self.keyword = KeywordRetriever()
        self.semantic = EmbeddingRetriever(embed_texts, threshold) if embed_texts else None
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight

    async def search(
        self,
        graph: GraphManager,
        query: str,
        node_type: NodeType | None = None,
        limit: int = 50,
    ) -> list[RankedNode]:
        combined: dict[str, RankedNode] = {}
        for ranked in await self.keyword.search(graph, query, node_type, limit):
            ranked.score = round(ranked.score * self.keyword_weight, 3)
            combined[ranked.node.id] = ranked
        if self.semantic is not None:
            try:
                sem = await self.semantic.search(graph, query, node_type, limit)
            except Exception:  # noqa: BLE001 — semantic is best-effort, keyword stands
                sem = []
            for ranked in sem:
                if ranked.node.id in combined:
                    existing = combined[ranked.node.id]
                    existing.score = round(existing.score + ranked.score * self.semantic_weight, 3)
                    existing.reasons = sorted(set(existing.reasons + ranked.reasons))
                else:
                    ranked.score = round(ranked.score * self.semantic_weight, 3)
                    combined[ranked.node.id] = ranked
        # Graph relevance: bonus for candidates connected to other candidates.
        ids = set(combined)
        for ranked in combined.values():
            try:
                edges = await graph.get_edges(ranked.node.id)
            except Exception:  # noqa: BLE001 — bonus skipped when edges unreadable
                edges = []
            if any(e.source_id in ids or e.target_id in ids for e in edges):
                ranked.score = round(ranked.score + 0.1, 3)
                if "graph_connected" not in ranked.reasons:
                    ranked.reasons.append("graph_connected")
        ordered = sorted(combined.values(), key=lambda r: (-r.score, r.node.title))
        return ordered[:limit]
