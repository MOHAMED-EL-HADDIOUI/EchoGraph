from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from backend import obs
from backend.graph.manager import GraphManager
from backend.models.knowledge_graph import EdgeType, NodeType
from backend.services.extraction import run_extraction
from backend.services.notifications import OWNERLESS_TYPES, OWNERSHIP_EDGES

logger = logging.getLogger(__name__)


class ExpectedNode(BaseModel):
    type: NodeType
    title: str


class ExpectedEdge(BaseModel):
    source: str
    target: str
    edge_type: EdgeType


class CaseExpectations(BaseModel):
    nodes: list[ExpectedNode] = Field(default_factory=list)
    edges: list[ExpectedEdge] = Field(default_factory=list)
    ownerless: list[str] = Field(default_factory=list)
    contradictions: list[list[str]] = Field(default_factory=list)


class EvalCase(BaseModel):
    id: str
    description: str = ""
    transcript: str
    recorded: list[dict] = Field(default_factory=list)
    expected: CaseExpectations = Field(default_factory=CaseExpectations)


class CaseMetrics(BaseModel):
    node_precision: float = 0.0
    node_recall: float = 0.0
    edge_precision: float = 0.0
    edge_recall: float = 0.0
    owner_precision: float = 0.0
    owner_recall: float = 0.0
    decision_recall: float = 0.0
    contradiction_recall: float = 0.0
    fabricated: list[str] = Field(default_factory=list)
    unexpected_edges: list[str] = Field(default_factory=list)
    unsupported_quotes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    passed: bool = False


class CaseResult(BaseModel):
    case_id: str
    metrics: CaseMetrics


class EvalReport(BaseModel):
    cases: list[CaseResult] = Field(default_factory=list)
    totals: dict[str, float] = Field(default_factory=dict)
    passed: bool = False
    provider: str = "fake"
    model: str = ""
    extraction_prompt_version: str = ""
    answer_prompt_version: str = ""
    timestamp: str = ""
    runtime_s: float = 0.0


def load_cases(cases_dir: str | Path) -> list[EvalCase]:
    cases = []
    for path in sorted(Path(cases_dir).glob("*.json")):
        cases.append(EvalCase.model_validate(json.loads(path.read_text(encoding="utf-8"))))
    return cases


def _pr(found: set, expected: set) -> tuple[float, float]:
    precision = len(found & expected) / len(found) if found else 1.0
    recall = len(found & expected) / len(expected) if expected else 1.0
    return precision, recall


async def run_case(graph: GraphManager, case: EvalCase) -> CaseResult:
    """Run one case with recorded LLM payloads (no network)."""
    payloads = list(case.recorded)

    async def recorded_complete(system: str, user: str) -> dict:
        if not payloads:
            return {"nodes": [], "edges": []}
        return payloads.pop(0)

    timer = obs.Timer()
    result = await run_extraction(
        graph, case.transcript, recorded_complete, ingestion_id=f"eval-{case.id}"
    )
    metrics = await score_case(graph, case, result)
    obs.log_event(
        logger,
        "eval.case",
        case_id=case.id,
        passed=metrics.passed,
        latency_ms=round(timer.elapsed_ms(), 1),
    )
    return CaseResult(case_id=case.id, metrics=metrics)


async def score_case(graph: GraphManager, case: EvalCase, result) -> CaseMetrics:
    """Score an extraction result against expectations (shared by fake/live)."""
    from backend.services.extraction import ExtractionResult

    assert isinstance(result, ExtractionResult)
    exp = case.expected

    got_nodes = {(n.type.value, n.title.lower()) for n in result.nodes}
    want_nodes = {(n.type.value, n.title.lower()) for n in exp.nodes}
    node_p, node_r = _pr(got_nodes, want_nodes)

    title_by_id = {n.id: n.title for n in result.nodes}
    got_edges = {
        (
            title_by_id.get(e.source_id, "").lower(),
            e.edge_type.value,
            title_by_id.get(e.target_id, "").lower(),
        )
        for e in result.edges
    }
    want_edges = {(e.source.lower(), e.edge_type.value, e.target.lower()) for e in exp.edges}
    edge_p, edge_r = _pr(got_edges, want_edges)

    ownerless: set[str] = set()
    for n in result.nodes:
        if n.type not in OWNERLESS_TYPES:
            continue
        node_edges = await graph.get_edges(n.id)
        if not any(
            e.edge_type in OWNERSHIP_EDGES and (e.target_id == n.id or e.source_id == n.id)
            for e in node_edges
        ):
            ownerless.add(n.title.lower())
    owner_p, owner_r = _pr(ownerless, {t.lower() for t in exp.ownerless})

    want_decisions = {n.title.lower() for n in exp.nodes if n.type == NodeType.DECISION}
    got_decisions = {n.title.lower() for n in result.nodes if n.type == NodeType.DECISION}
    decision_r = (
        len(got_decisions & want_decisions) / len(want_decisions) if want_decisions else 1.0
    )

    contra_pairs = {
        tuple(
            sorted(
                [title_by_id.get(e.source_id, "").lower(), title_by_id.get(e.target_id, "").lower()]
            )
        )
        for e in result.edges
        if e.edge_type == EdgeType.CONTRADICTS
    }
    want_contra = {tuple(sorted([a.lower(), b.lower()])) for a, b in exp.contradictions}
    contra_r = len(contra_pairs & want_contra) / len(want_contra) if want_contra else 1.0

    fabricated = sorted(f"{t}:{ti}" for (t, ti) in got_nodes - want_nodes)
    unexpected_edges = sorted(f"{s} -{t}-> {u}" for (s, t, u) in got_edges - want_edges)
    lowered_transcript = case.transcript.lower()
    unsupported_quotes = []
    for n in result.nodes:
        for ev in n.evidence:
            if ev.quote and ev.quote.lower() not in lowered_transcript:
                unsupported_quotes.append(f"node:{n.title}:{ev.quote[:60]}")
    for e in result.edges:
        if e.evidence and e.evidence.lower() not in lowered_transcript:
            unsupported_quotes.append(f"edge:{e.id[:8]}:{e.evidence[:60]}")
    unsupported_quotes = sorted(set(unsupported_quotes))

    metrics = CaseMetrics(
        node_precision=node_p,
        node_recall=node_r,
        edge_precision=edge_p,
        edge_recall=edge_r,
        owner_precision=owner_p,
        owner_recall=owner_r,
        decision_recall=decision_r,
        contradiction_recall=contra_r,
        fabricated=fabricated,
        unexpected_edges=unexpected_edges,
        unsupported_quotes=unsupported_quotes,
        errors=result.errors,
    )
    metrics.passed = (
        node_p == 1.0
        and node_r == 1.0
        and edge_p == 1.0
        and edge_r == 1.0
        and not fabricated
        and not unexpected_edges
        and not unsupported_quotes
        and not result.errors
    )
    return metrics


def summarize(
    results: list[CaseResult],
    *,
    provider: str = "fake",
    model: str = "",
    prompt_version: str = "",
    answer_prompt_version: str = "",
    runtime_s: float = 0.0,
) -> EvalReport:
    import datetime as dt

    def avg(field: str) -> float:
        vals = [getattr(r.metrics, field) for r in results]
        return sum(vals) / len(vals) if vals else 0.0

    totals = {
        f: avg(f)
        for f in (
            "node_precision",
            "node_recall",
            "edge_precision",
            "edge_recall",
            "owner_precision",
            "owner_recall",
            "decision_recall",
            "contradiction_recall",
        )
    }
    totals["fabricated_total"] = float(sum(len(r.metrics.fabricated) for r in results))
    totals["unexpected_edges_total"] = float(sum(len(r.metrics.unexpected_edges) for r in results))
    totals["unsupported_quotes_total"] = float(
        sum(len(r.metrics.unsupported_quotes) for r in results)
    )
    return EvalReport(
        cases=results,
        totals=totals,
        passed=bool(results) and all(r.metrics.passed for r in results),
        provider=provider,
        model=model,
        extraction_prompt_version=prompt_version,
        answer_prompt_version=answer_prompt_version,
        timestamp=dt.datetime.utcnow().isoformat(),
        runtime_s=runtime_s,
    )
