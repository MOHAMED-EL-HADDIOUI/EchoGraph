from __future__ import annotations

from pathlib import Path

from backend.graph.networkx_backend import NetworkXGraphManager
from backend.services.evaluation import load_cases, run_case, summarize
from backend.services.extraction import run_extraction

CASES_DIR = Path(__file__).resolve().parent.parent / "eval" / "cases"


async def test_eval_suite_is_green(tmp_path):
    cases = load_cases(CASES_DIR)
    assert len(cases) >= 7
    results = []
    for i, case in enumerate(cases):
        graph = NetworkXGraphManager(data_path=str(tmp_path / f"eval-{i}.json"))
        results.append(await run_case(graph, case))
    report = summarize(results)
    assert report.passed
    for name in (
        "node_precision",
        "node_recall",
        "edge_precision",
        "edge_recall",
        "owner_precision",
        "owner_recall",
        "decision_recall",
        "contradiction_recall",
    ):
        assert report.totals[name] == 1.0, name
    assert report.totals["fabricated_total"] == 0.0


async def test_malformed_llm_payload_is_recorded_not_raised(tmp_path):
    async def garbage_complete(system: str, user: str) -> dict:
        return {"nodes": [{"type": "NOT_A_TYPE", "title": "x"}]}

    graph = NetworkXGraphManager(data_path=str(tmp_path / "g.json"))
    result = await run_extraction(graph, "hello", garbage_complete)
    assert result.nodes_created == 0
    assert result.edges_created == 0
    assert len(result.errors) == 1


async def test_malformed_payload_job_still_completes(client):
    async def garbage_complete(system: str, user: str) -> dict:
        return {"nodes": "not-a-list"}

    from backend.app import app
    from backend.deps import get_extraction_complete

    app.dependency_overrides[get_extraction_complete] = lambda: garbage_complete
    try:
        r = await client.post("/ingestion", json={"source_type": "SLACK", "content": "x"})
        body = (await client.post(f"/ingestion/{r.json()['job_id']}/process")).json()
        assert body["status"] == "COMPLETED"
        assert body["nodes_created"] == 0
        assert body["error"] != ""
    finally:
        app.dependency_overrides.pop(get_extraction_complete, None)
