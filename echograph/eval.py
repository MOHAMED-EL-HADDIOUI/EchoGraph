"""Run the frozen extraction evaluation.

Fake mode (default) replays recorded payloads: no network, deterministic.
Live mode calls the real extraction model explicitly (spends API credits).

Usage:
  python -m echograph.eval [--cases eval/cases] [--report reports/eval.json]
  python -m echograph.eval --provider live [--model gpt-4o]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

from backend.config import settings
from backend.graph.networkx_backend import NetworkXGraphManager
from backend.services.evaluation import load_cases, run_case, score_case, summarize
from backend.services.extraction import run_extraction


async def _run(cases_dir: Path, provider: str, model: str) -> tuple[int, dict]:
    from backend.services.extraction import make_openai_complete

    cases = load_cases(cases_dir)
    complete = (
        make_openai_complete(model or settings.OPENAI_EXTRACTION_MODEL, settings.OPENAI_API_KEY)
        if provider == "live"
        else None
    )
    started = time.perf_counter()
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, case in enumerate(cases):
            graph = NetworkXGraphManager(data_path=str(Path(tmp) / f"eval-{i}.json"))
            if provider == "live":
                assert complete is not None
                result = await run_extraction(
                    graph, case.transcript, complete, ingestion_id=f"eval-{case.id}"
                )
                metrics = await score_case(graph, case, result)
                from backend.services.evaluation import CaseResult

                results.append(CaseResult(case_id=case.id, metrics=metrics))
            else:
                results.append(await run_case(graph, case))
    runtime_s = time.perf_counter() - started
    report = summarize(
        results,
        provider=("openai" if provider == "live" else "fake"),
        model=(model or settings.OPENAI_EXTRACTION_MODEL) if provider == "live" else "",
        prompt_version=settings.EXTRACTION_PROMPT_VERSION,
        answer_prompt_version=settings.ANSWER_PROMPT_VERSION,
        runtime_s=round(runtime_s, 2),
    )
    return (0 if report.passed else 1), report.model_dump(mode="json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EchoGraph extraction evaluation")
    parser.add_argument("--cases", default="eval/cases")
    parser.add_argument("--report", default="reports/eval.json")
    parser.add_argument("--provider", choices=["fake", "live"], default="fake")
    parser.add_argument("--model", default="")
    args = parser.parse_args(argv)

    if args.provider == "live" and not settings.OPENAI_API_KEY:
        print("live provider requires OPENAI_API_KEY (explicit opt-in)", file=sys.stderr)
        return 2
    code, payload = asyncio.run(_run(Path(args.cases), args.provider, args.model))
    if "error" in payload:
        return code
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"cases={len(payload['cases'])} passed={payload['passed']} provider={payload['provider']}"
    )
    for name, value in payload["totals"].items():
        print(f"  {name}={value:.3f}")
    for case in payload["cases"]:
        m = case["metrics"]
        mark = "PASS" if m["passed"] else "FAIL"
        print(f"  [{mark}] {case['case_id']} fabricated={m['fabricated']}")
    print(f"report: {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
