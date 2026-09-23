"""Run the frozen extraction evaluation without network access.

Usage:  python -m echograph.eval [--cases eval/cases] [--report reports/eval.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from backend.graph.networkx_backend import NetworkXGraphManager
from backend.services.evaluation import load_cases, run_case, summarize


async def _run(cases_dir: Path) -> tuple[int, dict]:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, case in enumerate(load_cases(cases_dir)):
            graph = NetworkXGraphManager(data_path=str(Path(tmp) / f"eval-{i}.json"))
            results.append(await run_case(graph, case))
    report = summarize(results)
    return (0 if report.passed else 1), report.model_dump(mode="json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EchoGraph extraction evaluation")
    parser.add_argument("--cases", default="eval/cases")
    parser.add_argument("--report", default="reports/eval.json")
    args = parser.parse_args(argv)

    code, payload = asyncio.run(_run(Path(args.cases)))
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"cases={len(payload['cases'])} passed={payload['passed']}")
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
