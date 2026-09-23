# Evaluation

> Synthetic fixture scores demonstrate deterministic pipeline correctness,
> not real-world live LLM quality. Do not claim production extraction
> accuracy from these numbers — run `--provider live` before making
> quality claims.

EchoGraph treats extraction quality as a measured property, not a vibe.
`eval/cases/*.json` holds synthetic organizational transcripts with **recorded**
LLM payloads and hand-checked expectations — separate from implementation
prompts, so prompt changes are judged objectively.

## Run it

```bash
python -m echograph.eval                 # uses eval/cases, writes reports/eval.json
python -m echograph.eval --cases eval/cases --report reports/eval.json
pytest tests/test_evaluation.py -q      # same suite inside the test run
```

No network access is needed: the runner replays recorded payloads.

## Metrics (per case + micro-averaged totals)

| Metric | Meaning |
|---|---|
| `node_precision` / `node_recall` | `(type, title)` set overlap |
| `edge_precision` / `edge_recall` | `(source, edge_type, target)` title-tuple overlap |
| `owner_precision` / `owner_recall` | ownerless-node detection vs expected |
| `decision_recall` | expected DECISION titles recovered |
| `contradiction_recall` | expected CONTRADICTS pairs recovered |
| `fabricated` | extracted `(type, title)` not in expected — must be empty |
| `errors` | extraction errors — must be empty |

A case passes only on perfect scores with zero fabrications and zero errors;
the report's `passed` flag (and CLI exit code) reflects the whole suite.

## Adding a case

1. Write a realistic transcript (`transcript`).
2. Paste a real or hand-built LLM payload into `recorded` (one entry per chunk).
3. Fill `expected`: `nodes` (`type`+`title`), `edges` (`source`/`target` titles +
   `edge_type`), `ownerless` titles, `contradictions` title pairs.
4. Run the CLI and confirm the suite stays green.

## Prompt workflow

Change `SYSTEM_PROMPT` → run `python -m echograph.eval` → inspect
`reports/eval.json` (committed for history). If a metric drops, the prompt
change regressed quality. `tests/test_eval_quality.py` additionally freezes
two cases in-code as a fast regression tripwire.
