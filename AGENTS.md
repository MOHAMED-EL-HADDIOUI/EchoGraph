# AGENTS.md — EchoGraph

FastAPI + knowledge-graph backend. Entrypoint `backend/app.py:app`
(`uvicorn backend.app:app --reload`; factory form `uvicorn backend.app:create_app --factory` also works).

## Setup
- Requires Python `>=3.12`. `cp .env.example .env`, then `pip install -e ".[dev]"`.
- Dev defaults need no Docker: `GRAPH_BACKEND=networkx`,
  `DATABASE_URL=sqlite+aiosqlite:///./echograph.db`. Full stack via
  `docker compose --profile full up` (postgres, redis, neo4j, worker);
  plain `docker compose up` runs the API only.
- Verify: `ruff check .`, `ruff format --check .`, `pytest -q`,
  `python -m echograph.eval` (same four steps run in `.github/workflows/ci.yml`,
  which also starts Neo4j so the contract suite runs on both backends).

## Architecture
- `backend/app.py` — `create_app()` + lifespan (`init_db()`, graph singleton on
  `app.state.graph`, Neo4j `init_constraints()`/`close()`). Routers: `health`, `graph`,
  `ingestion`, `notifications`. Request-ID middleware logs method/path/status/latency.
- `backend/deps.py` — `get_graph_manager` (from `app.state`), `get_db`,
  `get_extraction_complete` (None without `OPENAI_API_KEY`; override in tests with a fake),
  answer/embedder/transcriber seams. `require_api_key` guards graph/ingestion/notifications
  routers (`X-API-Key`); empty `API_KEY` = open server (dev). `/health` is always public,
  `/ready` returns per-dependency status (503 otherwise).
- `backend/exceptions.py` — `EchoGraphError` tree (config/graph/validation/ingestion/
  extraction/provider/transcription/auth). Service errors subclass it; routes map to HTTP.
- `backend/api/graph.py` — nodes CRUD + merge/update, `GET /nodes` (`limit`/`offset`),
  `GET /search`, `POST /edges`, `DELETE /edges/{id}`, neighbors, history timeline,
  unresolved questions, `GET+POST /query`,
  subgraph, export, lineage, statistics.
- `backend/services/graph_ops.py` — `add_edge_validated()` is the single choke point
  for edge writes (endpoint existence + `validate_edge()`); routes and workers must use
  it, never `graph.add_edge` directly.
- `backend/services/processing.py` — `process_job_core()` shared by the inline
  route and the Celery task (attempts, PROCESSING, extraction, notifications, run
  history in job metadata). `dry_run_job()` previews against an ephemeral seeded
  copy; `build_diff()` shapes the response diff.
- `backend/services/extraction.py` — sync LLM extraction. `CompleteJson` seam
  `(system, user) -> dict`; `run_extraction()` returns `ExtractionResult` and never
  raises on bad LLM output (caps errors at `MAX_EXTRACTION_ERRORS`, per-chunk timeout via
  `LLM_TIMEOUT_SECONDS`), rejects empty titles, verifies quotes against the chunk
  (drops + records mismatches), dedupes by normalized `(type, title)` via `merge_node`,
  optionally by embedding cosine (`embed_texts` + `ENABLE_EMBEDDING_DEDUP`, threshold
  `EMBEDDING_DEDUP_THRESHOLD`). Same-triple repeats merge evidence instead of
  duplicating edges. Node metadata carries lineage (`created_by`, `prompt_version`,
  `source_ingestion_id`, `extraction_run_id`, `merged_from`); pass `ingestion_id`
  and `run_id` from the caller. Chunking via `CHUNK_SIZE`/`CHUNK_OVERLAP`/`MAX_CHUNKS`;
  optional in-process cache (`ENABLE_EXTRACTION_CACHE`, keyed by prompt version + chunk hash).
  Ingestion flow: `POST /ingestion` (stores `content` + sha, PENDING; 413 over
  `MAX_TEXT_CHARS`) → `POST /ingestion/{id}/process` (idempotent:
  COMPLETED returns stored outcome, PROCESSING 409, attempts counted; runs
  extraction inline, then `generate_notifications()`; response carries
  `notifications_created` + `diff`). Audio via `POST /ingestion/transcribe`
  (`services/transcription.py` delegating to `providers/` classes; api mode needs key,
  local mode 501 without `faster-whisper`; `MAX_AUDIO_BYTES` cap; full transcript metadata).
- `backend/services/query.py` + `retrieval.py` — hybrid retriever (keyword ∪ semantic
  + graph-connectedness bonus, ranked with reasons) over `Retriever` protocol;
  `answer_query()` honoring `node_type`/`ingestion_id` filters. `find_unresolved_questions()`
  lists QUESTION nodes with no incoming ANSWERS edge. Pass `CompleteText`
  (`get_answer_complete`, None without `OPENAI_API_KEY` → retrieval-only) for grounded
  LLM answers parsed into `AnswerPayload` (answer/citations/uncertainties/conflicts;
  plain-text degrades gracefully, failures abstain). The LLM is never called on empty
  retrieval. `verdict` ∈ supported/uncertain/contradictory; `citations` resolve
  `[Title]` → node IDs; `include_evidence` returns citable `evidence` items;
  `explain=true` returns retrieval debug. Prompts separate SYSTEM/TASK/CONTEXT;
  source text is untrusted data.
- `backend/services/history.py` — `get_decision_history()` BFS over explicit
  SUPERSEDES/CONTRADICTS edges (bounded depth, both directions); never inferred
  from ingestion order.
- `backend/services/notifications.py` — deterministic post-extraction rules:
  CONTRADICTS edge → HIGH CONTRADICTION; ownerless ACTION_ITEM/DECISION →
  MEDIUM MISSING_OWNER (ownership: OWNS/DECIDED_BY/ASSIGNED_TO/DECIDES). Rows carry
  severity, `ingestion_id`, `evidence_ids`, `read_at`; dedupe via `find_duplicate()`
  (same type + nodes + ingestion, unread only). Pass `ingestion_id` from the caller.
  `to_read()` is the single ORM→API conversion (used by routes + explain).
- `backend/providers/` — `llm.py` (`ExtractionLLM`/`AnswerLLM` protocols, OpenAI +
  Fake providers with call recording), `embeddings.py`, `transcription.py`
  (`WhisperAPITranscriber`, `FasterWhisperTranscriber`, `TranscriptResult`).
  Services keep callable seams; deps wire providers. Tests must use Fakes.
- `backend/repositories.py` — `IngestionJobRepository`, `NotificationRepository`;
  routes and services persist through these, not inline SQL.
- `backend/services/sources.py` — `NormalizedDocument` + `SourceAdapter` protocol
  (`ManualAdapter`, `AudioAdapter`); all inputs converge here before extraction.
- `backend/services/evaluation.py` + `eval/cases/*.json` + `python -m echograph.eval`
  (`--provider fake|live`) → `reports/eval.json` (node/edge P-R, owner/decision/
  contradiction recall, fabricated/unexpected-edges/unsupported-quotes lists,
  provider/model/prompt-version/timestamp metadata; case passes only on perfect
  scores). Docs: `docs/evaluation.md`.
- `backend/services/deduplication.py` — `normalize_title()`, `quotes_match()`.
- `backend/obs.py` — `request_id` context var + middleware (echoes `X-Request-ID`),
  `log_event()` (structured, lengths/hashes only — never raw transcripts;
  `fingerprint()`/`fingerprint_content()` for content identity), `Timer`,
  LLM usage + `estimate_cost_usd()` logging. `docs/architecture.md` has Mermaid diagrams.
- `backend/database.py` — async SQLAlchemy, `init_db()` at startup. JSON in `Text`
  columns; `ingestion_jobs` stores raw `content` + sha + run history in metadata.
  Schema changes go through Alembic (`alembic revision --autogenerate`,
  `alembic upgrade head` shares `Base.metadata`; `alembic/versions/` is ruff-excluded
  generated code; downgrade covered by test).
- `backend/graph/schema.py` — `VALID_EDGES` (13 node types, 14 edge types),
  except `RELATES_TO` (open-schema).
- `backend/models/` — pure Pydantic v2 only; ORM rows live in `backend/database.py`.
  API responses use `IngestionJobRead` / `NotificationRead` models (not dicts);
  keep field names stable — tests and the eval report depend on them.
- `backend/worker.py` — lazy Celery app (`celery -A backend.worker:celery_app`,
  `--pool=solo` on Windows); `USE_BACKGROUND_JOBS=true` makes `POST /process`
  enqueue (202) instead of extracting inline. Broker deps stay optional for dev.

## Neo4j gotchas (verified by smoke test, don't regress)
- Cypher 5: use `CREATE (n:Label) SET n = $props`, never `CREATE (n:Label $props)`.
- `metadata` dicts and node `evidence` lists are JSON-stringified on write,
  parsed on read (`_serialize_props`/`_deserialize_props`/`_record_to_edge`);
  embeddings round-trip as float arrays — `find_similar_nodes` depends on
  reading them back (pure-Python cosine).
- Always `await result.consume()` on fire-and-forget writes, else Cypher errors
  surface nowhere.
- `depth` in `get_subgraph` is f-string-interpolated (Cypher has no bind param for
  hop bounds) — safe only because the API validates `1..5`.

## Conventions
- All modules start with `from __future__ import annotations`; codebase standardizes
  on `datetime.utcnow` (ruff `DTZ003` ignored) and FastAPI `Depends` defaults (ruff
  `B008` ignored); tolerant graph reads ignore `BLE001`/`S110` per-file;
  health probes ignore `BLE001`.
- Dependency direction: API → services → providers/graph/repos/models. Routers hold
  no business logic; no Cypher outside backends; no service imports from `api`.
- New graph operations go on both backends behind the `GraphManager` ABC (+ contract
  test in `tests/graph/test_contract.py`; Neo4j params skip without a server).
- Tests: `tests/conftest.py` overrides graph (tmp JSON) + DB (tmp sqlite); never hit
  real services — inject fakes via `app.dependency_overrides` (prefer the `Fake*`
  providers for new tests).
  `tests/test_eval_quality.py` is the frozen quality set (transcript → recorded LLM
  payload → expected graph); prompt tweaks must keep it green. New eval cases go
  in `eval/cases/*.json` (run via `python -m echograph.eval` or
  `tests/test_evaluation.py`).
- Config: canonical names per `.env.example`; legacy names work via aliases.
  Prompt versions bump deliberately (recorded in run metadata + eval reports).
