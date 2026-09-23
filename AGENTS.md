# AGENTS.md — EchoGraph

FastAPI + knowledge-graph backend. Entrypoint `backend/app.py:app`
(`uvicorn backend.app:app --reload`).

## Setup
- Requires Python `>=3.12`. `cp .env.example .env`, then `pip install -e ".[dev]"`.
- Dev defaults need no Docker: `GRAPH_BACKEND=networkx`,
  `DATABASE_URL=sqlite+aiosqlite:///./echograph.db`. Neo4j via
  `docker compose up -d neo4j` (`bolt://localhost:7687`, `neo4j/password`, APOC included).
- Verify: `ruff check backend tests`, `ruff format --check backend tests`, `pytest -q`
  (same three steps run in `.github/workflows/ci.yml`).

## Architecture
- `backend/app.py` — `create_app()` + lifespan (`init_db()`, graph singleton on
  `app.state.graph`, Neo4j `init_constraints()`/`close()`). Routers: `health`, `graph`,
  `ingestion`, `notifications`.
- `backend/deps.py` — `get_graph_manager` (from `app.state`), `get_db`,
  `get_extraction_complete` (503 without `OPENAI_API_KEY`; override in tests with a fake).
- `backend/api/graph.py` — nodes CRUD + merge, `GET /nodes` (`limit`/`offset`),
  `GET /search`, `POST /edges`, `POST /query`, subgraph, full graph, statistics.
- `backend/services/graph_ops.py` — `add_edge_validated()` is the single choke point
  for edge writes (endpoint existence + `validate_edge()`); routes and workers must use
  it, never `graph.add_edge` directly.
- `backend/services/extraction.py` — sync LLM extraction. `CompleteJson` seam
  `(system, user) -> dict`; `run_extraction()` never raises on bad LLM output
  (caps errors at `MAX_ERRORS`), dedupes by `(type, title)` via `merge_node`.
  Ingestion flow: `POST /ingestion` (stores `content`, PENDING) →
  `POST /ingestion/{id}/process` (runs extraction inline, COMPLETED/FAILED).
- `backend/services/query.py` — keyword `answer_query()` honoring
  `{"node_type": ...}` filter; `answer` stays `None` until the LLM step lands.
- `backend/database.py` — async SQLAlchemy, `init_db()` at startup. JSON in `Text`
  columns; `ingestion_jobs` stores raw `content`. No migrations yet: delete
  `echograph.db` after model changes.
- `backend/graph/schema.py:30` — `VALID_EDGES`, except `RELATES_TO` (open-schema).
- `backend/models/` — pure Pydantic v2 only; ORM rows live in `backend/database.py`.

## Neo4j gotchas (verified by smoke test, don't regress)
- Cypher 5: use `CREATE (n:Label) SET n = $props`, never `CREATE (n:Label $props)`.
- `metadata` dicts are JSON-stringified on write, parsed on read
  (`_serialize_props`/`_deserialize_props`/`_record_to_edge`); embeddings round-trip
  as float arrays — `find_similar_nodes` depends on reading them back (pure-Python cosine).
- Always `await result.consume()` on fire-and-forget writes, else Cypher errors
  surface nowhere.
- `depth` in `get_subgraph` is f-string-interpolated (Cypher has no bind param for
  hop bounds) — safe only because the API validates `1..5`.

## Conventions
- All modules start with `from __future__ import annotations`; codebase standardizes
  on `datetime.utcnow` (ruff `DTZ003` ignored) and FastAPI `Depends` defaults (ruff
  `B008` ignored); tolerant graph reads ignore `BLE001`/`S110` per-file.
- New graph operations go on both backends behind the `GraphManager` ABC.
- Tests: `tests/conftest.py` overrides graph (tmp JSON) + DB (tmp sqlite); never hit
  real services — inject fakes via `app.dependency_overrides`.
