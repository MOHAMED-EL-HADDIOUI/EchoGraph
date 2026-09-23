# EchoGraph — Organizational memory with evidence.

EchoGraph turns meeting notes, chat, email, and audio into a provenance-first
knowledge graph: every fact keeps its source quote and ingestion ID, edges are
schema-validated, contradictions and missing owners raise alerts, and answers
are grounded in retrievable evidence — or explicitly abstained from.

## Key features

- **Structured LLM extraction** (strict JSON, Pydantic-validated) with chunking,
  overlap, timeouts, and per-ingestion error caps — malformed output is recorded,
  never fatal.
- **Provenance-first graph**: `evidence[]` on nodes, `ingestion_id` + quotes on
  edges, merge appends (never overwrites) history, lineage API per node.
- **Validated mutations**: every edge passes `add_edge_validated()` against
  `VALID_EDGES` (`RELATES_TO` stays open-schema); duplicate triples merge
  evidence instead of duplicating relations.
- **Deterministic intelligence**: contradiction (HIGH) and missing-owner
  (MEDIUM) notifications with severity, ingestion, and evidence IDs; per-ingestion
  dedupe; idempotent processing with attempts and run history.
- **Evidence-aware querying**: keyword + hybrid semantic retrieval, 1-hop closure,
  `verdict` (supported/uncertain/contradictory), `[Title]` citations, optional
  evidence items and retrieval debug output.
- **Evaluation-driven**: 11 recorded-payload cases → `reports/eval.json` with
  precision/recall, fabrication detection, and reproducibility metadata; fake and
  live provider modes.
- **Operable**: API-key auth, `/health` + dependency-aware `/ready`, request IDs,
  structured logs (never raw transcripts), token/cost tracking, Celery workers
  behind a flag, Alembic migrations, Docker Compose for api/postgres/redis/neo4j.

## Quickstart

```bash
cp .env.example .env
pip install -e ".[dev]"
uvicorn backend.app:app --reload
```

Open http://localhost:8000/docs. Defaults (NetworkX + SQLite) need no Docker.
Set `API_KEY` for shared deployments (`X-API-Key` header).

## End-to-end example

```bash
# 1. Submit the Atlas standup (stores content, returns PENDING job)
curl -X POST localhost:8000/ingestion -H 'Content-Type: application/json' -d '{
  "source_type": "MEETING", "title": "Atlas standup",
  "content": "Atlas standup: Sarah owns the migration to PostgreSQL. We decided to move away from MySQL by October."
}'

# 2. Extract inline (needs OPENAI_API_KEY)
curl -X POST localhost:8000/ingestion/<job_id>/process
# -> {"status": "COMPLETED", "nodes_created": 4, "edges_created": 1,
#     "notifications_created": 1,
#     "diff": {"nodes_added": 4, "contradictions": 0, "missing_owners": 1, ...}}

# 3. Dry-run the next meeting first (no mutation)
curl -X POST localhost:8000/ingestion/<job_id2>/dry-run
# -> {"diff": {...}, "proposed_nodes": [...], "errors": []}

# 4. Query with evidence: who owns the migration, and what supports it?
curl -X POST localhost:8000/graph/query -H 'Content-Type: application/json' \
  -d '{"query": "migration owner", "include_evidence": true}'
# -> nodes + edges + verdict + evidence[{ingestion_id, quote, ...}]

# 5. Why does this node exist?
curl localhost:8000/graph/nodes/<node_id>/lineage
# -> {origin: {ingestion_id, quote, prompt_version, extraction_run_id}, ...}
```

Contradiction flow: ingest a meeting stating the opposite decision → a
`CONTRADICTS` edge is extracted → HIGH notification with both titles, evidence
quotes, and edge ID → queries over the topic return `verdict: "contradictory"`.

## Docker setup

```bash
docker compose up --build            # api only (SQLite + NetworkX)
docker compose --profile full up --build   # + postgres, redis, neo4j, worker
```

Production env: `GRAPH_BACKEND=neo4j`, `DATABASE_URL=postgresql+asyncpg://…`,
`USE_BACKGROUND_JOBS=true`, `API_KEY=…`, then `alembic upgrade head`.
Worker: `celery -A backend.worker:celery_app worker --queues=echograph`
(`--pool=solo` on Windows).

## Evaluation

```bash
python -m echograph.eval                       # fake mode, offline, deterministic
python -m echograph.eval --provider live       # explicit real-LLM run (spends credits)
```

See `docs/evaluation.md`. Synthetic scores prove pipeline determinism, not
live-model quality — run live mode before making quality claims.

## Testing and configuration

```bash
ruff check backend tests echograph eval
ruff format --check backend tests echograph eval
pytest -q
```

All 90+ tests are hermetic (fake providers, tmp graph/DB, Neo4j contract
skipped without a server). Full option reference: `.env.example`.

## Design decisions

- No edge write bypasses `add_edge_validated()`; no fact without provenance.
- DiGraph (not MultiDiGraph): one logical relation per triple; repeats merge
  evidence. Parallel distinct-type edges between one pair collapse — known limit.
- No `{data, meta}` envelope: typed models + explicit `limit`/`offset`.
- Sync-by-default processing; Celery reuses the identical core.
- SQLite `create_all` still runs at startup for zero-config dev; Alembic owns
  real schema evolution.

## Limitations

- Keyword retrieval is substring-based; semantic ranking needs
  `ENABLE_EMBEDDING_DEDUP` + key (off by default until evaluated).
- `find_similar_nodes` on Neo4j is pure-Python over all nodes (no vector index).
- Dry-run seeds an in-memory copy capped at 2000 nodes.
- Duplicate-relation merge is per-triple; distinct edge types between one node
  pair still collapse on the NetworkX backend.

## Roadmap

Live-LLM quality measurement on real transcripts → embedding dedupe by default
→ richer temporal queries → (much later) optional vector index, UI.
