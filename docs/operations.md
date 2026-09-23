# Operations

## Minimal (no Docker)

```bash
pip install -e ".[dev]"
uvicorn backend.app:app --reload   # SQLite + NetworkX
```

## Containers

```bash
docker compose up --build          # api only (SQLite + NetworkX)
docker compose --profile full up --build   # + postgres, redis, neo4j, worker
```

For the full profile, point the API at production backends:
`GRAPH_BACKEND=neo4j`, `DATABASE_URL=postgresql+asyncpg://…`,
`USE_BACKGROUND_JOBS=true`, `API_KEY=…`, then `alembic upgrade head`.

## Workers

```bash
celery -A backend.worker:celery_app worker --loglevel=info --queues=echograph
# Windows: append --pool=solo
```

Worker and inline route share `process_job_core()`; 202 means accepted.

## Migrations

```bash
alembic revision --autogenerate -m "..."
alembic upgrade head
python -m pytest tests/test_features.py::test_alembic_downgrade_roundtrip -q
```

## Supervision

- Liveness: `GET /health` (public).
- Readiness: `GET /ready` returns per-dependency status (`database`, `graph`,
  `redis` when workers enabled); non-ok → 503 with detail.
- Logs: structured events with `request_id`, latencies, token usage, cost
  estimates. Transcripts are never logged (lengths + SHA only).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `POST /process` → 503 | `OPENAI_API_KEY` unset |
| `POST /transcribe` → 501 | local mode without `faster-whisper` |
| Job stuck PROCESSING | worker down (check `celery inspect active`); reset row to FAILED to retry |
| Neo4j Cypher errors | writes without `result.consume()`; nested maps (use `_serialize_props`) |
