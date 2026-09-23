# EchoGraph

Autonomous knowledge-graph builder: ingest organizational noise (meetings, chat,
email) → LLM extraction → queryable graph of decisions, people, topics, and actions.

## Quickstart

```bash
cp .env.example .env
pip install -e ".[dev]"
uvicorn backend.app:app --reload
```

Open http://localhost:8000/docs. Default dev backend is in-memory NetworkX +
SQLite — no Docker needed.

## Ingest → query

```bash
# 1. Submit (stores content, returns PENDING job)
curl -X POST localhost:8000/ingestion -H 'Content-Type: application/json' \
  -d '{"source_type": "MEETING", "title": "Planning", "content": "Ada decided to ship v1."}'

# 2. Extract inline (needs OPENAI_API_KEY; returns COMPLETED with counts)
curl -X POST localhost:8000/ingestion/<job_id>/process

# 3. Query
curl -X POST localhost:8000/graph/query -H 'Content-Type: application/json' \
  -d '{"query": "ship"}'
```

## Production backend

```bash
docker compose up -d neo4j
GRAPH_BACKEND=neo4j uvicorn backend.app:app
```

## Verify

```bash
ruff check backend tests
ruff format --check backend tests
pytest -q
```
