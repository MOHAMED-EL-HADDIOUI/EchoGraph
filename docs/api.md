# API contracts

All responses are typed Pydantic models (`IngestionJobRead`,
`NotificationRead`, graph models). No `{data, meta}` envelope: list
endpoints return arrays with explicit `limit`/`offset` pagination instead —
documented here as the project convention.

## Status codes

| Code | Meaning | Where |
|---|---|---|
| 200 | OK (incl. idempotent COMPLETED reprocess) | everywhere |
| 201 | (unused — creates return 200 with the object) | — |
| 202 | Accepted for background processing | `POST /process` with flag |
| 400 | Invalid edge triple | `POST /edges` |
| 401 | Missing/invalid `X-API-Key` | guarded routers |
| 404 | Unknown node/edge/job/notification | getters, deletes, merges |
| 409 | Job already PROCESSING / not processable | `POST /process` |
| 413 | Over `MAX_TEXT_CHARS` or audio cap | submit, transcribe |
| 422 | Missing content, invalid payloads | process, dry-run, model validation |
| 500 | Unhandled domain error (`EchoGraphError` mapped, no traces) | anywhere |
| 501 | Celery missing / local transcriber missing | background, transcribe |
| 503 | No `OPENAI_API_KEY` / dependency down | process, transcribe, `/ready` |

## Key endpoints

- `POST /ingestion` → `POST /ingestion/{id}/process` → `POST /ingestion/{id}/dry-run`
  (preview), `POST /ingestion/transcribe`, `GET /ingestion` (`limit`/`offset`)
- `POST /graph/query` and `GET /graph/query?q=` (`node_type`, `ingestion_id`,
  `include_evidence`, `explain`); `GET /graph/nodes/{id}/lineage`,
  `GET /graph/nodes/{id}/history` (decision evolution timeline),
  `GET /graph/questions/unresolved` (organizational gaps),
  `GET /graph/export`, `GET /graph/nodes/{id}/neighbors`
- `GET /notifications`, `PATCH|POST /notifications/{id}/read`,
  `GET /notifications/{id}/explain`
- `GET /health` (public), `GET /ready` (dependency breakdown)
