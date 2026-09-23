# Security

- **Auth**: `X-API-Key` (constant-time compare) on graph/ingestion/notifications;
  empty `API_KEY` = open dev server with a startup warning; `/health` public.
  Keys are never logged.
- **Input limits**: `MAX_TEXT_CHARS` (413), 25 MB audio cap, `limit`/`offset`
  pagination everywhere, subgraph `depth` 1–5, query `limit` ≤ 200, `MAX_CHUNKS`.
- **Injection**: SQL via SQLAlchemy parameters; Cypher via parameters (`depth`
  is interpolated only after 1–5 int validation); no shell execution; uploads
  use suffix-only tempfiles; source text is untrusted data in prompts
  (SYSTEM INSTRUCTIONS / SOURCE DATA / TASK / OUTPUT CONTRACT separation).
- **LLM boundaries**: strict JSON contracts, Pydantic validation, per-chunk
  timeouts, bounded context, no answer on empty retrieval, no external calls
  in tests.
- **Data**: transcripts never logged; metadata/evidence JSON-stringified for
  Neo4j; export endpoint excludes secrets.
