# Graph model

Node types: `PERSON TEAM ORGANIZATION PROJECT MEETING MESSAGE EVENT TOPIC
DOCUMENT DECISION RATIONALE QUESTION ACTION_ITEM`.

Edge types: `DECIDED_BY DECIDES RELATES_TO CONTRADICTS SUPERSEDES OWNS
ASSIGNED_TO PART_OF MENTIONS DISCUSSES BLOCKED_BY REFERENCES DERIVED_FROM
ANSWERS`. All triples except `RELATES_TO` (open-schema) must appear in
`VALID_EDGES` (`backend/graph/schema.py`); writes go through
`add_edge_validated()`.

## Node fields

`id, type, title, content, source_ref, source_type, confidence, metadata,
embedding, evidence[], valid_from, valid_until, observed_at, state,
created_at, updated_at`. Titles are stored verbatim; matching uses
`normalize_title()` (trimmed, whitespace-collapsed, casefolded).

## Evidence

Every source-derived node carries `evidence[]` (`ingestion_id`,
`source_type`, `source_id`, `quote`, `confidence`); every edge carries
`ingestion_id` + quote. Quotes must be exact source substrings — the
extractor drops anything else and records an error. Merges append evidence
(deduplicated); metadata never silently overwritten (`merged_from` tracked).

## Temporal lifecycle

`observed_at` is set at extraction; `valid_from`/`valid_until` bound a claim
when the text states them; `state` ∈ active/superseded/uncertain tracks
decision evolution. Evolution is represented explicitly (`SUPERSEDES`,
`CONTRADICTS` edges) — never inferred from ingestion order.
