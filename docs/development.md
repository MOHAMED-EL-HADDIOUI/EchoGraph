# Development

```bash
cp .env.example .env
pip install -e ".[dev]"
uvicorn backend.app:app --reload
ruff check backend tests echograph eval
ruff format --check backend tests echograph eval
pytest -q
python -m echograph.eval
```

Conventions: `from __future__ import annotations`; pure-Pydantic models;
`GraphManager` ABC for new graph ops (both backends); injectable provider
seams (`app.dependency_overrides` in tests); no real network in tests.

Key files: `AGENTS.md` (agent playbook), `docs/architecture.md` (pipeline +
diagrams), `docs/evaluation.md` (quality workflow), `docs/api.md` (contracts).
