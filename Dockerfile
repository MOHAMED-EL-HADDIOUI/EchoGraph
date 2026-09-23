# EchoGraph API/worker image. Dev needs no Docker; this is for
# production-like deployments (Postgres + Neo4j + Redis + workers).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY backend/ ./backend/
COPY echograph/ ./echograph/
COPY alembic/ ./alembic/
COPY alembic.ini ./

RUN pip install --no-cache-dir -e ".[production]"

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
