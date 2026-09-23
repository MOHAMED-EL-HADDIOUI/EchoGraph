from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import graph, health, ingestion, notifications
from backend.config import settings
from backend.database import init_db
from backend.graph.manager import create_graph_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    app.state.graph = create_graph_manager(settings.GRAPH_BACKEND)
    init_constraints = getattr(app.state.graph, "init_constraints", None)
    if callable(init_constraints):
        await init_constraints()
    yield
    close = getattr(app.state.graph, "close", None)
    if callable(close):
        await close()


def create_app() -> FastAPI:
    app = FastAPI(title="EchoGraph", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(graph.router)
    app.include_router(ingestion.router)
    app.include_router(notifications.router)
    return app


app = create_app()
