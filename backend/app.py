from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import graph, health, ingestion, notifications
from backend.config import settings
from backend.database import init_db
from backend.deps import require_api_key
from backend.graph.manager import create_graph_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if not settings.API_KEY:
        logging.getLogger(__name__).warning("API_KEY unset: server is open (dev mode)")
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
    authed = [Depends(require_api_key)]
    app.include_router(graph.router, dependencies=authed)
    app.include_router(ingestion.router, dependencies=authed)
    app.include_router(notifications.router, dependencies=authed)
    return app


app = create_app()
