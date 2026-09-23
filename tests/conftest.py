from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app import app
from backend.database import Base, get_db
from backend.deps import get_graph_manager
from backend.graph.networkx_backend import NetworkXGraphManager


@pytest.fixture()
def tmp_graph(tmp_path):
    manager = NetworkXGraphManager(data_path=str(tmp_path / "test_graph.json"))
    app.state.graph = manager
    app.dependency_overrides[get_graph_manager] = lambda: manager
    yield manager
    app.dependency_overrides.pop(get_graph_manager, None)


@pytest.fixture()
async def test_db(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    yield session_factory
    app.dependency_overrides.pop(get_db, None)
    await engine.dispose()


@pytest.fixture()
async def client(tmp_graph, test_db):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
