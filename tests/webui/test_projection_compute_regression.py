"""Regression test for compute_projection session reuse bug."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

    import pytest

import packages.webui.tasks.projection as projection_module


class _DummySession:
    """Minimal async session stub supporting closed-session detection."""

    def __init__(self) -> None:
        self._closed = False

    def ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Session is closed")

    async def commit(self) -> None:
        self.ensure_open()

    async def rollback(self) -> None:
        self.ensure_open()

    async def close(self) -> None:
        self._closed = True


class _BaseRepo:
    def __init__(self, session: _DummySession) -> None:
        self._session = session

    def _ensure_open(self) -> None:
        self._session.ensure_open()


class _ProjectionRunRepository(_BaseRepo):
    def __init__(self, session: _DummySession) -> None:
        super().__init__(session)
        self._runs: dict[str, SimpleNamespace] = {}

    def _get_or_create(self, projection_uuid: str) -> SimpleNamespace:
        run = self._runs.get(projection_uuid)
        if run is None:
            run = SimpleNamespace(
                uuid=projection_uuid,
                collection_id="collection-1",
                config={},
                reducer="pca",
                operation_uuid="operation-1",
                meta={},
                status=None,
            )
            self._runs[projection_uuid] = run
        return run

    async def get_by_uuid(self, projection_uuid: str) -> SimpleNamespace:
        self._ensure_open()
        return self._get_or_create(projection_uuid)

    async def update_status(
        self,
        projection_uuid: str,
        *,
        status: Any,
        error_message: str | None = None,
        started_at: Any = None,
        completed_at: Any = None,
    ) -> None:
        self._ensure_open()
        run = self._get_or_create(projection_uuid)
        run.status = status
        run.error_message = error_message
        run.started_at = started_at
        run.completed_at = completed_at

    async def update_metadata(
        self,
        projection_uuid: str,
        *,
        storage_path: str | None = None,
        point_count: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self._ensure_open()
        run = self._get_or_create(projection_uuid)
        if storage_path is not None:
            run.storage_path = storage_path
        if point_count is not None:
            run.point_count = point_count
        if meta:
            merged = dict(run.meta)
            merged.update(meta)
            run.meta = merged


class _OperationRepository(_BaseRepo):
    async def update_status(self, *args: Any, **kwargs: Any) -> None:
        self._ensure_open()


class _CollectionRepository(_BaseRepo):
    async def get_by_uuid(self, collection_uuid: str) -> SimpleNamespace:
        self._ensure_open()
        return SimpleNamespace(
            uuid=collection_uuid,
            vector_store_name="vector-collection",
            vector_count=2,
        )


class _FakeQdrantClient:
    def __init__(self) -> None:
        self._call_count = 0

    def scroll(self, *args: Any, **kwargs: Any) -> tuple[list[Any], Any]:
        if self._call_count:
            return [], None
        self._call_count += 1
        record1 = SimpleNamespace(id="1", vector=[0.1, 0.2, 0.3], payload={"doc_id": "doc-1"})
        record2 = SimpleNamespace(id="2", vector=[0.4, 0.5, 0.6], payload={"doc_id": "doc-2"})
        return [record1, record2], None


class _FakeQdrantManager:
    def __init__(self) -> None:
        self.client = _FakeQdrantClient()


class _FakePostgresConnectionManager:
    def __init__(self, session_factory: Callable[[], AsyncIterator[_DummySession]]) -> None:
        self._sessionmaker = session_factory

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None


@asynccontextmanager
async def _session_factory() -> AsyncIterator[_DummySession]:
    session = _DummySession()
    try:
        yield session
    finally:
        await session.close()


class _FakeUpdater:
    async def send_update(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def close(self) -> None:
        return None


@asynccontextmanager
async def _fake_operation_updates(operation_id: str | None) -> AsyncIterator[_FakeUpdater | None]:
    if not operation_id:
        yield None
        return
    updater = _FakeUpdater()
    try:
        yield updater
    finally:
        await updater.close()


def _resolve_sync(value: Any) -> Any:
    if inspect.isawaitable(value):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(value)
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
    return value


def test_compute_projection_reports_closed_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Compute projection should surface the closed-session failure from the finally block."""

    monkeypatch.setattr(projection_module, "settings", SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(
        projection_module,
        "PostgresConnectionManager",
        lambda: _FakePostgresConnectionManager(_session_factory),
    )
    monkeypatch.setattr(projection_module, "ProjectionRunRepository", _ProjectionRunRepository)
    monkeypatch.setattr(projection_module, "OperationRepository", _OperationRepository)
    monkeypatch.setattr(projection_module, "CollectionRepository", _CollectionRepository)
    monkeypatch.setattr(projection_module, "resolve_qdrant_manager", lambda: _FakeQdrantManager())
    monkeypatch.setattr(projection_module, "_operation_updates", _fake_operation_updates)
    monkeypatch.setattr(projection_module, "_write_binary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(projection_module, "_write_meta", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(projection_module, "resolve_awaitable_sync", _resolve_sync)

    result = projection_module.compute_projection("run-123")

    assert result["status"] == "failed"
    assert "Session is closed" in (result.get("message") or "")

