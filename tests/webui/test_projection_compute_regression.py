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

import json

import numpy as np

import webui.tasks.projection as projection_module


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


class _NoGuardSession(_DummySession):
    """Session variant whose ensure_open does not raise after close."""

    def ensure_open(self) -> None:  # type: ignore[override]
        # Intentionally a no-op for happy-path artifact tests.
        return None


class _BaseRepo:
    def __init__(self, session: _DummySession) -> None:
        self._session = session

    def _ensure_open(self) -> None:
        self._session.ensure_open()


class _ProjectionRunRepository(_BaseRepo):
    def __init__(self, session: _DummySession) -> None:
        super().__init__(session)
        self._runs: dict[str, SimpleNamespace] = {}

    # Defaults can be overridden in individual tests to exercise
    # different reducer/config combinations without changing wiring.
    default_collection_id: str = "collection-1"
    default_config: dict[str, Any] = {}
    default_reducer: str = "pca"
    default_operation_uuid: str = "operation-1"
    default_meta: dict[str, Any] = {}

    def _get_or_create(self, projection_uuid: str) -> SimpleNamespace:
        run = self._runs.get(projection_uuid)
        if run is None:
            run = SimpleNamespace(
                uuid=projection_uuid,
                collection_id=self.default_collection_id,
                config=dict(self.default_config),
                reducer=self.default_reducer,
                operation_uuid=self.default_operation_uuid,
                meta=dict(self.default_meta),
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


@asynccontextmanager
async def _happy_session_factory() -> AsyncIterator[_NoGuardSession]:
    """Session factory that does not fail on ensure_open after close."""

    session = _NoGuardSession()
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


def test_compute_projection_writes_canonical_artifacts_and_meta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Successful projection writes canonical artifacts and meta.json."""

    monkeypatch.setattr(projection_module, "settings", SimpleNamespace(data_dir=tmp_path))
    # Use a session factory that does not trigger the closed-session error so we
    # can inspect successful artifact generation.
    monkeypatch.setattr(
        projection_module,
        "PostgresConnectionManager",
        lambda: _FakePostgresConnectionManager(_happy_session_factory),
    )
    monkeypatch.setattr(projection_module, "ProjectionRunRepository", _ProjectionRunRepository)
    monkeypatch.setattr(projection_module, "OperationRepository", _OperationRepository)
    monkeypatch.setattr(projection_module, "CollectionRepository", _CollectionRepository)
    monkeypatch.setattr(projection_module, "resolve_qdrant_manager", lambda: _FakeQdrantManager())
    monkeypatch.setattr(projection_module, "_operation_updates", _fake_operation_updates)
    monkeypatch.setattr(projection_module, "resolve_awaitable_sync", _resolve_sync)

    projection_id = "run-artifacts"
    result = projection_module.compute_projection(projection_id)

    assert result["status"] == "completed"

    run_dir = tmp_path / "semantik" / "projections" / "collection-1" / projection_id
    assert run_dir.is_dir()

    x_path = run_dir / "x.f32.bin"
    y_path = run_dir / "y.f32.bin"
    ids_path = run_dir / "ids.i32.bin"
    cat_path = run_dir / "cat.u8.bin"
    meta_path = run_dir / "meta.json"

    # All canonical artifacts must be present
    for path in (x_path, y_path, ids_path, cat_path, meta_path):
        assert path.is_file(), f"Missing projection artifact: {path.name}"

    # Binary arrays must share the same point_count
    x_values = np.fromfile(x_path, dtype=np.float32)
    y_values = np.fromfile(y_path, dtype=np.float32)
    ids_values = np.fromfile(ids_path, dtype=np.int32)
    cat_values = np.fromfile(cat_path, dtype=np.uint8)

    point_count = len(x_values)
    assert point_count >= 2
    assert len(y_values) == point_count
    assert len(ids_values) == point_count
    assert len(cat_values) == point_count

    # Meta payload must align with artifacts
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta_payload["projection_id"] == projection_id
    assert meta_payload["collection_id"] == "collection-1"
    assert meta_payload["point_count"] == point_count
    assert meta_payload["shown_count"] == point_count
    assert meta_payload["files"]["x"] == "x.f32.bin"
    assert meta_payload["files"]["y"] == "y.f32.bin"
    assert meta_payload["files"]["ids"] == "ids.i32.bin"
    assert meta_payload["files"]["categories"] == "cat.u8.bin"
    assert meta_payload["original_ids"] == ["1", "2"]
    # Category counts should be consistent with the cat.u8.bin array
    category_counts = {int(idx): count for idx, count in meta_payload["category_counts"].items()}
    for idx in cat_values:
        assert category_counts[int(idx)] >= 1


def test_compute_projection_umap_failure_falls_back_to_pca(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """UMAP reducer failures should fall back to PCA and mark run degraded."""

    monkeypatch.setattr(projection_module, "settings", SimpleNamespace(data_dir=tmp_path))
    # Use happy-path session factory so artifacts are written.
    monkeypatch.setattr(
        projection_module,
        "PostgresConnectionManager",
        lambda: _FakePostgresConnectionManager(_happy_session_factory),
    )

    # Configure the stub repository so that the projection run requests UMAP
    # with explicit reducer config that should be preserved when falling back.
    _ProjectionRunRepository.default_reducer = "umap"
    _ProjectionRunRepository.default_config = {
        "color_by": "document_id",
        "n_neighbors": 10,
        "min_dist": 0.05,
        "metric": "cosine",
    }

    monkeypatch.setattr(projection_module, "ProjectionRunRepository", _ProjectionRunRepository)
    monkeypatch.setattr(projection_module, "OperationRepository", _OperationRepository)
    monkeypatch.setattr(projection_module, "CollectionRepository", _CollectionRepository)
    monkeypatch.setattr(projection_module, "resolve_qdrant_manager", lambda: _FakeQdrantManager())
    monkeypatch.setattr(projection_module, "_operation_updates", _fake_operation_updates)
    monkeypatch.setattr(projection_module, "resolve_awaitable_sync", _resolve_sync)

    # Force the UMAP path to fail so that the implementation exercises the
    # PCA fallback branch and records a fallback_reason.
    def _failing_umap(*_: Any, **__: Any) -> dict[str, Any]:
        raise RuntimeError("umap reducer exploded")

    monkeypatch.setattr(projection_module, "_compute_umap_projection", _failing_umap)

    projection_id = "run-umap-fallback"
    result = projection_module.compute_projection(projection_id)

    assert result["status"] == "completed"
    assert result["projection_id"] == projection_id

    run_dir = tmp_path / "semantik" / "projections" / "collection-1" / projection_id
    meta_path = run_dir / "meta.json"
    assert meta_path.is_file()

    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))

    # Reducer metadata should record that UMAP was requested but PCA used.
    assert meta_payload["reducer_requested"] == "umap"
    assert meta_payload["reducer_used"] == "pca"
    assert meta_payload["reducer_params"] == {
        "n_neighbors": 10,
        "min_dist": 0.05,
        "metric": "cosine",
    }
    assert meta_payload.get("fallback_reason")
    assert meta_payload.get("degraded") is True


class _OverflowQdrantClient:
    """Qdrant stub that produces many distinct categories to hit overflow."""

    def __init__(self, num_points: int = 260) -> None:
        self._records: list[Any] = []
        for idx in range(num_points):
            self._records.append(
                SimpleNamespace(
                    id=str(idx + 1),
                    vector=[0.1 * (idx + 1), 0.2, 0.3],
                    payload={"doc_id": f"doc-{idx + 1}"},
                )
            )
        self._offset = 0

    def scroll(self, *, limit: int = 1000, **_: Any) -> tuple[list[Any], Any]:
        if self._offset >= len(self._records):
            return [], None
        batch = self._records[self._offset : self._offset + limit]
        self._offset += len(batch)
        return batch, None


class _OverflowQdrantManager:
    def __init__(self, num_points: int = 260) -> None:
        self.client = _OverflowQdrantClient(num_points=num_points)


def test_compute_projection_uses_overflow_category_bucket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Legend entries and category counts must respect the overflow bucket semantics."""

    monkeypatch.setattr(projection_module, "settings", SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(
        projection_module,
        "PostgresConnectionManager",
        lambda: _FakePostgresConnectionManager(_happy_session_factory),
    )

    _ProjectionRunRepository.default_reducer = "pca"
    _ProjectionRunRepository.default_config = {"color_by": "document_id"}

    monkeypatch.setattr(projection_module, "ProjectionRunRepository", _ProjectionRunRepository)
    monkeypatch.setattr(projection_module, "OperationRepository", _OperationRepository)
    monkeypatch.setattr(projection_module, "CollectionRepository", _CollectionRepository)
    monkeypatch.setattr(projection_module, "resolve_qdrant_manager", lambda: _OverflowQdrantManager())
    monkeypatch.setattr(projection_module, "_operation_updates", _fake_operation_updates)
    monkeypatch.setattr(projection_module, "resolve_awaitable_sync", _resolve_sync)

    projection_id = "run-overflow-categories"
    result = projection_module.compute_projection(projection_id)

    assert result["status"] == "completed"

    run_dir = tmp_path / "semantik" / "projections" / "collection-1" / projection_id
    cat_path = run_dir / "cat.u8.bin"
    meta_path = run_dir / "meta.json"
    assert cat_path.is_file()
    assert meta_path.is_file()

    cat_values = np.fromfile(cat_path, dtype=np.uint8)
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))

    legend = meta_payload["legend"]
    legend_by_index = {entry["index"]: entry for entry in legend}

    # Every non-overflow index present in cat.u8.bin must have a legend entry.
    from webui.tasks.projection import OVERFLOW_CATEGORY_INDEX, OVERFLOW_LEGEND_LABEL

    for idx in {int(v) for v in cat_values if int(v) < OVERFLOW_CATEGORY_INDEX}:
        assert idx in legend_by_index

    # Overflow bucket must be present with the canonical label and aggregated count.
    assert OVERFLOW_CATEGORY_INDEX in legend_by_index
    overflow_entry = legend_by_index[OVERFLOW_CATEGORY_INDEX]
    assert overflow_entry["label"] == OVERFLOW_LEGEND_LABEL
    assert overflow_entry["count"] == int((cat_values == OVERFLOW_CATEGORY_INDEX).sum())

    # No legend entries should use indices beyond the overflow bucket.
    assert all(entry["index"] <= OVERFLOW_CATEGORY_INDEX for entry in legend)
