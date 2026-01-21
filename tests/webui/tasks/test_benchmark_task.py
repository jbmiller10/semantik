"""Tests for webui.tasks.benchmark task orchestration.

These tests are intentionally unit-level: we patch repositories/services to avoid requiring
Postgres/Redis/Qdrant, and validate the control flow and status updates.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from shared.database.models import BenchmarkStatus, OperationStatus
from webui.tasks import benchmark as benchmark_module


class MockUpdater:
    def __init__(self, operation_id: str):
        self.operation_id = operation_id
        self.user_id: int | None = None
        self.closed = False

    def set_user_id(self, user_id: int | None) -> None:
        self.user_id = user_id

    async def __aenter__(self) -> MockUpdater:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def close(self) -> None:
        self.closed = True


@asynccontextmanager
async def _mock_session_context(session: Any):
    yield session


def _make_session_factory(session: Any):
    class _Factory:
        def __call__(self):  # noqa: ANN001
            return _mock_session_context(session)

    return _Factory()


@pytest.mark.asyncio()
async def test_run_benchmark_returns_failed_when_operation_missing(monkeypatch) -> None:
    session = SimpleNamespace(commit=AsyncMock())
    monkeypatch.setattr(
        benchmark_module, "_resolve_session_factory", AsyncMock(return_value=_make_session_factory(session))
    )
    monkeypatch.setattr(benchmark_module, "CeleryTaskWithOperationUpdates", MockUpdater)

    op_repo = AsyncMock()
    op_repo.get_by_uuid = AsyncMock(return_value=None)

    class FakeOperationRepo:  # noqa: D401
        """Fake OperationRepository."""

        def __init__(self, _session: Any):
            pass

        async def get_by_uuid(self, uuid: str):  # noqa: ANN001
            return await op_repo.get_by_uuid(uuid)

    monkeypatch.setattr("shared.database.repositories.operation_repository.OperationRepository", FakeOperationRepo)
    monkeypatch.setattr("shared.database.repositories.benchmark_repository.BenchmarkRepository", lambda _s: AsyncMock())
    monkeypatch.setattr(
        "shared.database.repositories.benchmark_dataset_repository.BenchmarkDatasetRepository",
        lambda _s: AsyncMock(),
    )
    monkeypatch.setattr(
        "shared.database.repositories.collection_repository.CollectionRepository", lambda _s: AsyncMock()
    )
    monkeypatch.setattr("webui.services.search_service.SearchService", lambda **_kw: AsyncMock())
    monkeypatch.setattr("webui.services.benchmark_executor.BenchmarkExecutor", lambda **_kw: AsyncMock())

    result = await benchmark_module._run_benchmark_async("op-missing", "bench-1", "task-1")
    assert result["status"] == "failed"
    assert result["error"] == "Operation not found: op-missing"


@pytest.mark.asyncio()
async def test_run_benchmark_updates_operation_completed_on_success(monkeypatch) -> None:
    session = SimpleNamespace(commit=AsyncMock())
    monkeypatch.setattr(
        benchmark_module, "_resolve_session_factory", AsyncMock(return_value=_make_session_factory(session))
    )
    monkeypatch.setattr(benchmark_module, "CeleryTaskWithOperationUpdates", MockUpdater)

    operation = SimpleNamespace(user_id=123)

    operation_repo = AsyncMock()
    operation_repo.get_by_uuid = AsyncMock(return_value=operation)
    operation_repo.set_task_id = AsyncMock()
    operation_repo.update_status = AsyncMock()

    benchmark_repo = AsyncMock()

    class FakeOperationRepo:
        def __init__(self, _session: Any):
            pass

        async def get_by_uuid(self, uuid: str):  # noqa: ANN001
            return await operation_repo.get_by_uuid(uuid)

        async def set_task_id(self, uuid: str, task_id: str):  # noqa: ANN001
            return await operation_repo.set_task_id(uuid, task_id)

        async def update_status(
            self, uuid: str, status: OperationStatus, error_message: str | None = None
        ):  # noqa: ANN001
            return await operation_repo.update_status(uuid, status, error_message=error_message)

    class FakeBenchmarkRepo:
        def __init__(self, _session: Any):
            pass

        async def update_status(self, benchmark_id: str, status: BenchmarkStatus):  # noqa: ANN001
            return await benchmark_repo.update_status(benchmark_id, status)

    class FakeBenchmarkDatasetRepo:
        def __init__(self, _session: Any):
            pass

    class FakeCollectionRepo:
        def __init__(self, _session: Any):
            pass

    class FakeSearchService:
        def __init__(self, **_kwargs: Any):
            pass

    executor = AsyncMock()
    executor.execute_benchmark = AsyncMock(return_value={"status": BenchmarkStatus.COMPLETED.value, "error": None})

    class FakeExecutor:
        def __init__(self, **_kwargs: Any):
            pass

        async def execute_benchmark(self, benchmark_id: str):  # noqa: ANN001
            return await executor.execute_benchmark(benchmark_id)

    monkeypatch.setattr("shared.database.repositories.operation_repository.OperationRepository", FakeOperationRepo)
    monkeypatch.setattr("shared.database.repositories.benchmark_repository.BenchmarkRepository", FakeBenchmarkRepo)
    monkeypatch.setattr(
        "shared.database.repositories.benchmark_dataset_repository.BenchmarkDatasetRepository",
        FakeBenchmarkDatasetRepo,
    )
    monkeypatch.setattr("shared.database.repositories.collection_repository.CollectionRepository", FakeCollectionRepo)
    monkeypatch.setattr("webui.services.search_service.SearchService", FakeSearchService)
    monkeypatch.setattr("webui.services.benchmark_executor.BenchmarkExecutor", FakeExecutor)

    result = await benchmark_module._run_benchmark_async("op-1", "bench-1", "task-1")
    assert result["status"] == BenchmarkStatus.COMPLETED.value
    operation_repo.update_status.assert_any_call("op-1", OperationStatus.PROCESSING, error_message=None)
    operation_repo.update_status.assert_any_call("op-1", OperationStatus.COMPLETED, error_message=None)


@pytest.mark.asyncio()
async def test_run_benchmark_updates_operation_failed_on_executor_exception(monkeypatch) -> None:
    session = SimpleNamespace(commit=AsyncMock())
    monkeypatch.setattr(
        benchmark_module, "_resolve_session_factory", AsyncMock(return_value=_make_session_factory(session))
    )
    monkeypatch.setattr(benchmark_module, "CeleryTaskWithOperationUpdates", MockUpdater)

    operation = SimpleNamespace(user_id=123)

    operation_repo = AsyncMock()
    operation_repo.get_by_uuid = AsyncMock(return_value=operation)
    operation_repo.set_task_id = AsyncMock()
    operation_repo.update_status = AsyncMock()

    benchmark_repo = AsyncMock()
    benchmark_repo.update_status = AsyncMock()

    class FakeOperationRepo:
        def __init__(self, _session: Any):
            pass

        async def get_by_uuid(self, uuid: str):  # noqa: ANN001
            return await operation_repo.get_by_uuid(uuid)

        async def set_task_id(self, uuid: str, task_id: str):  # noqa: ANN001
            return await operation_repo.set_task_id(uuid, task_id)

        async def update_status(
            self, uuid: str, status: OperationStatus, error_message: str | None = None
        ):  # noqa: ANN001
            return await operation_repo.update_status(uuid, status, error_message=error_message)

    class FakeBenchmarkRepo:
        def __init__(self, _session: Any):
            pass

        async def update_status(self, benchmark_id: str, status: BenchmarkStatus):  # noqa: ANN001
            return await benchmark_repo.update_status(benchmark_id, status)

    class FakeBenchmarkDatasetRepo:
        def __init__(self, _session: Any):
            pass

    class FakeCollectionRepo:
        def __init__(self, _session: Any):
            pass

    class FakeSearchService:
        def __init__(self, **_kwargs: Any):
            pass

    class FakeExecutor:
        def __init__(self, **_kwargs: Any):
            pass

        async def execute_benchmark(self, _benchmark_id: str):  # noqa: ANN001
            raise RuntimeError("boom")

    monkeypatch.setattr("shared.database.repositories.operation_repository.OperationRepository", FakeOperationRepo)
    monkeypatch.setattr("shared.database.repositories.benchmark_repository.BenchmarkRepository", FakeBenchmarkRepo)
    monkeypatch.setattr(
        "shared.database.repositories.benchmark_dataset_repository.BenchmarkDatasetRepository",
        FakeBenchmarkDatasetRepo,
    )
    monkeypatch.setattr("shared.database.repositories.collection_repository.CollectionRepository", FakeCollectionRepo)
    monkeypatch.setattr("webui.services.search_service.SearchService", FakeSearchService)
    monkeypatch.setattr("webui.services.benchmark_executor.BenchmarkExecutor", FakeExecutor)

    result = await benchmark_module._run_benchmark_async("op-1", "bench-1", "task-1")
    assert result["status"] == "failed"
    assert "boom" in (result["error"] or "")
    benchmark_repo.update_status.assert_awaited()
    operation_repo.update_status.assert_awaited()
