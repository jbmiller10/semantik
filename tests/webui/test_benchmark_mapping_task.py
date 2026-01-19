from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from shared.database.models import OperationStatus
from webui.tasks import benchmark_mapping


class _FakeProgressReporter:
    def __init__(self, operation_uuid: str):
        self.operation_uuid = operation_uuid
        self._closed = False

    def set_user_id(self, _user_id: int) -> None:
        return None

    def set_collection_id(self, _collection_id: str | None) -> None:
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def close(self) -> None:
        self._closed = True


@pytest.mark.asyncio()
async def test_resolve_mapping_async_returns_failed_when_operation_missing(monkeypatch) -> None:
    session = AsyncMock()
    session.commit = AsyncMock()

    @asynccontextmanager
    async def session_cm():
        yield session

    session_factory = MagicMock(return_value=session_cm())

    operation_repo = AsyncMock()
    operation_repo.get_by_uuid.return_value = None

    with (
        patch.object(benchmark_mapping, "_resolve_session_factory", AsyncMock(return_value=session_factory)),
        patch("shared.database.repositories.operation_repository.OperationRepository", return_value=operation_repo),
        patch("shared.database.repositories.benchmark_dataset_repository.BenchmarkDatasetRepository", return_value=AsyncMock()),
        patch("shared.database.repositories.collection_repository.CollectionRepository", return_value=AsyncMock()),
        patch("shared.database.repositories.document_repository.DocumentRepository", return_value=AsyncMock()),
        patch.object(benchmark_mapping, "CeleryTaskWithOperationUpdates", _FakeProgressReporter),
    ):
        result = await benchmark_mapping._resolve_mapping_async(
            operation_uuid="op-1",
            mapping_id=10,
            user_id=1,
            task_id="task-1",
        )

    assert result["status"] == "failed"
    assert "operation not found" in (result["error"] or "").lower()


@pytest.mark.asyncio()
async def test_resolve_mapping_async_marks_completed_and_returns_result(monkeypatch) -> None:
    session = AsyncMock()
    session.commit = AsyncMock()

    @asynccontextmanager
    async def session_cm():
        yield session

    session_factory = MagicMock(return_value=session_cm())

    operation = SimpleNamespace(uuid="op-1", collection_id="col-1")

    operation_repo = AsyncMock()
    operation_repo.get_by_uuid.return_value = operation
    operation_repo.set_task_id = AsyncMock()
    operation_repo.update_status = AsyncMock()

    service_instance = AsyncMock()
    service_instance.resolve_mapping_with_progress.return_value = {"mapping_id": 10, "mapped_count": 1, "total_count": 2}

    BenchmarkDatasetService = Mock(return_value=service_instance)

    with (
        patch.object(benchmark_mapping, "_resolve_session_factory", AsyncMock(return_value=session_factory)),
        patch("shared.database.repositories.operation_repository.OperationRepository", return_value=operation_repo),
        patch("shared.database.repositories.benchmark_dataset_repository.BenchmarkDatasetRepository", return_value=AsyncMock()),
        patch("shared.database.repositories.collection_repository.CollectionRepository", return_value=AsyncMock()),
        patch("shared.database.repositories.document_repository.DocumentRepository", return_value=AsyncMock()),
        patch("webui.services.benchmark_dataset_service.BenchmarkDatasetService", BenchmarkDatasetService),
        patch.object(benchmark_mapping, "CeleryTaskWithOperationUpdates", _FakeProgressReporter),
    ):
        result = await benchmark_mapping._resolve_mapping_async(
            operation_uuid="op-1",
            mapping_id=10,
            user_id=1,
            task_id="task-1",
        )

    assert result["status"] == "completed"
    assert result["mapping_id"] == 10
    operation_repo.update_status.assert_any_await("op-1", OperationStatus.PROCESSING)
    operation_repo.update_status.assert_any_await("op-1", OperationStatus.COMPLETED)


@pytest.mark.asyncio()
async def test_resolve_mapping_async_sets_failed_status_on_exception() -> None:
    session = AsyncMock()
    session.commit = AsyncMock()

    @asynccontextmanager
    async def session_cm():
        yield session

    session_factory = MagicMock(return_value=session_cm())

    operation = SimpleNamespace(uuid="op-1", collection_id="col-1")
    operation_repo = AsyncMock()
    operation_repo.get_by_uuid.return_value = operation
    operation_repo.set_task_id = AsyncMock()
    operation_repo.update_status = AsyncMock()

    service_instance = AsyncMock()
    service_instance.resolve_mapping_with_progress.side_effect = RuntimeError("boom")
    BenchmarkDatasetService = Mock(return_value=service_instance)

    with (
        patch.object(benchmark_mapping, "_resolve_session_factory", AsyncMock(return_value=session_factory)),
        patch("shared.database.repositories.operation_repository.OperationRepository", return_value=operation_repo),
        patch("shared.database.repositories.benchmark_dataset_repository.BenchmarkDatasetRepository", return_value=AsyncMock()),
        patch("shared.database.repositories.collection_repository.CollectionRepository", return_value=AsyncMock()),
        patch("shared.database.repositories.document_repository.DocumentRepository", return_value=AsyncMock()),
        patch("webui.services.benchmark_dataset_service.BenchmarkDatasetService", BenchmarkDatasetService),
        patch.object(benchmark_mapping, "CeleryTaskWithOperationUpdates", _FakeProgressReporter),
    ):
        result = await benchmark_mapping._resolve_mapping_async(
            operation_uuid="op-1",
            mapping_id=10,
            user_id=1,
            task_id="task-1",
        )

    assert result["status"] == "failed"
    assert result["error"] == "boom"
    operation_repo.update_status.assert_any_await("op-1", OperationStatus.FAILED, error_message="boom"[:1000])

