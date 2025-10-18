"""Task-level tests for :mod:`packages.webui.chunking_tasks`."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from celery.exceptions import SoftTimeLimitExceeded

from packages.shared.database.models import (
    CollectionStatus,
    DocumentStatus,
    OperationStatus,
    OperationType,
)
from packages.webui import chunking_tasks  # Import order intentional to avoid slowapi import at module load
from packages.webui.api.chunking_exceptions import ChunkingDependencyError, ChunkingTimeoutError
from packages.webui.chunking_tasks import ChunkingTask, _execute_chunking_task
from packages.webui.services.chunking.operation_manager import ChunkingOperationManager


class _StubClassifier:
    def as_code(self, exc: Exception) -> str:  # noqa: D401 - minimal stub
        return exc.__class__.__name__.replace("Chunking", "").lower()


def _patch_task_request(
    monkeypatch: pytest.MonkeyPatch,
    stub: SimpleNamespace,
) -> None:
    """Replace `ChunkingTask.request` property to return the provided stub."""

    def _request_property(self: ChunkingTask) -> SimpleNamespace:
        _ = self  # satisfy Ruff about unused parameter
        return stub

    monkeypatch.setattr(ChunkingTask, "request", property(_request_property))


def test_process_chunking_operation_invokes_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    task = ChunkingTask()

    class _Manager:
        def __init__(self) -> None:
            self.allow_called = 0

        def allow_execution(self) -> bool:
            self.allow_called += 1
            return True

    mgr = _Manager()
    monkeypatch.setattr(task, "_ensure_operation_manager", lambda: mgr)

    async def fake_async(**_: Any) -> dict[str, str]:
        return {"status": "success"}

    monkeypatch.setattr(
        "packages.webui.chunking_tasks._process_chunking_operation_async",
        fake_async,
    )

    request_stub = SimpleNamespace(id="task-1")
    _patch_task_request(monkeypatch, request_stub)

    result = _execute_chunking_task(task, "op-1", "corr-1")

    assert result["status"] == "success"
    assert mgr.allow_called == 1


def test_process_chunking_operation_circuit_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    task = ChunkingTask()

    class _Manager:
        def __init__(self) -> None:
            self.allow_called = 0

        def allow_execution(self) -> bool:
            self.allow_called += 1
            return False

    mgr = _Manager()
    monkeypatch.setattr(task, "_ensure_operation_manager", lambda: mgr)

    request_stub = SimpleNamespace(id="task-2")
    _patch_task_request(monkeypatch, request_stub)

    with pytest.raises(ChunkingDependencyError):
        _execute_chunking_task(task, "op-2", "corr-2")

    assert mgr.allow_called == 1


def test_chunking_task_on_retry_calls_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    task = ChunkingTask()
    manager = SimpleNamespace(handle_retry=MagicMock())
    monkeypatch.setattr(task, "_ensure_operation_manager", lambda: manager)
    request_stub = SimpleNamespace(retries=2)
    _patch_task_request(monkeypatch, request_stub)

    exc = ChunkingTimeoutError(detail="timeout", correlation_id="corr", operation_id="op")

    task.on_retry(exc, "task-id", ("op",), {"correlation_id": "corr"}, None)

    manager.handle_retry.assert_called_once()


def test_chunking_task_on_failure_publishes_dlq(monkeypatch: pytest.MonkeyPatch) -> None:
    redis_stub = MagicMock()
    classifier = _StubClassifier()
    manager = ChunkingOperationManager(
        redis_client=redis_stub,
        error_handler=MagicMock(),
        error_classifier=classifier,
        logger=logging.getLogger("test"),
        expected_circuit_breaker_exceptions=(ChunkingDependencyError,),
        failure_threshold=1,
        recovery_timeout=1,
    )

    task = ChunkingTask()
    monkeypatch.setattr(task, "_ensure_operation_manager", lambda: manager)
    request_stub = SimpleNamespace(retries=task.max_retries)
    _patch_task_request(monkeypatch, request_stub)

    exc = ChunkingDependencyError(detail="fail", correlation_id="corr", operation_id="op")

    task.on_failure(
        exc,
        "task-77",
        ("op",),
        {"correlation_id": "corr"},
        None,
    )

    assert redis_stub.rpush.call_count == 1
    dlq_key, payload = redis_stub.rpush.call_args.args
    assert dlq_key == "chunking:dlq:tasks"

    body = json.loads(payload)
    assert body["operation_id"] == "op"
    assert body["task_id"] == "task-77"
    assert body["args"] == ["op"]
    assert body["kwargs"] == {"correlation_id": "corr"}


def test_circuit_breaker_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTime:
        def __init__(self) -> None:
            self.value = 0.0

        def time(self) -> float:
            return self.value

        def advance(self, seconds: float) -> None:
            self.value += seconds

    time_stub = _FakeTime()
    redis_stub = MagicMock()
    manager = ChunkingOperationManager(
        redis_client=redis_stub,
        error_handler=MagicMock(),
        error_classifier=_StubClassifier(),
        logger=logging.getLogger("test"),
        expected_circuit_breaker_exceptions=(ChunkingDependencyError,),
        failure_threshold=1,
        recovery_timeout=5,
        time_module=time_stub,
    )

    task = ChunkingTask()
    task._operation_manager = manager
    monkeypatch.setattr(task, "_ensure_operation_manager", lambda: manager)

    async def fake_async(**_: Any) -> dict[str, str]:
        return {"status": "ok"}

    monkeypatch.setattr(
        "packages.webui.chunking_tasks._process_chunking_operation_async",
        fake_async,
    )

    # First call succeeds and keeps circuit closed.
    request_stub = SimpleNamespace(id="task-3", retries=0)
    _patch_task_request(monkeypatch, request_stub)

    result = _execute_chunking_task(task, "op-3", "corr-3")
    assert result["status"] == "ok"

    # Simulate failure to open the circuit.
    failure_exc = ChunkingDependencyError(detail="boom", correlation_id="corr-3", operation_id="op-3")
    request_stub.retries = task.max_retries
    task.on_failure(failure_exc, "task-1", ("op-3",), {"correlation_id": "corr-3"}, None)

    with pytest.raises(ChunkingDependencyError):
        _execute_chunking_task(task, "op-3", "corr-3")

    # Advance time to half-open state and ensure execution resumes.
    time_stub.advance(10)
    request_stub.retries = 0
    result_again = _execute_chunking_task(task, "op-3", "corr-3")
    assert result_again["status"] == "ok"


@pytest.mark.asyncio()
async def test_process_chunking_operation_async_updates_collection_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    task = ChunkingTask()

    class _Manager:
        def allow_execution(self) -> bool:
            return True

        async def check_resource_limits(self, **_: Any) -> None:
            return None

        async def monitor_resources(self, **_: Any) -> None:
            return None

        async def calculate_batch_size(self) -> int:
            return 1

    manager = _Manager()
    monkeypatch.setattr(task, "_ensure_operation_manager", lambda: manager)

    # Stub repositories to capture interactions
    operation = SimpleNamespace(
        collection_id="col-1",
        status=OperationStatus.PENDING,
        meta={},
        config={},
        type=OperationType.INDEX,
    )

    class _OperationRepo:
        def __init__(self) -> None:
            self.status_updates: list[tuple[str, OperationStatus]] = []

        async def get_by_uuid(self, _uuid: str) -> Any:
            return operation

        async def set_task_id(self, *_: Any, **__: Any) -> None:
            return None

        async def update_status(
            self,
            operation_id: str,
            status: OperationStatus,
            **_: Any,
        ) -> None:
            self.status_updates.append((operation_id, status))

    class _CollectionRepo:
        def __init__(self) -> None:
            self.update_calls: list[tuple[str, CollectionStatus]] = []

        async def get_by_uuid(self, _uuid: str) -> Any:
            return SimpleNamespace(id="col-1")

        async def update_status(self, collection_id: str, status: CollectionStatus) -> None:
            self.update_calls.append((collection_id, status))

    class _DocumentRepo:
        def __init__(self) -> None:
            self.status_updates: list[tuple[str, DocumentStatus]] = []

        async def list_by_collection(self, *_: Any, **__: Any) -> tuple[list[Any], int]:
            document = SimpleNamespace(
                id="doc-1",
                chunk_count=0,
                status=DocumentStatus.PENDING,
                file_path="dummy.txt",
            )
            return [document], 1

        async def update_status(
            self,
            document_id: str,
            status: DocumentStatus,
            **_: Any,
        ) -> None:
            self.status_updates.append((document_id, status))

    class _ChunkRepo:
        def __init__(self) -> None:
            self.bulk_calls: list[list[dict[str, Any]]] = []

        async def create_chunks_bulk(self, rows: list[dict[str, Any]]) -> int:
            self.bulk_calls.append(rows)
            return len(rows)

    operation_repo = _OperationRepo()
    collection_repo = _CollectionRepo()
    document_repo = _DocumentRepo()
    chunk_repo = _ChunkRepo()

    monkeypatch.setattr(chunking_tasks, "OperationRepository", lambda _db: operation_repo)
    monkeypatch.setattr(chunking_tasks, "CollectionRepository", lambda _db: collection_repo)
    monkeypatch.setattr(chunking_tasks, "DocumentRepository", lambda _db: document_repo)
    monkeypatch.setattr(chunking_tasks, "ChunkRepository", lambda _db: chunk_repo)

    async def fake_doc_chunking(
        *,
        chunk_repo: Any,
        document: Any,
        **__: Any,
    ) -> tuple[int, dict[str, Any]]:
        await chunk_repo.create_chunks_bulk(
            [
                {
                    "document_id": document.id,
                    "collection_id": operation.collection_id,
                    "chunk_index": 0,
                    "content": "stub",
                    "metadata": {},
                }
            ]
        )
        return 1, {"strategy": "test"}

    monkeypatch.setattr(chunking_tasks, "_process_document_chunking", fake_doc_chunking)

    async def _noop_progress(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(chunking_tasks, "_send_progress_update", _noop_progress)
    monkeypatch.setattr(chunking_tasks, "get_redis_client", lambda: SimpleNamespace())

    class _DbSession:
        async def commit(self) -> None:
            return None

        async def rollback(self) -> None:
            return None

        async def flush(self) -> None:
            return None

    class _SessionContext:
        async def __aenter__(self) -> Any:
            return _DbSession()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    def _session_factory() -> _SessionContext:
        return _SessionContext()

    monkeypatch.setattr(chunking_tasks, "AsyncSessionLocal", _session_factory, raising=False)

    async def fake_resolve_service(*_: Any, **__: Any) -> Any:
        return SimpleNamespace()

    monkeypatch.setattr(
        chunking_tasks,
        "resolve_celery_chunking_service",
        fake_resolve_service,
    )

    # Ensure initialization guard passes
    monkeypatch.setattr(chunking_tasks.pg_connection_manager, "_sessionmaker", object())
    monkeypatch.setattr(
        chunking_tasks.pg_connection_manager,
        "initialize",
        lambda: None,
    )

    result = await chunking_tasks._process_chunking_operation_async(
        operation_id="op-123",
        correlation_id="corr-123",
        celery_task=task,
    )

    assert result["status"] == "success"
    assert collection_repo.update_calls == [("col-1", CollectionStatus.READY)]
    assert operation_repo.status_updates[-1][1] is OperationStatus.COMPLETED
    assert chunk_repo.bulk_calls, "chunk rows should be persisted"


def test_execute_chunking_task_soft_timeout_invokes_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    task = ChunkingTask()

    class _Manager:
        def __init__(self) -> None:
            self.allow_called = 0

        def allow_execution(self) -> bool:
            self.allow_called += 1
            return True

    manager = _Manager()
    monkeypatch.setattr(task, "_ensure_operation_manager", lambda: manager)

    request_stub = SimpleNamespace(id="task-soft")
    _patch_task_request(monkeypatch, request_stub)

    handler_calls: list[tuple[str, str]] = []

    def _fake_handle_soft_timeout(operation_id: str, correlation_id: str, celery_task: ChunkingTask) -> None:
        handler_calls.append((operation_id, correlation_id))
        assert celery_task is task

    async def _fake_async(**_: Any) -> None:
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr(chunking_tasks, "_handle_soft_timeout_sync", _fake_handle_soft_timeout)
    monkeypatch.setattr(chunking_tasks, "_process_chunking_operation_async", _fake_async)

    with pytest.raises(ChunkingTimeoutError) as exc_info:
        _execute_chunking_task(task, "op-soft", "corr-soft")

    assert manager.allow_called == 1
    assert handler_calls == [("op-soft", "corr-soft")]
    assert exc_info.value.operation_id == "op-soft"
