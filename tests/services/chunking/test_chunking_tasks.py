"""Task-level tests for :mod:`webui.chunking_tasks`."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from webui.api.chunking_exceptions import ChunkingDependencyError, ChunkingTimeoutError
from webui.chunking_tasks import ChunkingTask, _execute_chunking_task
from webui.services.chunking.operation_manager import ChunkingOperationManager


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
        "webui.chunking_tasks._process_chunking_operation_async",
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
        "webui.chunking_tasks._process_chunking_operation_async",
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
