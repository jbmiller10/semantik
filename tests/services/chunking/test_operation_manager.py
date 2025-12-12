"""Unit tests for :mod:`webui.services.chunking.operation_manager`."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shared.chunking.exceptions import (
    ChunkingDependencyError,
    ChunkingMemoryError,
    ChunkingResourceLimitError,
    ChunkingTimeoutError,
)
from webui.services.chunking.operation_manager import ChunkingOperationManager


class _StubClassifier:
    def __init__(self, code: str = "dependency_error") -> None:
        self._code = code

    def as_code(self, _exc: Exception) -> str:
        return self._code


class _StubErrorHandler:
    def __init__(self, *, resource_action: str = "ok", batch_size: int = 5) -> None:
        self._resource_action = resource_action
        self._batch_size = batch_size

    async def handle_resource_exhaustion(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(action=self._resource_action, wait_time=0)

    def _calculate_adaptive_batch_size(self, **_kwargs: object) -> int:
        return self._batch_size


class _FakeTime:
    def __init__(self) -> None:
        self.value = 0.0

    def time(self) -> float:  # noqa: D401 - simple proxy
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _FakeProcess:
    def __init__(self, *, memory_rss: int, cpu_user: float = 0.0, cpu_system: float = 0.0) -> None:
        self._memory_rss = memory_rss
        self._cpu_user = cpu_user
        self._cpu_system = cpu_system

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=self._memory_rss)

    def cpu_times(self) -> SimpleNamespace:
        return SimpleNamespace(user=self._cpu_user, system=self._cpu_system)


class _FakePsutil:
    def __init__(self, *, memory_percent: float, cpu_percent_value: float) -> None:
        self._memory_percent = memory_percent
        self._cpu_percent_value = cpu_percent_value

    def virtual_memory(self) -> SimpleNamespace:
        return SimpleNamespace(percent=self._memory_percent)

    def cpu_percent(self, *, interval: float) -> float:  # noqa: ARG002
        return self._cpu_percent_value


def _make_manager(**overrides: object) -> ChunkingOperationManager:
    redis_client = overrides.pop("redis_client", MagicMock())
    error_handler = overrides.pop("error_handler", _StubErrorHandler())
    classifier = overrides.pop("error_classifier", _StubClassifier())
    logger_ = overrides.pop("logger", logging.getLogger("tests.operation_manager"))

    return ChunkingOperationManager(
        redis_client=redis_client,
        error_handler=error_handler,  # type: ignore[arg-type]
        error_classifier=classifier,  # type: ignore[arg-type]
        logger=logger_,
        expected_circuit_breaker_exceptions=(ChunkingDependencyError,),
        **overrides,
    )


def test_handle_failure_opens_circuit_and_sends_dlq() -> None:
    redis_client = MagicMock()
    manager = _make_manager(redis_client=redis_client, failure_threshold=1, recovery_timeout=300)

    error_code = manager.handle_failure(
        exc=ChunkingDependencyError(detail="boom"),
        task_id="task-1",
        operation_id="op-1",
        correlation_id="corr-1",
        retry_count=1,
        max_retries=1,
        args=("op-1",),
        kwargs={},
    )

    assert error_code == "dependency_error"
    redis_client.rpush.assert_called_once()
    assert not manager.allow_execution()


def test_handle_retry_persists_state() -> None:
    redis_client = MagicMock()
    manager = _make_manager(redis_client=redis_client)

    manager.handle_retry(
        exc=ChunkingTimeoutError(detail="timeout"),
        task_id="task-2",
        operation_id="op-2",
        correlation_id="corr-2",
        retry_count=2,
    )

    redis_client.hset.assert_called_once()
    redis_client.expire.assert_called_once()


def test_circuit_breaker_moves_to_half_open_after_timeout() -> None:
    fake_time = _FakeTime()
    manager = _make_manager(failure_threshold=1, recovery_timeout=10, time_module=fake_time)

    manager.handle_failure(
        exc=ChunkingDependencyError(detail="boom"),
        task_id="task-3",
        operation_id="op-3",
        correlation_id="corr-3",
        retry_count=0,
        max_retries=3,
        args=(),
        kwargs={},
    )
    assert not manager.allow_execution()

    fake_time.advance(11)
    assert manager.allow_execution()


@pytest.mark.asyncio()
async def test_check_resource_limits_raises_on_memory_exhaustion() -> None:
    manager = _make_manager(
        psutil_module=_FakePsutil(memory_percent=95.0, cpu_percent_value=5.0),
        error_handler=_StubErrorHandler(resource_action="fail"),
    )

    with pytest.raises(ChunkingResourceLimitError):
        await manager.check_resource_limits(operation_id="op-4", correlation_id="corr-4")


@pytest.mark.asyncio()
async def test_monitor_resources_enforces_limits() -> None:
    manager = _make_manager()
    process = _FakeProcess(memory_rss=5 * 1024**3, cpu_user=2000, cpu_system=0)

    with pytest.raises(ChunkingMemoryError):
        await manager.monitor_resources(
            process=process,
            operation_id="op-5",
            initial_memory=0,
            initial_cpu_time=0,
            correlation_id="corr-5",
        )
