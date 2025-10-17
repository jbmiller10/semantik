#!/usr/bin/env python3
"""Slim unit tests for lightweight helpers inside chunking_tasks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from packages.webui.chunking_tasks import (
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
    ChunkingTask,
    _calculate_batch_size,
)


@pytest.mark.anyio()
async def test_calculate_batch_size_delegates_to_error_handler(monkeypatch) -> None:
    """_calculate_batch_size should delegate to the error handler with current usage."""
    fake_memory = SimpleNamespace(percent=57.5)
    monkeypatch.setattr("packages.webui.chunking_tasks.psutil.virtual_memory", lambda: fake_memory)

    handler = MagicMock()
    handler._calculate_adaptive_batch_size.return_value = 8

    batch_size = await _calculate_batch_size(handler, initial_memory=0)

    handler._calculate_adaptive_batch_size.assert_called_once_with(current_usage=fake_memory.percent, limit=100)
    assert batch_size == 8


def test_chunking_task_circuit_breaker_transitions(monkeypatch) -> None:
    """Circuit breaker should transition closed -> open -> half-open after timeout."""
    task = ChunkingTask()

    # Trigger enough failures to open the circuit
    for _ in range(CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        task._update_circuit_breaker_state()

    assert task._circuit_breaker_state == "open"
    assert task._check_circuit_breaker() is False

    # Advance time beyond recovery timeout to transition to half-open
    last_failure = task._circuit_breaker_last_failure_time
    assert last_failure is not None

    monkeypatch.setattr(
        "packages.webui.chunking_tasks.time.time",
        lambda: last_failure + CIRCUIT_BREAKER_RECOVERY_TIMEOUT + 1,
    )

    assert task._check_circuit_breaker() is True
    assert task._circuit_breaker_state == "half_open"


def test_chunking_task_handle_shutdown_sets_flag() -> None:
    """_handle_shutdown should mark the task for graceful shutdown."""
    task = ChunkingTask()
    assert task._graceful_shutdown is False

    task._handle_shutdown(signum=15, frame=None)

    assert task._graceful_shutdown is True
