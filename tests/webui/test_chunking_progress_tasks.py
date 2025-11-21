"""Tests for chunking task progress helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from webui import chunking_tasks
from webui.services.progress_manager import ProgressSendResult


class FakeProgressManager:
    def __init__(self, result: ProgressSendResult = ProgressSendResult.SENT) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self._result = result

    def send_sync_update(self, payload, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append((payload, kwargs))
        return self._result


def test_send_progress_update_sync_uses_shared_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = FakeProgressManager()
    redis_client = MagicMock()

    monkeypatch.setattr(chunking_tasks, "get_progress_update_manager", lambda: manager)

    chunking_tasks._send_progress_update_sync(
        redis_client,
        operation_id="op-1",
        correlation_id="corr-1",
        progress=10,
        message="hello",
    )

    assert len(manager.calls) == 1
    payload, kwargs = manager.calls[0]
    assert payload.operation_id == "op-1"
    assert payload.progress == 10
    assert kwargs["redis_client"] is redis_client
    assert kwargs["stream_template"] == "stream:chunking:{operation_id}"
    assert kwargs["maxlen"] == 1000
    assert kwargs["ttl"] is None


@pytest.mark.asyncio()
async def test_send_progress_update_async_delegates_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    manager = FakeProgressManager(result=ProgressSendResult.FAILED)
    redis_client = MagicMock()

    monkeypatch.setattr(chunking_tasks, "get_progress_update_manager", lambda: manager)

    caplog.set_level("ERROR")
    await chunking_tasks._send_progress_update(
        redis_client,
        operation_id="op-2",
        correlation_id="corr-2",
        progress=55,
        message="progress",
    )

    assert len(manager.calls) == 1
    payload, kwargs = manager.calls[0]
    assert payload.operation_id == "op-2"
    assert kwargs["ttl"] == 3600
    assert kwargs["maxlen"] == 0
    assert "Failed to send progress update" in caplog.text


@pytest.mark.asyncio()
async def test_send_progress_update_async_success(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = FakeProgressManager()
    redis_client = MagicMock()

    monkeypatch.setattr(chunking_tasks, "get_progress_update_manager", lambda: manager)

    await chunking_tasks._send_progress_update(
        redis_client,
        operation_id="op-3",
        correlation_id="corr-3",
        progress=90,
        message="almost",
    )

    assert len(manager.calls) == 1
    payload, kwargs = manager.calls[0]
    assert payload.progress == 90
    assert kwargs["redis_client"] is redis_client
