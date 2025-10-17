#!/usr/bin/env python3
"""Integration tests covering chunking progress updates via Redis streams."""

from __future__ import annotations

import pytest

from packages.webui.chunking_tasks import _send_progress_update

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def test_send_progress_update_writes_stream_entry(use_fakeredis) -> None:
    """_send_progress_update should write a stream entry with progress details."""
    sync_client, _ = use_fakeredis

    await _send_progress_update(
        redis_client=sync_client,
        operation_id="op-123",
        correlation_id="corr-456",
        progress=40,
        message="chunking documents",
    )

    entries = sync_client.xrange("chunking:progress:op-123")
    assert len(entries) == 1
    entry_id, payload = entries[0]
    assert payload["correlation_id"] == "corr-456"
    assert payload["progress"] == "40"
    assert payload["message"] == "chunking documents"
    assert entry_id is not None
