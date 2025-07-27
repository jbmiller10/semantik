"""
Configuration for webui tests.
"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator

import pytest


@pytest.fixture(autouse=True)
async def _cleanup_pending_tasks() -> AsyncGenerator[None, None]:
    """
    Cleanup any pending asyncio tasks after each test to prevent hanging.
    """
    yield

    # Get all pending tasks
    try:
        # Python 3.9+
        pending = asyncio.all_tasks(asyncio.get_event_loop())
    except AttributeError:
        # Python 3.7-3.8
        # This API no longer exists, use asyncio.all_tasks() without loop
        pending = asyncio.all_tasks()

    current_task = asyncio.current_task()

    # Cancel all tasks except the current one
    tasks_to_cancel = []
    for task in pending:
        if task != current_task and not task.done():
            task.cancel()
            tasks_to_cancel.append(task)

    # Wait for all tasks to be cancelled
    if tasks_to_cancel:
        for task in tasks_to_cancel:
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=0.1)
