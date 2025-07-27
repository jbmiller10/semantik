"""
Configuration for webui tests.
"""

import asyncio
import pytest


@pytest.fixture(autouse=True)
async def cleanup_pending_tasks():
    """
    Cleanup any pending asyncio tasks after each test to prevent hanging.
    """
    yield
    
    # Get all pending tasks
    pending = asyncio.all_tasks(asyncio.get_event_loop())
    current_task = asyncio.current_task()
    
    # Cancel all tasks except the current one
    for task in pending:
        if task != current_task and not task.done():
            task.cancel()
    
    # Wait briefly for cancellation
    if pending:
        await asyncio.sleep(0.1)