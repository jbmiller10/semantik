"""
Configuration for webui tests.
"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


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


@pytest.fixture()
def authenticated_client_v2() -> TestClient:
    """
    Create a test client with authentication mocked.
    
    This fixture provides a TestClient with get_current_user dependency overridden
    to return a consistent test user without requiring actual JWT authentication.
    """
    from packages.webui.auth import get_current_user
    from packages.webui.main import app
    
    # Create a test user that will be returned by the mocked get_current_user
    test_user = {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "is_active": True,
        "is_superuser": False,
    }
    
    # Override the get_current_user dependency to return our test user
    async def override_get_current_user() -> dict[str, Any]:
        return test_user
    
    app.dependency_overrides[get_current_user] = override_get_current_user
    
    with TestClient(app) as client:
        yield client
    
    # Clean up the override after the test
    if get_current_user in app.dependency_overrides:
        del app.dependency_overrides[get_current_user]
