"""Configuration and shared fixtures for webui tests."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest

from packages.shared.database.models import CollectionStatus, OperationType

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture(autouse=True)
async def _cleanup_pending_tasks() -> AsyncGenerator[None, None]:
    """Ensure pending asyncio tasks are cancelled between tests."""

    yield

    try:
        pending = asyncio.all_tasks(asyncio.get_event_loop())
    except AttributeError:
        pending = asyncio.all_tasks()

    current_task = asyncio.current_task()

    to_cancel = []
    for task in pending:
        if task != current_task and not task.done():
            task.cancel()
            to_cancel.append(task)

    for task in to_cancel:
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.1)


@pytest.fixture(name="mock_repositories")
def fixture_mock_repositories():
    """Provide mock repository instances for tests that patch database access."""

    operation_repo = AsyncMock()
    operation_obj = Mock()
    operation_obj.uuid = "op-123"
    operation_obj.collection_id = "col-123"
    operation_obj.type = OperationType.INDEX
    operation_obj.config = {}
    operation_obj.user_id = 1
    operation_repo.get_by_uuid.return_value = operation_obj
    operation_repo.set_task_id = AsyncMock()
    operation_repo.update_status = AsyncMock()

    collection_repo = AsyncMock()
    collection_obj = Mock()
    collection_obj.id = "col-123"
    collection_obj.name = "Test Collection"
    collection_obj.vector_store_name = "test_collection_vec"
    collection_obj.config = {"vector_dim": 1024}
    collection_obj.status = CollectionStatus.READY
    collection_repo.get_by_uuid.return_value = collection_obj
    collection_repo.update = AsyncMock()
    collection_repo.update_status = AsyncMock()
    collection_repo.update_stats = AsyncMock()

    document_repo = AsyncMock()
    document_repo.get_stats_by_collection.return_value = {
        "total_documents": 10,
        "total_chunks": 100,
        "total_size_bytes": 1024000,
    }

    return {"operation": operation_repo, "collection": collection_repo, "document": document_repo}


@pytest.fixture(name="mock_celery_task")
def fixture_mock_celery_task():
    """Provide a mocked Celery task object with retry instrumentation."""

    task = Mock()
    task.request.id = "celery-task-123"
    task.retry = Mock(side_effect=Exception("Retry called"))
    return task
