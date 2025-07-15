"""Shared test configuration and fixtures."""

import os
import sys
from datetime import UTC
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set required environment variables for tests
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("DEFAULT_COLLECTION", "test_collection")
os.environ.setdefault("USE_MOCK_EMBEDDINGS", "true")
os.environ.setdefault("DISABLE_AUTH", "true")


@pytest.fixture()
def test_client(test_user):
    """Create a test client for the FastAPI app with auth mocked."""
    from packages.webui.auth import get_current_user
    from packages.webui.main import app

    # Override the authentication dependency
    async def override_get_current_user():
        return test_user

    app.dependency_overrides[get_current_user] = override_get_current_user

    client = TestClient(app)

    # Ensure we clean up after the test
    yield client

    app.dependency_overrides.clear()


@pytest.fixture()
def unauthenticated_test_client():
    """Create a test client without authentication override."""
    from packages.webui.main import app

    # Clear any existing overrides
    app.dependency_overrides.clear()

    return TestClient(app)


@pytest.fixture()
def test_client_with_mocks(
    test_user,
    mock_job_repository,
    mock_file_repository,
    mock_collection_repository,
    mock_user_repository,
    mock_auth_repository,
):
    """Create a test client with mocked repositories and auth."""
    from shared.database.factory import (
        create_auth_repository,
        create_collection_repository,
        create_file_repository,
        create_job_repository,
        create_user_repository,
    )

    from packages.webui.auth import get_current_user
    from packages.webui.main import app

    # Override the authentication dependency
    async def override_get_current_user():
        return test_user

    # Override repository dependencies
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[create_job_repository] = lambda: mock_job_repository
    app.dependency_overrides[create_file_repository] = lambda: mock_file_repository
    app.dependency_overrides[create_collection_repository] = lambda: mock_collection_repository
    app.dependency_overrides[create_user_repository] = lambda: mock_user_repository
    app.dependency_overrides[create_auth_repository] = lambda: mock_auth_repository

    client = TestClient(app)

    # Ensure we clean up after the test
    yield client

    app.dependency_overrides.clear()


@pytest.fixture()
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock = MagicMock()
    mock.get_collections.return_value = MagicMock(collections=[])
    mock.search.return_value = []
    return mock


@pytest.fixture()
def test_user():
    """Test user data."""
    from datetime import datetime

    return {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "disabled": False,
        "created_at": datetime.now(UTC).isoformat(),
    }


@pytest.fixture()
def auth_headers(test_user):
    """Create authorization headers with a test JWT token."""
    from webui.auth import create_access_token

    token = create_access_token(data={"sub": test_user["username"]})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def temp_test_file(tmp_path):
    """Create a temporary test file."""
    test_file = tmp_path / "test_document.txt"
    test_file.write_text("This is a test document.")
    return test_file


@pytest.fixture()
def mock_embedding_service():
    """Mock embedding service."""
    mock = MagicMock()
    mock.embed_texts.return_value = [[0.1] * 384]  # Mock embedding vector
    mock.embed_documents.return_value = [[0.1] * 384]
    return mock


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset any singleton instances between tests."""
    # This helps ensure test isolation
    return
    # Cleanup code here if needed


def create_async_mock(return_value=None):
    """Helper to create an async mock that returns a value."""

    async def async_mock(*_args, **_kwargs):
        return return_value

    return MagicMock(side_effect=async_mock)


@pytest.fixture()
def mock_job_repository():
    """Create a mock JobRepository for testing."""
    mock = MagicMock()
    # Set up async methods
    mock.create_job = create_async_mock()
    mock.get_job = create_async_mock()
    mock.update_job = create_async_mock()
    mock.delete_job = create_async_mock()
    mock.list_jobs = create_async_mock([])
    mock.get_all_job_ids = create_async_mock([])
    return mock


@pytest.fixture()
def mock_file_repository():
    """Create a mock FileRepository for testing."""
    mock = MagicMock()
    mock.add_files_to_job = create_async_mock()
    mock.get_job_files = create_async_mock([])
    mock.update_file_status = create_async_mock()
    mock.get_job_total_vectors = create_async_mock(0)
    mock.get_duplicate_files_in_collection = create_async_mock(set())
    return mock


@pytest.fixture()
def mock_collection_repository():
    """Create a mock CollectionRepository for testing."""
    mock = MagicMock()
    mock.list_collections = create_async_mock([])
    mock.get_collection_details = create_async_mock()
    mock.get_collection_files = create_async_mock({})
    mock.rename_collection = create_async_mock(True)
    mock.delete_collection = create_async_mock({})
    mock.get_collection_metadata = create_async_mock()
    return mock


@pytest.fixture()
def mock_user_repository():
    """Create a mock UserRepository for testing."""
    mock = MagicMock()
    mock.create_user = create_async_mock()
    mock.get_user = create_async_mock()
    mock.get_user_by_username = create_async_mock()
    mock.update_user = create_async_mock()
    mock.delete_user = create_async_mock(False)
    return mock


@pytest.fixture()
def mock_auth_repository():
    """Create a mock AuthRepository for testing."""
    mock = MagicMock()
    mock.save_refresh_token = create_async_mock()
    mock.verify_refresh_token = create_async_mock()
    mock.revoke_refresh_token = create_async_mock()
    mock.update_user_last_login = create_async_mock()
    return mock


@pytest.fixture()
def mock_redis_client():
    """Create a mock Redis client for testing WebSocket functionality."""
    from unittest.mock import AsyncMock

    import redis.asyncio as redis

    class MockRedisStreams:
        def __init__(self):
            self.streams = {}
            self.consumer_groups = {}
            self.message_counter = 0

    mock_streams = MockRedisStreams()

    async def mock_xadd(stream_key, data, maxlen=None):
        if stream_key not in mock_streams.streams:
            mock_streams.streams[stream_key] = []

        # Generate message ID
        mock_streams.message_counter += 1
        msg_id = f"{mock_streams.message_counter}-0"
        mock_streams.streams[stream_key].append((msg_id, data))

        # Trim to maxlen if specified
        if maxlen and len(mock_streams.streams[stream_key]) > maxlen:
            mock_streams.streams[stream_key] = mock_streams.streams[stream_key][-maxlen:]

        return msg_id

    async def mock_xrange(stream_key, min="-", max="+", count=None):  # noqa: ARG001
        if stream_key not in mock_streams.streams:
            return []

        messages = mock_streams.streams[stream_key]
        if count:
            messages = messages[-count:]

        return messages

    async def mock_xgroup_create(stream_key, group_name, id="0"):
        if stream_key not in mock_streams.consumer_groups:
            mock_streams.consumer_groups[stream_key] = {}
        mock_streams.consumer_groups[stream_key][group_name] = {"last_delivered_id": id, "consumers": {}}

    async def mock_xreadgroup(group_name, consumer_name, streams, count=None, block=None):  # noqa: ARG001
        results = []

        for stream_key, last_id in streams.items():
            if stream_key not in mock_streams.streams:
                continue

            if stream_key not in mock_streams.consumer_groups:
                continue

            if group_name not in mock_streams.consumer_groups[stream_key]:
                continue

            group_info = mock_streams.consumer_groups[stream_key][group_name]

            # Track this consumer
            if consumer_name not in group_info["consumers"]:
                group_info["consumers"][consumer_name] = {"last_ack": None}

            # Get new messages since last delivered to this group
            all_messages = mock_streams.streams[stream_key]
            new_messages = []

            if last_id == ">":
                # Find messages after the group's last delivered ID
                last_delivered = group_info["last_delivered_id"]
                for msg_id, data in all_messages:
                    if msg_id > last_delivered:
                        new_messages.append((msg_id, data))

                # Update last delivered ID for the group
                if new_messages:
                    group_info["last_delivered_id"] = new_messages[-1][0]

            if new_messages:
                if count:
                    new_messages = new_messages[:count]
                results.append((stream_key, new_messages))

        return results

    mock = AsyncMock(spec=redis.Redis)
    mock.ping = AsyncMock(return_value=True)
    mock.xadd = AsyncMock(side_effect=mock_xadd)
    mock.expire = AsyncMock()
    mock.xrange = AsyncMock(side_effect=mock_xrange)
    mock.xreadgroup = AsyncMock(side_effect=mock_xreadgroup)
    mock.xgroup_create = AsyncMock(side_effect=mock_xgroup_create)
    mock.xack = AsyncMock()
    mock.xgroup_delconsumer = AsyncMock()
    mock.delete = AsyncMock(return_value=1)
    mock.xinfo_groups = AsyncMock(return_value=[])
    mock.xgroup_destroy = AsyncMock()
    mock.close = AsyncMock()

    # Attach the streams object for test inspection
    mock._mock_streams = mock_streams

    return mock


@pytest.fixture()
def mock_websocket():
    """Create a mock WebSocket connection."""
    from unittest.mock import AsyncMock

    from fastapi import WebSocket

    mock = AsyncMock(spec=WebSocket)
    mock.accept = AsyncMock()
    mock.send_json = AsyncMock()
    mock.close = AsyncMock()
    mock.receive_json = AsyncMock()
    return mock


@pytest.fixture()
def mock_websocket_manager(mock_redis_client):
    """Create a mock WebSocket manager with Redis client."""
    from webui.websocket_manager import RedisStreamWebSocketManager

    manager = RedisStreamWebSocketManager()
    manager.redis = mock_redis_client
    return manager


@pytest.fixture()
def websocket_test_client(test_client):
    """Create a test client with WebSocket support."""

    # TestClient already supports WebSocket testing
    return test_client
