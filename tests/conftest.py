"""Shared test configuration and fixtures."""

import os
import sys
from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load test environment if available
test_env_path = Path(__file__).parent.parent / ".env.test"
if test_env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(test_env_path, override=True)

# Set required environment variables for tests
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("DEFAULT_COLLECTION", "test_collection")
os.environ.setdefault("USE_MOCK_EMBEDDINGS", "true")
os.environ.setdefault("DISABLE_AUTH", "true")
os.environ.setdefault("DISABLE_RATE_LIMITING", "true")


@pytest.fixture()
def test_client(test_user) -> None:
    """Create a test client for the FastAPI app with auth mocked."""
    from unittest.mock import AsyncMock, patch, MagicMock
    from packages.webui.auth import get_current_user
    from packages.webui.main import app
    from packages.shared.database import get_db
    
    # Mock the lifespan events to prevent real connections
    with patch('packages.webui.main.pg_connection_manager') as mock_pg:
        with patch('packages.webui.main.ws_manager') as mock_ws:
            # Mock the async methods
            mock_pg.initialize = AsyncMock()
            mock_ws.startup = AsyncMock()
            mock_ws.shutdown = AsyncMock()
            
            # Override dependencies
            async def override_get_current_user():
                return test_user
            
            async def override_get_db():
                # Return a mock database session
                mock_db = AsyncMock()
                # Mock common async methods
                mock_db.execute = AsyncMock()
                mock_db.scalar = AsyncMock()
                mock_db.scalars = AsyncMock()
                mock_db.commit = AsyncMock()
                mock_db.rollback = AsyncMock()
                mock_db.flush = AsyncMock()
                mock_db.refresh = AsyncMock()
                mock_db.add = MagicMock()
                mock_db.delete = MagicMock()
                yield mock_db

            app.dependency_overrides[get_current_user] = override_get_current_user
            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)

            # Ensure we clean up after the test
            yield client

            app.dependency_overrides.clear()


@pytest.fixture()
def unauthenticated_test_client() -> None:
    """Create a test client without authentication override."""
    from packages.webui.main import app

    # Clear any existing overrides
    app.dependency_overrides.clear()

    return TestClient(app)


@pytest.fixture()
def test_client_with_mocks(
    test_user,
    mock_collection_repository,
    mock_user_repository,
    mock_auth_repository,
) -> None:
    """Create a test client with mocked repositories and auth."""
    from packages.shared.database.factory import (
        create_auth_repository,
        create_collection_repository,
        create_user_repository,
    )
    from packages.webui.auth import get_current_user
    from packages.webui.main import app

    # Override the authentication dependency
    async def override_get_current_user():
        return test_user

    # Override repository dependencies
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[create_collection_repository] = lambda: mock_collection_repository
    app.dependency_overrides[create_user_repository] = lambda: mock_user_repository
    app.dependency_overrides[create_auth_repository] = lambda: mock_auth_repository

    client = TestClient(app)

    # Ensure we clean up after the test
    yield client

    app.dependency_overrides.clear()


@pytest.fixture()
def mock_qdrant_client() -> None:
    """Mock Qdrant client for testing."""
    mock = MagicMock()
    mock.get_collections.return_value = MagicMock(collections=[])
    mock.search.return_value = []
    return mock


@pytest.fixture()
def test_user() -> None:
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
def auth_headers(test_user) -> None:
    """Create authorization headers with a test JWT token."""
    from packages.webui.auth import create_access_token

    token = create_access_token(data={"sub": test_user["username"]})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def test_user_headers(auth_headers) -> None:
    """Alias for auth_headers to match test expectations."""
    return auth_headers


@pytest_asyncio.fixture
async def async_client(test_user):
    """Create an async test client for the FastAPI app with auth mocked."""
    from packages.webui.auth import get_current_user
    from packages.webui.main import app

    # Override the authentication dependency
    async def override_get_current_user():
        return test_user

    app.dependency_overrides[get_current_user] = override_get_current_user

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture()
def temp_test_file(tmp_path) -> None:
    """Create a temporary test file."""
    test_file = tmp_path / "test_document.txt"
    test_file.write_text("This is a test document.")
    return test_file


@pytest.fixture()
def mock_embedding_service() -> None:
    """Mock embedding service."""
    mock = MagicMock()
    mock.embed_texts.return_value = [[0.1] * 384]  # Mock embedding vector
    mock.embed_documents.return_value = [[0.1] * 384]
    return mock


@pytest.fixture(autouse=True)
def _reset_singletons() -> None:
    """Reset any singleton instances between tests."""
    # This helps ensure test isolation
    return
    # Cleanup code here if needed


def create_async_mock(return_value=None) -> None:
    """Helper to create an async mock that returns a value."""

    async def async_mock(*_args, **_kwargs):
        return return_value

    return MagicMock(side_effect=async_mock)


@pytest.fixture()
def mock_collection_repository() -> None:
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
def mock_user_repository() -> None:
    """Create a mock UserRepository for testing."""
    mock = MagicMock()
    mock.create_user = create_async_mock()
    mock.get_user = create_async_mock()
    mock.get_user_by_username = create_async_mock()
    mock.update_user = create_async_mock()
    mock.delete_user = create_async_mock(False)
    return mock


@pytest.fixture()
def mock_auth_repository() -> None:
    """Create a mock AuthRepository for testing."""
    mock = MagicMock()
    mock.save_refresh_token = create_async_mock()
    mock.verify_refresh_token = create_async_mock()
    mock.revoke_refresh_token = create_async_mock()
    mock.update_user_last_login = create_async_mock()
    return mock


@pytest.fixture()
def mock_redis_client() -> None:
    """Create a mock Redis client for testing WebSocket functionality."""

    import redis.asyncio as redis

    class MockRedisStreams:
        def __init__(self) -> None:
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
def mock_websocket() -> None:
    """Create a mock WebSocket connection."""

    from fastapi import WebSocket

    mock = AsyncMock(spec=WebSocket)
    mock.accept = AsyncMock()
    mock.send_json = AsyncMock()
    mock.close = AsyncMock()
    mock.receive_json = AsyncMock()
    return mock


@pytest.fixture()
def mock_websocket_manager(mock_redis_client) -> None:
    """Create a mock WebSocket manager with Redis client."""
    from packages.webui.websocket_manager import RedisStreamWebSocketManager

    manager = RedisStreamWebSocketManager()
    manager.redis = mock_redis_client
    return manager


@pytest.fixture()
def websocket_test_client(test_client) -> None:
    """Create a test client with WebSocket support."""

    # TestClient already supports WebSocket testing
    return test_client


# Additional fixtures for collection deletion tests
@pytest_asyncio.fixture
async def db_session():
    """Create a new database session for testing."""
    # Check if we have a test database available
    import asyncpg
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from packages.shared.database.models import Base
    from packages.shared.config.postgres import postgres_config

    # Use PostgreSQL for tests - get URL from environment or config
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        # Convert to async URL if needed
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    else:
        # Use default test database configuration
        database_url = postgres_config.async_database_url

    # Try to connect to the database
    try:
        # Test connection
        conn = await asyncpg.connect(database_url.replace("postgresql+asyncpg://", "postgresql://"))
        await conn.close()
    except (asyncpg.InvalidPasswordError, OSError) as e:
        # If we can't connect to a real database, skip these tests
        pytest.skip(f"PostgreSQL test database not available: {e}")
        return

    engine = create_async_engine(database_url, echo=False)

    # Drop all tables and recreate for each test to ensure isolation
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session
        await session.rollback()

    await engine.dispose()


@pytest_asyncio.fixture
async def test_user_db(db_session):
    """Create a test user in the database."""
    from packages.shared.database.models import User
    from datetime import datetime
    import random

    # Use random ID to avoid conflicts
    user_id = random.randint(1000, 9999)
    user = User(
        id=user_id,
        username=f"testuser_{user_id}",
        hashed_password="hashed_password",
        email=f"test_{user_id}@example.com",
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def other_user_db(db_session):
    """Create another test user in the database."""
    from packages.shared.database.models import User
    from datetime import datetime
    import random

    # Use random ID to avoid conflicts
    user_id = random.randint(10000, 19999)
    user = User(
        id=user_id,
        username=f"otheruser_{user_id}",
        hashed_password="hashed_password",
        email=f"other_{user_id}@example.com",
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def collection_factory(db_session):
    """Factory for creating test collections."""
    from packages.shared.database.models import Collection, CollectionStatus
    from datetime import datetime
    from uuid import uuid4

    created_collections = []

    async def _create_collection(**kwargs):
        # owner_id must be provided - no default
        if "owner_id" not in kwargs:
            raise ValueError("owner_id must be provided when creating a collection")
            
        defaults = {
            "id": str(uuid4()),  # Changed from "uuid" to "id"
            "name": f"Test Collection {len(created_collections)}",
            "description": "Test collection description",
            "vector_store_name": f"col_{uuid4().hex[:16]}",
            "embedding_model": "test-model",
            "quantization": "float16",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "is_public": False,
            "status": CollectionStatus.READY,
            "document_count": 0,
            "vector_count": 0,
            "total_size_bytes": 0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        defaults.update(kwargs)

        collection = Collection(**defaults)
        db_session.add(collection)
        await db_session.commit()
        await db_session.refresh(collection)

        created_collections.append(collection)
        return collection

    yield _create_collection


@pytest_asyncio.fixture
async def document_factory(db_session):
    """Factory for creating test documents."""
    from packages.shared.database.models import Document, DocumentStatus
    from datetime import datetime
    from uuid import uuid4

    created_documents = []

    async def _create_document(**kwargs):
        defaults = {
            "id": str(uuid4()),  # Add UUID for document ID
            "collection_id": 1,
            "file_name": f"test_doc_{len(created_documents)}.txt",
            "file_path": f"/test/path/test_doc_{len(created_documents)}.txt",
            "file_size": 1024,
            "mime_type": "text/plain",
            "content_hash": f"hash_{uuid4().hex[:8]}",
            "status": DocumentStatus.COMPLETED,
            "chunk_count": 10,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        defaults.update(kwargs)

        document = Document(**defaults)
        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)

        created_documents.append(document)
        return document

    yield _create_document


@pytest_asyncio.fixture
async def operation_factory(db_session):
    """Factory for creating test operations."""
    from packages.shared.database.models import Operation, OperationType, OperationStatus
    from datetime import datetime
    from uuid import uuid4

    created_operations = []

    async def _create_operation(**kwargs):
        # user_id must be provided - no default
        if "user_id" not in kwargs:
            raise ValueError("user_id must be provided when creating an operation")
        
        defaults = {
            "uuid": str(uuid4()),
            "collection_id": 1,
            "type": OperationType.INDEX,
            "status": OperationStatus.COMPLETED,
            "config": {},
            "created_at": datetime.utcnow(),
            "started_at": datetime.utcnow(),
            "completed_at": datetime.utcnow(),
        }
        defaults.update(kwargs)

        # Handle string status conversion
        if isinstance(defaults.get("status"), str):
            defaults["status"] = OperationStatus(defaults["status"])

        operation = Operation(**defaults)
        db_session.add(operation)
        await db_session.commit()
        await db_session.refresh(operation)

        created_operations.append(operation)
        return operation

    yield _create_operation


@pytest.fixture
def mock_qdrant_deletion():
    """Mock Qdrant client specifically for deletion tests."""
    mock = MagicMock()

    # Mock get_collections response
    mock_collections_response = MagicMock()
    mock_collections_response.collections = []
    mock.get_collections.return_value = mock_collections_response

    # Mock other methods
    mock.delete_collection = AsyncMock()
    mock.create_collection = AsyncMock()

    # Patch the qdrant manager
    from packages.webui.utils.qdrant_manager import qdrant_manager

    original_get_client = qdrant_manager.get_client
    qdrant_manager.get_client = lambda: mock

    yield mock

    # Restore original
    qdrant_manager.get_client = original_get_client


@pytest.fixture
def mock_celery_for_deletion():
    """Mock Celery app for deletion tests."""
    mock_app = MagicMock()
    mock_app.send_task = MagicMock()

    # Patch the celery app
    import packages.webui.celery_app as celery_module

    original_app = celery_module.celery_app
    celery_module.celery_app = mock_app

    yield mock_app

    # Restore original
    celery_module.celery_app = original_app
