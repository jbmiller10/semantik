"""Shared test configuration and fixtures."""

import os
import random
import sys
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse
from uuid import uuid4

# Set test environment BEFORE any app imports
os.environ["TESTING"] = "true"
os.environ["ENV"] = "test"
os.environ["DISABLE_RATE_LIMITING"] = "true"
os.environ["REDIS_URL"] = "redis://localhost:6379"

import asyncpg  # noqa: E402
import fakeredis  # noqa: E402
import fakeredis.aioredis  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
import redis.asyncio as redis  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from fastapi import WebSocket  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from httpx import AsyncClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

import packages.webui.celery_app as celery_module  # noqa: E402
from packages.shared.database import get_db  # noqa: E402
from packages.shared.database.factory import (  # noqa: E402
    create_auth_repository,
    create_collection_repository,
    create_user_repository,
)
from packages.shared.database.models import (  # noqa: E402
    Base,
    Collection,
    CollectionStatus,
    Document,
    DocumentStatus,
    Operation,
    OperationStatus,
    OperationType,
    User,
)
from packages.webui.auth import create_access_token, get_current_user  # noqa: E402
from packages.webui.main import app  # noqa: E402
from packages.webui.utils.qdrant_manager import qdrant_manager  # noqa: E402
from packages.webui.websocket_manager import RedisStreamWebSocketManager  # noqa: E402

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load test environment if available
test_env_path = Path(__file__).parent.parent / ".env.test"
if test_env_path.exists():

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
def use_fakeredis():
    """Opt-in fixture to use fakeredis for a specific test."""
    fake_sync_redis = fakeredis.FakeRedis(decode_responses=True)
    fake_async_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)

    # Import the sync redis module for proper patching
    import redis as sync_redis  # Import the sync redis module

    with (
        # Patch sync redis
        patch("redis.from_url", return_value=fake_sync_redis),
        patch("redis.ConnectionPool.from_url", return_value=fake_sync_redis.connection_pool),
        # Patch async redis
        patch("redis.asyncio.from_url", return_value=fake_async_redis),
        patch("redis.asyncio.ConnectionPool.from_url", return_value=fake_async_redis.connection_pool),
        # Also patch the WebSocket manager's Redis imports
        patch("packages.webui.websocket.scalable_manager.redis.from_url", return_value=fake_async_redis),
        patch("packages.webui.websocket_manager.redis.from_url", return_value=fake_async_redis),
        patch("packages.webui.websocket_manager.aioredis.from_url", return_value=fake_async_redis),
        # Patch service manager imports
        patch("packages.webui.services.redis_manager.aioredis.from_url", return_value=fake_async_redis),
        patch("packages.webui.services.redis_manager.redis.from_url", return_value=fake_sync_redis),
    ):
        # Also need to handle Redis() constructor with connection pool
        original_sync_redis_init = sync_redis.Redis.__init__
        original_async_redis_init = redis.Redis.__init__  # redis is already redis.asyncio

        def fake_redis_init(self, *args, connection_pool=None, **kwargs):
            if connection_pool == fake_sync_redis.connection_pool:
                # Initialize with fakeredis
                fake_sync_redis.__init__(*args, **kwargs)
                self.__dict__.update(fake_sync_redis.__dict__)
            else:
                original_sync_redis_init(self, *args, connection_pool=connection_pool, **kwargs)

        def fake_async_redis_init(self, *args, connection_pool=None, **kwargs):
            if connection_pool == fake_async_redis.connection_pool:
                # Initialize with fakeredis
                fake_async_redis.__init__(*args, **kwargs)
                self.__dict__.update(fake_async_redis.__dict__)
            else:
                original_async_redis_init(self, *args, connection_pool=connection_pool, **kwargs)

        sync_redis.Redis.__init__ = fake_redis_init
        redis.Redis.__init__ = fake_async_redis_init  # redis is already redis.asyncio

        try:
            yield fake_sync_redis, fake_async_redis
        finally:
            sync_redis.Redis.__init__ = original_sync_redis_init
            redis.Redis.__init__ = original_async_redis_init


@pytest.fixture()
def fake_redis_client():
    """Provide a fake Redis client for tests that need direct access."""
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture()
def real_redis_client():
    """Provide real Redis client for integration tests.

    Only use this for tests that MUST have real Redis behavior.
    """
    import redis.asyncio as aioredis

    return aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)


@pytest.fixture()
def test_client(test_user) -> None:
    """Create a test client for the FastAPI app with auth mocked."""

    # Mock the lifespan events to prevent real connections
    with (
        patch("packages.webui.main.pg_connection_manager") as mock_pg,
        patch("packages.webui.main.ws_manager") as mock_ws,
    ):
        # Mock the async methods
        mock_pg.initialize = AsyncMock()
        mock_ws.startup = AsyncMock()
        mock_ws.shutdown = AsyncMock()

        # Override dependencies
        async def override_get_current_user() -> None:
            return test_user

        async def override_get_db() -> Generator[Any, None, None]:
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

    # Override the authentication dependency
    async def override_get_current_user() -> None:
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

    token = create_access_token(data={"sub": test_user["username"]})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def test_user_headers(auth_headers) -> None:
    """Alias for auth_headers to match test expectations."""
    return auth_headers


@pytest_asyncio.fixture
async def async_client(test_user) -> None:
    """Create an async test client for the FastAPI app with auth mocked."""

    # Override the authentication dependency
    async def override_get_current_user() -> None:
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
    # Clear Prometheus metrics registry to avoid duplicate metric registration
    from prometheus_client import REGISTRY
    from packages.shared.metrics.prometheus import registry
    
    # Clear all collectors from the custom registry
    collectors_to_remove = list(registry._collector_to_names.keys())
    for collector in collectors_to_remove:
        try:
            registry.unregister(collector)
        except Exception:
            pass  # Ignore if already unregistered
    
    # Also clear the default registry if needed
    collectors_to_remove = list(REGISTRY._collector_to_names.keys())
    for collector in collectors_to_remove:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass  # Ignore if already unregistered


def create_async_mock(return_value=None) -> None:
    """Helper to create an async mock that returns a value."""

    async def async_mock(*_args, **_kwargs) -> None:
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

    class MockRedisStreams:
        def __init__(self) -> None:
            self.streams = {}
            self.consumer_groups = {}
            self.message_counter = 0

    mock_streams = MockRedisStreams()

    async def mock_xadd(stream_key, data, maxlen=None) -> None:
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

    async def mock_xrange(stream_key, min="-", max="+", count=None) -> None:  # noqa: ARG001
        if stream_key not in mock_streams.streams:
            return []

        messages = mock_streams.streams[stream_key]
        if count:
            messages = messages[-count:]

        return messages

    async def mock_xgroup_create(stream_key, group_name, id="0") -> None:
        if stream_key not in mock_streams.consumer_groups:
            mock_streams.consumer_groups[stream_key] = {}
        mock_streams.consumer_groups[stream_key][group_name] = {"last_delivered_id": id, "consumers": {}}

    async def mock_xreadgroup(group_name, consumer_name, streams, count=None, block=None) -> None:  # noqa: ARG001
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

    mock = AsyncMock(spec=WebSocket)
    mock.accept = AsyncMock()
    mock.send_json = AsyncMock()
    mock.close = AsyncMock()
    mock.receive_json = AsyncMock()
    return mock


@pytest.fixture()
def mock_websocket_manager(mock_redis_client) -> None:
    """Create a mock WebSocket manager with Redis client."""

    manager = RedisStreamWebSocketManager()
    manager.redis = mock_redis_client
    return manager


@pytest.fixture()
def websocket_test_client(test_client) -> None:
    """Create a test client with WebSocket support."""

    # TestClient already supports WebSocket testing
    return test_client


# Additional fixtures for collection deletion tests
@pytest.fixture()
def db_isolation():
    """Marker fixture to indicate tests that require database isolation."""
    pass


@pytest_asyncio.fixture
async def db_session() -> None:
    """Create a new database session for testing."""
    # Check if we have a test database available

    # Get database URL from environment, prioritizing DATABASE_URL
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        # Construct from individual components if DATABASE_URL not set
        postgres_user = os.environ.get("POSTGRES_USER", "postgres")
        postgres_password = os.environ.get("POSTGRES_PASSWORD", "postgres")
        postgres_db = os.environ.get("POSTGRES_DB", "semantik_test")
        postgres_host = os.environ.get("POSTGRES_HOST", "localhost")
        postgres_port = os.environ.get("POSTGRES_PORT", "5432")

        if postgres_password:
            database_url = (
                f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
            )
        else:
            database_url = f"postgresql://{postgres_user}@{postgres_host}:{postgres_port}/{postgres_db}"

    # Convert to async URL for SQLAlchemy
    if database_url.startswith("postgresql://"):
        async_database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    else:
        async_database_url = database_url

    # Try to connect to the database
    try:
        # Parse the URL to extract connection parameters for asyncpg
        parsed = urlparse(database_url)
        conn_params = {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip("/") if parsed.path else "semantik_test",
            "user": parsed.username or "postgres",
        }
        if parsed.password:
            conn_params["password"] = parsed.password

        # Test connection with asyncpg
        conn = await asyncpg.connect(**conn_params)
        await conn.close()
    except (asyncpg.InvalidPasswordError, OSError, Exception) as e:
        # If we can't connect to a real database, skip these tests
        pytest.skip(f"PostgreSQL test database not available: {e}")
        return

    engine = create_async_engine(async_database_url, echo=False)

    # Helper function to drop views before tables
    async def drop_views_and_tables(conn) -> None:
        # Drop views first (in dependency order)
        views_to_drop = [
            "DROP VIEW IF EXISTS partition_hot_spots CASCADE",
            "DROP VIEW IF EXISTS partition_health_summary CASCADE",
            "DROP VIEW IF EXISTS partition_size_distribution CASCADE",
            "DROP VIEW IF EXISTS partition_chunk_distribution CASCADE",
            "DROP VIEW IF EXISTS partition_distribution CASCADE",
            "DROP VIEW IF EXISTS partition_health CASCADE",
            "DROP VIEW IF EXISTS active_chunking_configs CASCADE",
            "DROP MATERIALIZED VIEW IF EXISTS collection_chunking_stats CASCADE",
        ]

        for view_sql in views_to_drop:
            await conn.execute(text(view_sql))

        # Now drop all tables
        await conn.run_sync(Base.metadata.drop_all)

    # Create tables once for the test run
    async with engine.begin() as conn:
        # Only drop and recreate if running in CI or explicitly requested
        if os.environ.get("CI") or os.environ.get("RECREATE_TEST_DB"):
            await drop_views_and_tables(conn)
            await conn.run_sync(Base.metadata.create_all)
        else:
            # Just ensure tables exist
            await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Create a new session for the test with proper transaction isolation
    async with async_session() as session:
        # Start a transaction
        async with session.begin():
            # Create a savepoint
            nested = await session.begin_nested()
            
            yield session
            
            # Rollback to savepoint
            if nested.is_active:
                await nested.rollback()
            
            # Don't commit the outer transaction - it will rollback

    await engine.dispose()


@pytest_asyncio.fixture
async def test_user_db(db_session) -> None:
    """Create a test user in the database."""

    # Use random ID to avoid conflicts
    user_id = random.randint(1000, 9999)
    user = User(
        id=user_id,
        username=f"testuser_{user_id}",
        hashed_password="hashed_password",
        email=f"test_{user_id}@example.com",
        is_active=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def other_user_db(db_session) -> None:
    """Create another test user in the database."""

    # Use random ID to avoid conflicts
    user_id = random.randint(10000, 19999)
    user = User(
        id=user_id,
        username=f"otheruser_{user_id}",
        hashed_password="hashed_password",
        email=f"other_{user_id}@example.com",
        is_active=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def collection_factory(db_session) -> None:
    """Factory for creating test collections."""

    created_collections = []

    async def _create_collection(**kwargs) -> None:
        # owner_id must be provided - no default
        if "owner_id" not in kwargs:
            raise ValueError("owner_id must be provided when creating a collection")

        collection_uuid = str(uuid4())
        defaults = {
            "id": collection_uuid,  # Changed from "uuid" to "id"
            "name": f"Test Collection {collection_uuid[:8]}",  # Use UUID to ensure uniqueness
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
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
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
async def document_factory(db_session) -> None:
    """Factory for creating test documents."""

    created_documents = []

    async def _create_document(**kwargs) -> None:
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
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
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
async def operation_factory(db_session) -> None:
    """Factory for creating test operations."""

    created_operations = []

    async def _create_operation(**kwargs) -> None:
        # user_id must be provided - no default
        if "user_id" not in kwargs:
            raise ValueError("user_id must be provided when creating an operation")

        defaults = {
            "uuid": str(uuid4()),
            "collection_id": 1,
            "type": OperationType.INDEX,
            "status": OperationStatus.COMPLETED,
            "config": {},
            "created_at": datetime.now(UTC),
            "started_at": datetime.now(UTC),
            "completed_at": datetime.now(UTC),
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


@pytest.fixture()
def mock_qdrant_deletion() -> Generator[Any, None, None]:
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

    original_get_client = qdrant_manager.get_client
    qdrant_manager.get_client = lambda: mock

    yield mock

    # Restore original
    qdrant_manager.get_client = original_get_client


@pytest.fixture()
def mock_celery_for_deletion() -> Generator[Any, None, None]:
    """Mock Celery app for deletion tests."""
    mock_app = MagicMock()
    mock_app.send_task = MagicMock()

    # Patch the celery app

    original_app = celery_module.celery_app
    celery_module.celery_app = mock_app

    yield mock_app

    # Restore original
    celery_module.celery_app = original_app
