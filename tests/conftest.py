"""Shared test configuration and fixtures."""

import contextlib
import fcntl
import hashlib
import os
import subprocess
import sys
import warnings
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import quote, urlparse
from uuid import uuid4

# Set test environment BEFORE any app imports
os.environ["TESTING"] = "true"
os.environ["ENV"] = "test"
os.environ["DISABLE_RATE_LIMITING"] = "true"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ.setdefault("PROMETHEUS_DISABLE_SERVER", "true")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")

import importlib  # noqa: E402

import asyncpg  # noqa: E402
import fakeredis  # noqa: E402
import fakeredis.aioredis  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
import redis.asyncio as redis  # noqa: E402
from dotenv import dotenv_values, load_dotenv  # noqa: E402
from fastapi import WebSocket  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from httpx import AsyncClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

celery_module = importlib.import_module("webui.celery_app")  # noqa: E402
from shared.database import get_db  # noqa: E402
from shared.database.factory import (  # noqa: E402
    create_auth_repository,
    create_collection_repository,
    create_user_repository,
)
from shared.database.models import (  # noqa: E402
    Base,
    Collection,
    CollectionStatus,
    Document,
    DocumentStatus,
    Entity,
    Operation,
    OperationStatus,
    OperationType,
    Relationship,
    User,
)
from shared.database.partition_utils import PartitionAwareMixin  # noqa: E402
from webui.auth import create_access_token, get_current_user  # noqa: E402
from webui.main import app  # noqa: E402
from webui.qdrant import qdrant_manager  # noqa: E402
from webui.websocket_manager import RedisStreamWebSocketManager  # noqa: E402

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load test environment with fallback and sane defaults
project_root = Path(__file__).parent.parent
env_file_loaded: Path | None = None

test_env = project_root / ".env.test"
default_env = project_root / ".env"

if test_env.exists():
    load_dotenv(test_env, override=True)
    env_file_loaded = test_env
elif default_env.exists():
    candidate_values = dotenv_values(default_env)
    host = (candidate_values.get("POSTGRES_HOST") or "").strip()
    database_url = (candidate_values.get("DATABASE_URL") or "").strip()

    is_local_host = not host or host in {"localhost", "127.0.0.1"} or host.startswith("127.")
    is_local_url = not database_url or "localhost" in database_url or "127." in database_url

    if is_local_host and is_local_url:
        load_dotenv(default_env, override=True)
        env_file_loaded = default_env
    else:
        warnings.warn(
            "Skipping .env for tests because POSTGRES_HOST/DATABASE_URL are not local.",
            stacklevel=1,
        )

# Ensure Postgres defaults are present whenever env files omit them
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")

if "POSTGRES_USER" not in os.environ:
    default_user = "semantik" if env_file_loaded else "postgres"
    os.environ["POSTGRES_USER"] = default_user

if "POSTGRES_DB" not in os.environ:
    default_db = "semantik_test" if env_file_loaded else "postgres"
    os.environ["POSTGRES_DB"] = default_db

if env_file_loaded and env_file_loaded.name == ".env":
    # Ensure tests never inherit production/staging DATABASE_URL values.
    os.environ.pop("DATABASE_URL", None)

    postgres_user = os.environ.get("POSTGRES_USER", "semantik")
    postgres_password = os.environ.get("POSTGRES_PASSWORD")
    # Force test database to avoid mutating non-test schemas.
    postgres_db = "semantik_test"
    os.environ["POSTGRES_DB"] = postgres_db
    postgres_host = os.environ.get("POSTGRES_HOST", "localhost")
    postgres_port = os.environ.get("POSTGRES_PORT", "5432")

    encoded_user = quote(postgres_user, safe="")
    encoded_db = quote(postgres_db, safe="")

    if postgres_password:
        encoded_password = quote(postgres_password, safe="")
        database_url = f"postgresql://{encoded_user}:{encoded_password}@{postgres_host}:{postgres_port}/{encoded_db}"
    else:
        database_url = f"postgresql://{encoded_user}@{postgres_host}:{postgres_port}/{encoded_db}"

    os.environ["DATABASE_URL"] = database_url

# Set required environment variables for tests
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("DEFAULT_COLLECTION", "test_collection")
os.environ.setdefault("USE_MOCK_EMBEDDINGS", "true")
os.environ.setdefault("DISABLE_AUTH", "true")
os.environ.setdefault("DISABLE_RATE_LIMITING", "true")


async def _ensure_chunk_partition_triggers(conn) -> None:
    """Ensure the test database can compute partition keys like production.

    Tests provide partition_key explicitly, so keep the helper functions but
    drop the trigger that would otherwise overwrite or reject manual values.
    """

    try:
        await conn.execute(
            text(
                """
            CREATE OR REPLACE FUNCTION compute_partition_key()
            RETURNS trigger AS $$
            BEGIN
                IF NEW.collection_id IS NULL THEN
                    RAISE EXCEPTION 'collection_id cannot be null for chunks';
                END IF;

                IF NEW.partition_key IS NULL THEN
                    NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """
            )
        )

        await conn.execute(
            text(
                """
            DROP TRIGGER IF EXISTS set_partition_key ON chunks;
        """
            )
        )
    except Exception as exc:
        warnings.warn(f"Unable to ensure chunk partition trigger: {exc}", stacklevel=1)


@pytest.fixture(scope="session", autouse=True)
def _migrate_test_database_schema() -> None:
    """Ensure the test database schema is migrated to the latest Alembic revision.

    Many DB-backed tests assume the database schema matches the current models.
    SQLAlchemy's create_all() is not a schema migration tool; it won't add new
    columns to existing tables. When running against a persistent test DB, we
    apply Alembic migrations once per test session.
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return

    parsed = urlparse(database_url)
    db_name = (parsed.path or "").lstrip("/")
    if db_name and "test" not in db_name:
        warnings.warn(
            f"Refusing to auto-migrate non-test database '{db_name}'. Set DATABASE_URL to a test DB to enable.",
            stacklevel=1,
        )
        return

    lock_path = project_root / ".pytest-alembic-migrate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a simple file lock for xdist/multi-process safety.
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            warnings.warn(f"Unable to lock migration file {lock_path}: {exc}", stacklevel=1)

        try:
            # If the database was previously initialized via SQLAlchemy create_all(),
            # it may contain "future" tables (like entities/relationships) without
            # the corresponding Alembic revision applied, which causes migrations
            # to fail. Clean up known GraphRAG tables when the collections table
            # doesn't yet have graph_enabled.
            try:
                from sqlalchemy import create_engine as _create_engine

                sync_url = database_url
                if sync_url.startswith("postgresql+asyncpg://"):
                    sync_url = sync_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
                elif sync_url.startswith("postgresql://"):
                    sync_url = sync_url.replace("postgresql://", "postgresql+psycopg2://", 1)

                engine = _create_engine(sync_url)
                with engine.begin() as conn:
                    graph_enabled_exists = conn.execute(
                        text(
                            """
                            SELECT 1
                            FROM information_schema.columns
                            WHERE table_name = 'collections'
                              AND column_name = 'graph_enabled'
                            LIMIT 1
                            """
                        )
                    ).scalar_one_or_none()
                    if graph_enabled_exists is None:
                        conn.execute(text("DROP TABLE IF EXISTS relationships CASCADE"))
                        conn.execute(text("DROP TABLE IF EXISTS entities CASCADE"))
                engine.dispose()
            except Exception as exc:
                warnings.warn(f"Unable to pre-clean GraphRAG tables before migration: {exc}", stacklevel=1)

            # Avoid importing the Alembic Python package directly here because this
            # repository has an `alembic/` package that shadows the third-party
            # dependency in pytest (sys.path[0] is the repo root).
            alembic_bin = str(Path(sys.executable).with_name("alembic"))
            _ = subprocess.run(
                [alembic_bin, "upgrade", "head"],
                cwd=str(project_root),
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            # If the database is unavailable, db_session will skip DB tests.
            if isinstance(exc, subprocess.CalledProcessError):
                warnings.warn(
                    "Alembic migration skipped/failed for tests: "
                    f"{exc}\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}",
                    stacklevel=1,
                )
            else:
                warnings.warn(f"Alembic migration skipped/failed for tests: {exc}", stacklevel=1)
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


@pytest.fixture(autouse=True)
def stub_celery_send_task(monkeypatch):
    """Ensure Celery does not require a live broker during tests."""

    send_task_mock = MagicMock(name="celery_send_task_stub")
    send_task_mock.return_value = MagicMock(name="celery_async_result_stub")

    monkeypatch.setattr(celery_module.celery_app, "send_task", send_task_mock, raising=False)

    # Expose the stub to tests that may want to inspect dispatch calls.
    return send_task_mock


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
        patch("webui.websocket.scalable_manager.redis.from_url", return_value=fake_async_redis),
        patch("webui.websocket_manager.redis.from_url", return_value=fake_async_redis),
        patch("webui.websocket_manager.aioredis.from_url", return_value=fake_async_redis),
        # Patch service manager imports
        patch("webui.services.redis_manager.aioredis.from_url", return_value=fake_async_redis),
        patch("webui.services.redis_manager.redis.from_url", return_value=fake_sync_redis),
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
        patch("webui.main.pg_connection_manager") as mock_pg,
        patch("webui.main.ws_manager") as mock_ws,
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
async def async_client(test_user_db, db_session: AsyncSession) -> None:
    """Create an async test client for the FastAPI app with auth + DB wired up."""

    current_user = {
        "id": test_user_db.id,
        "username": test_user_db.username,
    }

    # Override the authentication dependency
    async def override_get_current_user() -> None:
        return current_user

    async def override_get_db() -> Generator[Any, None, None]:
        yield db_session

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db

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

    from shared.metrics.prometheus import registry

    # Clear all collectors from the custom registry
    collectors_to_remove = list(registry._collector_to_names.keys())
    for collector in collectors_to_remove:
        with contextlib.suppress(Exception):
            registry.unregister(collector)

    # Also clear the default registry if needed
    collectors_to_remove = list(REGISTRY._collector_to_names.keys())
    for collector in collectors_to_remove:
        with contextlib.suppress(Exception):
            REGISTRY.unregister(collector)


def create_async_mock(return_value=None) -> MagicMock:
    """Helper to create an async mock that returns a value."""
    from typing import Any

    async def async_mock(*_args: Any, **_kwargs: Any) -> Any:
        return return_value

    return MagicMock(side_effect=async_mock)


@pytest.fixture()
def mock_collection_repository() -> MagicMock:
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
def mock_user_repository() -> MagicMock:
    """Create a mock UserRepository for testing."""
    mock = MagicMock()
    mock.create_user = create_async_mock()
    mock.get_user = create_async_mock()
    mock.get_user_by_username = create_async_mock()
    mock.update_user = create_async_mock()
    mock.delete_user = create_async_mock(False)
    return mock


@pytest.fixture()
def mock_auth_repository() -> MagicMock:
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
def _db_isolation():
    """Marker fixture to indicate tests that require database isolation."""


@pytest_asyncio.fixture
async def db_session():
    """Create a new database session for testing."""
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

    # Create engine with isolation level for better concurrency
    engine = create_async_engine(
        async_database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=1,  # Small pool size per test
        max_overflow=0,  # No overflow connections
    )

    # Create tables if they don't exist (idempotent operation)
    async with engine.begin() as conn:
        # When running against a migrated database, rely on Alembic-managed schema.
        # SQLAlchemy create_all() won't apply schema changes (e.g., adding columns),
        # and can create tables that conflict with pending migrations.
        try:
            result = await conn.execute(text("SELECT to_regclass('public.alembic_version')"))
            has_alembic = result.scalar_one_or_none() is not None
        except Exception:
            has_alembic = False

        if not has_alembic:
            await conn.run_sync(Base.metadata.create_all)
        await _ensure_chunk_partition_triggers(conn)

    # Create session for this test
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session
        # Rollback any uncommitted changes
        if session.in_transaction():
            await session.rollback()
        await session.close()

    # Dispose of the engine to close all connections
    await engine.dispose()


@pytest_asyncio.fixture
async def test_user_db(db_session) -> None:
    """Create a test user in the database."""

    unique_suffix = uuid4().hex[:8]
    user = User(
        username=f"testuser_{unique_suffix}",
        hashed_password="hashed_password",
        email=f"test_{unique_suffix}@example.com",
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

    unique_suffix = uuid4().hex[:8]
    user = User(
        username=f"otheruser_{unique_suffix}",
        hashed_password="hashed_password",
        email=f"other_{unique_suffix}@example.com",
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


@pytest.fixture()
def test_documents_fixture() -> Path:
    """Provide path to test documents directory."""
    test_data_path = Path(__file__).parent / "test_data"
    if not test_data_path.exists():
        # Docker fallback
        docker_path = Path("/mnt/docs")
        if docker_path.exists():
            return docker_path
        pytest.skip(f"Test data directory not found at {test_data_path}")
    return test_data_path


# ---------------------------------------------------------------------------
# Graph Testing Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def entity_factory(db_session, collection_factory, document_factory, test_user_db) -> None:
    """Factory for creating Entity instances for testing.

    Creates Entity instances with automatic computation of:
    - partition_key: abs(hashtext(collection_id::text)) % 100 (via DB when available; deterministic fallback otherwise)
    - name_hash: SHA256 of "{entity_type}:{name.lower().strip()}"
    - name_normalized: name.lower().strip()

    If collection_id is not provided, creates a new collection with graph_enabled=True.
    If document_id is not provided, creates a new document in the collection.

    Usage:
        entity = await entity_factory(name="John Smith", entity_type="PERSON")
        entity = await entity_factory(
            name="Acme Corp",
            entity_type="ORG",
            collection_id=existing_collection.id,
            confidence=0.95,
        )
    """
    created_entities = []
    # Cache for collections/documents created by this factory
    _default_collection = None
    _default_document = None

    async def _create_entity(
        name: str = "Test Entity",
        entity_type: str = "PERSON",
        collection_id: str | None = None,
        document_id: str | None = None,
        chunk_id: int | None = None,
        confidence: float = 0.85,
        start_offset: int | None = None,
        end_offset: int | None = None,
        canonical_id: int | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> Entity:
        nonlocal _default_collection, _default_document

        # Create or use cached collection
        if collection_id is None:
            if _default_collection is None:
                _default_collection = await collection_factory(
                    owner_id=test_user_db.id,
                    graph_enabled=True,
                )
            collection_id = _default_collection.id

        # Create or use cached document
        if document_id is None:
            if _default_document is None or _default_document.collection_id != collection_id:
                _default_document = await document_factory(
                    collection_id=collection_id,
                    file_name="test_entity_doc.txt",
                )
            document_id = _default_document.id

        # Compute partition_key
        partition_key = await PartitionAwareMixin.compute_partition_key(db_session, collection_id)

        # Compute name_normalized and name_hash
        name_normalized = name.lower().strip()
        hash_input = f"{entity_type}:{name_normalized}"
        name_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        entity = Entity(
            collection_id=collection_id,
            partition_key=partition_key,
            document_id=document_id,
            chunk_id=chunk_id,
            name=name,
            name_normalized=name_normalized,
            name_hash=name_hash,
            entity_type=entity_type,
            confidence=confidence,
            start_offset=start_offset,
            end_offset=end_offset,
            canonical_id=canonical_id,
            metadata_=metadata or {},
            **kwargs,
        )
        db_session.add(entity)
        await db_session.flush()
        await db_session.refresh(entity)
        created_entities.append(entity)
        return entity

    yield _create_entity
    # Cleanup handled by transaction rollback


@pytest_asyncio.fixture
async def relationship_factory(db_session, entity_factory) -> None:
    """Factory for creating Relationship instances for testing.

    Creates Relationship instances with automatic computation of:
    - partition_key: abs(hashtext(collection_id::text)) % 100 (via DB when available; deterministic fallback otherwise)
    - relationship_type: normalized to uppercase

    If source_entity_id or target_entity_id are not provided, creates new entities.

    Usage:
        relationship = await relationship_factory(
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="WORKS_FOR",
        )
        # Or let the factory create entities automatically:
        relationship = await relationship_factory(relationship_type="LOCATED_IN")
    """
    created_relationships = []
    # Cache for entities created by this factory
    _cached_source_entity = None
    _cached_target_entity = None

    async def _create_relationship(
        source_entity_id: int | None = None,
        target_entity_id: int | None = None,
        collection_id: str | None = None,
        relationship_type: str = "RELATED_TO",
        confidence: float = 0.7,
        extraction_method: str | None = "dependency",
        metadata: dict | None = None,
        **kwargs,
    ) -> Relationship:
        nonlocal _cached_source_entity, _cached_target_entity

        # Create source entity if not provided
        if source_entity_id is None:
            if _cached_source_entity is None:
                _cached_source_entity = await entity_factory(
                    name="Source Entity",
                    entity_type="PERSON",
                    collection_id=collection_id,
                )
            source_entity_id = _cached_source_entity.id
            if collection_id is None:
                collection_id = _cached_source_entity.collection_id

        # Create target entity if not provided
        if target_entity_id is None:
            if _cached_target_entity is None:
                _cached_target_entity = await entity_factory(
                    name="Target Entity",
                    entity_type="ORG",
                    collection_id=collection_id,
                )
            target_entity_id = _cached_target_entity.id
            if collection_id is None:
                collection_id = _cached_target_entity.collection_id

        # Must have collection_id at this point
        if collection_id is None:
            raise ValueError("collection_id must be provided or inferred from entities")

        # Compute partition_key
        partition_key = await PartitionAwareMixin.compute_partition_key(db_session, collection_id)

        # Normalize relationship_type to uppercase
        relationship_type_normalized = relationship_type.upper()

        relationship = Relationship(
            collection_id=collection_id,
            partition_key=partition_key,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type_normalized,
            confidence=confidence,
            extraction_method=extraction_method,
            metadata_=metadata or {},
            **kwargs,
        )
        db_session.add(relationship)
        await db_session.flush()
        await db_session.refresh(relationship)
        created_relationships.append(relationship)
        return relationship

    yield _create_relationship
    # Cleanup handled by transaction rollback


@pytest_asyncio.fixture
async def collection_with_graph(collection_factory, test_user_db) -> None:
    """Convenience fixture for creating a graph-enabled collection.

    Usage:
        collection = await collection_with_graph()
        collection = await collection_with_graph(name="My Graph Collection")
    """

    async def _create_collection(**kwargs) -> None:
        defaults = {
            "owner_id": test_user_db.id,
            "graph_enabled": True,
        }
        defaults.update(kwargs)
        return await collection_factory(**defaults)

    yield _create_collection


@dataclass
class GraphTestData:
    """Container for a complete test graph with collection, entities, and relationships.

    Note: Named GraphTestData (not TestGraph) to avoid pytest collection warnings.
    """

    collection: Any
    entities: list = field(default_factory=list)
    relationships: list = field(default_factory=list)

    @property
    def entity_count(self) -> int:
        """Return the number of entities in the graph."""
        return len(self.entities)

    @property
    def relationship_count(self) -> int:
        """Return the number of relationships in the graph."""
        return len(self.relationships)

    def get_entity_by_name(self, name: str) -> Any | None:
        """Find an entity by name (case-insensitive)."""
        name_lower = name.lower()
        for entity in self.entities:
            if entity.name.lower() == name_lower:
                return entity
        return None

    def get_entities_by_type(self, entity_type: str) -> list:
        """Get all entities of a specific type."""
        return [e for e in self.entities if e.entity_type == entity_type]


@pytest_asyncio.fixture
async def graph_factory(
    db_session,
    collection_factory,
    document_factory,
    entity_factory,
    relationship_factory,
    test_user_db,
) -> None:
    """Factory for creating complete test graphs.

    Creates a collection with graph_enabled=True, multiple entities of different types,
    and relationships connecting them in a chain structure.

    Args:
        num_entities: Number of entities to create (default: 5)
        num_relationships: Number of relationships to create (default: num_entities - 1)
        entity_types: List of entity types to cycle through (default: ["PERSON", "ORG", "GPE"])
        relationship_types: List of relationship types to cycle through (default: ["WORKS_FOR", "LOCATED_IN", "AFFILIATED_WITH"])

    Usage:
        graph = await graph_factory()
        graph = await graph_factory(num_entities=10, num_relationships=9)

    Returns:
        GraphTestData dataclass with collection, entities, and relationships.
    """
    created_graphs = []

    async def _create_graph(
        num_entities: int = 5,
        num_relationships: int | None = None,
        entity_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        collection_name: str | None = None,
        **collection_kwargs,
    ) -> GraphTestData:
        # Default values
        if entity_types is None:
            entity_types = ["PERSON", "ORG", "GPE"]
        if relationship_types is None:
            relationship_types = ["WORKS_FOR", "LOCATED_IN", "AFFILIATED_WITH"]
        if num_relationships is None:
            num_relationships = max(0, num_entities - 1)

        # Create collection with graph enabled
        collection_defaults = {
            "owner_id": test_user_db.id,
            "graph_enabled": True,
        }
        if collection_name:
            collection_defaults["name"] = collection_name
        collection_defaults.update(collection_kwargs)
        collection = await collection_factory(**collection_defaults)

        # Create a document for the entities
        document = await document_factory(
            collection_id=collection.id,
            file_name="test_graph_doc.txt",
        )

        # Create entities cycling through types
        entities = []
        for i in range(num_entities):
            entity_type = entity_types[i % len(entity_types)]
            entity_name = f"Entity_{i}_{entity_type}"

            entity = await entity_factory(
                name=entity_name,
                entity_type=entity_type,
                collection_id=collection.id,
                document_id=document.id,
                confidence=0.85 + (i * 0.01),  # Slightly varying confidence
            )
            entities.append(entity)

        # Create relationships in a chain structure (entity[0] -> entity[1] -> entity[2] -> ...)
        relationships = []
        for i in range(min(num_relationships, len(entities) - 1)):
            rel_type = relationship_types[i % len(relationship_types)]

            relationship = await relationship_factory(
                source_entity_id=entities[i].id,
                target_entity_id=entities[i + 1].id,
                collection_id=collection.id,
                relationship_type=rel_type,
                confidence=0.7 + (i * 0.05),  # Slightly varying confidence
            )
            relationships.append(relationship)

        graph = GraphTestData(
            collection=collection,
            entities=entities,
            relationships=relationships,
        )
        created_graphs.append(graph)
        return graph

    yield _create_graph
    # Cleanup handled by transaction rollback
