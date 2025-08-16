"""
Shared fixtures for integration tests.

Provides database sessions, Redis clients, test users, and other
common fixtures needed for integration testing.
"""

import asyncio
import os
import uuid
from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio
import redis.asyncio as redis_async
from faker import Faker
from httpx import ASGITransport, AsyncClient
from shared.database import get_db
from shared.database.models import Base, User
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool
from webui.auth import create_access_token, get_password_hash
from webui.main import app

fake = Faker()

# Test database configuration
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+asyncpg://semantik:semantik@localhost:5432/semantik_test"
)
TEST_DATABASE_URL_SYNC = TEST_DATABASE_URL.replace("+asyncpg", "")

# Redis configuration
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Create an async database engine for testing."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=NullPool,  # Disable connection pooling for tests
        echo=False,
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Clean up tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async database session for testing."""
    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture()
def sync_engine():
    """Create a sync database engine for testing."""
    engine = create_engine(
        TEST_DATABASE_URL_SYNC,
        poolclass=NullPool,
        echo=False,
    )

    # Create tables
    Base.metadata.create_all(bind=engine)

    yield engine

    # Clean up tables
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture()
def sync_session(sync_engine) -> Generator[Session, None, None]:
    """Create a sync database session for testing."""
    session_factory = sessionmaker(bind=sync_engine)
    session = session_factory()

    yield session

    session.rollback()
    session.close()


@pytest_asyncio.fixture(scope="function")
async def redis_client() -> AsyncGenerator[Any, None]:
    """Create a Redis client for testing."""
    client = await redis_async.from_url(
        TEST_REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )

    # Clear test database
    await client.flushdb()

    yield client

    # Clean up
    await client.flushdb()
    await client.close()


@pytest_asyncio.fixture(scope="function")
async def test_user(async_session: AsyncSession) -> dict[str, Any]:
    """Create a test user for authentication."""
    user = User(
        id=1,
        username=f"test_user_{uuid.uuid4().hex[:8]}",
        email=f"test_{uuid.uuid4().hex[:8]}@example.com",
        hashed_password=get_password_hash("test_password"),
        is_active=True,
        is_superuser=False,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "password": "test_password",
    }


@pytest_asyncio.fixture(scope="function")
async def admin_user(async_session: AsyncSession) -> dict[str, Any]:
    """Create an admin test user."""
    user = User(
        id=2,
        username=f"admin_{uuid.uuid4().hex[:8]}",
        email=f"admin_{uuid.uuid4().hex[:8]}@example.com",
        hashed_password=get_password_hash("admin_password"),
        is_active=True,
        is_superuser=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "password": "admin_password",
    }


@pytest.fixture()
def auth_token(test_user: dict[str, Any]) -> str:
    """Create an authentication token for the test user."""
    return create_access_token(
        data={
            "sub": test_user["username"],
            "user_id": test_user["id"],
        }
    )


@pytest.fixture()
def auth_headers(auth_token: str) -> dict[str, str]:
    """Create authorization headers for API requests."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest_asyncio.fixture(scope="function")
async def async_client(
    async_session: AsyncSession,
    _redis_client: Any,
) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing."""

    # Override database dependency
    async def override_get_db():
        yield async_session

    app.dependency_overrides[get_db] = override_get_db

    # Create client
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=30.0,
    ) as client:
        yield client

    # Clean up
    app.dependency_overrides.clear()


@pytest.fixture()
def test_server_url() -> str:
    """Get test server URL for WebSocket tests."""
    return os.getenv("TEST_SERVER_URL", "ws://localhost:8080")


@pytest_asyncio.fixture(scope="function")
async def test_websocket_server():
    """Start a test WebSocket server (mock implementation)."""
    # This would normally start an actual test server
    # For now, we'll just provide a mock
    from unittest.mock import AsyncMock, MagicMock

    server = MagicMock()
    server.url = "ws://test:8080"
    server.start = AsyncMock()
    server.stop = AsyncMock()

    await server.start()

    yield server

    await server.stop()


@pytest.fixture(autouse=True)
async def _cleanup_after_test(async_session: AsyncSession, redis_client: Any):
    """Automatically clean up after each test."""
    yield

    # Clean up database
    await async_session.execute(text("TRUNCATE TABLE chunks CASCADE"))
    await async_session.execute(text("TRUNCATE TABLE operations CASCADE"))
    await async_session.execute(text("TRUNCATE TABLE documents CASCADE"))
    await async_session.execute(text("TRUNCATE TABLE collections CASCADE"))
    await async_session.commit()

    # Clean up Redis
    await redis_client.flushdb()


@pytest.fixture()
def small_test_content() -> str:
    """Generate small test content for fast tests."""
    return fake.text(max_nb_chars=500)


@pytest.fixture()
def medium_test_content() -> str:
    """Generate medium test content."""
    return fake.text(max_nb_chars=5000)


@pytest.fixture()
def large_test_content() -> str:
    """Generate large test content (use sparingly)."""
    # Only generate if explicitly needed
    return fake.text(max_nb_chars=50000)


# Performance test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "load: marks tests as load tests")
    config.addinivalue_line("markers", "memory: marks tests as memory tests")


# Test data factories
class TestDataFactory:
    """Factory for generating test data efficiently."""

    @staticmethod
    def create_test_documents(count: int = 10, size: str = "small") -> list:
        """Create test documents with configurable size."""
        sizes = {
            "small": 500,
            "medium": 5000,
            "large": 50000,
        }

        max_chars = sizes.get(size, 500)
        documents = []

        for i in range(count):
            documents.append(
                {
                    "id": str(uuid.uuid4()),
                    "name": f"test_doc_{i}.txt",
                    "content": fake.text(max_nb_chars=max_chars),
                    "type": "text",
                    "size": max_chars,
                }
            )

        return documents

    @staticmethod
    def create_test_chunks(collection_id: str, document_id: str, count: int = 10) -> list:
        """Create test chunks efficiently."""
        chunks = []

        for i in range(count):
            chunks.append(
                {
                    "collection_id": collection_id,
                    "document_id": document_id,
                    "content": f"Test chunk {i}: {fake.text(max_nb_chars=200)}",
                    "chunk_index": i,
                    "start_offset": i * 200,
                    "end_offset": (i + 1) * 200,
                    "token_count": 40,
                }
            )

        return chunks


@pytest.fixture()
def test_data_factory():
    """Provide test data factory to tests."""
    return TestDataFactory()


# Environment setup check
@pytest.fixture(scope="session", autouse=True)
def _check_test_environment():
    """Check that test environment is properly configured."""
    required_env_vars = []

    missing = [var for var in required_env_vars if not os.getenv(var)]

    if missing:
        pytest.skip(f"Missing required environment variables: {', '.join(missing)}")

    # Check database connection
    try:
        import psycopg2

        conn = psycopg2.connect(TEST_DATABASE_URL_SYNC.replace("+asyncpg", ""))
        conn.close()
    except Exception as e:
        pytest.skip(f"Cannot connect to test database: {e}")

    # Check Redis connection
    try:
        import redis

        r = redis.from_url(TEST_REDIS_URL)
        r.ping()
        r.close()
    except Exception as e:
        pytest.skip(f"Cannot connect to Redis: {e}")
