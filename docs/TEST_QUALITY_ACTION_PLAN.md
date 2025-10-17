# Test Quality Improvement Action Plan

## Executive Summary

This document provides a detailed, step-by-step action plan to address critical test quality issues identified in the Semantik test suite. Each action includes code examples, rationale, and implementation guidance.

**Timeline**: 3 sprints (6 weeks)
**Priority**: High - Current tests provide false confidence and miss real bugs

---

## IMMEDIATE ACTIONS (Sprint 1 - Week 1-2)

### Action 1: Delete `tests/test_auth.py` ⚠️ CRITICAL

**Priority**: P0 - Do this first
**Effort**: 5 minutes
**Risk**: None - functionality already covered elsewhere

**Rationale**: This file has:
- Zero assertions (only print statements)
- Creates persistent test pollution
- Makes real HTTP requests without E2E marker
- Completely redundant with existing tests

**Steps**:

```bash
cd /home/john/semantik

# 1. Verify existing coverage is sufficient
pytest tests/unit/test_auth.py tests/integration/test_auth_api.py -v --cov=packages.webui.auth

# 2. Delete the problematic file
git rm tests/test_auth.py

# 3. Commit
git commit -m "refactor(tests): remove redundant test_auth.py

- Deleted tests/test_auth.py (E2E test without assertions)
- Functionality already covered by:
  - tests/unit/test_auth.py (comprehensive unit tests)
  - tests/integration/test_auth_api.py (proper integration tests)
- Removes test pollution and external dependencies"
```

**Verification**: Run full auth test suite:
```bash
pytest tests/unit/test_auth.py tests/integration/test_auth_api.py -v
```

---

### Action 2: Fix `test_collection_repository.py` - Convert to Real DB Tests ⚠️ CRITICAL

**Priority**: P0
**Effort**: 2-3 days
**Risk**: Medium - requires database setup understanding

**Current Problem**: File mocks entire database layer, tests mock behavior instead of real repository logic.

**Target State**: Integration tests using real PostgreSQL with transaction rollback.

#### Step 2.1: Update Test Class Structure

**Before** (`tests/unit/test_collection_repository.py:18-56`):
```python
class TestCollectionRepository:
    """Test cases for CollectionRepository."""

    @pytest.fixture()
    def mock_session(self) -> None:  # ❌ Wrong: mocks database
        session = AsyncMock()
        session.commit = AsyncMock()
        return session

    @pytest.fixture()
    def repository(self, mock_session) -> CollectionRepository:
        return CollectionRepository(mock_session)
```

**After** (new file: `tests/integration/repositories/test_collection_repository.py`):
```python
"""Integration tests for CollectionRepository with real database."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from shared.database.repositories import create_collection_repository
from shared.database.models import Collection, CollectionStatus
from shared.exceptions import EntityAlreadyExistsError, EntityNotFoundError


@pytest.mark.integration
class TestCollectionRepositoryIntegration:
    """Integration tests for CollectionRepository using real database."""

    @pytest.fixture
    async def repository(self, async_session: AsyncSession):
        """Create repository with real database session."""
        return create_collection_repository(async_session)

    @pytest.fixture
    async def test_user(self, async_session: AsyncSession):
        """Create a test user for collection ownership."""
        from shared.database.models import User
        from shared.auth import hash_password

        user = User(
            username=f"test_user_{uuid4().hex[:8]}",
            email=f"test_{uuid4().hex[:8]}@example.com",
            hashed_password=hash_password("test_password"),
            is_active=True
        )
        async_session.add(user)
        await async_session.commit()
        await async_session.refresh(user)
        return user
```

#### Step 2.2: Rewrite Core Tests with Real Database

**Before** (`tests/unit/test_collection_repository.py:59-89`):
```python
async def test_create_collection_success(self, repository, mock_session):
    # Mock no existing collection
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    # Mock UUID generation
    with patch("shared.database.repositories.collection_repository.uuid4") as mock_uuid:
        mock_uuid.return_value = "test-uuid-1234"

        collection = await repository.create(
            name=name,
            owner_id=user_id,
            description=description,
        )

    # Assert mock calls
    mock_session.add.assert_called_once()
    mock_session.flush.assert_called_once()
```

**After**:
```python
@pytest.mark.asyncio
async def test_create_collection_success(
    self,
    repository,
    async_session: AsyncSession,
    test_user
):
    """Test successful collection creation with real database."""
    # Arrange
    collection_name = "test-collection"
    description = "Test description"

    # Act - no mocks, real database interaction
    collection = await repository.create(
        name=collection_name,
        owner_id=test_user.id,
        description=description,
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        quantization="float16",
    )

    await async_session.commit()

    # Assert - verify against real data
    assert collection.name == collection_name
    assert collection.owner_id == test_user.id
    assert collection.status == CollectionStatus.PENDING
    assert collection.description == description
    assert collection.id is not None  # Real UUID generated
    assert collection.vector_store_name.startswith("collection_")

    # Verify persistence with fresh query
    retrieved = await repository.get_by_uuid(collection.id)
    assert retrieved is not None
    assert retrieved.name == collection_name
    assert retrieved.owner_id == test_user.id


@pytest.mark.asyncio
async def test_create_collection_duplicate_name_fails(
    self,
    repository,
    async_session: AsyncSession,
    test_user
):
    """Test that creating collection with duplicate name raises error."""
    # Arrange - create first collection
    await repository.create(
        name="duplicate-name",
        owner_id=test_user.id
    )
    await async_session.commit()

    # Act & Assert - second collection with same name should fail
    with pytest.raises(EntityAlreadyExistsError) as exc_info:
        await repository.create(
            name="duplicate-name",
            owner_id=test_user.id
        )

    # Verify error message is informative
    assert "duplicate-name" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_create_collection_validates_chunk_size(
    self,
    repository,
    test_user
):
    """Test that invalid chunk_size is rejected."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        await repository.create(
            name="test",
            owner_id=test_user.id,
            chunk_size=0  # Invalid
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size,chunk_overlap,expected_error", [
    (0, 200, "chunk_size must be positive"),
    (1000, -1, "chunk_overlap cannot be negative"),
    (100, 100, "chunk_overlap must be less than chunk_size"),
    (100, 150, "chunk_overlap must be less than chunk_size"),
])
async def test_create_collection_chunk_validation(
    repository,
    test_user,
    chunk_size,
    chunk_overlap,
    expected_error
):
    """Test chunk size and overlap validation."""
    with pytest.raises(ValueError, match=expected_error):
        await repository.create(
            name="test",
            owner_id=test_user.id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
```

#### Step 2.3: Add Missing Test Coverage for `update()` Method

**Current State**: No tests for `update()` method (lines 402-476 in collection_repository.py)

**Add These Tests**:
```python
@pytest.mark.asyncio
async def test_update_collection_single_field(
    repository,
    async_session: AsyncSession,
    test_user
):
    """Test updating a single field."""
    # Arrange
    collection = await repository.create(
        name="original-name",
        owner_id=test_user.id
    )
    await async_session.commit()

    # Act
    updated = await repository.update(
        collection_id=collection.id,
        user_id=test_user.id,
        updates={"name": "new-name"}
    )
    await async_session.commit()

    # Assert
    assert updated.name == "new-name"
    assert updated.description == collection.description  # Unchanged

    # Verify persistence
    retrieved = await repository.get_by_uuid(collection.id)
    assert retrieved.name == "new-name"


@pytest.mark.asyncio
async def test_update_collection_multiple_fields(
    repository,
    async_session: AsyncSession,
    test_user
):
    """Test updating multiple fields atomically."""
    # Arrange
    collection = await repository.create(
        name="original",
        owner_id=test_user.id,
        chunk_size=1000,
        chunk_overlap=200
    )
    await async_session.commit()

    # Act
    updated = await repository.update(
        collection_id=collection.id,
        user_id=test_user.id,
        updates={
            "name": "updated",
            "description": "new description",
            "chunk_size": 500,
            "chunk_overlap": 100
        }
    )
    await async_session.commit()

    # Assert - all fields updated atomically
    assert updated.name == "updated"
    assert updated.description == "new description"
    assert updated.chunk_size == 500
    assert updated.chunk_overlap == 100


@pytest.mark.asyncio
async def test_update_collection_invalid_field_rejected(
    repository,
    async_session: AsyncSession,
    test_user
):
    """Test that invalid fields are rejected."""
    # Arrange
    collection = await repository.create(
        name="test",
        owner_id=test_user.id
    )
    await async_session.commit()

    # Act & Assert
    with pytest.raises(ValueError, match="Invalid update field"):
        await repository.update(
            collection_id=collection.id,
            user_id=test_user.id,
            updates={"nonexistent_field": "value"}
        )


@pytest.mark.asyncio
async def test_update_collection_validates_cross_field_constraints(
    repository,
    async_session: AsyncSession,
    test_user
):
    """Test that cross-field validation is enforced."""
    # Arrange
    collection = await repository.create(
        name="test",
        owner_id=test_user.id,
        chunk_size=1000,
        chunk_overlap=200
    )
    await async_session.commit()

    # Act & Assert - cannot set overlap >= size
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        await repository.update(
            collection_id=collection.id,
            user_id=test_user.id,
            updates={
                "chunk_size": 500,
                "chunk_overlap": 500  # Invalid: equal to size
            }
        )
```

#### Step 2.4: Migration Checklist

- [ ] Create new directory: `tests/integration/repositories/`
- [ ] Create `tests/integration/repositories/__init__.py`
- [ ] Create `tests/integration/repositories/test_collection_repository.py`
- [ ] Implement all repository tests with real database
- [ ] Run tests to verify they pass: `pytest tests/integration/repositories/test_collection_repository.py -v`
- [ ] Delete old mock-based tests: `git rm tests/unit/test_collection_repository.py`
- [ ] Update any CI configuration to run integration tests
- [ ] Document the change in commit message

**Estimated Time**: 2-3 days

---

### Action 3: Fix `test_collection_service.py` - Reduce Mocking ⚠️ CRITICAL

**Priority**: P0
**Effort**: 3-4 days
**Risk**: Medium

**Current Problem**: 904 lines, mocks repositories (should use real ones), tests mock setup not business logic.

#### Step 3.1: Create Factory Fixtures

**Create**: `tests/fixtures/factories.py`

```python
"""Test data factories to reduce duplication."""

from datetime import datetime, UTC
from unittest.mock import Mock
from uuid import uuid4
from shared.database.models import Collection, Operation, CollectionStatus, OperationType


def create_mock_collection(
    id: str = None,
    name: str = "Test Collection",
    owner_id: int = 123,
    status: CollectionStatus = CollectionStatus.PENDING,
    **overrides
) -> Collection:
    """Create a mock Collection with sensible defaults.

    Usage:
        collection = create_mock_collection(name="Custom", status=CollectionStatus.READY)
    """
    defaults = {
        "id": id or str(uuid4()),
        "name": name,
        "description": "Test description",
        "owner_id": owner_id,
        "status": status,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "quantization": "float16",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "chunking_strategy": None,
        "chunking_config": None,
        "is_public": False,
        "meta": None,
        "vector_store_name": f"collection_{uuid4().hex[:8]}",
        "document_count": 0,
        "vector_count": 0,
        "total_size_bytes": 0,
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    defaults.update(overrides)

    mock = Mock(spec=Collection)
    for key, value in defaults.items():
        setattr(mock, key, value)
    return mock


def create_mock_operation(
    uuid: str = None,
    collection_id: str = None,
    user_id: int = 123,
    type: OperationType = OperationType.INDEX,
    status: str = "PENDING",
    **overrides
) -> Operation:
    """Create a mock Operation with sensible defaults."""
    defaults = {
        "uuid": uuid or str(uuid4()),
        "collection_id": collection_id or str(uuid4()),
        "user_id": user_id,
        "type": type,
        "status": status,
        "progress": 0,
        "result": None,
        "error": None,
        "celery_task_id": str(uuid4()),
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    defaults.update(overrides)

    mock = Mock(spec=Operation)
    for key, value in defaults.items():
        setattr(mock, key, value)
    return mock
```

#### Step 3.2: Convert to Integration Tests with Real Repositories

**Before** (`tests/unit/test_collection_service.py:55-132`):
```python
async def test_create_collection_success(self, collection_service, mock_collection_repo, mock_operation_repo, mock_celery_app):
    # 80 lines of mock setup...
    mock_collection = Mock(spec=Collection)
    mock_collection.id = 999
    mock_collection.name = "Test Collection"
    # ... 15+ more attributes

    mock_collection_repo.create.return_value = mock_collection
    mock_operation_repo.create.return_value = mock_operation

    # Test
    collection_dict, operation_dict = await collection_service.create_collection(...)

    # Assert mock calls
    mock_collection_repo.create.assert_called_once_with(...)
```

**After** (new: `tests/integration/services/test_collection_service.py`):
```python
"""Integration tests for CollectionService with real repositories."""

import pytest
from unittest.mock import patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession
from packages.webui.services.collection_service import CollectionService
from shared.database.repositories import (
    create_collection_repository,
    create_operation_repository,
    create_document_repository,
)


@pytest.mark.integration
class TestCollectionServiceIntegration:
    """Integration tests for CollectionService."""

    @pytest.fixture
    async def collection_service(self, async_session: AsyncSession):
        """Create service with real repositories."""
        return CollectionService(
            db_session=async_session,
            collection_repo=create_collection_repository(async_session),
            operation_repo=create_operation_repository(async_session),
            document_repo=create_document_repository(async_session),
        )

    @pytest.fixture
    async def test_user(self, async_session: AsyncSession):
        """Create test user."""
        from shared.database.models import User
        from shared.auth import hash_password

        user = User(
            username=f"test_{uuid4().hex[:8]}",
            email=f"test_{uuid4().hex[:8]}@example.com",
            hashed_password=hash_password("password"),
            is_active=True
        )
        async_session.add(user)
        await async_session.commit()
        await async_session.refresh(user)
        return user

    @pytest.mark.asyncio
    async def test_create_collection_success(
        self,
        collection_service,
        async_session: AsyncSession,
        test_user
    ):
        """Test successful collection creation with real database."""
        # Arrange - only mock external services (Celery)
        with patch("packages.webui.services.collection_service.celery_app") as mock_celery:
            mock_celery.send_task.return_value = AsyncMock(id="task-123")

            # Act
            collection_dict, operation_dict = await collection_service.create_collection(
                user_id=test_user.id,
                name="Test Collection",
                description="Test description",
                embedding_model="Qwen/Qwen3-Embedding-0.6B",
                source_paths=["/data/documents"],
            )

            await async_session.commit()

        # Assert - verify actual database state
        assert collection_dict["name"] == "Test Collection"
        assert collection_dict["status"] == CollectionStatus.PENDING.value
        assert collection_dict["owner_id"] == test_user.id

        assert operation_dict["type"] == OperationType.INDEX.value
        assert operation_dict["status"] == "PENDING"
        assert operation_dict["collection_id"] == collection_dict["id"]

        # Verify persistence - fetch from database
        from shared.database.models import Collection, Operation
        from sqlalchemy import select

        result = await async_session.execute(
            select(Collection).where(Collection.id == collection_dict["id"])
        )
        db_collection = result.scalar_one()
        assert db_collection.name == "Test Collection"
        assert db_collection.status == CollectionStatus.PENDING

        result = await async_session.execute(
            select(Operation).where(Operation.uuid == operation_dict["uuid"])
        )
        db_operation = result.scalar_one()
        assert db_operation.type == OperationType.INDEX

        # Verify Celery was called correctly
        mock_celery.send_task.assert_called_once()
        call_args = mock_celery.send_task.call_args
        assert call_args[0][0] == "tasks.index_collection"


    @pytest.mark.asyncio
    async def test_create_collection_validates_name(
        self,
        collection_service,
        test_user
    ):
        """Test that empty collection name is rejected."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            await collection_service.create_collection(
                user_id=test_user.id,
                name="",  # Invalid
                source_paths=["/data"]
            )


    @pytest.mark.asyncio
    async def test_create_collection_duplicate_name_fails(
        self,
        collection_service,
        async_session: AsyncSession,
        test_user
    ):
        """Test that duplicate collection name raises error."""
        # Arrange - create first collection
        with patch("packages.webui.services.collection_service.celery_app"):
            await collection_service.create_collection(
                user_id=test_user.id,
                name="Duplicate Name",
                source_paths=["/data"]
            )
            await async_session.commit()

        # Act & Assert - second collection should fail
        with pytest.raises(EntityAlreadyExistsError):
            with patch("packages.webui.services.collection_service.celery_app"):
                await collection_service.create_collection(
                    user_id=test_user.id,
                    name="Duplicate Name",
                    source_paths=["/data"]
                )


    @pytest.mark.asyncio
    async def test_create_collection_transaction_rollback_on_error(
        self,
        collection_service,
        async_session: AsyncSession,
        test_user
    ):
        """Test that transaction rolls back when Celery task dispatch fails."""
        # Arrange - mock Celery to fail
        with patch("packages.webui.services.collection_service.celery_app") as mock_celery:
            mock_celery.send_task.side_effect = Exception("Celery connection failed")

            # Act & Assert
            with pytest.raises(Exception, match="Celery connection failed"):
                await collection_service.create_collection(
                    user_id=test_user.id,
                    name="Test",
                    source_paths=["/data"]
                )

        # Verify no collection was created (transaction rolled back)
        from shared.database.models import Collection
        from sqlalchemy import select

        result = await async_session.execute(
            select(Collection).where(Collection.owner_id == test_user.id)
        )
        collections = result.scalars().all()
        assert len(collections) == 0, "Collection should not exist after rollback"
```

#### Step 3.3: Keep Unit Tests for Pure Business Logic Only

**Create**: `tests/unit/services/test_collection_service_validation.py`

```python
"""Pure unit tests for CollectionService validation logic."""

import pytest
from packages.webui.services.collection_service import CollectionService


class TestCollectionServiceValidation:
    """Unit tests for validation logic that doesn't require database."""

    @pytest.mark.parametrize("invalid_name", [
        "",
        "   ",
        None,
        "a" * 256,  # Too long
    ])
    def test_validate_collection_name_rejects_invalid(self, invalid_name):
        """Test that invalid names are rejected."""
        with pytest.raises(ValueError):
            CollectionService._validate_collection_name(invalid_name)


    def test_validate_collection_name_accepts_valid(self):
        """Test that valid names are accepted."""
        valid_names = ["Test Collection", "Collection-123", "My_Collection"]
        for name in valid_names:
            # Should not raise
            CollectionService._validate_collection_name(name)


    @pytest.mark.parametrize("chunk_size,chunk_overlap,should_pass", [
        (1000, 200, True),   # Valid
        (500, 100, True),    # Valid
        (1000, 0, True),     # Valid - no overlap
        (100, 100, False),   # Invalid - overlap equals size
        (100, 150, False),   # Invalid - overlap exceeds size
        (0, 100, False),     # Invalid - zero size
        (-100, 50, False),   # Invalid - negative size
    ])
    def test_validate_chunk_config(self, chunk_size, chunk_overlap, should_pass):
        """Test chunk size and overlap validation."""
        if should_pass:
            CollectionService._validate_chunk_config(chunk_size, chunk_overlap)
        else:
            with pytest.raises(ValueError):
                CollectionService._validate_chunk_config(chunk_size, chunk_overlap)
```

#### Step 3.4: Migration Steps

1. **Week 1**: Create factory fixtures and helper functions
2. **Week 1-2**: Convert create/update/delete tests to integration tests with real repos
3. **Week 2**: Extract pure validation logic to unit tests
4. **Week 2**: Add transaction rollback tests
5. **Week 2**: Delete old mock-heavy tests

**Checklist**:
- [ ] Create `tests/fixtures/factories.py`
- [ ] Create `tests/integration/services/` directory
- [ ] Move validation logic tests to `tests/unit/services/test_collection_service_validation.py`
- [ ] Create `tests/integration/services/test_collection_service.py` with real repos
- [ ] Run full test suite: `pytest tests/integration/services/ tests/unit/services/ -v`
- [ ] Delete old `tests/unit/test_collection_service.py`
- [ ] Update imports in any dependent test files

**Estimated Time**: 3-4 days

---

## SHORT TERM ACTIONS (Sprint 1 - Rest of Week 2)

### Action 4: Reorganize Test Structure

**Priority**: P1
**Effort**: 1 day
**Risk**: Low

**Create New Structure**:

```bash
cd /home/john/semantik

# Create new directories
mkdir -p tests/unit/strategies
mkdir -p tests/integration/repositories
mkdir -p tests/integration/services
mkdir -p tests/e2e/websocket

# Move misplaced tests
git mv tests/test_embedding_integration.py tests/integration/test_embedding_integration.py
git mv tests/unit/test_all_chunking_strategies.py tests/integration/strategies/test_all_chunking_strategies.py

# Update test markers
# (See step 4.1 below for details)

# Commit
git commit -m "refactor(tests): reorganize test structure

- Move integration tests to tests/integration/
- Move chunking strategy tests (they use real implementations)
- Create logical grouping by test type
- Follows project CLAUDE.md guidelines"
```

#### Step 4.1: Add Proper Test Markers

**Edit**: `tests/integration/test_embedding_integration.py:1`

Add marker at top of file:
```python
"""Integration tests for embedding service."""

import pytest

pytestmark = pytest.mark.integration  # Mark all tests in file as integration
```

**Edit**: `tests/integration/strategies/test_all_chunking_strategies.py:1`

```python
"""Integration tests for chunking strategies with real implementations."""

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow  # These tests take time
]
```

#### Step 4.2: Update `pytest.ini`

**Edit**: `pytest.ini`

```ini
[pytest]
markers =
    unit: Unit tests (fast, no external dependencies)
    integration: Integration tests (use database, may be slower)
    e2e: End-to-end tests (require full stack running)
    slow: Tests that take >1 second
    performance: Performance/benchmark tests (run separately)

# Run unit tests by default
addopts =
    -v
    --strict-markers
    -m "not e2e"  # Don't run e2e by default

# Separate test directories
testpaths = tests

# Async support
asyncio_mode = auto
```

**Usage**:
```bash
# Run only unit tests (fast)
pytest -m unit

# Run integration tests
pytest -m integration

# Run everything except E2E
pytest -m "not e2e"

# Run E2E tests (requires services)
pytest -m e2e
```

---

### Action 5: Extract Shared E2E Fixtures

**Priority**: P1
**Effort**: 4 hours
**Risk**: Low

**Current Problem**: Auth helpers duplicated in:
- `tests/e2e/test_websocket_integration.py:38-84`
- `tests/e2e/test_websocket_reindex.py:38-84`

**Solution**: Create shared E2E conftest

#### Step 5.1: Create E2E Fixtures

**Create**: `tests/e2e/conftest.py`

```python
"""Shared fixtures for E2E tests."""

import os
import random
import pytest
import requests
from uuid import uuid4


@pytest.fixture(scope="session")
def api_base_url():
    """Get API base URL from environment."""
    return os.getenv("API_BASE_URL", "http://localhost:8080")


@pytest.fixture(scope="session")
def ws_base_url(api_base_url):
    """Get WebSocket base URL from API URL."""
    return api_base_url.replace("http://", "ws://").replace("https://", "wss://")


@pytest.fixture
def e2e_test_user(api_base_url):
    """Create a unique test user for E2E tests with automatic cleanup.

    Returns credentials dict: {"username": "...", "password": "..."}
    """
    unique_id = f"{random.randint(10000, 99999)}_{uuid4().hex[:8]}"
    username = f"e2e_user_{unique_id}"
    email = f"e2e_{unique_id}@example.com"
    password = "test_password_123"

    # Register user
    response = requests.post(
        f"{api_base_url}/api/auth/register",
        json={
            "username": username,
            "email": email,
            "password": password,
            "full_name": f"E2E Test User {unique_id}",
        },
        timeout=10,
    )

    if response.status_code != 200:
        pytest.fail(f"Failed to create E2E test user: {response.text}")

    return {"username": username, "password": password, "email": email}


@pytest.fixture
def e2e_auth_headers(api_base_url, e2e_test_user):
    """Get authentication headers for E2E test user."""
    # Login
    login_response = requests.post(
        f"{api_base_url}/api/auth/login",
        json={
            "username": e2e_test_user["username"],
            "password": e2e_test_user["password"],
        },
        timeout=10,
    )

    if login_response.status_code != 200:
        pytest.fail(f"Failed to login E2E user: {login_response.text}")

    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def e2e_auth_token(e2e_auth_headers):
    """Extract just the token from auth headers."""
    return e2e_auth_headers["Authorization"].replace("Bearer ", "")


@pytest.fixture(scope="session")
def skip_if_services_unavailable(api_base_url):
    """Skip tests if required services are not running."""
    try:
        response = requests.get(f"{api_base_url}/health", timeout=2)
        if response.status_code != 200:
            pytest.skip("Services not available")
    except requests.exceptions.RequestException:
        pytest.skip("Services not running at {api_base_url}")


@pytest.fixture
def cleanup_collections(api_base_url, e2e_auth_headers):
    """Cleanup fixture that deletes created collections after test."""
    created_collection_ids = []

    def track_collection(collection_id: str):
        """Track a collection for cleanup."""
        created_collection_ids.append(collection_id)

    yield track_collection

    # Cleanup after test
    for collection_id in created_collection_ids:
        try:
            requests.delete(
                f"{api_base_url}/api/v2/collections/{collection_id}",
                headers=e2e_auth_headers,
                timeout=10,
            )
        except Exception:
            pass  # Best effort cleanup
```

#### Step 5.2: Update E2E Tests to Use Shared Fixtures

**Before** (`tests/e2e/test_websocket_integration.py:38-84`):
```python
def _get_auth_headers(self) -> dict[str, str]:
    # 40+ lines of authentication logic
    ...
```

**After**:
```python
# Remove all auth helper methods - use fixtures instead

@pytest.mark.e2e
class TestWebSocketIntegration:
    """E2E tests for WebSocket functionality."""

    def test_collection_creation_with_websocket_progress(
        self,
        api_base_url,
        ws_base_url,
        e2e_auth_headers,
        e2e_auth_token,
        cleanup_collections,
        skip_if_services_unavailable,  # Auto-skip if services down
    ):
        """Test collection creation with WebSocket progress tracking."""
        # Create collection
        response = requests.post(
            f"{api_base_url}/api/v2/collections",
            headers=e2e_auth_headers,
            json={
                "name": f"ws-test-{uuid4().hex[:8]}",
                "source_paths": ["/test/data"],
            },
        )
        assert response.status_code == 200

        collection_data = response.json()
        cleanup_collections(collection_data["id"])  # Track for cleanup

        # Connect to WebSocket with token
        ws_url = f"{ws_base_url}/ws/operations/{operation_id}?token={e2e_auth_token}"
        # ... rest of test
```

**Estimated Time**: 4 hours

---

## SPRINT 2 ACTIONS (Week 3-4)

### Action 6: Add Missing Negative Test Cases

**Priority**: P1
**Effort**: 2-3 days
**Risk**: Low

#### Step 6.1: Add Error Handling Tests for Search Service

**Create**: `tests/integration/services/test_search_service_errors.py`

```python
"""Error handling tests for SearchService."""

import pytest
from unittest.mock import patch, AsyncMock
import httpx
from packages.webui.services.search_service import SearchService


@pytest.mark.integration
class TestSearchServiceErrorHandling:
    """Test error scenarios and recovery."""

    @pytest.fixture
    async def search_service(self, async_session):
        """Create search service."""
        from shared.database.repositories import create_collection_repository
        return SearchService(
            db_session=async_session,
            collection_repo=create_collection_repository(async_session)
        )

    @pytest.mark.asyncio
    async def test_search_handles_vecpipe_timeout(
        self,
        search_service,
        test_collection
    ):
        """Test that search handles vecpipe timeout gracefully."""
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timeout")

            with pytest.raises(ServiceUnavailableError, match="Search service timeout"):
                await search_service.single_collection_search(
                    user_id=1,
                    collection_uuid=test_collection.id,
                    query="test",
                    k=10
                )


    @pytest.mark.asyncio
    async def test_search_handles_vecpipe_500_error(
        self,
        search_service,
        test_collection
    ):
        """Test that search handles internal server errors."""
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500 Error", request=None, response=mock_response
            )
            mock_post.return_value = mock_response

            with pytest.raises(ServiceUnavailableError):
                await search_service.single_collection_search(
                    user_id=1,
                    collection_uuid=test_collection.id,
                    query="test",
                    k=10
                )


    @pytest.mark.asyncio
    async def test_search_validates_k_parameter(self, search_service, test_collection):
        """Test that invalid k values are rejected."""
        invalid_k_values = [0, -1, 10001]

        for k in invalid_k_values:
            with pytest.raises(ValueError, match="k must be between 1 and 10000"):
                await search_service.single_collection_search(
                    user_id=1,
                    collection_uuid=test_collection.id,
                    query="test",
                    k=k
                )


    @pytest.mark.asyncio
    async def test_search_handles_empty_query(self, search_service, test_collection):
        """Test that empty queries are handled."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await search_service.single_collection_search(
                user_id=1,
                collection_uuid=test_collection.id,
                query="",
                k=10
            )


    @pytest.mark.asyncio
    async def test_search_handles_nonexistent_collection(
        self,
        search_service
    ):
        """Test that searching nonexistent collection raises error."""
        with pytest.raises(EntityNotFoundError):
            await search_service.single_collection_search(
                user_id=1,
                collection_uuid="nonexistent-uuid",
                query="test",
                k=10
            )
```

#### Step 6.2: Add Concurrent Operation Tests

**Create**: `tests/integration/test_concurrent_operations.py`

```python
"""Tests for concurrent operation handling."""

import pytest
import asyncio
from shared.database.repositories import create_collection_repository


@pytest.mark.integration
class TestConcurrentCollectionOperations:
    """Test concurrent access to collections."""

    @pytest.mark.asyncio
    async def test_concurrent_collection_creation_different_names(
        self,
        async_session
    ):
        """Test that multiple collections can be created concurrently."""
        repository = create_collection_repository(async_session)

        async def create_collection(name: str):
            return await repository.create(
                name=name,
                owner_id=1
            )

        # Create 10 collections concurrently
        tasks = [
            create_collection(f"concurrent-{i}")
            for i in range(10)
        ]

        collections = await asyncio.gather(*tasks)
        await async_session.commit()

        # All should succeed
        assert len(collections) == 10
        assert len(set(c.id for c in collections)) == 10  # All unique IDs


    @pytest.mark.asyncio
    async def test_concurrent_collection_creation_same_name_fails(
        self,
        async_session
    ):
        """Test that concurrent creation with same name is handled."""
        repository = create_collection_repository(async_session)

        async def create_collection():
            return await repository.create(
                name="duplicate-concurrent",
                owner_id=1
            )

        # Try to create same name concurrently
        tasks = [create_collection() for _ in range(5)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Exactly one should succeed, others should fail
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]

        assert len(successes) == 1, "Exactly one creation should succeed"
        assert len(failures) == 4, "Four should fail with duplicate error"
```

**Estimated Time**: 2-3 days for comprehensive negative test coverage

---

### Action 7: Fix Performance Test Flakiness

**Priority**: P2
**Effort**: 4 hours
**Risk**: Low

#### Step 7.1: Fix Path Traversal Performance Test

**Edit**: `tests/security/test_path_traversal.py:239-256`

**Before**:
```python
def test_performance_under_10ms(self) -> None:
    """Test that validation completes within 10ms performance requirement."""
    for paths in test_cases:
        start_time = time.perf_counter()
        with contextlib.suppress(ValidationError):
            ChunkingSecurityValidator.validate_file_paths(paths)

        elapsed_time = (time.perf_counter() - start_time) * 1000
        assert elapsed_time < 10, f"Validation took {elapsed_time:.2f}ms"
```

**After**:
```python
import statistics
from typing import List

@pytest.mark.performance  # Separate marker
def test_performance_median_under_10ms(self) -> None:
    """Test that validation median time is under 10ms.

    Uses median (p50) and p95 to handle CI variability.
    Runs multiple iterations for statistical significance.
    """
    test_cases = [
        ["normal/path/file.txt"],
        ["%2e%2e%2f%2e%2e%2fetc%2fpasswd"],
        ["file\x00.txt"],
        ["C:\\Windows\\System32"],
        ["documents/file.txt", "data/other.pdf", "test.doc"],
    ]

    timings: List[float] = []
    WARMUP_ITERATIONS = 10
    TEST_ITERATIONS = 100

    # Warmup to stabilize timing (JIT, caching, etc.)
    for _ in range(WARMUP_ITERATIONS):
        with contextlib.suppress(ValidationError):
            ChunkingSecurityValidator.validate_file_paths(["warmup/path.txt"])

    # Actual timing runs
    for paths in test_cases:
        for _ in range(TEST_ITERATIONS):
            start_time = time.perf_counter()
            with contextlib.suppress(ValidationError):
                ChunkingSecurityValidator.validate_file_paths(paths)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            timings.append(elapsed_ms)

    # Statistical analysis
    p50 = statistics.median(timings)
    p95 = sorted(timings)[int(len(timings) * 0.95)]
    p99 = sorted(timings)[int(len(timings) * 0.99)]

    # More lenient thresholds accounting for CI variability
    assert p50 < 10, (
        f"Median (p50) validation time {p50:.2f}ms exceeds 10ms target\n"
        f"Distribution: p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms"
    )
    assert p95 < 20, (
        f"P95 validation time {p95:.2f}ms exceeds 20ms threshold (2x median budget)\n"
        f"This indicates inconsistent performance"
    )

    # Log performance for monitoring
    print(f"\nPerformance metrics:")
    print(f"  p50: {p50:.2f}ms")
    print(f"  p95: {p95:.2f}ms")
    print(f"  p99: {p99:.2f}ms")
    print(f"  min: {min(timings):.2f}ms")
    print(f"  max: {max(timings):.2f}ms")
```

**Usage**:
```bash
# Run performance tests separately
pytest -m performance -v

# Run with specific thresholds in CI
pytest -m performance --performance-threshold-multiplier=1.5
```

---

## SPRINT 3 ACTIONS (Week 5-6)

### Action 8: Break Down Giant Tests

**Priority**: P2
**Effort**: 2 days
**Risk**: Low

**Target**: `tests/e2e/test_websocket_integration.py:86-183` (98 lines)

#### Step 8.1: Extract Helper Methods

**Create shared helpers in conftest**:

```python
# tests/e2e/helpers.py
"""Helper utilities for E2E tests."""

import time
from typing import Callable


def wait_for_condition(
    check_fn: Callable[[], bool],
    timeout: float = 60,
    interval: float = 0.5,
    backoff: float = 1.5,
    max_interval: float = 5,
) -> bool:
    """Wait for condition with exponential backoff.

    Args:
        check_fn: Function that returns True when condition is met
        timeout: Maximum time to wait in seconds
        interval: Initial polling interval
        backoff: Multiplier for interval after each check
        max_interval: Maximum interval between checks

    Returns:
        True if condition met, False if timeout
    """
    start_time = time.time()
    current_interval = interval

    while time.time() - start_time < timeout:
        if check_fn():
            return True

        time.sleep(current_interval)
        current_interval = min(current_interval * backoff, max_interval)

    return False


def wait_for_operation_complete(
    api_base_url: str,
    operation_id: str,
    headers: dict,
    timeout: float = 60
) -> dict:
    """Wait for operation to complete and return final state.

    Raises:
        TimeoutError: If operation doesn't complete in timeout
        RuntimeError: If operation fails
    """
    def check_complete():
        response = requests.get(
            f"{api_base_url}/api/v2/operations/{operation_id}",
            headers=headers
        )
        if response.status_code == 200:
            data = response.json()
            return data["status"] in ["completed", "failed"]
        return False

    if not wait_for_condition(check_complete, timeout):
        raise TimeoutError(f"Operation {operation_id} did not complete in {timeout}s")

    # Fetch final state
    response = requests.get(
        f"{api_base_url}/api/v2/operations/{operation_id}",
        headers=headers
    )
    data = response.json()

    if data["status"] == "failed":
        raise RuntimeError(f"Operation failed: {data.get('error', 'Unknown error')}")

    return data
```

#### Step 8.2: Split Giant Test

**Before** (lines 86-183):
```python
def test_collection_creation_with_websocket_progress(self):
    # 98 lines doing:
    # - Auth setup
    # - Collection creation
    # - WebSocket connection
    # - Progress monitoring
    # - Final status verification
```

**After** - Split into focused tests:

```python
from tests.e2e.helpers import wait_for_operation_complete

class TestWebSocketCollectionOperations:
    """E2E tests for WebSocket integration with collection operations."""

    def test_websocket_receives_progress_messages(
        self,
        api_base_url,
        ws_base_url,
        e2e_auth_headers,
        e2e_auth_token,
        cleanup_collections,
    ):
        """Test that WebSocket receives progress messages during indexing."""
        # Create collection
        response = requests.post(
            f"{api_base_url}/api/v2/collections",
            headers=e2e_auth_headers,
            json={
                "name": f"ws-progress-test-{uuid4().hex[:8]}",
                "source_paths": ["/test/data"],
            },
        )
        assert response.status_code == 200

        data = response.json()
        collection_id = data["id"]
        operation_id = data["operation_id"]
        cleanup_collections(collection_id)

        # Connect to WebSocket
        messages = []
        ws_url = f"{ws_base_url}/ws/operations/{operation_id}?token={e2e_auth_token}"

        def on_message(ws, message):
            messages.append(json.loads(message))

        ws = websocket.WebSocketApp(ws_url, on_message=on_message)
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        try:
            # Wait for operation to complete
            wait_for_operation_complete(
                api_base_url,
                operation_id,
                e2e_auth_headers,
                timeout=60
            )
        finally:
            ws.close()
            ws_thread.join(timeout=2)

        # Assert - focus only on WebSocket messages
        assert len(messages) > 0, "Should receive at least one progress message"
        assert any(0 <= msg.get("progress", -1) <= 100 for msg in messages)


    def test_collection_becomes_ready_after_indexing(
        self,
        api_base_url,
        e2e_auth_headers,
        cleanup_collections,
    ):
        """Test that collection status transitions to READY after indexing."""
        # Create collection
        response = requests.post(
            f"{api_base_url}/api/v2/collections",
            headers=e2e_auth_headers,
            json={
                "name": f"status-test-{uuid4().hex[:8]}",
                "source_paths": ["/test/data"],
            },
        )
        assert response.status_code == 200

        data = response.json()
        collection_id = data["id"]
        operation_id = data["operation_id"]
        cleanup_collections(collection_id)

        # Wait for completion
        wait_for_operation_complete(
            api_base_url,
            operation_id,
            e2e_auth_headers,
            timeout=60
        )

        # Assert - focus only on final collection state
        response = requests.get(
            f"{api_base_url}/api/v2/collections/{collection_id}",
            headers=e2e_auth_headers
        )
        assert response.status_code == 200

        collection = response.json()
        assert collection["status"] == "ready"
        assert collection["document_count"] > 0


    def test_indexing_creates_documents(
        self,
        api_base_url,
        e2e_auth_headers,
        cleanup_collections,
    ):
        """Test that indexing operation creates documents in collection."""
        # Create collection
        response = requests.post(
            f"{api_base_url}/api/v2/collections",
            headers=e2e_auth_headers,
            json={
                "name": f"docs-test-{uuid4().hex[:8]}",
                "source_paths": ["/test/data"],
            },
        )
        assert response.status_code == 200

        data = response.json()
        collection_id = data["id"]
        operation_id = data["operation_id"]
        cleanup_collections(collection_id)

        # Wait for completion
        wait_for_operation_complete(
            api_base_url,
            operation_id,
            e2e_auth_headers,
            timeout=60
        )

        # Assert - focus only on document creation
        response = requests.get(
            f"{api_base_url}/api/v2/collections/{collection_id}/documents",
            headers=e2e_auth_headers
        )
        assert response.status_code == 200

        documents = response.json()
        assert len(documents) > 0, "Should create at least one document"
        assert all("file_path" in doc for doc in documents)
```

**Benefits**:
- Each test has single clear purpose
- Easier to identify what broke when test fails
- Can run specific tests in isolation
- Better test names describe actual behavior

### Action 3: Quarantine Script-Style "Tests" That Hit Live Services ⚠️ CRITICAL

**Priority**: P0  
**Effort**: 0.5 day to relocate/remove, follow-up stories for real coverage  
**Risk**: Low - these files already provide no automated validation

**Problem**: Several top-level files under `tests/` and `apps/webui-react/tests/` are executable scripts rather than assertions. They call live HTTP endpoints, sleep, or print results, and they run as part of CI because they live under `tests/`. Examples:
- `tests/test_metrics.py`, `tests/test_metrics_update.py`, `tests/test_search.py` – issue only `requests` calls with bare `print`, zero assertions.
- `tests/test_embedding_performance.py`, `tests/test_embedding_full_integration.py` – long-running benchmarks with timers, create real services, no validation.
- `tests/streaming/validate_streaming_pipeline.py` – generates 100MB files, runs async loops, and prints results without assertions.
- `apps/webui-react/tests/api_test_suite.py` – asynchronous smoke harness that depends on a fully running stack.

**Impact**: Pytest treats these files as passing even when nothing is asserted, masking regressions and lengthening suite runtime. Some call external services (`localhost:8080`, `:9092`) and will hang if the stack isn't running locally.

**Target State**:
1. Move these scripts into a dedicated `manual_tests/` (or `qa_scripts/`) directory excluded from pytest discovery.
2. Replace each with real automated coverage stories (tracked separately) before re-introducing them to automated runs.
3. Document how to run the manual scripts when needed.

**Steps**:
1. Create `manual_tests/README.md` explaining intent.
2. Relocate the script files (or delete if superseded) and update imports.
3. Update CI (pytest invocation) to ignore the new manual directory explicitly.
4. File follow-up tickets for proper automated coverage (see Tracking doc).

### Action 4: Replace Placeholder Reranking "E2E" Suite With Real Coverage ⚠️ CRITICAL

**Priority**: P0  
**Effort**: 1-2 days for a real API+service reranking flow  
**Risk**: Medium - requires orchestrating integration fixtures

**Problem**: `tests/test_reranking_e2e.py` (now deleted) previously contained only `assert True` statements referencing "verified by code inspection". No request was performed, so the reranking pipeline remains untested end-to-end.

**Target State**:
- Implement an async integration test that uses existing fixtures to issue a real search request with reranking enabled and validates that reranking parameters flow through WebUI → VecPipe and that reranking metrics return.
- Ensure the test is tagged (`@pytest.mark.integration`) and placed in `tests/integration/search/` (or equivalent canonical location).
- Remove any placeholder scaffolding once genuine coverage exists.

**Verification**: Run `uv run pytest tests/integration/search/test_reranking_flow.py -v`.

### Action 5: Deduplicate Celery Helper Suites & Audit Assertions

**Priority**: P1  
**Effort**: 1 day  
**Risk**: Low

**Problem**: `tests/webui/test_tasks_helpers.py` and `tests/webui/test_tasks_helpers_original.py` diverge while covering overlapping behavior. Keeping both causes maintenance drift and inconsistent assertions.

**Target State**:
1. Agree on a single canonical helper suite.
2. Merge meaningful scenarios, drop redundant ones, and tighten assertions (avoid `assert True` placeholders).
3. Remove the obsolete file and update imports.

### Action 6: Stabilize WebSocket/Redis Tests (Remove Sleeps & Real Network Reliance)

**Priority**: P1  
**Effort**: 2-3 days across suites  
**Risk**: Medium - concurrency logic is sensitive

**Problem**: WebSocket suites (`tests/webui/test_chunking_websocket.py`, `tests/websocket/test_performance.py`, `tests/websocket/websocket_load_test.py`, etc.) rely on `time.sleep`, direct `redis://localhost:6379/15`, and even aiohttp clients. While CI spins up Redis, these tests are effectively integration/load tests hiding under unit directories and remain flaky locally.

**Target State**:
- Introduce fake Redis pub/sub fixtures or use `fakeredis` to avoid real network calls.
- Replace `time.sleep` with controllable time mocking (e.g., `freezegun`, `asyncio` clock helpers).
- Extract true load/stress scripts into performance tooling outside pytest default runs.

**Verification**: Run `uv run pytest tests/websocket -v` locally without Redis; ensure tests pass using fakes/mocks.

**Estimated Time**: 2 days to refactor all giant tests

### Action 7: Make Streaming Memory Pool Tests Deterministic ⚠️ HIGH

**Priority**: P1  
**Effort**: 1 day  
**Risk**: Medium - leak detection relies on background tasks

**Problem**: `packages/shared/chunking/infrastructure/streaming/test_memory_pool.py` uses real `asyncio.sleep`, threaded leak detection, and `gc.collect()` to assert behavior (e.g., lines 90-149, 196-233). This slows the suite and causes flaky timing.

**Target State**:
1. Inject a controllable clock/scheduler into `MemoryPool` so tests can advance leak timers without sleeping.
2. Replace `asyncio.sleep` calls with patched versions (e.g., using `pytest.mark.parametrize` + virtual time helpers).
3. Assert leak detection via deterministic hooks rather than relying on garbage collection side effects.

**Steps**:
1. Refactor `MemoryPool` to accept an optional timing helper (e.g., `loop_time_fn`).
2. Update tests to drive leak detection manually (e.g., call a `check_leaks()` helper).
3. Remove real sleeps/threading from tests; use `pytest` fixtures to clean up background tasks.

**Verification**: `uv run pytest packages/shared/chunking/infrastructure/streaming/test_memory_pool.py -q` should complete quickly and reliably on repeat runs.

### Action 8: Promote Use-Case & Validation Suites to Real Integration Coverage ⚠️ HIGH

**Priority**: P1  
**Effort**: 2-3 days  
**Risk**: Medium - requires wiring real repositories/unit-of-work

**Problem**: `tests/application/test_*_use_case.py` suites (e.g., `test_process_document_use_case.py`, `test_preview_chunking_use_case.py`), `tests/webui/test_ingestion_chunking_integration.py`, `tests/webui/services/test_collection_service.py`, and `packages/webui/tests/test_collection_service_chunking_validation.py` mock every dependency, duplicate service logic, and largely assert on call counts. They provide little confidence that wiring works with real repositories or DB transactions.

**Target State**:
1. Move critical scenarios into integration tests that exercise the actual unit-of-work, repositories, and notification services.
2. Keep only genuine unit tests for pure validation helpers; remove redundant mocks.
3. Align coverage with service/integration suites to avoid duplication.

**Steps**:
1. Introduce shared fixtures for unit-of-work + test database (reuse existing async session fixtures).
2. Rewrite the highest-value scenarios (success path, validation failures, checkpoint resume) to use real models.
3. Delete or drastically slim down the mock-heavy files once integration coverage exists.

**Verification**: `uv run pytest tests/integration/use_cases/ -v` (new suite) passes and captures the scenarios currently mocked.

### Action 9: Enable Rate-Limit Tests in CI ⚠️ CRITICAL

**Priority**: P0  
**Effort**: 1 day  
**Risk**: Low - requires fixture adjustments

**Problem**: `tests/api/test_rate_limits.py` is skipped in CI via `@pytest.mark.skipif(os.getenv(\"CI\") == \"true\")`, so we lack automated coverage of rate limiting and circuit-breaker behavior.

**Target State**:
1. Provide deterministic Redis/SlowAPI fixtures that run in CI (e.g., use the existing Redis service, or mock the limiter).
2. Remove the `skipif` guard once the tests pass reliably in CI.
3. Ensure bypass token logic and failure counters are fully exercised.

**Steps**:
1. Create fixture to reset limiter state and use in tests.
2. Run the suite locally with `CI=true` env to confirm behavior.
3. Update docs/CI expectations.

**Verification**: CI run should execute `tests/api/test_rate_limits.py` without skips; confirm via workflow logs.

### Action 10: Tighten Domain Strategy Assertions & Remove Placeholder Checks ⚠️ MEDIUM

**Priority**: P2  
**Effort**: 1 day  
**Risk**: Low

**Problem**: `tests/domain/test_chunking_strategies.py` still contains placeholder assertions (`assert True`) and minimal checks on metadata/weights, letting regressions slip by. Similar redundant validation exists in `packages/webui/tests/test_collection_service_chunking_validation.py`.

**Target State**:
1. Replace placeholder assertions with concrete expectations (e.g., ensure `strategies_used` metadata exists, weights sum correctly).
2. Deduplicate coverage by moving overlapping scenarios into integration tests (see Action 8).

**Verification**: Updated tests fail when metadata fields regress (simulate by intentionally mutating production code during development).

### Action 11: Align Chunking/Search API Suites With Integration Coverage ⚠️ HIGH

**Priority**: P1  
**Effort**: 3-4 days  
**Risk**: Medium - requires reorganizing large suites

**Problem**: `tests/webui/api/v2/test_chunking.py`, `tests/webui/api/v2/test_chunking_simple_integration.py`, `tests/webui/api/v2/test_chunking_direct.py`, and `tests/webui/services/test_search_service.py` are massive mock-based suites (500+ tests) that patch FastAPI dependencies, override singletons, and assert on implementation details. They overlap heavily with newer integration suites yet remain brittle and slow.

**Target State**:
1. Collapse redundant API/service tests into focused integration coverage that uses real dependency overrides (database + service fixtures).
2. Keep only lightweight unit tests for schema/validation and error mapping.
3. Split monolithic files into smaller modules grouped by endpoint behavior (strategies, previews, analytics, operations).

**Steps**:
1. Inventory overlapping scenarios with existing integration suites (`tests/webui/api/v2/test_chunking_integration.py`, service-level integration tests).
2. Port high-value gaps to integration tests, delete redundant mock-heavy cases.
3. Introduce shared fixtures for FastAPI dependency overrides to avoid per-test patch cascades.

**Verification**: `uv run pytest tests/webui/api/v2/ -k chunking` executes with manageable runtime and minimal mocking.

### Action 12: Refocus Metrics & Chunker Unit Suites ⚠️ MEDIUM

**Priority**: P2  
**Effort**: 2 days  
**Risk**: Low

**Problem**:
- `tests/webui/test_chunking_metrics.py` manipulates Prometheus internals (`metric._metrics.clear()`) and mocks `ChunkingService`, providing little behavioral coverage.
- `tests/unit/test_hierarchical_chunker.py`, `tests/unit/test_hybrid_chunker.py`, and related suites rely on heavy mocking/patching of llama-index internals and cover massive permutations.
- Metrics assertions duplicate logic better handled via integration tests that exercise real chunking flows.

**Target State**:
1. Move Prometheus metric validation into integration tests that run actual chunking operations via service fixtures.
2. Reduce unit chunker tests to focused, deterministic scenarios with lightweight fixtures or property-based checks.
3. Remove direct manipulation of metric internals; instead, use the Prometheus registry isolation fixtures.

**Verification**: Metrics coverage verified via new integration tests; unit chunker suites execute quickly with limited fixture setup.

**Progress (2025-10-17)**:
1. Added `tests/integration/chunking/test_ingestion_metrics.py`

**Pending (2025-10-17)**:
- Recursive runs in `tests/integration/chunking/test_ingestion_metrics.py` still fall back to `TokenChunker` when driven by the default fixtures. Need to trace ChunkingConfigBuilder vs. ChunkingStrategyFactory resolution to stop the fallback and satisfy the new assertions.

 to exercise real chunking flows with isolated Prometheus registries and DB/fakeredis fixtures.
2. Replaced mock-heavy chunker suites with `tests/unit/chunking/test_hierarchical_chunker_validations.py` plus expanded integration assertions under `tests/integration/strategies/`.
3. Retired metrics/error suites that manipulated private `_metrics`, ensuring new coverage relies on CollectorRegistry fixtures.

### Action 13: Rationalize Search/Rate-Limiter Unit Suites ⚠️ MEDIUM

**Priority**: P2  
**Effort**: 2 days  
**Risk**: Low

**Problem**:
- `tests/unit/test_search_service.py` recreates `SearchService` with `AsyncMock` dependencies, patches `httpx.AsyncClient`, and asserts on internal calls rather than actual HTTP responses. Most scenarios overlap with API/integration coverage.
- `tests/unit/test_rate_limiter.py` manipulates global limiter state (`limiter._limiter.storage.storage`) and relies on environment-dependent behavior, which can leak across tests and diverge from production configuration.

**Target State**:
1. Convert high-value search scenarios into integration tests that hit `/api/v2/search` with dependency overrides; keep a slim unit file for pure validation.
2. Provide isolated rate-limiter fixtures (fresh `Limiter` instance, fakeredis or in-memory storage) and rework tests to assert observable behavior without touching private attributes.
3. Remove redundant call-count assertions once integration coverage covers the flows.

**Verification**: `uv run pytest tests/integration/search/ tests/unit/services/test_search_service_validation.py` runs green with minimal mocks; rate-limiter unit tests pass without mutating global state.

### Action 14: Consolidate Directory/Document Auxiliary Suites ⚠️ HIGH

**Priority**: P1  
**Effort**: 2-3 days  
**Risk**: Medium - requires coordinating Celery/WS fixtures

**Problem**: Tests such as `tests/webui/test_document_chunk_count_updates.py`, `tests/webui/services/test_directory_scan_service.py`, `tests/webui/services/test_execute_ingestion_chunking.py`, and the corresponding unit repository/service suites (`tests/unit/test_document_scanning_service.py`, `tests/unit/test_directory_scan_service.py`, `tests/unit/test_operation_repository.py`, etc.) rely on heavy mocking, custom filesystem setups, and duplicate scenarios already covered (or planned) in integration tests. They assert on implementation details (call counts, mock attributes) rather than observable outcomes.

**Target State**:
1. Move Celery/ingestion verification into integration tests using fakeredis + async session fixtures.
2. Provide lightweight, behavior-focused unit tests only for pure helper logic (e.g., pattern filtering).
3. Replace bespoke filesystem scaffolding with reusable fixtures or integration-level smoke tests.

**Verification**: A consolidated integration suite (e.g., `tests/integration/tasks/test_document_ingestion.py`) validates chunk count updates and directory scans end-to-end; unit suites shrink to minimal validation helpers.

**Progress (2025-10-17)**:
1. Introduced `tests/integration/chunking/test_ingestion_metrics.py` to cover chunk count updates and service metrics using async DB + fakeredis fixtures.
2. Removed `tests/webui/test_document_chunk_count_updates.py` and `tests/webui/services/test_execute_ingestion_chunking.py`, migrating high-value assertions to integration coverage.
3. Documented remaining Celery orchestration gaps for follow-up (see tracking table entries for directory scan and task orchestration).

---

## Action Plan Summary

### Week 1-2 (Sprint 1)
- [ ] Delete `test_auth.py` (5 min)
- [ ] Fix `test_collection_repository.py` (2-3 days)
- [ ] Start fixing `test_collection_service.py` (3-4 days)
- [ ] Quarantine manual/script-style test files (0.5 day)
- [ ] Replace placeholder reranking E2E suite with real coverage (1-2 days)
- [ ] Enable rate-limit tests in CI (1 day)
- [ ] Kick off chunking/search API suite consolidation (inventory overlaps, draft plan)

### Week 3-4 (Sprint 2)
- [ ] Complete `test_collection_service.py` refactor
- [ ] Reorganize test structure (1 day)
- [ ] Extract shared E2E fixtures (4 hours)
- [ ] Add missing negative test cases (2-3 days)
- [ ] Fix performance test flakiness (4 hours)
- [ ] Deduplicate Celery helper suites & tighten assertions (1 day)
- [ ] Stabilize WebSocket/Redis tests with fakes and async time control (2-3 days)
- [ ] Make streaming memory pool tests deterministic (1 day)
- [ ] Promote use-case & validation suites to real integration coverage (2-3 days)
- [ ] Consolidate chunking/search API suites into integration-focused coverage (3-4 days)
- [ ] Consolidate directory/document auxiliary suites into integration coverage (2-3 days)

### Week 5-6 (Sprint 3)
- [ ] Break down giant tests (2 days)
- [ ] Add concurrent operation tests (2 days)
- [ ] Documentation updates (1 day)
- [ ] Final verification and cleanup (1 day)

### Success Metrics

**Before**:
- 40-60% false confidence (tests pass but don't catch bugs)
- ~2000 lines of mock setup code
- Tests in wrong directories
- No concurrent operation testing

**After**:
- 90%+ real integration coverage
- <100 lines of mock setup (only for external services)
- Proper test organization
- Comprehensive error and edge case coverage
- Fast unit tests (<10ms each)
- Reliable integration tests (<1s each)

### Risk Mitigation

1. **Database setup complexity**: Use existing `async_session` fixture from conftest.py
2. **Test failures during migration**: Keep old tests until new ones pass
3. **Time constraints**: Prioritize P0 actions first, defer P2 if needed
4. **Breaking changes**: Run full suite after each action to catch issues early

---

## Next Steps

1. Review this plan with team
2. Create tracking tickets for each action
3. Start with Action 1 (delete test_auth.py)
4. Proceed through actions in priority order
5. Review progress after Sprint 1

**Questions?** See `/home/john/semantik/docs/TEST_QUALITY_TRACKING.md` for detailed findings and issue tracking.
