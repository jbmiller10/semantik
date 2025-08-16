"""
Authentication mocking infrastructure for tests.

This module provides a centralized, robust authentication mocking solution
for all integration and unit tests. It ensures consistent behavior across
test environments, regardless of DISABLE_AUTH settings.

Key Features:
- Automatic user creation and authentication for tests
- Owner ID consistency between resources and API calls
- Support for multiple test users with different permissions
- Works in all environments (local, CI/CD, Docker)
- Thread-safe and async-compatible

Usage:
    from packages.webui.tests.auth_mock import mock_auth, create_test_user

    async def test_my_endpoint(mock_auth):
        # Auth is automatically mocked with a default test user
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            response = await client.post("/api/v2/collections", json={...})
            assert response.status_code == 201
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI


class TestUser:
    """Represents a test user with consistent attributes."""

    def __init__(
        self,
        user_id: int = 1,
        username: str = "testuser",
        email: str = "test@example.com",
        full_name: str = "Test User",
        is_active: bool = True,
        is_superuser: bool = False,
    ):
        self.id = user_id
        self.username = username
        self.email = email
        self.full_name = full_name
        self.is_active = is_active
        self.is_superuser = is_superuser
        self.created_at = datetime.now(UTC).isoformat()
        self.last_login = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format expected by the auth system."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_superuser": self.is_superuser,
            "created_at": self.created_at,
            "last_login": self.last_login,
        }


# Pre-defined test users for common scenarios
DEFAULT_TEST_USER = TestUser(user_id=1, username="testuser")
ADMIN_TEST_USER = TestUser(user_id=2, username="admin", email="admin@example.com", is_superuser=True)
INACTIVE_TEST_USER = TestUser(user_id=3, username="inactive", email="inactive@example.com", is_active=False)
SECOND_TEST_USER = TestUser(user_id=4, username="testuser2", email="test2@example.com")


class AuthMocker:
    """
    Manages authentication mocking for tests.
    
    This class provides methods to mock authentication at different levels:
    - Application level (FastAPI dependency overrides)
    - Module level (patching imports)
    - Function level (direct mocking)
    """

    def __init__(self, user: TestUser | None = None):
        """
        Initialize the auth mocker with a test user.
        
        Args:
            user: The test user to use for authentication. Defaults to DEFAULT_TEST_USER.
        """
        self.user = user or DEFAULT_TEST_USER
        self._patches: list[Any] = []

    async def get_current_user(self) -> dict[str, Any]:
        """Mock implementation of get_current_user dependency."""
        return self.user.to_dict()

    async def get_current_user_websocket(self, token: str | None = None) -> dict[str, Any]:
        """Mock implementation of get_current_user_websocket."""
        return self.user.to_dict()

    def override_fastapi_auth(self, app: FastAPI) -> None:
        """
        Override FastAPI authentication dependencies.
        
        This method should be called on a FastAPI test app to replace
        authentication dependencies with mocked versions.
        
        Args:
            app: The FastAPI application instance to modify.
        """
        from packages.webui.auth import get_current_user

        # Override the dependency with our mock
        app.dependency_overrides[get_current_user] = self.get_current_user

    def patch_module_auth(self, module_path: str = "packages.webui.auth") -> None:
        """
        Patch authentication functions at the module level.
        
        This is useful when you need to mock auth for code that imports
        auth functions directly rather than using dependency injection.
        
        Args:
            module_path: The module path to patch.
        """
        # Patch get_current_user
        user_patch = patch(f"{module_path}.get_current_user", new=AsyncMock(return_value=self.user.to_dict()))
        self._patches.append(user_patch)
        user_patch.start()

        # Patch get_current_user_websocket
        ws_patch = patch(
            f"{module_path}.get_current_user_websocket", new=AsyncMock(return_value=self.user.to_dict())
        )
        self._patches.append(ws_patch)
        ws_patch.start()

        # Patch authenticate_user to always succeed
        auth_patch = patch(f"{module_path}.authenticate_user", new=AsyncMock(return_value=self.user.to_dict()))
        self._patches.append(auth_patch)
        auth_patch.start()

        # Patch verify_token to always return username
        verify_patch = patch(f"{module_path}.verify_token", new=MagicMock(return_value=self.user.username))
        self._patches.append(verify_patch)
        verify_patch.start()

    def stop_patches(self) -> None:
        """Stop all active patches."""
        for p in self._patches:
            p.stop()
        self._patches.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup patches."""
        self.stop_patches()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup patches."""
        self.stop_patches()


@pytest.fixture
def mock_auth() -> AuthMocker:
    """
    Pytest fixture that provides an AuthMocker instance.
    
    This fixture automatically cleans up patches after the test completes.
    
    Usage:
        def test_something(mock_auth):
            mock_auth.override_fastapi_auth(app)
            # Run your test
    """
    mocker = AuthMocker()
    yield mocker
    mocker.stop_patches()


@pytest.fixture
def mock_auth_admin() -> AuthMocker:
    """
    Pytest fixture that provides an AuthMocker with admin user.
    
    Usage:
        def test_admin_function(mock_auth_admin):
            mock_auth_admin.override_fastapi_auth(app)
            # Run your test with admin privileges
    """
    mocker = AuthMocker(user=ADMIN_TEST_USER)
    yield mocker
    mocker.stop_patches()


@pytest.fixture
def mock_auth_second_user() -> AuthMocker:
    """
    Pytest fixture that provides an AuthMocker with a second test user.
    
    Useful for testing multi-user scenarios like access control.
    
    Usage:
        def test_access_control(mock_auth, mock_auth_second_user):
            # Create resource with first user
            # Try to access with second user
    """
    mocker = AuthMocker(user=SECOND_TEST_USER)
    yield mocker
    mocker.stop_patches()


@asynccontextmanager
async def mock_authenticated_user(
    app: FastAPI, user: TestUser | None = None
) -> AsyncGenerator[tuple[FastAPI, TestUser], None]:
    """
    Async context manager for temporarily mocking authentication.
    
    This is useful for inline test scenarios where you need to quickly
    mock auth without using fixtures.
    
    Args:
        app: The FastAPI app to mock auth for.
        user: Optional test user. Defaults to DEFAULT_TEST_USER.
        
    Yields:
        Tuple of (app, user) for use in tests.
        
    Usage:
        async with mock_authenticated_user(app) as (app, user):
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/api/v2/collections")
    """
    user = user or DEFAULT_TEST_USER
    mocker = AuthMocker(user=user)
    mocker.override_fastapi_auth(app)
    
    try:
        yield app, user
    finally:
        # Clean up overrides
        from packages.webui.auth import get_current_user
        if get_current_user in app.dependency_overrides:
            del app.dependency_overrides[get_current_user]


def create_test_user(
    user_id: int = 1,
    username: str = "testuser",
    email: str | None = None,
    **kwargs: Any,
) -> TestUser:
    """
    Factory function to create custom test users.
    
    Args:
        user_id: The user's ID.
        username: The username.
        email: The email address. Defaults to {username}@example.com.
        **kwargs: Additional user attributes.
        
    Returns:
        A configured TestUser instance.
        
    Usage:
        user = create_test_user(user_id=10, username="custom", is_superuser=True)
    """
    if email is None:
        email = f"{username}@example.com"
    
    return TestUser(
        user_id=user_id,
        username=username,
        email=email,
        **kwargs,
    )


# Utility functions for common test scenarios

async def create_authenticated_app(user: TestUser | None = None) -> FastAPI:
    """
    Create a FastAPI app with authentication pre-mocked.
    
    Args:
        user: Optional test user. Defaults to DEFAULT_TEST_USER.
        
    Returns:
        A FastAPI app with auth dependencies overridden.
        
    Usage:
        app = await create_authenticated_app()
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v2/collections")
    """
    from packages.webui.api.v2.collections import router as collections_router

    app = FastAPI()
    app.include_router(collections_router)
    
    mocker = AuthMocker(user=user or DEFAULT_TEST_USER)
    mocker.override_fastapi_auth(app)
    
    return app


def ensure_owner_consistency(resource: dict[str, Any], user: TestUser) -> dict[str, Any]:
    """
    Ensure that a resource has the correct owner_id for the test user.
    
    This utility function helps maintain consistency between the mocked
    user and any resources created in tests.
    
    Args:
        resource: The resource dictionary to update.
        user: The test user who should own the resource.
        
    Returns:
        The updated resource dictionary.
        
    Usage:
        collection = {"name": "Test", "description": "Test collection"}
        collection = ensure_owner_consistency(collection, mock_auth.user)
    """
    resource["owner_id"] = user.id
    return resource


class MockDatabase:
    """
    Mock database that maintains consistency with auth mocking.
    
    This class helps create mock database responses that are consistent
    with the authenticated user.
    """

    def __init__(self, user: TestUser | None = None):
        """Initialize with a test user."""
        self.user = user or DEFAULT_TEST_USER
        self.collections: dict[str, dict[str, Any]] = {}
        self.operations: dict[str, dict[str, Any]] = {}

    def create_collection(self, **kwargs: Any) -> dict[str, Any]:
        """Create a mock collection owned by the test user."""
        import uuid
        collection = {
            "uuid": kwargs.get("uuid", str(uuid.uuid4())),
            "name": kwargs.get("name", "Test Collection"),
            "description": kwargs.get("description", "Test description"),
            "owner_id": self.user.id,  # Ensure owner consistency
            "vector_store_name": kwargs.get("vector_store_name", "test_vector_store"),
            "embedding_model": kwargs.get("embedding_model", "text-embedding-ada-002"),
            "quantization": kwargs.get("quantization", "scalar"),
            "chunk_size": kwargs.get("chunk_size", 512),
            "chunk_overlap": kwargs.get("chunk_overlap", 50),
            "chunking_strategy": kwargs.get("chunking_strategy"),
            "chunking_config": kwargs.get("chunking_config"),
            "is_public": kwargs.get("is_public", False),
            "metadata": kwargs.get("metadata", {}),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "document_count": kwargs.get("document_count", 0),
            "vector_count": kwargs.get("vector_count", 0),
            "status": kwargs.get("status", "pending"),
            "status_message": kwargs.get("status_message"),
        }
        self.collections[collection["uuid"]] = collection
        return collection

    def create_operation(self, collection_id: str, **kwargs: Any) -> dict[str, Any]:
        """Create a mock operation for a collection."""
        import uuid
        operation = {
            "uuid": kwargs.get("uuid", str(uuid.uuid4())),
            "collection_id": collection_id,
            "type": kwargs.get("type", "index"),
            "status": kwargs.get("status", "pending"),
            "config": kwargs.get("config", {}),
            "created_at": datetime.now(UTC).isoformat(),
            "started_at": kwargs.get("started_at"),
            "completed_at": kwargs.get("completed_at"),
            "error_message": kwargs.get("error_message"),
        }
        self.operations[operation["uuid"]] = operation
        return operation


# Export commonly used items
__all__ = [
    "AuthMocker",
    "TestUser",
    "DEFAULT_TEST_USER",
    "ADMIN_TEST_USER",
    "INACTIVE_TEST_USER",
    "SECOND_TEST_USER",
    "mock_auth",
    "mock_auth_admin",
    "mock_auth_second_user",
    "mock_authenticated_user",
    "create_test_user",
    "create_authenticated_app",
    "ensure_owner_consistency",
    "MockDatabase",
]