"""
Pytest configuration and shared fixtures for webui tests.

This module provides common fixtures and configuration for all tests in the
webui package. It uses the centralized auth mocking infrastructure to ensure
consistent testing behavior across all environments.
"""

import asyncio
from typing import Any, AsyncGenerator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient

from packages.webui.tests.auth_mock import (
    AuthMocker,
    DEFAULT_TEST_USER,
    SECOND_TEST_USER,
    MockDatabase,
    create_test_user,
)


# Configure pytest-asyncio to use auto mode
pytest_asyncio.fixture_loop_scope = "function"


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_user():
    """Provide the default test user for consistency across tests."""
    return DEFAULT_TEST_USER


@pytest.fixture
def second_test_user():
    """Provide a second test user for multi-user scenarios."""
    return SECOND_TEST_USER


@pytest.fixture
def test_app(mock_auth) -> FastAPI:
    """
    Create a test FastAPI app with mocked authentication.
    
    This fixture creates a minimal FastAPI app with the collections router
    and authentication pre-mocked using the default test user.
    
    Usage:
        def test_endpoint(test_app):
            # test_app has auth already mocked
    """
    from packages.webui.api.v2.collections import router as collections_router

    app = FastAPI()
    
    # Apply auth mocking
    mock_auth.override_fastapi_auth(app)
    
    # Include routers
    app.include_router(collections_router)
    
    return app


@pytest.fixture
def test_app_with_custom_user() -> FastAPI:
    """
    Create a test FastAPI app that can be configured with different users.
    
    This fixture returns a factory function that creates apps with
    specific user authentication.
    
    Usage:
        def test_multi_user(test_app_with_custom_user):
            app1 = test_app_with_custom_user(user_id=1)
            app2 = test_app_with_custom_user(user_id=2)
    """
    def _create_app(user_id: int = 1, username: str = "testuser", **user_kwargs) -> FastAPI:
        from packages.webui.api.v2.collections import router as collections_router

        app = FastAPI()
        
        # Create custom user and mocker
        user = create_test_user(user_id=user_id, username=username, **user_kwargs)
        mocker = AuthMocker(user=user)
        mocker.override_fastapi_auth(app)
        
        # Include routers
        app.include_router(collections_router)
        
        # Store the mocker and user on the app for reference
        app.state.auth_mocker = mocker
        app.state.test_user = user
        
        return app
    
    return _create_app


@pytest_asyncio.fixture
async def authenticated_client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """
    Create an authenticated async HTTP client.
    
    This client is pre-configured with the test app that has mocked
    authentication, ready for making API calls.
    
    Usage:
        async def test_api_call(authenticated_client):
            response = await authenticated_client.get("/api/v2/collections")
            assert response.status_code == 200
    """
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def authenticated_client_factory():
    """
    Factory fixture for creating authenticated clients with custom users.
    
    This allows tests to create multiple clients with different users
    for testing access control and multi-user scenarios.
    
    Usage:
        async def test_access_control(authenticated_client_factory):
            client1 = await authenticated_client_factory(user_id=1)
            client2 = await authenticated_client_factory(user_id=2)
    """
    clients = []
    
    async def _create_client(user_id: int = 1, username: str = "testuser", **user_kwargs) -> AsyncClient:
        from packages.webui.api.v2.collections import router as collections_router

        app = FastAPI()
        
        # Create custom user and mocker
        user = create_test_user(user_id=user_id, username=username, **user_kwargs)
        mocker = AuthMocker(user=user)
        mocker.override_fastapi_auth(app)
        
        # Include routers
        app.include_router(collections_router)
        
        # Create client
        client = AsyncClient(app=app, base_url="http://test")
        client.app = app  # Store app reference
        client.test_user = user  # Store user reference
        clients.append(client)
        
        return client
    
    yield _create_client
    
    # Cleanup all clients
    for client in clients:
        await client.aclose()


@pytest.fixture
def mock_db(test_user) -> MockDatabase:
    """
    Provide a mock database that maintains consistency with auth.
    
    This fixture creates a MockDatabase instance that automatically
    ensures owner_id consistency with the authenticated test user.
    
    Usage:
        def test_with_mock_db(mock_db):
            collection = mock_db.create_collection(name="Test")
            assert collection["owner_id"] == mock_db.user.id
    """
    return MockDatabase(user=test_user)


@pytest.fixture
def mock_collection_service(mock_db):
    """
    Create a mock CollectionService for testing.
    
    This fixture provides a mock service that returns consistent
    data aligned with the authenticated user.
    """
    from unittest.mock import AsyncMock, MagicMock
    
    service = MagicMock()
    
    # Mock create_collection
    async def mock_create_collection(user_id: int, name: str, description: str, config: dict[str, Any]):
        collection = mock_db.create_collection(
            name=name,
            description=description,
            **config
        )
        operation = mock_db.create_operation(collection["uuid"])
        return collection, operation
    
    service.create_collection = AsyncMock(side_effect=mock_create_collection)
    
    # Mock list_for_user
    async def mock_list_for_user(user_id: int, offset: int = 0, limit: int = 50, include_public: bool = True):
        # Return only collections owned by the user
        user_collections = [
            c for c in mock_db.collections.values()
            if c["owner_id"] == user_id or (include_public and c.get("is_public", False))
        ]
        return user_collections[offset:offset + limit], len(user_collections)
    
    service.list_for_user = AsyncMock(side_effect=mock_list_for_user)
    
    # Mock update
    async def mock_update(collection_id: str, user_id: int, updates: dict[str, Any]):
        if collection_id not in mock_db.collections:
            from packages.shared.database.exceptions import EntityNotFoundError
            raise EntityNotFoundError("collection", collection_id)
        
        collection = mock_db.collections[collection_id]
        if collection["owner_id"] != user_id:
            from packages.shared.database.exceptions import AccessDeniedError
            raise AccessDeniedError(
                user_id=str(user_id),
                resource_type="collection",
                resource_id=collection_id
            )
        
        collection.update(updates)
        return collection
    
    service.update = AsyncMock(side_effect=mock_update)
    
    # Mock delete_collection
    async def mock_delete_collection(collection_id: str, user_id: int):
        if collection_id not in mock_db.collections:
            from packages.shared.database.exceptions import EntityNotFoundError
            raise EntityNotFoundError("collection", collection_id)
        
        collection = mock_db.collections[collection_id]
        if collection["owner_id"] != user_id:
            from packages.shared.database.exceptions import AccessDeniedError
            raise AccessDeniedError(
                user_id=str(user_id),
                resource_type="collection",
                resource_id=collection_id
            )
        
        del mock_db.collections[collection_id]
    
    service.delete_collection = AsyncMock(side_effect=mock_delete_collection)
    
    return service


@pytest.fixture
def override_service_dependency(test_app):
    """
    Fixture that provides a function to override service dependencies.
    
    This makes it easy to inject mock services into the FastAPI app.
    
    Usage:
        def test_with_mock_service(test_app, mock_collection_service, override_service_dependency):
            override_service_dependency(test_app, mock_collection_service)
            # Now the app uses the mock service
    """
    def _override(app: FastAPI, service):
        from packages.webui.services.factory import get_collection_service
        app.dependency_overrides[get_collection_service] = lambda: service
        return app
    
    return _override


# Markers for different test types
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "auth: mark test as requiring authentication"
    )


# Test data fixtures
@pytest.fixture
def sample_collection_data():
    """Provide sample collection data for tests."""
    return {
        "name": "Test Collection",
        "description": "A test collection for unit tests",
        "embedding_model": "text-embedding-ada-002",
        "quantization": "scalar",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "is_public": False,
        "metadata": {"test": True},
    }


@pytest.fixture
def sample_chunking_configs():
    """Provide sample chunking configurations for tests."""
    return {
        "fixed_size": {
            "chunking_strategy": "fixed_size",
            "chunking_config": {
                "chunk_size": 1000,
                "chunk_overlap": 100,
            },
        },
        "semantic": {
            "chunking_strategy": "semantic",
            "chunking_config": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "similarity_threshold": 0.7,
                "min_chunk_size": 100,
                "max_chunk_size": 2000,
            },
        },
        "recursive": {
            "chunking_strategy": "recursive",
            "chunking_config": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", " ", ""],
            },
        },
    }


# Export auth mocking utilities for direct use
from packages.webui.tests.auth_mock import mock_auth, mock_auth_admin, mock_auth_second_user

__all__ = [
    "mock_auth",
    "mock_auth_admin", 
    "mock_auth_second_user",
    "test_app",
    "authenticated_client",
    "mock_db",
    "mock_collection_service",
]