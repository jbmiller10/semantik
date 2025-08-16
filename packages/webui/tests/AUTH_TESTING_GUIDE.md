# Authentication Testing Guide

## Overview

The Semantik webui package includes a comprehensive authentication mocking infrastructure that ensures consistent, reliable testing across all environments. This guide explains how to use the auth mocking system effectively.

## Key Features

- **Environment Independent**: Works regardless of `DISABLE_AUTH` settings
- **Owner Consistency**: Automatically ensures resources are owned by the authenticated user
- **Multi-User Support**: Easy testing of access control and permissions
- **Zero Configuration**: Works out of the box with sensible defaults
- **CI/CD Ready**: Designed for automated testing environments

## Quick Start

### Basic Usage

The simplest way to test authenticated endpoints:

```python
@pytest.mark.asyncio()
async def test_my_endpoint(authenticated_client):
    """Test with default authenticated user."""
    response = await authenticated_client.get("/api/v2/collections")
    assert response.status_code == 200
```

The `authenticated_client` fixture automatically:
- Creates a FastAPI test app
- Mocks authentication with a default test user (ID: 1, username: "testuser")
- Provides an HTTP client ready for API calls

### Testing with Different Users

For multi-user scenarios:

```python
@pytest.mark.asyncio()
async def test_access_control(authenticated_client_factory):
    """Test that users can only modify their own resources."""
    # Create two clients with different users
    owner_client = await authenticated_client_factory(user_id=1, username="owner")
    other_client = await authenticated_client_factory(user_id=2, username="other")
    
    # Test access control logic here
```

## Core Components

### 1. TestUser Class

Represents a test user with consistent attributes:

```python
from packages.webui.tests.auth_mock import create_test_user

user = create_test_user(
    user_id=10,
    username="custom_user",
    email="custom@example.com",
    is_superuser=True
)
```

### 2. AuthMocker Class

Manages authentication mocking at different levels:

```python
from packages.webui.tests.auth_mock import AuthMocker, DEFAULT_TEST_USER

mocker = AuthMocker(user=DEFAULT_TEST_USER)
mocker.override_fastapi_auth(app)  # Override FastAPI dependencies
```

### 3. MockDatabase Class

Provides consistent mock data that respects ownership:

```python
def test_with_mock_db(mock_db):
    collection = mock_db.create_collection(name="Test")
    # collection["owner_id"] automatically matches the test user
```

## Available Fixtures

### Basic Fixtures

- `mock_auth`: Provides an AuthMocker instance
- `test_app`: FastAPI app with mocked auth
- `authenticated_client`: HTTP client with mocked auth
- `mock_db`: Mock database with ownership consistency

### Advanced Fixtures

- `mock_auth_admin`: AuthMocker with admin privileges
- `authenticated_client_factory`: Factory for creating multiple clients
- `mock_collection_service`: Mock CollectionService with auth logic
- `override_service_dependency`: Helper to inject mock services

## Common Testing Patterns

### 1. Single User CRUD Operations

```python
@pytest.mark.asyncio()
async def test_crud_operations(authenticated_client, mock_collection_service, test_app, override_service_dependency):
    # Setup mock service
    override_service_dependency(test_app, mock_collection_service)
    
    # Create resource
    response = await authenticated_client.post(
        "/api/v2/collections",
        json={"name": "Test", "description": "Test"}
    )
    assert response.status_code == 201
    
    # Update resource
    collection_id = response.json()["id"]
    response = await authenticated_client.put(
        f"/api/v2/collections/{collection_id}",
        json={"description": "Updated"}
    )
    assert response.status_code == 200
    
    # Delete resource
    response = await authenticated_client.delete(f"/api/v2/collections/{collection_id}")
    assert response.status_code == 204
```

### 2. Access Control Testing

```python
@pytest.mark.asyncio()
async def test_access_control(authenticated_client_factory, mock_collection_service):
    # Create resources with different users
    owner_client = await authenticated_client_factory(user_id=1)
    other_client = await authenticated_client_factory(user_id=2)
    
    # Owner creates private resource
    collection, _ = await mock_collection_service.create_collection(
        user_id=1,
        name="Private",
        description="Private collection",
        config={"is_public": False}
    )
    
    # Other user cannot modify
    from packages.shared.database.exceptions import AccessDeniedError
    with pytest.raises(AccessDeniedError):
        await mock_collection_service.update(
            collection_id=collection["uuid"],
            user_id=2,
            updates={"name": "Hacked"}
        )
```

### 3. Testing with Mock Services

```python
@pytest.mark.asyncio()
async def test_with_mock_service(test_app, authenticated_client, override_service_dependency):
    from unittest.mock import AsyncMock, MagicMock
    
    # Create custom mock service
    mock_service = MagicMock()
    mock_service.create_collection = AsyncMock(
        return_value=(
            {"uuid": "123", "name": "Test", "owner_id": 1},
            {"uuid": "op-123", "status": "pending"}
        )
    )
    
    # Inject mock service
    override_service_dependency(test_app, mock_service)
    
    # Test endpoint
    response = await authenticated_client.post(
        "/api/v2/collections",
        json={"name": "Test", "description": "Test"}
    )
    assert response.status_code == 201
    mock_service.create_collection.assert_called_once()
```

## Best Practices

### 1. Always Use Fixtures

Don't manually create auth mocks. Use the provided fixtures:

```python
# Good
async def test_endpoint(authenticated_client):
    response = await authenticated_client.get("/api/v2/collections")
    
# Bad - manual mocking
async def test_endpoint():
    app = FastAPI()
    async def mock_user():
        return {"id": 1}
    app.dependency_overrides[get_current_user] = mock_user
```

### 2. Ensure Owner Consistency

When creating test data, use the helper functions:

```python
from packages.webui.tests.auth_mock import ensure_owner_consistency

def test_ownership(test_user):
    resource = {"name": "Test Resource"}
    resource = ensure_owner_consistency(resource, test_user)
    assert resource["owner_id"] == test_user.id
```

### 3. Test Both Success and Failure Cases

```python
async def test_authorization(authenticated_client_factory):
    owner = await authenticated_client_factory(user_id=1)
    other = await authenticated_client_factory(user_id=2)
    
    # Test success (owner can modify)
    # Test failure (other cannot modify)
```

### 4. Use Appropriate Fixtures for Scope

- `test_app`: When you need to configure the app
- `authenticated_client`: For simple API testing
- `authenticated_client_factory`: For multi-user scenarios
- `mock_db`: For testing with consistent mock data

## Troubleshooting

### Issue: Tests Pass Locally but Fail in CI

**Solution**: The auth mocking infrastructure is designed to work consistently across environments. Ensure you're using the provided fixtures rather than relying on environment variables.

### Issue: Owner ID Mismatch

**Solution**: Use `ensure_owner_consistency()` or `MockDatabase` to maintain ownership:

```python
collection = mock_db.create_collection(name="Test")
# owner_id is automatically set correctly
```

### Issue: Multiple Users in Same Test

**Solution**: Use the factory fixtures:

```python
client1 = await authenticated_client_factory(user_id=1)
client2 = await authenticated_client_factory(user_id=2)
```

## Advanced Usage

### Custom Test Users

Create specialized test users for specific scenarios:

```python
from packages.webui.tests.auth_mock import create_test_user

premium_user = create_test_user(
    user_id=100,
    username="premium",
    email="premium@example.com",
    metadata={"subscription": "premium"}
)
```

### Context Manager Pattern

For temporary auth mocking:

```python
from packages.webui.tests.auth_mock import mock_authenticated_user

async with mock_authenticated_user(app) as (app, user):
    # Auth is mocked within this context
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/v2/collections")
```

### Module-Level Patching

For testing code that imports auth functions directly:

```python
def test_with_patched_module(mock_auth):
    mock_auth.patch_module_auth("packages.webui.auth")
    # All auth imports are now mocked
```

## Migration from Old Tests

If you have existing tests that manually mock authentication:

### Old Pattern
```python
async def override_get_current_user():
    return {"id": 1, "username": "testuser"}
app.dependency_overrides[get_current_user] = override_get_current_user
```

### New Pattern
```python
# Just use the fixture - it handles everything
async def test_endpoint(authenticated_client):
    response = await authenticated_client.get("/api/v2/collections")
```

## Summary

The auth mocking infrastructure provides:
- Consistent testing across all environments
- Automatic owner ID management
- Easy multi-user testing
- Clean, maintainable test code

By using the provided fixtures and following these patterns, you can write reliable, maintainable tests that work correctly in all environments.