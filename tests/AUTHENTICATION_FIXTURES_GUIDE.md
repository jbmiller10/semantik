# Test Authentication Fixtures Guide

## Problem
Tests were failing with `AccessDeniedError` because:
1. Collections were created with `owner_id=1` (from test_user)
2. When `DISABLE_AUTH=true`, the app returns a dev user with `id=0`
3. This mismatch caused authentication errors

## Solution
New authentication fixtures in `/tests/integration/conftest.py` ensure consistency:

### Available Fixtures

#### For Async Tests
- **`authenticated_user`**: Returns a consistent user dict with id=1
- **`authenticated_async_client`**: AsyncClient with proper auth overrides
- **`authenticated_collection`**: Collection owned by authenticated_user
- **`auth_headers`**: Bearer token headers for the authenticated user
- **`auth_token`**: Raw JWT token for custom use

#### For Sync Tests  
- **`authenticated_client`**: TestClient with proper auth overrides
- **`test_user`**: Creates a user in the database (id=1)
- **`admin_user`**: Creates an admin user (id=2)

## Migration Steps

### 1. Remove Duplicate Fixtures
Remove these from your test files (they're in conftest now):
```python
# REMOVE THESE:
@pytest.fixture
def mock_user():
    return {"id": 1, "username": "testuser", ...}

@pytest.fixture  
def mock_collection():
    return {"owner_id": 1, ...}
```

### 2. Use Authenticated Clients
Replace manual client creation:
```python
# OLD:
async def test_something():
    app.dependency_overrides[get_current_user] = lambda: mock_user
    async with AsyncClient(...) as client:
        ...

# NEW:
async def test_something(authenticated_async_client):
    response = await authenticated_async_client.get(...)
```

### 3. Use Authenticated Collections
Replace manual collection creation:
```python
# OLD:
collection = Collection(owner_id=1, ...)
db.add(collection)

# NEW:
async def test_something(authenticated_collection):
    # Collection already created with correct owner_id
    response = await client.get(f"/api/v2/collections/{authenticated_collection['id']}")
```

### 4. Complete Example
```python
@pytest.mark.asyncio
async def test_chunking_endpoint(
    authenticated_async_client: AsyncClient,
    authenticated_collection: dict,
    auth_headers: dict,
):
    """Test with proper authentication."""
    response = await authenticated_async_client.post(
        f"/api/v2/collections/{authenticated_collection['id']}/chunking/apply",
        json={"strategy": "fixed_size"},
        headers=auth_headers,
    )
    
    assert response.status_code == 200  # No AccessDeniedError!
```

## Common Patterns

### Testing Unauthorized Access
```python
async def test_unauthorized(async_client, authenticated_collection):
    # Use regular async_client without auth overrides
    response = await async_client.get(
        f"/api/v2/collections/{authenticated_collection['id']}"
    )
    assert response.status_code == 401
```

### Testing Different Users
```python
async def test_access_denied(
    authenticated_async_client,
    async_session,
):
    # Create collection owned by different user
    other_collection = Collection(owner_id=999, ...)
    async_session.add(other_collection)
    await async_session.commit()
    
    # Try to access with authenticated_user (id=1)
    response = await authenticated_async_client.get(
        f"/api/v2/collections/{other_collection.id}"
    )
    assert response.status_code == 403  # Access denied
```

### Custom User Properties
```python
async def test_with_custom_user(
    async_client,
    async_session,
):
    from webui.auth import get_current_user
    
    custom_user = {
        "id": 42,
        "username": "custom",
        "is_superuser": True,
        ...
    }
    
    async def override():
        return custom_user
        
    app.dependency_overrides[get_current_user] = override
    # ... test logic ...
    app.dependency_overrides.clear()
```

## Troubleshooting

### Still Getting AccessDeniedError?
1. Check you're using `authenticated_async_client` not `async_client`
2. Check you're using `authenticated_collection` not a manual one
3. Verify the collection's owner_id matches authenticated_user's id

### Tests Pass Locally but Fail in CI?
1. Ensure `DISABLE_AUTH` environment variable is consistent
2. Check database is properly cleaned between tests
3. Verify fixtures are properly scoped

### Need Different User IDs?
Create a parameterized fixture:
```python
@pytest.fixture
def user_with_id(request):
    return {
        "id": request.param,
        "username": f"user_{request.param}",
        ...
    }

@pytest.mark.parametrize("user_with_id", [1, 2, 3], indirect=True)
async def test_multiple_users(user_with_id):
    ...
```

## Benefits
- **Consistency**: User ID always matches collection owner_id
- **Simplicity**: No manual dependency overrides in each test
- **Reliability**: Fewer authentication-related test failures
- **Maintainability**: Central place to update auth logic