"""Integration tests for the v2 API keys endpoints."""

import os
from uuid import uuid4

import pytest
from httpx import AsyncClient

# =============================================================================
# Create API Key Tests
# =============================================================================


@pytest.mark.asyncio()
async def test_create_api_key_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Creating an API key should return the key details including the raw key."""
    response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "Test Key"},
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["name"] == "Test Key"
    assert payload["is_active"] is True
    assert "api_key" in payload
    assert payload["api_key"].startswith("smtk_")
    assert "id" in payload
    assert "created_at" in payload
    assert "expires_at" in payload
    # key_hash should never be in response
    assert "key_hash" not in payload


@pytest.mark.asyncio()
async def test_create_api_key_with_default_expiry(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Creating an API key without expires_in_days uses the default expiry."""
    response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "Default Expiry Key"},
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["expires_at"] is not None


@pytest.mark.asyncio()
async def test_create_api_key_with_custom_expiry(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Creating an API key with custom expires_in_days sets the expiration."""
    response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "Custom Expiry Key", "expires_in_days": 30},
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["expires_at"] is not None


@pytest.mark.asyncio()
async def test_create_api_key_duplicate_name_fails(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Creating an API key with duplicate name should return 409."""
    # Create first key
    response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "Duplicate Key"},
    )
    assert response.status_code == 201, response.text

    # Try to create duplicate
    response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "Duplicate Key"},
    )
    assert response.status_code == 409, response.text


@pytest.mark.asyncio()
async def test_create_api_key_duplicate_name_case_insensitive(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Duplicate name check should be case-insensitive."""
    # Create first key
    response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "My Key"},
    )
    assert response.status_code == 201, response.text

    # Try to create with different case
    response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "MY KEY"},
    )
    assert response.status_code == 409, response.text


@pytest.mark.asyncio()
async def test_create_api_key_invalid_name_empty(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Creating an API key with empty name should return 422."""
    response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": ""},
    )
    assert response.status_code == 422, response.text


@pytest.mark.asyncio()
async def test_create_api_key_invalid_name_too_long(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Creating an API key with name > 100 chars should return 422."""
    response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "x" * 101},
    )
    assert response.status_code == 422, response.text


@pytest.mark.asyncio()
async def test_create_api_key_invalid_expiry_too_low(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Creating an API key with expires_in_days < 1 should return 422."""
    response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "Invalid Expiry", "expires_in_days": 0},
    )
    assert response.status_code == 422, response.text


@pytest.mark.asyncio()
async def test_create_api_key_invalid_expiry_too_high(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Creating an API key with expires_in_days > 3650 should return 422."""
    response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "Invalid Expiry", "expires_in_days": 3651},
    )
    assert response.status_code == 422, response.text


@pytest.mark.asyncio()
async def test_create_api_key_limit_exceeded(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    """Creating API keys beyond the limit should return 400."""
    # Set a very low limit for testing
    monkeypatch.setattr("shared.config.settings.API_KEY_MAX_PER_USER", 2)

    # Create keys up to the limit
    for i in range(2):
        response = await api_client.post(
            "/api/v2/api-keys",
            headers=api_auth_headers,
            json={"name": f"Key {i}"},
        )
        assert response.status_code == 201, response.text

    # Try to exceed the limit
    response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "Exceeding Key"},
    )
    assert response.status_code == 400, response.text
    assert "Maximum" in response.text or "maximum" in response.text


# =============================================================================
# List API Keys Tests
# =============================================================================


@pytest.mark.asyncio()
async def test_list_api_keys_returns_owned_keys(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Listing API keys should return all keys owned by the user."""
    # Create two keys
    for name in ["Key One", "Key Two"]:
        response = await api_client.post(
            "/api/v2/api-keys",
            headers=api_auth_headers,
            json={"name": name},
        )
        assert response.status_code == 201, response.text

    response = await api_client.get("/api/v2/api-keys", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] >= 2
    names = {k["name"] for k in payload["api_keys"]}
    assert "Key One" in names
    assert "Key Two" in names


@pytest.mark.asyncio()
async def test_list_api_keys_excludes_raw_key_and_hash(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """List response should not include api_key or key_hash."""
    # Create a key
    await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "Secret Key"},
    )

    response = await api_client.get("/api/v2/api-keys", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    payload = response.json()
    for key in payload["api_keys"]:
        assert "api_key" not in key
        assert "key_hash" not in key


# =============================================================================
# Get API Key Tests
# =============================================================================


@pytest.mark.asyncio()
async def test_get_api_key_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Getting an API key should return its details."""
    # Create a key
    create_response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "Get Me"},
    )
    key_id = create_response.json()["id"]

    response = await api_client.get(f"/api/v2/api-keys/{key_id}", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["id"] == key_id
    assert payload["name"] == "Get Me"


@pytest.mark.asyncio()
async def test_get_api_key_excludes_raw_key(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Get response should not include api_key."""
    create_response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "No Raw Key"},
    )
    key_id = create_response.json()["id"]

    response = await api_client.get(f"/api/v2/api-keys/{key_id}", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "api_key" not in payload
    assert "key_hash" not in payload


@pytest.mark.asyncio()
async def test_get_api_key_not_found(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Getting a nonexistent API key should return 404."""
    fake_id = str(uuid4())
    response = await api_client.get(f"/api/v2/api-keys/{fake_id}", headers=api_auth_headers)
    assert response.status_code == 404, response.text


# =============================================================================
# Update API Key Tests
# =============================================================================


@pytest.mark.asyncio()
async def test_update_api_key_revoke_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Revoking an API key should set is_active to false."""
    # Create a key
    create_response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "To Revoke"},
    )
    key_id = create_response.json()["id"]

    # Revoke the key
    response = await api_client.patch(
        f"/api/v2/api-keys/{key_id}",
        headers=api_auth_headers,
        json={"is_active": False},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["is_active"] is False


@pytest.mark.asyncio()
async def test_update_api_key_reactivate_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Reactivating an API key should set is_active to true."""
    # Create and revoke a key
    create_response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "To Reactivate"},
    )
    key_id = create_response.json()["id"]

    await api_client.patch(
        f"/api/v2/api-keys/{key_id}",
        headers=api_auth_headers,
        json={"is_active": False},
    )

    # Reactivate the key
    response = await api_client.patch(
        f"/api/v2/api-keys/{key_id}",
        headers=api_auth_headers,
        json={"is_active": True},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["is_active"] is True


@pytest.mark.asyncio()
async def test_update_api_key_not_found(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Updating a nonexistent API key should return 404."""
    fake_id = str(uuid4())
    response = await api_client.patch(
        f"/api/v2/api-keys/{fake_id}",
        headers=api_auth_headers,
        json={"is_active": False},
    )
    assert response.status_code == 404, response.text


# =============================================================================
# Authentication Tests
# =============================================================================


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes"),
    reason="Auth is disabled in test environment",
)
async def test_all_endpoints_require_authentication(
    api_client_unauthenticated: AsyncClient,
) -> None:
    """All API key endpoints should require authentication."""
    endpoints = [
        ("POST", "/api/v2/api-keys"),
        ("GET", "/api/v2/api-keys"),
        ("GET", "/api/v2/api-keys/some-id"),
        ("PATCH", "/api/v2/api-keys/some-id"),
    ]

    for method, path in endpoints:
        if method == "GET":
            response = await api_client_unauthenticated.get(path)
        elif method == "POST":
            response = await api_client_unauthenticated.post(path, json={"name": "test"})
        elif method == "PATCH":
            response = await api_client_unauthenticated.patch(path, json={"is_active": False})
        else:
            continue

        assert response.status_code == 401, f"{method} {path} should require auth: {response.text}"


# =============================================================================
# Cross-User Isolation Tests (403 Forbidden)
# =============================================================================


@pytest.mark.asyncio()
@pytest.mark.usefixtures("test_user_db")
async def test_get_api_key_owned_by_other_user_returns_403(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    db_session,
) -> None:
    """User A cannot view User B's API key - returns 403 Forbidden."""
    from shared.database.models import ApiKey

    # Create an API key owned by other_user directly in the database
    key_id = str(uuid4())
    api_key = ApiKey(
        id=key_id,
        user_id=other_user_db.id,
        name="other-user-key",
        key_hash="fake_hash_for_testing",
        is_active=True,
    )
    db_session.add(api_key)
    await db_session.flush()

    # test_user tries to access other_user's key - should get 403
    response = await api_client.get(
        f"/api/v2/api-keys/{key_id}",
        headers=api_auth_headers,
    )
    assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"


@pytest.mark.asyncio()
@pytest.mark.usefixtures("test_user_db")
async def test_update_api_key_owned_by_other_user_returns_403(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    db_session,
) -> None:
    """User A cannot update User B's API key - returns 403 Forbidden."""
    from shared.database.models import ApiKey

    # Create an API key owned by other_user directly in the database
    key_id = str(uuid4())
    api_key = ApiKey(
        id=key_id,
        user_id=other_user_db.id,
        name="other-user-key",
        key_hash="fake_hash_for_testing",
        is_active=True,
    )
    db_session.add(api_key)
    await db_session.flush()

    # test_user tries to update other_user's key - should get 403
    response = await api_client.patch(
        f"/api/v2/api-keys/{key_id}",
        headers=api_auth_headers,
        json={"is_active": False},
    )
    assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"


@pytest.mark.asyncio()
@pytest.mark.usefixtures("test_user_db")
async def test_list_api_keys_only_returns_owned_keys(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    db_session,
) -> None:
    """List API keys should not return other users' keys."""
    from shared.database.models import ApiKey

    # test_user creates a key via API
    await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "my-key"},
    )

    # Create an API key owned by other_user directly in the database
    other_key_id = str(uuid4())
    other_api_key = ApiKey(
        id=other_key_id,
        user_id=other_user_db.id,
        name="other-key",
        key_hash="fake_hash_for_testing_other",
        is_active=True,
    )
    db_session.add(other_api_key)
    await db_session.flush()

    # test_user lists keys - should only see their own
    response = await api_client.get(
        "/api/v2/api-keys",
        headers=api_auth_headers,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    key_names = {k["name"] for k in payload["api_keys"]}
    assert "my-key" in key_names
    assert "other-key" not in key_names, "Should not see other user's keys"


# =============================================================================
# Key Format Tests
# =============================================================================


@pytest.mark.asyncio()
async def test_api_key_format(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """API key should follow the expected format: smtk_<prefix>_<secret>."""
    response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "Format Test Key"},
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    api_key = payload["api_key"]

    # Check format: smtk_<8chars>_<secret>
    # The secret can contain underscores (url-safe base64), so split only on first 2 underscores
    assert api_key.startswith("smtk_"), f"Key should start with 'smtk_', got '{api_key[:10]}...'"

    # Split only first two underscores to get prefix, uuid portion, and rest
    parts = api_key.split("_", 2)
    assert len(parts) == 3, f"Expected at least 3 parts in key, got {len(parts)}"
    assert parts[0] == "smtk", f"Key should start with 'smtk', got '{parts[0]}'"
    assert len(parts[1]) == 8, f"UUID prefix should be 8 chars, got {len(parts[1])}"
    # The secret part should be ~43 characters (base64 encoded 32 bytes)
    # but may contain underscores, so we check total length of remaining part
    assert len(parts[2]) >= 40, f"Secret should be at least 40 chars, got {len(parts[2])}"
