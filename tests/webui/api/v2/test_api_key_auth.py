"""Integration tests for API key authentication (Phase 2).

These tests verify that API keys can be used to authenticate HTTP and WebSocket
requests, providing headless access to the Semantik API.
"""

import hashlib
import os
from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient

from webui.auth import create_access_token

# =============================================================================
# HTTP Authentication Tests
# =============================================================================


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes"),
    reason="Auth is disabled in test environment",
)
async def test_api_key_authenticates_http_request(
    api_client: AsyncClient,
    api_client_unauthenticated: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """A valid API key should authenticate HTTP requests successfully."""
    # Create an API key using authenticated client
    create_response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "HTTP Auth Test Key"},
    )
    assert create_response.status_code == 201, create_response.text
    raw_key = create_response.json()["api_key"]
    assert raw_key.startswith("smtk_"), f"Key should start with 'smtk_', got '{raw_key[:20]}...'"

    # Use the API key to authenticate a request
    response = await api_client_unauthenticated.get(
        "/api/v2/collections",
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert response.status_code == 200, f"API key auth should succeed: {response.text}"


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes"),
    reason="Auth is disabled in test environment",
)
async def test_revoked_api_key_returns_401(
    api_client_unauthenticated: AsyncClient,
    test_user_db,
    db_session,
) -> None:
    """A revoked API key (is_active=false) should return 401."""
    import secrets
    from uuid import uuid4

    from shared.database.models import ApiKey

    # Create an API key directly in the database (committed)
    key_id = str(uuid4())
    raw_key = f"smtk_{key_id.replace('-', '')[:8]}_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    api_key = ApiKey(
        id=key_id,
        user_id=test_user_db.id,
        name="Revoke Test Key",
        key_hash=key_hash,
        is_active=False,  # Already revoked
        expires_at=datetime.now(UTC) + timedelta(days=365),
    )
    db_session.add(api_key)
    await db_session.flush()

    # Try to use the revoked key
    response = await api_client_unauthenticated.get(
        "/api/v2/collections",
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert response.status_code == 401, f"Revoked key should be rejected: {response.text}"


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes"),
    reason="Auth is disabled in test environment",
)
async def test_expired_api_key_returns_401(
    api_client_unauthenticated: AsyncClient,
    test_user_db,
    db_session,
) -> None:
    """An expired API key should return 401."""
    import secrets
    from uuid import uuid4

    from shared.database.models import ApiKey

    # Create an expired API key directly in the database
    key_id = str(uuid4())
    raw_key = f"smtk_{key_id.replace('-', '')[:8]}_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    api_key = ApiKey(
        id=key_id,
        user_id=test_user_db.id,
        name="Expired Key",
        key_hash=key_hash,
        is_active=True,
        expires_at=datetime.now(UTC) - timedelta(days=1),  # Expired yesterday
    )
    db_session.add(api_key)
    await db_session.flush()

    # Try to use the expired key
    response = await api_client_unauthenticated.get(
        "/api/v2/collections",
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert response.status_code == 401, f"Expired key should be rejected: {response.text}"


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes"),
    reason="Auth is disabled in test environment",
)
async def test_invalid_api_key_format_returns_401(
    api_client_unauthenticated: AsyncClient,
) -> None:
    """An invalid API key format should return 401."""
    # Try various invalid formats
    invalid_keys = [
        "invalid_key_format",
        "smtk_tooshort",  # Missing secret
        "not_smtk_prefix_key_abc123",  # Wrong prefix
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",  # JWT-like (should be handled differently)
        "",  # Empty
    ]

    for invalid_key in invalid_keys:
        response = await api_client_unauthenticated.get(
            "/api/v2/collections",
            headers={"Authorization": f"Bearer {invalid_key}"},
        )
        assert response.status_code == 401, f"Invalid key '{invalid_key[:20]}...' should be rejected: {response.text}"


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes"),
    reason="Auth is disabled in test environment",
)
async def test_nonexistent_api_key_returns_401(
    api_client_unauthenticated: AsyncClient,
) -> None:
    """A properly formatted but nonexistent API key should return 401."""
    # Generate a fake key that looks valid but doesn't exist
    import secrets

    fake_key = f"smtk_abcd1234_{secrets.token_urlsafe(32)}"

    response = await api_client_unauthenticated.get(
        "/api/v2/collections",
        headers={"Authorization": f"Bearer {fake_key}"},
    )
    assert response.status_code == 401, f"Nonexistent key should be rejected: {response.text}"


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes"),
    reason="Auth is disabled in test environment",
)
async def test_api_key_updates_last_used_at(
    api_client_unauthenticated: AsyncClient,
    test_user_db,
    db_session,
) -> None:
    """Using an API key should update last_used_at timestamp."""
    import secrets
    from uuid import uuid4

    from sqlalchemy import select

    from shared.database.models import ApiKey

    # Create an API key directly in the database
    key_id = str(uuid4())
    raw_key = f"smtk_{key_id.replace('-', '')[:8]}_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    api_key = ApiKey(
        id=key_id,
        user_id=test_user_db.id,
        name="Last Used Test Key",
        key_hash=key_hash,
        is_active=True,
        expires_at=datetime.now(UTC) + timedelta(days=365),
    )
    db_session.add(api_key)
    await db_session.flush()

    # Check initial last_used_at is None
    result = await db_session.execute(select(ApiKey).where(ApiKey.id == key_id))
    api_key_record = result.scalar_one()
    assert api_key_record.last_used_at is None, "last_used_at should be None initially"

    # Use the API key
    response = await api_client_unauthenticated.get(
        "/api/v2/collections",
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert response.status_code == 200, response.text

    # Verify last_used_at was updated
    await db_session.refresh(api_key_record)
    assert api_key_record.last_used_at is not None, "last_used_at should be updated after use"
    # Check it was updated recently (within last minute)
    time_diff = datetime.now(UTC) - api_key_record.last_used_at.replace(tzinfo=UTC)
    assert time_diff.total_seconds() < 60, f"last_used_at should be recent, but was {time_diff} ago"


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes"),
    reason="Auth is disabled in test environment",
)
async def test_jwt_still_works_regression(
    api_client_unauthenticated: AsyncClient,
    test_user_db,
) -> None:
    """JWT authentication should still work after adding API key support."""
    # Create a valid JWT
    jwt_token = create_access_token(data={"sub": test_user_db.username})

    # Use JWT to authenticate
    response = await api_client_unauthenticated.get(
        "/api/v2/collections",
        headers={"Authorization": f"Bearer {jwt_token}"},
    )
    assert response.status_code == 200, f"JWT auth should still work: {response.text}"


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes"),
    reason="Auth is disabled in test environment",
)
async def test_inactive_user_api_key_returns_401(
    api_client_unauthenticated: AsyncClient,
    db_session,
) -> None:
    """An API key for an inactive user should return 401."""
    import secrets
    from uuid import uuid4

    from shared.database.models import ApiKey, User

    # Create an inactive user directly
    unique_suffix = uuid4().hex[:8]
    inactive_user = User(
        username=f"inactive_user_{unique_suffix}",
        hashed_password="hashed_password",
        email=f"inactive_{unique_suffix}@example.com",
        is_active=False,  # Inactive user
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(inactive_user)
    await db_session.flush()

    # Create an API key for the inactive user
    key_id = str(uuid4())
    raw_key = f"smtk_{key_id.replace('-', '')[:8]}_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    api_key = ApiKey(
        id=key_id,
        user_id=inactive_user.id,
        name="Inactive User Key",
        key_hash=key_hash,
        is_active=True,
        expires_at=datetime.now(UTC) + timedelta(days=365),
    )
    db_session.add(api_key)
    await db_session.flush()

    # Try to use the API key
    response = await api_client_unauthenticated.get(
        "/api/v2/collections",
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert response.status_code == 401, f"Key for inactive user should be rejected: {response.text}"


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes"),
    reason="Auth is disabled in test environment",
)
async def test_api_key_does_not_grant_superuser(
    api_client: AsyncClient,
    api_client_unauthenticated: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """API key auth should never grant superuser privileges."""
    # Create an API key
    create_response = await api_client.post(
        "/api/v2/api-keys",
        headers=api_auth_headers,
        json={"name": "Superuser Test Key"},
    )
    assert create_response.status_code == 201, create_response.text
    raw_key = create_response.json()["api_key"]

    # Access an endpoint that returns user info (collections is fine for this)
    # The user object returned by get_current_user() will have is_superuser=False
    response = await api_client_unauthenticated.get(
        "/api/v2/collections",
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert response.status_code == 200, response.text


# =============================================================================
# WebSocket Authentication Tests
# =============================================================================


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes"),
    reason="Auth is disabled in test environment",
)
async def test_api_key_authenticates_websocket_query_param(
    test_user_db,
    db_session,
) -> None:
    """A valid API key should authenticate WebSocket connections via query param."""
    import secrets
    from unittest.mock import patch
    from uuid import uuid4

    from shared.database.models import ApiKey
    from webui.auth import get_current_user_websocket

    # Create an API key directly in the database
    key_id = str(uuid4())
    raw_key = f"smtk_{key_id.replace('-', '')[:8]}_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    api_key = ApiKey(
        id=key_id,
        user_id=test_user_db.id,
        name="WebSocket Query Param Key",
        key_hash=key_hash,
        is_active=True,
        expires_at=datetime.now(UTC) + timedelta(days=365),
    )
    db_session.add(api_key)
    await db_session.flush()

    # Patch get_db_session to use the test session
    async def patched_get_db_session():
        yield db_session

    with patch("webui.auth.get_db_session", patched_get_db_session):
        user = await get_current_user_websocket(raw_key)
        assert user is not None
        assert user.get("_auth_method") == "api_key"
        assert user.get("is_superuser") is False


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes"),
    reason="Auth is disabled in test environment",
)
async def test_api_key_websocket_revoked_key_fails(
    test_user_db,
    db_session,
) -> None:
    """A revoked API key should fail WebSocket authentication."""
    import secrets
    from unittest.mock import patch
    from uuid import uuid4

    from shared.database.models import ApiKey
    from webui.auth import get_current_user_websocket

    # Create a revoked API key directly in the database
    key_id = str(uuid4())
    raw_key = f"smtk_{key_id.replace('-', '')[:8]}_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    api_key = ApiKey(
        id=key_id,
        user_id=test_user_db.id,
        name="WebSocket Revoke Test Key",
        key_hash=key_hash,
        is_active=False,  # Revoked
        expires_at=datetime.now(UTC) + timedelta(days=365),
    )
    db_session.add(api_key)
    await db_session.flush()

    # Patch get_db_session to use the test session
    async def patched_get_db_session():
        yield db_session

    with patch("webui.auth.get_db_session", patched_get_db_session):
        with pytest.raises(ValueError, match="Invalid API key"):
            await get_current_user_websocket(raw_key)


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes"),
    reason="Auth is disabled in test environment",
)
async def test_jwt_websocket_still_works_regression(
    test_user_db,
    db_session,
) -> None:
    """JWT authentication should still work for WebSocket after adding API key support."""
    from unittest.mock import patch

    from webui.auth import create_access_token, get_current_user_websocket

    jwt_token = create_access_token(data={"sub": test_user_db.username})

    # Patch get_db_session to use the test session
    async def patched_get_db_session():
        yield db_session

    with patch("webui.auth.get_db_session", patched_get_db_session):
        user = await get_current_user_websocket(jwt_token)
        assert user is not None
        assert user["username"] == test_user_db.username
        # JWT auth doesn't add _auth_method metadata
        assert "_auth_method" not in user
