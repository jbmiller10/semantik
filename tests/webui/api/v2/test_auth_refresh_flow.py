"""Integration coverage for refresh token rotation."""

from __future__ import annotations

from uuid import uuid4

import pytest


@pytest.mark.asyncio()
async def test_refresh_flow_rotates_tokens(api_client_unauthenticated) -> None:
    """Login + refresh should succeed and revoke the old refresh token."""
    suffix = uuid4().hex[:8]
    registration_data = {
        "username": f"refresh_user_{suffix}",
        "email": f"refresh_{suffix}@example.com",
        "password": "refresh-password-123",
        "full_name": "Refresh Flow User",
    }

    register_response = await api_client_unauthenticated.post("/api/auth/register", json=registration_data)
    assert register_response.status_code == 200

    login_response = await api_client_unauthenticated.post(
        "/api/auth/login",
        json={"username": registration_data["username"], "password": registration_data["password"]},
    )
    assert login_response.status_code == 200
    login_payload = login_response.json()
    refresh_token = login_payload["refresh_token"]

    refresh_response = await api_client_unauthenticated.post(
        "/api/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_response.status_code == 200
    refreshed_payload = refresh_response.json()
    assert refreshed_payload["access_token"]
    assert refreshed_payload["refresh_token"]

    # Old refresh token should now be revoked.
    second_refresh = await api_client_unauthenticated.post(
        "/api/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert second_refresh.status_code == 401
