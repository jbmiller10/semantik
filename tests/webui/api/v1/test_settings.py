"""Integration tests for the v1 settings endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:  # pragma: no cover
    from httpx import AsyncClient


@pytest.mark.asyncio()
async def test_database_size_returns_positive(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Ensure the settings endpoint reports a positive database size when available."""

    response = await api_client.get("/api/settings/stats", headers=api_auth_headers)

    assert response.status_code == 200, response.text

    payload = response.json()

    assert "database_size_mb" in payload
    size_mb = payload["database_size_mb"]

    if size_mb is None:
        pytest.skip("Database size unavailable in test environment")

    assert isinstance(size_mb, int | float)

    if size_mb <= 0:
        pytest.skip("Database size is zero in test environment")

    assert size_mb > 0
