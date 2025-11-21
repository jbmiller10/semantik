"""API-level authorization checks for partition monitoring endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest
from shared.config import settings
from webui.auth import get_current_user
from webui.main import app

if TYPE_CHECKING:  # pragma: no cover
    from httpx import AsyncClient

PARTITION_HEALTH_PATH = "/api/v2/partitions/health"


def _stub_monitoring_result() -> SimpleNamespace:
    """Create a minimal monitoring payload used by the mocked service."""

    return SimpleNamespace(
        status="HEALTHY",
        timestamp="2025-10-19T00:00:00Z",
        alerts=[],
        metrics={},
        error=None,
    )


@pytest.mark.asyncio()
async def test_partition_health_superuser_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    db_session,
) -> None:
    """Superusers should receive a 200 response with monitoring data."""

    original_flag = test_user_db.is_superuser

    test_user_db.is_superuser = True
    db_session.add(test_user_db)
    await db_session.commit()
    await db_session.refresh(test_user_db)

    async def override_superuser() -> dict[str, Any]:
        return {
            "id": test_user_db.id,
            "username": test_user_db.username,
            "email": test_user_db.email,
            "full_name": test_user_db.full_name,
            "is_superuser": True,
        }

    original_override = app.dependency_overrides.get(get_current_user)
    app.dependency_overrides[get_current_user] = override_superuser

    try:
        with patch("webui.api.v2.partition_monitoring.PartitionMonitoringService") as mock_service_cls:
            mock_instance = mock_service_cls.return_value
            mock_instance.check_partition_health = AsyncMock(return_value=_stub_monitoring_result())

            response = await api_client.get(PARTITION_HEALTH_PATH, headers=api_auth_headers)
    finally:
        if original_override is not None:
            app.dependency_overrides[get_current_user] = original_override
        else:
            app.dependency_overrides.pop(get_current_user, None)

        if test_user_db.is_superuser != original_flag:
            test_user_db.is_superuser = original_flag
            db_session.add(test_user_db)
            await db_session.commit()
            await db_session.refresh(test_user_db)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "HEALTHY"
    assert body["alerts"] == []


@pytest.mark.asyncio()
@pytest.mark.usefixtures("monitoring_service_stub")
async def test_partition_health_regular_user_forbidden(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Non-superusers must be rejected with HTTP 403."""

    response = await api_client.get(PARTITION_HEALTH_PATH, headers=api_auth_headers)

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin access required"


@pytest.mark.asyncio()
@pytest.mark.usefixtures("monitoring_service_stub")
async def test_partition_health_internal_api_key_allows_access(
    api_client: AsyncClient,
) -> None:
    """Valid internal API key header should grant access."""

    original_key = settings.INTERNAL_API_KEY
    settings.INTERNAL_API_KEY = "integration-test-key"

    try:
        headers = {"X-Internal-Api-Key": "integration-test-key"}
        response = await api_client.get(PARTITION_HEALTH_PATH, headers=headers)
    finally:
        settings.INTERNAL_API_KEY = original_key

    assert response.status_code == 200


@pytest.fixture()
def monitoring_service_stub():
    """Ensure service calls are stubbed to avoid DB dependencies."""

    with patch("webui.api.v2.partition_monitoring.PartitionMonitoringService") as mock_service_cls:
        mock_instance = mock_service_cls.return_value
        mock_instance.check_partition_health = AsyncMock(return_value=_stub_monitoring_result())
        yield mock_instance
