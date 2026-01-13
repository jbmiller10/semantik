"""Integration tests for system settings API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from shared.database import get_db
from shared.database.repositories.system_settings_repository import SystemSettingsRepository
from webui.auth import get_current_user
from webui.main import app
from webui.services.system_settings_service import reset_service_instance

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from httpx import AsyncClient as AsyncClientType
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.models import User


# API paths
SYSTEM_SETTINGS_PATH = "/api/v2/system-settings"
SYSTEM_SETTINGS_EFFECTIVE_PATH = "/api/v2/system-settings/effective"
SYSTEM_SETTINGS_DEFAULTS_PATH = "/api/v2/system-settings/defaults"


@pytest.fixture(autouse=True)
def _reset_settings_service():
    """Reset the singleton service instance before and after each test."""
    reset_service_instance()
    yield
    reset_service_instance()


@pytest_asyncio.fixture()
async def api_client_admin(
    db_session: AsyncSession,
    test_user_db: User,
    use_fakeredis,
    reset_redis_manager,
) -> AsyncGenerator[AsyncClientType, None]:
    """Provide an AsyncClient authenticated as an admin (superuser)."""
    _ = use_fakeredis
    _ = reset_redis_manager

    # Set user as superuser in DB
    test_user_db.is_superuser = True  # type: ignore[assignment]
    db_session.add(test_user_db)
    await db_session.commit()
    await db_session.refresh(test_user_db)

    async def override_get_db() -> AsyncGenerator[Any, None]:
        yield db_session

    async def override_get_current_user() -> dict[str, Any]:
        return {
            "id": test_user_db.id,
            "username": test_user_db.username,
            "email": test_user_db.email,
            "full_name": test_user_db.full_name,
            "is_superuser": True,
        }

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


class TestGetSystemSettings:
    """Tests for GET /api/v2/system-settings."""

    @pytest.mark.asyncio()
    async def test_admin_can_get_settings(
        self,
        api_client_admin: AsyncClientType,
    ) -> None:
        """Admin user should receive 200 with all settings."""
        response = await api_client_admin.get(SYSTEM_SETTINGS_PATH)

        assert response.status_code == 200
        body = response.json()
        assert "settings" in body

    @pytest.mark.asyncio()
    async def test_non_admin_forbidden(
        self,
        api_client: AsyncClientType,
        api_auth_headers: dict[str, str],
    ) -> None:
        """Non-admin users should receive 403."""
        response = await api_client.get(SYSTEM_SETTINGS_PATH, headers=api_auth_headers)

        assert response.status_code == 403
        assert "Admin access required" in response.json()["detail"]

    @pytest.mark.asyncio()
    async def test_settings_include_metadata(
        self,
        api_client_admin: AsyncClientType,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Settings response should include value, updated_at, updated_by."""
        # First set a value to ensure metadata is populated
        repo = SystemSettingsRepository(db_session)
        await repo.set_setting("max_collections_per_user", 25, user_id=cast(int, test_user_db.id))
        await db_session.commit()

        response = await api_client_admin.get(SYSTEM_SETTINGS_PATH)

        assert response.status_code == 200
        body = response.json()
        settings = body["settings"]

        # Check that max_collections_per_user has metadata
        if "max_collections_per_user" in settings:
            setting = settings["max_collections_per_user"]
            assert "value" in setting
            assert "updated_at" in setting
            assert "updated_by" in setting


class TestGetEffectiveSettings:
    """Tests for GET /api/v2/system-settings/effective."""

    @pytest.mark.asyncio()
    async def test_admin_can_get_effective(
        self,
        api_client_admin: AsyncClientType,
    ) -> None:
        """Admin user should receive 200 with resolved values."""
        response = await api_client_admin.get(SYSTEM_SETTINGS_EFFECTIVE_PATH)

        assert response.status_code == 200
        body = response.json()
        assert "settings" in body
        # Should have all default keys
        settings = body["settings"]
        assert "max_collections_per_user" in settings
        assert "cache_ttl_seconds" in settings

    @pytest.mark.asyncio()
    async def test_non_admin_forbidden(
        self,
        api_client: AsyncClientType,
        api_auth_headers: dict[str, str],
    ) -> None:
        """Non-admin users should receive 403."""
        response = await api_client.get(SYSTEM_SETTINGS_EFFECTIVE_PATH, headers=api_auth_headers)

        assert response.status_code == 403

    @pytest.mark.asyncio()
    async def test_db_value_takes_precedence(
        self,
        api_client_admin: AsyncClientType,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """DB value should override default."""
        # Set a custom value in DB
        repo = SystemSettingsRepository(db_session)
        await repo.set_setting("max_collections_per_user", 50, user_id=cast(int, test_user_db.id))
        await db_session.commit()

        response = await api_client_admin.get(SYSTEM_SETTINGS_EFFECTIVE_PATH)

        assert response.status_code == 200
        body = response.json()
        # The effective value should be 50 (from DB), not 10 (default)
        assert body["settings"]["max_collections_per_user"] == 50


class TestUpdateSystemSettings:
    """Tests for PATCH /api/v2/system-settings."""

    @pytest.mark.asyncio()
    async def test_admin_can_update_settings(
        self,
        api_client_admin: AsyncClientType,
        db_session: AsyncSession,
    ) -> None:
        """Admin should be able to update settings."""
        response = await api_client_admin.patch(
            SYSTEM_SETTINGS_PATH,
            json={"settings": {"max_collections_per_user": 30}},
        )

        assert response.status_code == 200
        body = response.json()
        assert "updated" in body
        assert "max_collections_per_user" in body["updated"]

        # Verify in DB
        repo = SystemSettingsRepository(db_session)
        value = await repo.get_setting("max_collections_per_user")
        assert value == 30

    @pytest.mark.asyncio()
    async def test_non_admin_forbidden(
        self,
        api_client: AsyncClientType,
        api_auth_headers: dict[str, str],
    ) -> None:
        """Non-admin users should receive 403."""
        response = await api_client.patch(
            SYSTEM_SETTINGS_PATH,
            json={"settings": {"max_collections_per_user": 30}},
            headers=api_auth_headers,
        )

        assert response.status_code == 403

    @pytest.mark.asyncio()
    async def test_rejects_empty_settings(
        self,
        api_client_admin: AsyncClientType,
    ) -> None:
        """Should reject empty settings dict."""
        response = await api_client_admin.patch(
            SYSTEM_SETTINGS_PATH,
            json={"settings": {}},
        )

        assert response.status_code == 400

    @pytest.mark.asyncio()
    async def test_rejects_unknown_keys(
        self,
        api_client_admin: AsyncClientType,
    ) -> None:
        """Should reject unknown setting keys."""
        response = await api_client_admin.patch(
            SYSTEM_SETTINGS_PATH,
            json={"settings": {"unknown_setting_key": 123}},
        )

        assert response.status_code == 400
        assert "unknown" in response.json()["detail"].lower()

    @pytest.mark.asyncio()
    async def test_null_resets_to_fallback(
        self,
        api_client_admin: AsyncClientType,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Setting null should clear DB value (use fallback)."""
        # First set a value
        repo = SystemSettingsRepository(db_session)
        await repo.set_setting("max_collections_per_user", 99, user_id=cast(int, test_user_db.id))
        await db_session.commit()

        # Now set to null
        response = await api_client_admin.patch(
            SYSTEM_SETTINGS_PATH,
            json={"settings": {"max_collections_per_user": None}},
        )

        assert response.status_code == 200

        # The DB value should now be null (will fall back to env/default)
        value = await repo.get_setting("max_collections_per_user")
        assert value is None


class TestGetDefaultSettings:
    """Tests for GET /api/v2/system-settings/defaults."""

    @pytest.mark.asyncio()
    async def test_admin_can_get_defaults(
        self,
        api_client_admin: AsyncClientType,
    ) -> None:
        """Admin should receive default values."""
        response = await api_client_admin.get(SYSTEM_SETTINGS_DEFAULTS_PATH)

        assert response.status_code == 200
        defaults = response.json()
        # Verify some known defaults
        assert defaults["max_collections_per_user"] == 10
        assert defaults["cache_ttl_seconds"] == 300

    @pytest.mark.asyncio()
    async def test_non_admin_forbidden(
        self,
        api_client: AsyncClientType,
        api_auth_headers: dict[str, str],
    ) -> None:
        """Non-admin users should receive 403."""
        response = await api_client.get(SYSTEM_SETTINGS_DEFAULTS_PATH, headers=api_auth_headers)

        assert response.status_code == 403
