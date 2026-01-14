"""Unit tests for SystemSettingsService."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from webui.services.system_settings_service import (
    SystemSettingsService,
    _parse_env_value,
    get_system_settings_service,
    reset_service_instance,
)


class TestParseEnvValue:
    """Tests for _parse_env_value function."""

    def test_parse_bool_true_values(self):
        """Parses 'true', '1', 'yes', 'on' as True."""
        for value in ("true", "TRUE", "True", "1", "yes", "on"):
            result = _parse_env_value("enable_cpu_offload", value)
            assert result is True, f"Expected True for '{value}'"

    def test_parse_bool_false_values(self):
        """Parses 'false', '0', 'no', 'off' as False."""
        for value in ("false", "FALSE", "False", "0", "no", "off"):
            result = _parse_env_value("enable_cpu_offload", value)
            assert result is False, f"Expected False for '{value}'"

    def test_parse_int(self):
        """Parses integer string to int."""
        result = _parse_env_value("max_collections_per_user", "20")
        assert result == 20
        assert isinstance(result, int)

    def test_parse_int_invalid(self):
        """Returns default for invalid int string."""
        result = _parse_env_value("max_collections_per_user", "not_a_number")
        assert result == 10  # Default from SYSTEM_SETTING_DEFAULTS

    def test_parse_float(self):
        """Parses float string to float."""
        result = _parse_env_value("gpu_memory_max_percent", "0.85")
        assert result == 0.85
        assert isinstance(result, float)

    def test_parse_float_invalid(self):
        """Returns default for invalid float string."""
        result = _parse_env_value("gpu_memory_max_percent", "not_a_float")
        assert result == 0.90  # Default from SYSTEM_SETTING_DEFAULTS

    def test_unknown_key_returns_string(self):
        """Returns string as-is for unknown keys."""
        result = _parse_env_value("unknown_key", "some_value")
        assert result == "some_value"


class TestSystemSettingsService:
    """Tests for SystemSettingsService."""

    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        """Reset the singleton instance before and after each test."""
        reset_service_instance()
        yield
        reset_service_instance()

    @pytest.fixture()
    def mock_session_factory(self):
        """Create a mock session factory."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None

        def factory():
            return mock_context

        return factory, mock_session

    @pytest.fixture()
    def service(self, mock_session_factory):
        """Create a service instance with mocked session factory."""
        factory, _ = mock_session_factory
        return SystemSettingsService(factory)

    @pytest.mark.asyncio()
    async def test_get_setting_from_cache(self, service, mock_session_factory):
        """Returns cached value when valid."""
        _, mock_session = mock_session_factory

        # Set up mock repository to return a value
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [
            MagicMock(key="max_collections_per_user", value=25),
        ]
        mock_session.execute = AsyncMock(return_value=mock_result)

        # First call triggers cache refresh
        result = await service.get_setting("max_collections_per_user")
        assert result == 25

        # Second call should use cached value (no additional DB call)
        mock_session.execute.reset_mock()
        result2 = await service.get_setting("max_collections_per_user")
        assert result2 == 25
        mock_session.execute.assert_not_called()

    @pytest.mark.asyncio()
    async def test_get_setting_env_fallback(self, service, mock_session_factory):
        """Falls back to env var when not in cache."""
        _, mock_session = mock_session_factory

        # Set up empty cache (no DB values)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Set env var
        with patch.dict("os.environ", {"MAX_COLLECTIONS_PER_USER": "30"}):
            result = await service.get_setting("max_collections_per_user")
            assert result == 30

    @pytest.mark.asyncio()
    async def test_get_setting_default_fallback(self, service, mock_session_factory):
        """Falls back to default when not in cache or env."""
        _, mock_session = mock_session_factory

        # Set up empty cache (no DB values)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        # No env var set, should use SYSTEM_SETTING_DEFAULTS
        result = await service.get_setting("max_collections_per_user")
        assert result == 10  # Default from SYSTEM_SETTING_DEFAULTS

    @pytest.mark.asyncio()
    async def test_get_setting_custom_default(self, service, mock_session_factory):
        """Uses custom default for unknown key."""
        _, mock_session = mock_session_factory

        # Set up empty cache
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Unknown key with custom default
        result = await service.get_setting("unknown_key", default=999)
        assert result == 999

    @pytest.mark.asyncio()
    async def test_cache_refresh_on_expiry(self, service, mock_session_factory):
        """Refreshes cache when expired."""
        _, mock_session = mock_session_factory

        # Set up mock to return different values on each call
        call_count = 0

        def mock_execute(*_):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            value = 10 if call_count == 1 else 20
            mock_result.scalars.return_value.all.return_value = [
                MagicMock(key="max_collections_per_user", value=value),
            ]
            return mock_result

        mock_session.execute = AsyncMock(side_effect=mock_execute)

        # First call
        result1 = await service.get_setting("max_collections_per_user")
        assert result1 == 10

        # Expire the cache
        service._cache_expiry = datetime.min.replace(tzinfo=UTC)

        # Second call should refresh cache
        result2 = await service.get_setting("max_collections_per_user")
        assert result2 == 20
        assert call_count == 2

    @pytest.mark.asyncio()
    async def test_cache_refresh_failure_uses_stale(self, service, mock_session_factory):
        """Gracefully handles DB errors and uses short retry interval."""
        _, mock_session = mock_session_factory

        # First call succeeds
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [
            MagicMock(key="max_collections_per_user", value=15),
        ]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result1 = await service.get_setting("max_collections_per_user")
        assert result1 == 15

        # Expire cache and make next DB call fail
        service._cache_expiry = datetime.min.replace(tzinfo=UTC)
        mock_session.execute = AsyncMock(side_effect=Exception("DB error"))

        # Should still return stale cached value
        result2 = await service.get_setting("max_collections_per_user")
        assert result2 == 15

        # Cache expiry should be set to retry soon (within 10 seconds)
        time_until_retry = (service._cache_expiry - datetime.now(UTC)).total_seconds()
        assert time_until_retry <= 10

    def test_invalidate_cache(self, service):
        """Resets expiry to timezone-aware datetime.min."""
        # Set a valid expiry
        service._cache_expiry = datetime.now(UTC) + timedelta(hours=1)

        # Invalidate
        service.invalidate_cache()

        assert service._cache_expiry == datetime.min.replace(tzinfo=UTC)

    @pytest.mark.asyncio()
    async def test_get_all_settings(self, service, mock_session_factory):
        """Returns all resolved settings."""
        _, mock_session = mock_session_factory

        # Set up cache with partial settings
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [
            MagicMock(key="max_collections_per_user", value=25),
            MagicMock(key="cache_ttl_seconds", value=600),
        ]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await service.get_all_settings()

        # Should include all default keys
        assert "max_collections_per_user" in result
        assert result["max_collections_per_user"] == 25  # From cache
        assert "cache_ttl_seconds" in result
        assert result["cache_ttl_seconds"] == 600  # From cache
        # Should include other defaults too
        assert "gpu_memory_max_percent" in result


class TestGetSystemSettingsService:
    """Tests for singleton factory function."""

    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        """Reset the singleton instance before and after each test."""
        reset_service_instance()
        yield
        reset_service_instance()

    def test_returns_same_instance(self):
        """Singleton pattern returns same instance."""
        mock_factory = MagicMock()

        service1 = get_system_settings_service(mock_factory)
        service2 = get_system_settings_service(mock_factory)

        assert service1 is service2

    def test_reset_clears_instance(self):
        """Reset creates new instance."""
        mock_factory = MagicMock()

        service1 = get_system_settings_service(mock_factory)
        reset_service_instance()
        service2 = get_system_settings_service(mock_factory)

        assert service1 is not service2
