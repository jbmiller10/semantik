"""Unit tests for SystemSettingsRepository."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.database.exceptions import DatabaseOperationError
from shared.database.models import SystemSettings
from shared.database.repositories.system_settings_repository import SystemSettingsRepository


class TestSystemSettingsRepository:
    """Tests for SystemSettingsRepository."""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.flush = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.fixture()
    def repo(self, mock_session):
        """Create a repository with mocked session."""
        return SystemSettingsRepository(mock_session)

    # =========================================================================
    # get_setting tests
    # =========================================================================

    @pytest.mark.asyncio()
    async def test_get_setting_found(self, repo, mock_session):
        """Returns value when setting exists."""
        mock_setting = MagicMock(spec=SystemSettings)
        mock_setting.key = "max_collections_per_user"
        mock_setting.value = 20

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_setting
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_setting("max_collections_per_user")
        assert result == 20

    @pytest.mark.asyncio()
    async def test_get_setting_not_found(self, repo, mock_session):
        """Returns None when setting doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_setting("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio()
    async def test_get_setting_returns_json_null(self, repo, mock_session):
        """Returns None for JSON null value (indicates env var fallback)."""
        mock_setting = MagicMock(spec=SystemSettings)
        mock_setting.key = "cache_ttl_seconds"
        mock_setting.value = None  # JSON null

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_setting
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_setting("cache_ttl_seconds")
        assert result is None

    @pytest.mark.asyncio()
    async def test_get_setting_exception(self, repo, mock_session):
        """Raises DatabaseOperationError on database error."""
        mock_session.execute = AsyncMock(side_effect=Exception("DB error"))

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.get_setting("any_key")

        assert "get" in str(exc_info.value)
        assert "SystemSettings" in str(exc_info.value)

    # =========================================================================
    # get_all_settings tests
    # =========================================================================

    @pytest.mark.asyncio()
    async def test_get_all_settings_returns_dict(self, repo, mock_session):
        """Returns dictionary of all settings."""
        mock_setting1 = MagicMock(spec=SystemSettings)
        mock_setting1.key = "max_collections_per_user"
        mock_setting1.value = 20

        mock_setting2 = MagicMock(spec=SystemSettings)
        mock_setting2.key = "cache_ttl_seconds"
        mock_setting2.value = 600

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_setting1, mock_setting2]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_all_settings()

        assert result == {
            "max_collections_per_user": 20,
            "cache_ttl_seconds": 600,
        }

    @pytest.mark.asyncio()
    async def test_get_all_settings_empty(self, repo, mock_session):
        """Returns empty dict when no settings exist."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_all_settings()
        assert result == {}

    @pytest.mark.asyncio()
    async def test_get_all_settings_exception(self, repo, mock_session):
        """Raises DatabaseOperationError on database error."""
        mock_session.execute = AsyncMock(side_effect=Exception("DB error"))

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.get_all_settings()

        assert "get_all" in str(exc_info.value)
        assert "SystemSettings" in str(exc_info.value)

    # =========================================================================
    # get_settings_with_metadata tests
    # =========================================================================

    @pytest.mark.asyncio()
    async def test_get_settings_with_metadata(self, repo, mock_session):
        """Returns settings with updated_at and updated_by metadata."""
        from datetime import UTC, datetime

        mock_setting = MagicMock(spec=SystemSettings)
        mock_setting.key = "max_collections_per_user"
        mock_setting.value = 20
        mock_setting.updated_at = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        mock_setting.updated_by = 1

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_setting]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_settings_with_metadata()

        assert "max_collections_per_user" in result
        assert result["max_collections_per_user"]["value"] == 20
        assert result["max_collections_per_user"]["updated_at"] == "2025-01-15T12:00:00+00:00"
        assert result["max_collections_per_user"]["updated_by"] == 1

    # =========================================================================
    # set_setting tests
    # =========================================================================

    @pytest.mark.asyncio()
    async def test_set_setting_creates_new(self, repo, mock_session):
        """Creates new record when setting doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        await repo.set_setting("new_setting", 100, user_id=1)

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

        # Verify the setting object was created correctly
        added_setting = mock_session.add.call_args[0][0]
        assert added_setting.key == "new_setting"
        assert added_setting.value == 100
        assert added_setting.updated_by == 1

    @pytest.mark.asyncio()
    async def test_set_setting_updates_existing(self, repo, mock_session):
        """Updates existing record when setting exists."""
        mock_setting = MagicMock(spec=SystemSettings)
        mock_setting.key = "max_collections_per_user"
        mock_setting.value = 10

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_setting
        mock_session.execute = AsyncMock(return_value=mock_result)

        await repo.set_setting("max_collections_per_user", 20, user_id=1)

        # Should not add new record
        mock_session.add.assert_not_called()
        mock_session.flush.assert_called_once()

        # Verify the setting was updated
        assert mock_setting.value == 20
        assert mock_setting.updated_by == 1

    @pytest.mark.asyncio()
    async def test_set_setting_key_too_long(self, repo, mock_session):
        """Raises DatabaseOperationError for keys over 64 characters."""
        long_key = "a" * 65

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.set_setting(long_key, 100)

        assert "exceeds 64 character limit" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_set_setting_exception(self, repo, mock_session):
        """Raises DatabaseOperationError on database error."""
        mock_session.execute = AsyncMock(side_effect=Exception("DB error"))

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.set_setting("any_key", 100)

        assert "set" in str(exc_info.value)
        assert "SystemSettings" in str(exc_info.value)

    # =========================================================================
    # set_settings tests (bulk)
    # =========================================================================

    @pytest.mark.asyncio()
    async def test_set_settings_bulk_update(self, repo, mock_session):
        """Updates multiple settings at once."""
        mock_existing = MagicMock(spec=SystemSettings)
        mock_existing.key = "existing_key"
        mock_existing.value = 10

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_existing]
        mock_session.execute = AsyncMock(return_value=mock_result)

        settings = {
            "existing_key": 20,
            "new_key": 100,
        }

        result = await repo.set_settings(settings, user_id=1)

        assert "existing_key" in result
        assert "new_key" in result
        mock_session.add.assert_called_once()  # Only new_key
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_set_settings_validates_key_length(self, repo):
        """Validates all keys upfront before any updates."""
        long_key = "a" * 65
        settings = {
            "valid_key": 10,
            long_key: 20,
        }

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.set_settings(settings)

        assert "exceeds 64 character limit" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_set_settings_exception(self, repo, mock_session):
        """Raises DatabaseOperationError on database error."""
        mock_session.execute = AsyncMock(side_effect=Exception("DB error"))

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.set_settings({"key": "value"})

        assert "set_settings" in str(exc_info.value)

    # =========================================================================
    # delete_setting tests
    # =========================================================================

    @pytest.mark.asyncio()
    async def test_delete_setting_found(self, repo, mock_session):
        """Returns True when setting is deleted."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.delete_setting("key_to_delete")

        assert result is True
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_delete_setting_not_found(self, repo, mock_session):
        """Returns False when setting doesn't exist."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.delete_setting("nonexistent_key")

        assert result is False
        mock_session.flush.assert_not_called()

    @pytest.mark.asyncio()
    async def test_delete_setting_exception(self, repo, mock_session):
        """Raises DatabaseOperationError on database error."""
        mock_session.execute = AsyncMock(side_effect=Exception("DB error"))

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.delete_setting("any_key")

        assert "delete" in str(exc_info.value)
        assert "SystemSettings" in str(exc_info.value)
