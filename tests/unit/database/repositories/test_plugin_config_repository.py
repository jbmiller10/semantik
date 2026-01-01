"""Unit tests for PluginConfigRepository."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.database.exceptions import DatabaseOperationError
from shared.database.models import PluginConfig
from shared.database.repositories.plugin_config_repository import PluginConfigRepository


class TestPluginConfigRepository:
    """Tests for PluginConfigRepository."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.flush = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.fixture
    def repo(self, mock_session):
        """Create a repository with mocked session."""
        return PluginConfigRepository(mock_session)

    @pytest.mark.asyncio
    async def test_list_configs_no_filters(self, repo, mock_session):
        """Test list_configs without filters."""
        mock_config = MagicMock(spec=PluginConfig)
        mock_config.id = "test-plugin"
        mock_config.type = "embedding"
        mock_config.enabled = True

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_config]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.list_configs()
        assert len(result) == 1
        assert result[0].id == "test-plugin"

    @pytest.mark.asyncio
    async def test_list_configs_filter_by_type(self, repo, mock_session):
        """Test list_configs with type filter."""
        mock_config = MagicMock(spec=PluginConfig)
        mock_config.id = "test-plugin"
        mock_config.type = "embedding"

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_config]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.list_configs(plugin_type="embedding")
        assert len(result) == 1
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_configs_filter_by_enabled(self, repo, mock_session):
        """Test list_configs with enabled filter."""
        mock_config = MagicMock(spec=PluginConfig)
        mock_config.id = "test-plugin"
        mock_config.enabled = True

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_config]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.list_configs(enabled=True)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_configs_exception(self, repo, mock_session):
        """Test list_configs raises DatabaseOperationError on exception."""
        mock_session.execute = AsyncMock(side_effect=Exception("DB error"))

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.list_configs()

        assert "list" in str(exc_info.value)
        assert "PluginConfig" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_config_found(self, repo, mock_session):
        """Test get_config returns config when found."""
        mock_config = MagicMock(spec=PluginConfig)
        mock_config.id = "test-plugin"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_config
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_config("test-plugin")
        assert result is not None
        assert result.id == "test-plugin"

    @pytest.mark.asyncio
    async def test_get_config_not_found(self, repo, mock_session):
        """Test get_config returns None when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_config("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_config_exception(self, repo, mock_session):
        """Test get_config raises DatabaseOperationError on exception."""
        mock_session.execute = AsyncMock(side_effect=Exception("DB error"))

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.get_config("test-plugin")

        assert "get" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_upsert_config_create_new(self, repo, mock_session):
        """Test upsert_config creates new record when not found."""
        # Mock get_config to return None (not found)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.upsert_config(
            plugin_id="new-plugin",
            plugin_type="embedding",
            enabled=True,
            config={"key": "value"},
        )

        assert result is not None
        assert result.id == "new-plugin"
        assert result.type == "embedding"
        assert result.enabled is True
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_config_update_existing(self, repo, mock_session):
        """Test upsert_config updates existing record."""
        existing = MagicMock(spec=PluginConfig)
        existing.id = "existing-plugin"
        existing.type = "embedding"
        existing.enabled = True
        existing.config = {}

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.upsert_config(
            plugin_id="existing-plugin",
            plugin_type="embedding",
            enabled=False,
            config={"updated": True},
        )

        assert result.enabled is False
        assert result.config == {"updated": True}
        mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_upsert_config_update_type(self, repo, mock_session):
        """Test upsert_config updates type if different."""
        existing = MagicMock(spec=PluginConfig)
        existing.id = "plugin"
        existing.type = "old-type"
        existing.enabled = True
        existing.config = {}

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.upsert_config(
            plugin_id="plugin",
            plugin_type="new-type",
        )

        assert result.type == "new-type"

    @pytest.mark.asyncio
    async def test_upsert_config_defaults(self, repo, mock_session):
        """Test upsert_config uses defaults for optional params."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.upsert_config(
            plugin_id="plugin",
            plugin_type="embedding",
        )

        assert result.enabled is True
        assert result.config == {}

    @pytest.mark.asyncio
    async def test_upsert_config_exception(self, repo, mock_session):
        """Test upsert_config raises DatabaseOperationError on exception."""
        mock_session.execute = AsyncMock(side_effect=Exception("DB error"))

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.upsert_config(plugin_id="plugin", plugin_type="embedding")

        assert "upsert" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_health(self, repo, mock_session):
        """Test update_health updates health status."""
        existing = MagicMock(spec=PluginConfig)
        existing.id = "plugin"
        existing.type = "embedding"
        existing.health_status = "unknown"
        existing.error_message = None
        existing.last_health_check = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.update_health(
            plugin_id="plugin",
            plugin_type="embedding",
            status="healthy",
            error_message=None,
        )

        assert result.health_status == "healthy"
        assert result.error_message is None
        # last_health_check is set to func.now() which is a SQLAlchemy expression
        mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_update_health_with_error(self, repo, mock_session):
        """Test update_health with error message."""
        existing = MagicMock(spec=PluginConfig)
        existing.id = "plugin"
        existing.type = "embedding"
        existing.health_status = "unknown"
        existing.error_message = None
        existing.last_health_check = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.update_health(
            plugin_id="plugin",
            plugin_type="embedding",
            status="unhealthy",
            error_message="Connection failed",
        )

        assert result.health_status == "unhealthy"
        assert result.error_message == "Connection failed"

    @pytest.mark.asyncio
    async def test_list_disabled_ids_no_filter(self, repo, mock_session):
        """Test list_disabled_ids without type filter."""
        mock_result = MagicMock()
        mock_result.all.return_value = [("plugin1",), ("plugin2",)]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.list_disabled_ids()

        assert result == {"plugin1", "plugin2"}

    @pytest.mark.asyncio
    async def test_list_disabled_ids_with_types(self, repo, mock_session):
        """Test list_disabled_ids with type filter."""
        mock_result = MagicMock()
        mock_result.all.return_value = [("plugin1",)]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.list_disabled_ids(plugin_types=["embedding", "chunking"])

        assert result == {"plugin1"}
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_disabled_ids_empty(self, repo, mock_session):
        """Test list_disabled_ids returns empty set."""
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.list_disabled_ids()

        assert result == set()

    @pytest.mark.asyncio
    async def test_list_disabled_ids_exception(self, repo, mock_session):
        """Test list_disabled_ids raises DatabaseOperationError on exception."""
        mock_session.execute = AsyncMock(side_effect=Exception("DB error"))

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.list_disabled_ids()

        assert "list" in str(exc_info.value)
