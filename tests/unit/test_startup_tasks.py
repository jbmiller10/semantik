"""Unit tests for startup_tasks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from webui.startup_tasks import ensure_default_chunking_strategies, ensure_default_data


class TestEnsureDefaultData:
    """Tests for ensure_default_data function."""

    @pytest.mark.asyncio
    async def test_ensure_default_data_success(self):
        """Test ensure_default_data runs successfully."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()

        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        async def mock_get_db():
            yield mock_session

        mock_registry = MagicMock()
        mock_registry.list_ids.return_value = []

        with patch("webui.startup_tasks.get_db", return_value=mock_get_db()):
            with patch("webui.startup_tasks.load_plugins", return_value=mock_registry):
                with patch("webui.startup_tasks.ensure_default_chunking_strategies", new=AsyncMock()):
                    await ensure_default_data()

    @pytest.mark.asyncio
    async def test_ensure_default_data_with_disabled_plugins(self):
        """Test ensure_default_data loads disabled plugin IDs."""
        mock_session = AsyncMock()

        mock_result = MagicMock()
        mock_result.all.return_value = [("plugin1",), ("plugin2",)]
        mock_session.execute = AsyncMock(return_value=mock_result)

        async def mock_get_db():
            yield mock_session

        mock_registry = MagicMock()
        mock_registry.list_ids.return_value = []

        with patch("webui.startup_tasks.get_db", return_value=mock_get_db()):
            with patch("webui.startup_tasks.load_plugins", return_value=mock_registry) as mock_load:
                with patch("webui.startup_tasks.ensure_default_chunking_strategies", new=AsyncMock()):
                    await ensure_default_data()

                # Check that disabled_plugin_ids was passed to load_plugins
                mock_load.assert_called_once()
                call_kwargs = mock_load.call_args.kwargs
                assert call_kwargs.get("disabled_plugin_ids") == {"plugin1", "plugin2"}

    @pytest.mark.asyncio
    async def test_ensure_default_data_with_embedding_plugins(self):
        """Test ensure_default_data logs embedding plugins."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        async def mock_get_db():
            yield mock_session

        mock_registry = MagicMock()
        mock_registry.list_ids.side_effect = lambda plugin_type, source: (
            ["embed-plugin"] if plugin_type == "embedding" else []
        )

        with patch("webui.startup_tasks.get_db", return_value=mock_get_db()):
            with patch("webui.startup_tasks.load_plugins", return_value=mock_registry):
                with patch("webui.startup_tasks.ensure_default_chunking_strategies", new=AsyncMock()):
                    with patch("webui.startup_tasks.logger") as mock_logger:
                        await ensure_default_data()
                        # Verify that embedding plugins were logged
                        calls = [call for call in mock_logger.info.call_args_list]
                        assert any("embedding plugins" in str(call).lower() for call in calls)

    @pytest.mark.asyncio
    async def test_ensure_default_data_with_chunking_plugins(self):
        """Test ensure_default_data logs chunking plugins."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        async def mock_get_db():
            yield mock_session

        mock_registry = MagicMock()
        mock_registry.list_ids.side_effect = lambda plugin_type, source: (
            ["chunk-plugin"] if plugin_type == "chunking" else []
        )

        with patch("webui.startup_tasks.get_db", return_value=mock_get_db()):
            with patch("webui.startup_tasks.load_plugins", return_value=mock_registry):
                with patch("webui.startup_tasks.ensure_default_chunking_strategies", new=AsyncMock()):
                    with patch("webui.startup_tasks.logger") as mock_logger:
                        await ensure_default_data()
                        calls = [call for call in mock_logger.info.call_args_list]
                        assert any("chunking plugins" in str(call).lower() for call in calls)

    @pytest.mark.asyncio
    async def test_ensure_default_data_with_connector_plugins(self):
        """Test ensure_default_data logs connector plugins."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        async def mock_get_db():
            yield mock_session

        mock_registry = MagicMock()
        mock_registry.list_ids.side_effect = lambda plugin_type, source: (
            ["conn-plugin"] if plugin_type == "connector" else []
        )

        with patch("webui.startup_tasks.get_db", return_value=mock_get_db()):
            with patch("webui.startup_tasks.load_plugins", return_value=mock_registry):
                with patch("webui.startup_tasks.ensure_default_chunking_strategies", new=AsyncMock()):
                    with patch("webui.startup_tasks.logger") as mock_logger:
                        await ensure_default_data()
                        calls = [call for call in mock_logger.info.call_args_list]
                        assert any("connector plugins" in str(call).lower() for call in calls)

    @pytest.mark.asyncio
    async def test_ensure_default_data_db_exception(self):
        """Test ensure_default_data handles database exceptions gracefully."""
        from sqlalchemy.exc import SQLAlchemyError

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=SQLAlchemyError("DB not ready"))

        async def mock_get_db():
            yield mock_session

        mock_registry = MagicMock()
        mock_registry.list_ids.return_value = []

        with patch("webui.startup_tasks.get_db", return_value=mock_get_db()):
            with patch("webui.startup_tasks.load_plugins", return_value=mock_registry) as mock_load:
                with patch("webui.startup_tasks.ensure_default_chunking_strategies", new=AsyncMock()):
                    await ensure_default_data()

                # Should load plugins with None disabled_plugin_ids
                mock_load.assert_called_once()
                call_kwargs = mock_load.call_args.kwargs
                assert call_kwargs.get("disabled_plugin_ids") is None


class TestEnsureDefaultChunkingStrategies:
    """Tests for ensure_default_chunking_strategies function."""

    @pytest.mark.asyncio
    async def test_ensure_strategies_success(self):
        """Test ensure_default_chunking_strategies creates strategies."""
        mock_session = AsyncMock()
        mock_service = MagicMock()
        mock_service.ensure_default_strategies = AsyncMock(return_value=5)

        with patch(
            "webui.startup_tasks.ChunkingStrategyService", return_value=mock_service
        ):
            with patch("webui.startup_tasks.logger") as mock_logger:
                await ensure_default_chunking_strategies(mock_session)
                mock_logger.info.assert_called()
                assert any("5" in str(call) for call in mock_logger.info.call_args_list)

    @pytest.mark.asyncio
    async def test_ensure_strategies_already_exist(self):
        """Test ensure_default_chunking_strategies when all exist."""
        mock_session = AsyncMock()
        mock_service = MagicMock()
        mock_service.ensure_default_strategies = AsyncMock(return_value=0)

        with patch(
            "webui.startup_tasks.ChunkingStrategyService", return_value=mock_service
        ):
            with patch("webui.startup_tasks.logger") as mock_logger:
                await ensure_default_chunking_strategies(mock_session)
                mock_logger.debug.assert_called()
                assert any("already exist" in str(call) for call in mock_logger.debug.call_args_list)

    @pytest.mark.asyncio
    async def test_ensure_strategies_exception(self):
        """Test ensure_default_chunking_strategies handles exceptions."""
        mock_session = AsyncMock()
        mock_service = MagicMock()
        mock_service.ensure_default_strategies = AsyncMock(
            side_effect=Exception("Service error")
        )

        with patch(
            "webui.startup_tasks.ChunkingStrategyService", return_value=mock_service
        ):
            with patch("webui.startup_tasks.logger") as mock_logger:
                # Should not raise - exceptions are caught and logged
                await ensure_default_chunking_strategies(mock_session)
                mock_logger.error.assert_called()
