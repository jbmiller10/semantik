"""Tests for SDK service module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestCreateSDKSession:
    """Test create_sdk_session function."""

    @pytest.mark.asyncio
    async def test_creates_session_with_unique_id(self) -> None:
        """Session ID is unique and starts with 'af_'."""
        from webui.services.assisted_flow.sdk_service import create_sdk_session

        mock_db = AsyncMock()
        source_stats = {
            "source_name": "Test Source",
            "source_type": "directory",
            "source_path": "/test",
            "source_config": {},
        }

        with (
            patch("webui.services.assisted_flow.sdk_service.ClaudeSDKClient") as MockClient,
            patch("webui.services.assisted_flow.sdk_service.ClaudeAgentOptions"),
            patch("webui.services.assisted_flow.sdk_service.create_mcp_server"),
            patch("webui.services.assisted_flow.sdk_service.session_manager") as mock_manager,
        ):
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            MockClient.return_value = mock_client
            mock_manager.store_client = AsyncMock()

            session_id, client = await create_sdk_session(
                db=mock_db,
                user_id=1,
                source_id=42,
                source_stats=source_stats,
            )

        assert session_id.startswith("af_")
        assert len(session_id) == 19  # "af_" + 16 hex chars
        assert client is mock_client

    @pytest.mark.asyncio
    async def test_stores_client_in_session_manager(self) -> None:
        """Client is stored in session manager."""
        from webui.services.assisted_flow.sdk_service import create_sdk_session

        mock_db = AsyncMock()
        source_stats = {
            "source_name": "Test",
            "source_type": "directory",
            "source_path": "/test",
            "source_config": {},
        }

        with (
            patch("webui.services.assisted_flow.sdk_service.ClaudeSDKClient") as MockClient,
            patch("webui.services.assisted_flow.sdk_service.ClaudeAgentOptions"),
            patch("webui.services.assisted_flow.sdk_service.create_mcp_server"),
            patch("webui.services.assisted_flow.sdk_service.session_manager") as mock_manager,
        ):
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            MockClient.return_value = mock_client
            mock_manager.store_client = AsyncMock()

            session_id, _ = await create_sdk_session(
                db=mock_db,
                user_id=1,
                source_id=42,
                source_stats=source_stats,
            )

        mock_manager.store_client.assert_called_once()
        call_args = mock_manager.store_client.call_args
        assert call_args[0][0] == session_id
        assert call_args[0][1] is mock_client

    @pytest.mark.asyncio
    async def test_raises_sdk_not_available_when_cli_missing(self) -> None:
        """Raises SDKNotAvailableError when CLI not installed."""
        from claude_agent_sdk import CLINotFoundError

        from webui.services.assisted_flow.sdk_service import (
            SDKNotAvailableError,
            create_sdk_session,
        )

        mock_db = AsyncMock()
        source_stats = {
            "source_name": "Test",
            "source_type": "directory",
            "source_path": "/test",
            "source_config": {},
        }

        with (
            patch("webui.services.assisted_flow.sdk_service.ClaudeSDKClient") as MockClient,
            patch("webui.services.assisted_flow.sdk_service.ClaudeAgentOptions"),
            patch("webui.services.assisted_flow.sdk_service.create_mcp_server"),
            patch("webui.services.assisted_flow.sdk_service.session_manager"),
        ):
            MockClient.side_effect = CLINotFoundError("CLI not found")

            with pytest.raises(SDKNotAvailableError):
                await create_sdk_session(
                    db=mock_db,
                    user_id=1,
                    source_id=42,
                    source_stats=source_stats,
                )


class TestGetSessionClient:
    """Test get_session_client function."""

    @pytest.mark.asyncio
    async def test_returns_client_from_manager(self) -> None:
        """Returns client from session manager."""
        from webui.services.assisted_flow.sdk_service import get_session_client

        mock_client = MagicMock()

        with patch("webui.services.assisted_flow.sdk_service.session_manager") as mock_manager:
            mock_manager.get_client = AsyncMock(return_value=mock_client)

            result = await get_session_client("session-123")

        assert result is mock_client
        mock_manager.get_client.assert_called_once_with("session-123")

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self) -> None:
        """Returns None when session not found."""
        from webui.services.assisted_flow.sdk_service import get_session_client

        with patch("webui.services.assisted_flow.sdk_service.session_manager") as mock_manager:
            mock_manager.get_client = AsyncMock(return_value=None)

            result = await get_session_client("nonexistent")

        assert result is None


class TestCloseSession:
    """Test close_session function."""

    @pytest.mark.asyncio
    async def test_disconnects_and_removes_client(self) -> None:
        """Disconnects client and removes from manager."""
        from webui.services.assisted_flow.sdk_service import close_session

        mock_client = MagicMock()

        with patch("webui.services.assisted_flow.sdk_service.session_manager") as mock_manager:
            mock_manager.get_client = AsyncMock(return_value=mock_client)
            mock_manager.remove_client = AsyncMock()

            await close_session("session-123")

        mock_client.disconnect.assert_called_once()
        mock_manager.remove_client.assert_called_once_with("session-123")
