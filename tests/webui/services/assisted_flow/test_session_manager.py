"""Tests for assisted flow session manager."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestSessionManager:
    """Test SessionManager class."""

    @pytest.mark.asyncio()
    async def test_store_and_get_client(self) -> None:
        """Can store and retrieve a client."""
        from webui.services.assisted_flow.session_manager import SessionManager

        manager = SessionManager()
        mock_client = MagicMock()

        await manager.store_client("session-123", mock_client, user_id=1)
        retrieved = await manager.get_client("session-123", user_id=1)

        assert retrieved is mock_client

    @pytest.mark.asyncio()
    async def test_get_nonexistent_client_returns_none(self) -> None:
        """Getting nonexistent client returns None."""
        from webui.services.assisted_flow.session_manager import SessionManager

        manager = SessionManager()
        retrieved = await manager.get_client("nonexistent", user_id=1)

        assert retrieved is None

    @pytest.mark.asyncio()
    async def test_remove_client(self) -> None:
        """Can remove a client."""
        from webui.services.assisted_flow.session_manager import SessionManager

        manager = SessionManager()
        mock_client = MagicMock()

        await manager.store_client("session-123", mock_client, user_id=1)
        await manager.remove_client("session-123")
        retrieved = await manager.get_client("session-123", user_id=1)

        assert retrieved is None

    @pytest.mark.asyncio()
    async def test_expired_client_returns_none(self) -> None:
        """Expired clients are not returned."""
        from webui.services.assisted_flow.session_manager import SessionManager

        manager = SessionManager(ttl_seconds=0)  # Immediate expiry
        mock_client = MagicMock()

        await manager.store_client("session-123", mock_client, user_id=1)
        await asyncio.sleep(0.01)  # Wait for expiry
        retrieved = await manager.get_client("session-123", user_id=1)

        assert retrieved is None

    @pytest.mark.asyncio()
    async def test_singleton_instance(self) -> None:
        """Module provides singleton instance."""
        from webui.services.assisted_flow.session_manager import session_manager

        assert session_manager is not None

    @pytest.mark.asyncio()
    async def test_cleanup_all_disconnects_all_sessions(self) -> None:
        """cleanup_all disconnects and removes all sessions."""
        from webui.services.assisted_flow.session_manager import SessionManager

        manager = SessionManager()
        mock_client1 = MagicMock()
        mock_client1.disconnect = AsyncMock()
        mock_client2 = MagicMock()
        mock_client2.disconnect = AsyncMock()

        await manager.store_client("session-1", mock_client1, user_id=1)
        await manager.store_client("session-2", mock_client2, user_id=2)

        assert manager.active_session_count == 2

        removed_count = await manager.cleanup_all()

        assert removed_count == 2
        assert manager.active_session_count == 0
        mock_client1.disconnect.assert_called_once()
        mock_client2.disconnect.assert_called_once()

    @pytest.mark.asyncio()
    async def test_cleanup_all_handles_disconnect_errors(self) -> None:
        """cleanup_all continues even if a disconnect fails."""
        from webui.services.assisted_flow.session_manager import SessionManager

        manager = SessionManager()
        mock_client1 = MagicMock()
        mock_client1.disconnect = AsyncMock(side_effect=RuntimeError("Disconnect failed"))
        mock_client2 = MagicMock()
        mock_client2.disconnect = AsyncMock()

        await manager.store_client("session-1", mock_client1, user_id=1)
        await manager.store_client("session-2", mock_client2, user_id=2)

        # Should not raise, and both clients should be cleared
        removed_count = await manager.cleanup_all()

        assert removed_count == 2
        assert manager.active_session_count == 0

    @pytest.mark.asyncio()
    async def test_cleanup_all_returns_zero_when_empty(self) -> None:
        """cleanup_all returns zero when no sessions exist."""
        from webui.services.assisted_flow.session_manager import SessionManager

        manager = SessionManager()

        removed_count = await manager.cleanup_all()

        assert removed_count == 0
