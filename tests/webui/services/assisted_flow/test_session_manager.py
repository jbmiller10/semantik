"""Tests for assisted flow session manager."""

import asyncio
from unittest.mock import MagicMock

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
