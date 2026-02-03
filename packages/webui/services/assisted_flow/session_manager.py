"""Session manager for ClaudeSDKClient instances.

This module provides a thread-safe session manager that stores
ClaudeSDKClient instances with TTL-based cleanup.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeSDKClient

logger = logging.getLogger(__name__)

# Default session TTL (1 hour)
DEFAULT_TTL_SECONDS = 3600


class SessionManager:
    """Manages ClaudeSDKClient instances for assisted flow sessions.

    Provides thread-safe storage and retrieval of SDK client instances
    with automatic TTL-based expiry.

    Attributes:
        ttl_seconds: Time-to-live for sessions in seconds (default: 1 hour)
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        """Initialize the session manager.

        Args:
            ttl_seconds: Session time-to-live in seconds
        """
        self._clients: dict[str, tuple[ClaudeSDKClient, datetime, int]] = {}
        self._lock = asyncio.Lock()
        self._ttl = timedelta(seconds=ttl_seconds)

    async def store_client(self, session_id: str, client: ClaudeSDKClient, *, user_id: int) -> None:
        """Store a client instance.

        Args:
            session_id: Unique session identifier
            client: ClaudeSDKClient instance to store
            user_id: ID of the user who owns this session
        """
        async with self._lock:
            self._clients[session_id] = (client, datetime.now(UTC), user_id)
            logger.debug(f"Stored client for session {session_id}")

    async def get_client(self, session_id: str, *, user_id: int | None = None) -> ClaudeSDKClient | None:
        """Get a client instance, returning None if expired or not found.

        Args:
            session_id: Unique session identifier
            user_id: Optional user ID to enforce session ownership

        Returns:
            ClaudeSDKClient instance or None if not found/expired
        """
        client_to_disconnect: ClaudeSDKClient | None = None
        client_to_return: ClaudeSDKClient | None = None

        async with self._lock:
            existing = self._clients.get(session_id)
            if existing is None:
                return None

            client, created_at, owner_user_id = existing

            if user_id is not None and owner_user_id != user_id:
                logger.warning("Session %s access denied for user_id=%s", session_id, user_id)
                return None

            if datetime.now(UTC) - created_at > self._ttl:
                # Session expired, remove it and disconnect outside the lock.
                self._clients.pop(session_id, None)
                client_to_disconnect = client
                logger.debug("Session %s expired", session_id)
            else:
                client_to_return = client

        if client_to_disconnect is not None:
            try:
                await client_to_disconnect.disconnect()
            except Exception:
                logger.debug("Failed to disconnect expired session %s", session_id, exc_info=True)
            return None

        return client_to_return

    async def remove_client(self, session_id: str) -> None:
        """Remove a client instance.

        Args:
            session_id: Unique session identifier
        """
        client_to_disconnect: ClaudeSDKClient | None = None
        async with self._lock:
            existing = self._clients.pop(session_id, None)
            if existing is None:
                return
            client_to_disconnect, _created_at, _owner_user_id = existing

        try:
            await client_to_disconnect.disconnect()
        except Exception:
            logger.debug("Failed to disconnect session %s", session_id, exc_info=True)
        logger.debug(f"Removed client for session {session_id}")

    async def cleanup_expired(self) -> int:
        """Remove all expired sessions.

        Returns:
            Number of sessions removed
        """
        expired_items: list[tuple[str, ClaudeSDKClient]] = []
        async with self._lock:
            now = datetime.now(UTC)
            expired_ids = [
                sid for sid, (_, created_at, _user_id) in self._clients.items() if now - created_at > self._ttl
            ]
            for sid in expired_ids:
                client, _created_at, _user_id = self._clients.pop(sid)
                expired_items.append((sid, client))

        for sid, client in expired_items:
            try:
                await client.disconnect()
            except Exception:
                logger.debug("Failed to disconnect expired session %s", sid, exc_info=True)

        if expired_items:
            logger.info("Cleaned up %s expired sessions", len(expired_items))

        return len(expired_items)

    @property
    def active_session_count(self) -> int:
        """Return the number of active sessions."""
        return len(self._clients)


# Singleton instance
session_manager = SessionManager()
