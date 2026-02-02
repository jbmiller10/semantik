"""Session manager for ClaudeSDKClient instances.

This module provides a thread-safe session manager that stores
ClaudeSDKClient instances with TTL-based cleanup.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

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
        self._clients: dict[str, tuple[Any, datetime]] = {}
        self._lock = asyncio.Lock()
        self._ttl = timedelta(seconds=ttl_seconds)

    async def store_client(self, session_id: str, client: ClaudeSDKClient) -> None:
        """Store a client instance.

        Args:
            session_id: Unique session identifier
            client: ClaudeSDKClient instance to store
        """
        async with self._lock:
            self._clients[session_id] = (client, datetime.now(UTC))
            logger.debug(f"Stored client for session {session_id}")

    async def get_client(self, session_id: str) -> ClaudeSDKClient | None:
        """Get a client instance, returning None if expired or not found.

        Args:
            session_id: Unique session identifier

        Returns:
            ClaudeSDKClient instance or None if not found/expired
        """
        async with self._lock:
            if session_id not in self._clients:
                return None

            client, created_at = self._clients[session_id]
            if datetime.now(UTC) - created_at > self._ttl:
                # Session expired, remove it
                del self._clients[session_id]
                logger.debug(f"Session {session_id} expired")
                return None

            return client

    async def remove_client(self, session_id: str) -> None:
        """Remove a client instance.

        Args:
            session_id: Unique session identifier
        """
        async with self._lock:
            if session_id in self._clients:
                del self._clients[session_id]
                logger.debug(f"Removed client for session {session_id}")

    async def cleanup_expired(self) -> int:
        """Remove all expired sessions.

        Returns:
            Number of sessions removed
        """
        async with self._lock:
            now = datetime.now(UTC)
            expired = [
                sid for sid, (_, created_at) in self._clients.items()
                if now - created_at > self._ttl
            ]
            for sid in expired:
                del self._clients[sid]

            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")

            return len(expired)

    @property
    def active_session_count(self) -> int:
        """Return the number of active sessions."""
        return len(self._clients)


# Singleton instance
session_manager = SessionManager()
