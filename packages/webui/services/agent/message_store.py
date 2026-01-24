"""Redis-based message store for agent conversations.

This module provides ephemeral message storage with TTL for agent conversations.
Messages are stored in Redis for performance, with a 24-hour TTL that extends
on activity. When messages expire, conversations can be recovered using the
summary stored in PostgreSQL.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from webui.services.agent.exceptions import MessageStoreError

logger = logging.getLogger(__name__)

# Default TTL for conversation messages (24 hours)
DEFAULT_MESSAGE_TTL_SECONDS = 24 * 60 * 60

# Redis key prefixes
KEY_PREFIX = "agent:conversation"


def _messages_key(conversation_id: str) -> str:
    """Get the Redis key for conversation messages."""
    return f"{KEY_PREFIX}:{conversation_id}:messages"


def _lock_key(conversation_id: str) -> str:
    """Get the Redis key for conversation lock."""
    return f"{KEY_PREFIX}:{conversation_id}:lock"


@dataclass
class ConversationMessage:
    """A message in an agent conversation.

    Attributes:
        role: Who sent this message (user, assistant, tool, subagent)
        content: The message content
        timestamp: When the message was created (ISO format)
        metadata: Additional data (tool_name, subagent_type, etc.)
    """

    role: Literal["user", "assistant", "tool", "subagent"]
    content: str
    timestamp: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationMessage:
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata"),
        )

    @classmethod
    def create(
        cls,
        role: Literal["user", "assistant", "tool", "subagent"],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        """Create a new message with current timestamp."""
        return cls(
            role=role,
            content=content,
            timestamp=datetime.now(UTC).isoformat(),
            metadata=metadata,
        )


class MessageStore:
    """Redis-based storage for conversation messages.

    Messages are stored as a JSON list in Redis with a configurable TTL.
    The TTL is extended on each write operation to keep active conversations
    alive.

    Usage:
        store = MessageStore(redis_client)

        # Add messages
        await store.append_message(conv_id, ConversationMessage.create("user", "Hello"))

        # Get all messages
        messages = await store.get_messages(conv_id)

        # Check if messages exist
        exists = await store.has_messages(conv_id)
    """

    def __init__(
        self,
        redis_client: Any,  # redis.asyncio.Redis
        ttl_seconds: int = DEFAULT_MESSAGE_TTL_SECONDS,
    ):
        """Initialize the message store.

        Args:
            redis_client: Async Redis client
            ttl_seconds: TTL for messages (default 24 hours)
        """
        self.redis = redis_client
        self.ttl = ttl_seconds

    async def get_messages(
        self,
        conversation_id: str,
    ) -> list[ConversationMessage]:
        """Get all messages for a conversation.

        Args:
            conversation_id: UUID of the conversation

        Returns:
            List of messages, empty if none or expired
        """
        try:
            key = _messages_key(conversation_id)
            data = await self.redis.get(key)

            if not data:
                return []

            messages_data = json.loads(data)
            return [ConversationMessage.from_dict(m) for m in messages_data]

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode messages for {conversation_id}: {e}"
            )
            raise MessageStoreError(f"Invalid message data: {e}") from e
        except Exception as e:
            logger.error(
                f"Failed to get messages for {conversation_id}: {e}",
                exc_info=True,
            )
            raise MessageStoreError(f"Failed to get messages: {e}") from e

    async def append_message(
        self,
        conversation_id: str,
        message: ConversationMessage,
    ) -> None:
        """Append a message to a conversation.

        This also extends the TTL on the conversation messages.

        Args:
            conversation_id: UUID of the conversation
            message: Message to append
        """
        try:
            key = _messages_key(conversation_id)

            # Get existing messages
            existing = await self.get_messages(conversation_id)
            existing.append(message)

            # Store updated list with TTL
            data = json.dumps([m.to_dict() for m in existing])
            await self.redis.setex(key, self.ttl, data)

        except MessageStoreError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to append message to {conversation_id}: {e}",
                exc_info=True,
            )
            raise MessageStoreError(f"Failed to append message: {e}") from e

    async def set_messages(
        self,
        conversation_id: str,
        messages: list[ConversationMessage],
    ) -> None:
        """Set all messages for a conversation (replaces existing).

        Args:
            conversation_id: UUID of the conversation
            messages: List of messages to store
        """
        try:
            key = _messages_key(conversation_id)
            data = json.dumps([m.to_dict() for m in messages])
            await self.redis.setex(key, self.ttl, data)

        except Exception as e:
            logger.error(
                f"Failed to set messages for {conversation_id}: {e}",
                exc_info=True,
            )
            raise MessageStoreError(f"Failed to set messages: {e}") from e

    async def has_messages(self, conversation_id: str) -> bool:
        """Check if a conversation has messages stored.

        Args:
            conversation_id: UUID of the conversation

        Returns:
            True if messages exist and haven't expired
        """
        try:
            key = _messages_key(conversation_id)
            return await self.redis.exists(key) > 0

        except Exception as e:
            logger.error(
                f"Failed to check messages for {conversation_id}: {e}",
                exc_info=True,
            )
            raise MessageStoreError(f"Failed to check messages: {e}") from e

    async def extend_ttl(self, conversation_id: str) -> bool:
        """Extend the TTL on conversation messages.

        Args:
            conversation_id: UUID of the conversation

        Returns:
            True if TTL was extended, False if key doesn't exist
        """
        try:
            key = _messages_key(conversation_id)
            return await self.redis.expire(key, self.ttl)

        except Exception as e:
            logger.error(
                f"Failed to extend TTL for {conversation_id}: {e}",
                exc_info=True,
            )
            raise MessageStoreError(f"Failed to extend TTL: {e}") from e

    async def delete_messages(self, conversation_id: str) -> None:
        """Delete all messages for a conversation.

        Args:
            conversation_id: UUID of the conversation
        """
        try:
            key = _messages_key(conversation_id)
            await self.redis.delete(key)

        except Exception as e:
            logger.error(
                f"Failed to delete messages for {conversation_id}: {e}",
                exc_info=True,
            )
            raise MessageStoreError(f"Failed to delete messages: {e}") from e

    async def get_message_count(self, conversation_id: str) -> int:
        """Get the number of messages in a conversation.

        Args:
            conversation_id: UUID of the conversation

        Returns:
            Number of messages, 0 if none or expired
        """
        messages = await self.get_messages(conversation_id)
        return len(messages)

    # Lock methods for concurrent access control

    async def acquire_lock(
        self,
        conversation_id: str,
        timeout_seconds: int = 30,
    ) -> bool:
        """Acquire a lock on a conversation for exclusive access.

        Args:
            conversation_id: UUID of the conversation
            timeout_seconds: Lock timeout

        Returns:
            True if lock acquired, False if already locked
        """
        try:
            key = _lock_key(conversation_id)
            # SET NX with expiry for atomic lock acquisition
            result = await self.redis.set(
                key, "locked", nx=True, ex=timeout_seconds
            )
            return result is not None

        except Exception as e:
            logger.error(
                f"Failed to acquire lock for {conversation_id}: {e}",
                exc_info=True,
            )
            raise MessageStoreError(f"Failed to acquire lock: {e}") from e

    async def release_lock(self, conversation_id: str) -> None:
        """Release a lock on a conversation.

        Args:
            conversation_id: UUID of the conversation
        """
        try:
            key = _lock_key(conversation_id)
            await self.redis.delete(key)

        except Exception as e:
            logger.error(
                f"Failed to release lock for {conversation_id}: {e}",
                exc_info=True,
            )
            raise MessageStoreError(f"Failed to release lock: {e}") from e

    async def is_locked(self, conversation_id: str) -> bool:
        """Check if a conversation is locked.

        Args:
            conversation_id: UUID of the conversation

        Returns:
            True if locked
        """
        try:
            key = _lock_key(conversation_id)
            return await self.redis.exists(key) > 0

        except Exception as e:
            logger.error(
                f"Failed to check lock for {conversation_id}: {e}",
                exc_info=True,
            )
            raise MessageStoreError(f"Failed to check lock: {e}") from e
