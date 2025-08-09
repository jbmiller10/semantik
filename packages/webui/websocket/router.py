"""Message routing for WebSocket connections."""

import hashlib
import json
import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels."""

    SYSTEM = 0  # Highest priority - system messages
    HIGH = 1  # Important user messages
    NORMAL = 2  # Regular messages
    LOW = 3  # Background updates


class RoutingRule(BaseModel):
    """Rule for message routing."""

    pattern: str
    priority: MessagePriority = MessagePriority.NORMAL
    ttl: int = 3600  # seconds
    max_retries: int = 3
    dedupe_window: int = 5  # seconds


class Message(BaseModel):
    """WebSocket message with routing metadata."""

    id: str
    channel: str
    type: str
    data: dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime
    source_instance: str | None = None
    target_user: str | None = None
    target_instance: str | None = None
    retry_count: int = 0
    metadata: dict[str, Any] = {}


class MessageRouter:
    """
    Routes messages between WebSocket connections.

    Features:
    - Pattern-based routing
    - Message prioritization
    - Deduplication
    - Dead letter queue
    - Rate limiting
    """

    def __init__(self) -> None:
        """Initialize the message router."""
        self.routing_rules: dict[str, RoutingRule] = self._default_routing_rules()
        self.message_cache: dict[str, datetime] = {}  # For deduplication
        self.dead_letter_queue: list[Message] = []
        self.max_dlq_size = 1000

    def _default_routing_rules(self) -> dict[str, RoutingRule]:
        """Get default routing rules."""
        return {
            "system:*": RoutingRule(pattern="system:*", priority=MessagePriority.SYSTEM, ttl=300, max_retries=5),
            "operation:*": RoutingRule(pattern="operation:*", priority=MessagePriority.HIGH, ttl=3600, max_retries=3),
            "collection:*": RoutingRule(
                pattern="collection:*", priority=MessagePriority.NORMAL, ttl=3600, max_retries=3
            ),
            "user:*": RoutingRule(pattern="user:*", priority=MessagePriority.NORMAL, ttl=3600, max_retries=3),
            "chunking:*": RoutingRule(pattern="chunking:*", priority=MessagePriority.LOW, ttl=1800, max_retries=2),
        }

    def get_routing_key(self, channel: str, target_user: str | None = None, target_instance: str | None = None) -> str:
        """
        Generate Redis Pub/Sub routing key.

        Args:
            channel: Base channel name
            target_user: Optional user targeting
            target_instance: Optional instance targeting

        Returns:
            Routing key for Redis Pub/Sub
        """
        if target_instance:
            return f"instance:{target_instance}"
        if target_user:
            return f"user:{target_user}"
        return channel

    def route_message(self, message: dict[str, Any]) -> Message:
        """
        Process and route a message.

        Args:
            message: Raw message data

        Returns:
            Processed message with routing metadata
        """
        # Generate message ID if not present
        if "id" not in message:
            message["id"] = self._generate_message_id(message)

        # Determine channel
        channel = message.get("channel", "default")

        # Find matching routing rule
        rule = self._match_routing_rule(channel)

        # Create message object
        msg = Message(
            id=message["id"],
            channel=channel,
            type=message.get("type", "message"),
            data=message.get("data", {}),
            priority=rule.priority if rule else MessagePriority.NORMAL,
            timestamp=datetime.now(UTC),
            source_instance=message.get("source_instance"),
            target_user=message.get("target_user"),
            target_instance=message.get("target_instance"),
            metadata=message.get("metadata", {}),
        )

        # Check for duplicates
        if self._is_duplicate(msg):
            logger.debug(f"Duplicate message detected: {msg.id}")
            return None

        # Store for deduplication
        self._store_message_id(msg)

        return msg

    def should_retry(self, message: Message) -> bool:
        """
        Check if message should be retried.

        Args:
            message: Message to check

        Returns:
            True if should retry
        """
        rule = self._match_routing_rule(message.channel)
        if not rule:
            return message.retry_count < 3  # Default

        return message.retry_count < rule.max_retries

    def add_to_dlq(self, message: Message, error: str) -> None:
        """
        Add message to dead letter queue.

        Args:
            message: Failed message
            error: Error description
        """
        message.metadata["dlq_error"] = error
        message.metadata["dlq_timestamp"] = datetime.now(UTC).isoformat()

        self.dead_letter_queue.append(message)

        # Trim queue if too large
        if len(self.dead_letter_queue) > self.max_dlq_size:
            self.dead_letter_queue = self.dead_letter_queue[-self.max_dlq_size :]

        logger.warning(f"Message {message.id} added to DLQ: {error}")

    def get_dlq_messages(self, limit: int = 100) -> list[Message]:
        """
        Get messages from dead letter queue.

        Args:
            limit: Maximum messages to return

        Returns:
            List of failed messages
        """
        return self.dead_letter_queue[-limit:]

    def clear_dlq(self) -> int:
        """
        Clear dead letter queue.

        Returns:
            Number of messages cleared
        """
        count = len(self.dead_letter_queue)
        self.dead_letter_queue.clear()
        return count

    def get_channel_pattern(self, channel: str) -> str:
        """
        Get routing pattern for channel.

        Args:
            channel: Channel name

        Returns:
            Matching pattern or channel itself
        """
        # Check for exact match
        if channel in self.routing_rules:
            return channel

        # Check for wildcard patterns
        parts = channel.split(":")
        for i in range(len(parts), 0, -1):
            pattern = ":".join(parts[: i - 1] + ["*"])
            if pattern in self.routing_rules:
                return pattern

        return channel

    def _match_routing_rule(self, channel: str) -> RoutingRule | None:
        """Find matching routing rule for channel."""
        pattern = self.get_channel_pattern(channel)
        return self.routing_rules.get(pattern)

    def _generate_message_id(self, message: dict[str, Any]) -> str:
        """Generate unique message ID."""
        # Create hash of message content and timestamp
        content = json.dumps(message, sort_keys=True)
        timestamp = datetime.now(UTC).isoformat()
        hash_input = f"{content}{timestamp}"

        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _is_duplicate(self, message: Message) -> bool:
        """Check if message is a duplicate."""
        # Check if we've seen this message ID recently
        if message.id in self.message_cache:
            last_seen = self.message_cache[message.id]
            rule = self._match_routing_rule(message.channel)
            dedupe_window = rule.dedupe_window if rule else 5

            if (datetime.now(UTC) - last_seen).total_seconds() < dedupe_window:
                return True

        return False

    def _store_message_id(self, message: Message) -> None:
        """Store message ID for deduplication."""
        self.message_cache[message.id] = datetime.now(UTC)

        # Clean old entries
        self._cleanup_message_cache()

    def _cleanup_message_cache(self) -> None:
        """Remove old entries from message cache."""
        now = datetime.now(UTC)
        max_age = 60  # seconds

        # Remove entries older than max_age
        self.message_cache = {
            msg_id: timestamp
            for msg_id, timestamp in self.message_cache.items()
            if (now - timestamp).total_seconds() < max_age
        }

    def get_stats(self) -> dict[str, Any]:
        """
        Get router statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "routing_rules": len(self.routing_rules),
            "message_cache_size": len(self.message_cache),
            "dlq_size": len(self.dead_letter_queue),
            "rules": {
                pattern: {"priority": rule.priority.name, "ttl": rule.ttl, "max_retries": rule.max_retries}
                for pattern, rule in self.routing_rules.items()
            },
        }
