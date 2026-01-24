"""Unit tests for MessageStore."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from webui.services.agent.exceptions import MessageStoreError
from webui.services.agent.message_store import (
    ConversationMessage,
    MessageStore,
)


@pytest.fixture()
def mock_redis():
    """Create a mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock()
    redis.exists = AsyncMock(return_value=0)
    redis.expire = AsyncMock(return_value=True)
    redis.delete = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    return redis


@pytest.fixture()
def message_store(mock_redis):
    """Create a MessageStore with mock Redis."""
    return MessageStore(mock_redis, ttl_seconds=3600)


class TestConversationMessage:
    """Tests for ConversationMessage dataclass."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = ConversationMessage.create("user", "Hello, agent!")

        assert msg.role == "user"
        assert msg.content == "Hello, agent!"
        assert msg.timestamp is not None
        assert msg.metadata is None

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = ConversationMessage.create("assistant", "Hello, user!")

        assert msg.role == "assistant"
        assert msg.content == "Hello, user!"

    def test_create_tool_message_with_metadata(self):
        """Test creating a tool message with metadata."""
        metadata = {"tool_name": "list_plugins", "duration_ms": 150}
        msg = ConversationMessage.create("tool", "Found 5 plugins", metadata=metadata)

        assert msg.role == "tool"
        assert msg.metadata == metadata

    def test_to_dict(self):
        """Test converting message to dictionary."""
        msg = ConversationMessage(
            role="user",
            content="Test",
            timestamp="2026-01-24T10:00:00",
            metadata={"key": "value"},
        )

        data = msg.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Test"
        assert data["timestamp"] == "2026-01-24T10:00:00"
        assert data["metadata"] == {"key": "value"}

    def test_from_dict(self):
        """Test creating message from dictionary."""
        data = {
            "role": "assistant",
            "content": "Response",
            "timestamp": "2026-01-24T10:00:00",
            "metadata": None,
        }

        msg = ConversationMessage.from_dict(data)

        assert msg.role == "assistant"
        assert msg.content == "Response"
        assert msg.timestamp == "2026-01-24T10:00:00"

    def test_from_dict_with_metadata(self):
        """Test creating message from dictionary with metadata."""
        data = {
            "role": "subagent",
            "content": "Analysis complete",
            "timestamp": "2026-01-24T10:00:00",
            "metadata": {"subagent_type": "source_analyzer"},
        }

        msg = ConversationMessage.from_dict(data)

        assert msg.metadata == {"subagent_type": "source_analyzer"}


class TestMessageStore:
    """Tests for MessageStore."""

    @pytest.mark.asyncio()
    async def test_get_messages_empty(self, message_store, mock_redis):
        """Test getting messages when none exist."""
        mock_redis.get.return_value = None

        messages = await message_store.get_messages(str(uuid4()))

        assert messages == []

    @pytest.mark.asyncio()
    async def test_get_messages_with_data(self, message_store, mock_redis):
        """Test getting messages with stored data."""
        import json

        conv_id = str(uuid4())
        stored_messages = [
            {"role": "user", "content": "Hello", "timestamp": "2026-01-24T10:00:00", "metadata": None},
            {"role": "assistant", "content": "Hi there", "timestamp": "2026-01-24T10:00:01", "metadata": None},
        ]
        mock_redis.get.return_value = json.dumps(stored_messages)

        messages = await message_store.get_messages(conv_id)

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"

    @pytest.mark.asyncio()
    async def test_get_messages_invalid_json(self, message_store, mock_redis):
        """Test getting messages with invalid JSON."""
        mock_redis.get.return_value = "not valid json"

        with pytest.raises(MessageStoreError, match="Invalid message data"):
            await message_store.get_messages(str(uuid4()))

    @pytest.mark.asyncio()
    async def test_append_message(self, message_store, mock_redis):
        """Test appending a message."""
        import json

        conv_id = str(uuid4())
        mock_redis.get.return_value = None  # No existing messages

        msg = ConversationMessage.create("user", "New message")
        await message_store.append_message(conv_id, msg)

        # Check that setex was called with the message
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == f"agent:conversation:{conv_id}:messages"
        assert call_args[0][1] == 3600  # TTL

        stored_data = json.loads(call_args[0][2])
        assert len(stored_data) == 1
        assert stored_data[0]["content"] == "New message"

    @pytest.mark.asyncio()
    async def test_append_to_existing_messages(self, message_store, mock_redis):
        """Test appending to existing messages."""
        import json

        conv_id = str(uuid4())
        existing = [{"role": "user", "content": "First", "timestamp": "2026-01-24T10:00:00", "metadata": None}]
        mock_redis.get.return_value = json.dumps(existing)

        msg = ConversationMessage.create("assistant", "Second")
        await message_store.append_message(conv_id, msg)

        call_args = mock_redis.setex.call_args
        stored_data = json.loads(call_args[0][2])
        assert len(stored_data) == 2
        assert stored_data[0]["content"] == "First"
        assert stored_data[1]["content"] == "Second"

    @pytest.mark.asyncio()
    async def test_set_messages(self, message_store, mock_redis):
        """Test setting all messages."""
        import json

        conv_id = str(uuid4())
        messages = [
            ConversationMessage.create("user", "Hello"),
            ConversationMessage.create("assistant", "Hi"),
        ]

        await message_store.set_messages(conv_id, messages)

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        stored_data = json.loads(call_args[0][2])
        assert len(stored_data) == 2

    @pytest.mark.asyncio()
    async def test_has_messages_true(self, message_store, mock_redis):
        """Test checking for messages when they exist."""
        mock_redis.exists.return_value = 1

        result = await message_store.has_messages(str(uuid4()))

        assert result is True

    @pytest.mark.asyncio()
    async def test_has_messages_false(self, message_store, mock_redis):
        """Test checking for messages when they don't exist."""
        mock_redis.exists.return_value = 0

        result = await message_store.has_messages(str(uuid4()))

        assert result is False

    @pytest.mark.asyncio()
    async def test_extend_ttl(self, message_store, mock_redis):
        """Test extending TTL on messages."""
        conv_id = str(uuid4())

        result = await message_store.extend_ttl(conv_id)

        assert result is True
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio()
    async def test_delete_messages(self, message_store, mock_redis):
        """Test deleting messages."""
        conv_id = str(uuid4())

        await message_store.delete_messages(conv_id)

        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio()
    async def test_get_message_count(self, message_store, mock_redis):
        """Test getting message count."""
        import json

        conv_id = str(uuid4())
        messages = [
            {"role": "user", "content": "1", "timestamp": "t1", "metadata": None},
            {"role": "assistant", "content": "2", "timestamp": "t2", "metadata": None},
            {"role": "user", "content": "3", "timestamp": "t3", "metadata": None},
        ]
        mock_redis.get.return_value = json.dumps(messages)

        count = await message_store.get_message_count(conv_id)

        assert count == 3

    @pytest.mark.asyncio()
    async def test_acquire_lock(self, message_store, mock_redis):
        """Test acquiring a conversation lock."""
        conv_id = str(uuid4())
        mock_redis.set.return_value = True

        result = await message_store.acquire_lock(conv_id)

        assert result is True
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert "lock" in call_args[0][0]
        assert call_args[1]["nx"] is True  # Only if not exists

    @pytest.mark.asyncio()
    async def test_acquire_lock_already_locked(self, message_store, mock_redis):
        """Test acquiring a lock when already locked."""
        mock_redis.set.return_value = None  # SET NX returns None if key exists

        result = await message_store.acquire_lock(str(uuid4()))

        assert result is False

    @pytest.mark.asyncio()
    async def test_release_lock(self, message_store, mock_redis):
        """Test releasing a conversation lock."""
        conv_id = str(uuid4())

        await message_store.release_lock(conv_id)

        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio()
    async def test_is_locked_true(self, message_store, mock_redis):
        """Test checking if conversation is locked."""
        mock_redis.exists.return_value = 1

        result = await message_store.is_locked(str(uuid4()))

        assert result is True

    @pytest.mark.asyncio()
    async def test_is_locked_false(self, message_store, mock_redis):
        """Test checking if conversation is not locked."""
        mock_redis.exists.return_value = 0

        result = await message_store.is_locked(str(uuid4()))

        assert result is False
