"""Tests for ClaudeAgentAdapter with mocked SDK."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.agents.exceptions import (
    AgentExecutionError,
    AgentInterruptedError,
)
from shared.agents.types import (
    AgentCapabilities,
    AgentContext,
    MessageRole,
    MessageType,
)


class AsyncIteratorMock:
    """Mock async iterator for testing."""

    def __init__(self, items: list[Any]):
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


@pytest.fixture(autouse=True)
def mock_claude_sdk():
    """Mock the claude_agent_sdk module."""
    # Create a mock module
    mock_sdk = ModuleType("claude_agent_sdk")
    mock_sdk.query = MagicMock()  # Will be configured per test
    mock_sdk.ClaudeAgentOptions = MagicMock()

    # Insert into sys.modules
    sys.modules["claude_agent_sdk"] = mock_sdk

    yield mock_sdk

    # Cleanup
    del sys.modules["claude_agent_sdk"]


class TestClaudeAgentAdapterImport:
    """Test adapter can be imported with mocked SDK."""

    def test_import_adapter(self, mock_claude_sdk: ModuleType) -> None:
        """Test that ClaudeAgentAdapter can be imported."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        assert ClaudeAgentAdapter is not None


class TestClaudeAgentAdapterCapabilities:
    """Tests for ClaudeAgentAdapter capabilities."""

    def test_get_capabilities(self, mock_claude_sdk: ModuleType) -> None:
        """Test get_capabilities returns Claude-specific capabilities."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        caps = ClaudeAgentAdapter.get_capabilities()

        assert isinstance(caps, AgentCapabilities)
        assert caps.supports_streaming is True
        assert caps.supports_tools is True
        assert caps.supports_sessions is True
        assert caps.supports_session_fork is True
        assert caps.supports_extended_thinking is True
        assert caps.max_context_tokens == 200000
        assert "claude-sonnet-4-20250514" in caps.supported_models
        assert caps.default_model == "claude-sonnet-4-20250514"

    def test_supported_models(self, mock_claude_sdk: ModuleType) -> None:
        """Test SUPPORTED_MODELS constant."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        assert "claude-sonnet-4-20250514" in ClaudeAgentAdapter.SUPPORTED_MODELS
        assert "claude-opus-4-20250514" in ClaudeAgentAdapter.SUPPORTED_MODELS
        assert "claude-haiku-3-5-20241022" in ClaudeAgentAdapter.SUPPORTED_MODELS


class TestClaudeAgentAdapterLifecycle:
    """Tests for adapter lifecycle (initialize, cleanup)."""

    @pytest.mark.asyncio()
    async def test_initialize_success(self, mock_claude_sdk: ModuleType) -> None:
        """Test successful initialization."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        adapter = ClaudeAgentAdapter()
        await adapter.initialize()

        assert adapter.is_initialized is True
        assert adapter._sdk_available is True

    @pytest.mark.asyncio()
    @pytest.mark.skip(reason="SDK unavailability is hard to test with module mocking")
    async def test_initialize_sdk_not_installed(self, mock_claude_sdk: ModuleType) -> None:  # noqa: ARG002
        """Test initialization fails when SDK not installed.

        Note: This test is skipped because Python's module caching makes it
        difficult to reliably test ImportError scenarios with autouse fixtures.
        The functionality is tested manually.
        """

    @pytest.mark.asyncio()
    async def test_cleanup(self, mock_claude_sdk: ModuleType) -> None:
        """Test cleanup resets state."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        adapter = ClaudeAgentAdapter()
        await adapter.initialize()

        await adapter.cleanup()

        assert adapter.is_initialized is False
        assert adapter._sdk_available is False


class TestClaudeAgentAdapterExecute:
    """Tests for execute method."""

    @pytest.fixture()
    def mock_sdk_message(self) -> MagicMock:
        """Create a mock SDK message."""
        msg = MagicMock()
        msg.subtype = "success"
        msg.content = "Hello, I'm Claude!"
        msg.session_id = "test-session-123"
        msg.is_partial = False
        # Explicitly set to None to avoid MagicMock truthy behavior
        msg.tool_use = None
        msg.tool_result = None
        msg.thinking = None
        msg.usage = MagicMock()
        msg.usage.input_tokens = 10
        msg.usage.output_tokens = 20
        msg.usage.cache_read_input_tokens = 0
        msg.usage.cache_creation_input_tokens = 0
        msg.usage.reasoning_tokens = 0
        return msg

    @pytest.mark.asyncio()
    async def test_execute_basic(
        self,
        mock_claude_sdk: ModuleType,
        mock_sdk_message: MagicMock,
    ) -> None:
        """Test basic execution."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        # Use AsyncIteratorMock to properly mock async iteration
        mock_claude_sdk.query.return_value = AsyncIteratorMock([mock_sdk_message])

        adapter = ClaudeAgentAdapter()
        await adapter.initialize()

        messages = []
        async for msg in adapter.execute("Hello"):
            messages.append(msg)

        assert len(messages) == 1
        assert messages[0].role == MessageRole.ASSISTANT
        assert messages[0].type == MessageType.FINAL
        assert messages[0].content == "Hello, I'm Claude!"
        assert adapter._current_session_id == "test-session-123"

    @pytest.mark.asyncio()
    async def test_execute_auto_initializes(
        self,
        mock_claude_sdk: ModuleType,
        mock_sdk_message: MagicMock,
    ) -> None:
        """Test execute auto-initializes if needed."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        mock_claude_sdk.query.return_value = AsyncIteratorMock([mock_sdk_message])

        adapter = ClaudeAgentAdapter()
        assert adapter.is_initialized is False

        messages = []
        async for msg in adapter.execute("Hello"):
            messages.append(msg)

        assert adapter.is_initialized is True
        assert len(messages) == 1

    @pytest.mark.asyncio()
    async def test_execute_with_context(
        self,
        mock_claude_sdk: ModuleType,
        mock_sdk_message: MagicMock,
    ) -> None:
        """Test execute with context injection."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        mock_claude_sdk.query.return_value = AsyncIteratorMock([mock_sdk_message])

        adapter = ClaudeAgentAdapter()
        context = AgentContext(
            request_id="test-req",
            collection_name="Test Collection",
            retrieved_chunks=[
                {"content": "Chunk 1 content", "score": 0.9, "source": "doc1.txt"},
            ],
        )

        messages = []
        async for msg in adapter.execute("Summarize this", context=context):
            messages.append(msg)

        # Verify query was called with injected context
        mock_claude_sdk.query.assert_called_once()
        call_kwargs = mock_claude_sdk.query.call_args
        prompt = call_kwargs.kwargs.get("prompt", call_kwargs.args[0] if call_kwargs.args else "")
        assert "<collection>Test Collection</collection>" in prompt
        assert "<retrieved_context>" in prompt
        assert "Chunk 1 content" in prompt

    @pytest.mark.asyncio()
    async def test_execute_interrupted(
        self,
        mock_claude_sdk: ModuleType,
    ) -> None:
        """Test execute respects interrupt."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        # Create messages that will be interrupted
        msg1 = MagicMock()
        msg1.subtype = None
        msg1.content = "First"
        msg1.is_partial = True
        msg1.session_id = None
        msg1.tool_use = None
        msg1.tool_result = None
        msg1.thinking = None

        msg2 = MagicMock()
        msg2.subtype = None
        msg2.content = "Second"
        msg2.is_partial = True
        msg2.session_id = None
        msg2.tool_use = None
        msg2.tool_result = None
        msg2.thinking = None

        mock_claude_sdk.query.return_value = AsyncIteratorMock([msg1, msg2])

        adapter = ClaudeAgentAdapter()
        await adapter.initialize()

        # Set interrupt after first message - should raise AgentInterruptedError
        messages: list[Any] = []

        async def collect_with_interrupt():
            async for msg in adapter.execute("Hello"):
                messages.append(msg)
                await adapter.interrupt()

        with pytest.raises(AgentInterruptedError):
            await collect_with_interrupt()

        # Should have received first message before interrupt was raised
        assert len(messages) == 1

    @pytest.mark.asyncio()
    async def test_execute_error_handling(
        self,
        mock_claude_sdk: ModuleType,
    ) -> None:
        """Test execute wraps SDK errors."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        class ErrorIterator:
            """Async iterator that raises an error."""

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise ValueError("SDK error")

        mock_claude_sdk.query.return_value = ErrorIterator()

        adapter = ClaudeAgentAdapter()
        await adapter.initialize()

        async def consume_generator():
            async for _ in adapter.execute("Hello"):
                pass

        with pytest.raises(AgentExecutionError, match="Claude execution failed"):
            await consume_generator()


class TestClaudeAgentAdapterMessageTranslation:
    """Tests for message translation."""

    @pytest.mark.asyncio()
    async def test_translate_text_message(self, mock_claude_sdk: ModuleType) -> None:
        """Test translating text message."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        msg = MagicMock()
        msg.subtype = None
        msg.content = "Hello!"
        msg.is_partial = False
        msg.session_id = None
        msg.tool_use = None
        msg.tool_result = None
        msg.thinking = None

        mock_claude_sdk.query.return_value = AsyncIteratorMock([msg])

        adapter = ClaudeAgentAdapter()
        messages = []
        async for m in adapter.execute("Hi"):
            messages.append(m)

        assert len(messages) == 1
        assert messages[0].role == MessageRole.ASSISTANT
        assert messages[0].type == MessageType.TEXT

    @pytest.mark.asyncio()
    async def test_translate_thinking_message(self, mock_claude_sdk: ModuleType) -> None:
        """Test translating thinking/reasoning message."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        msg = MagicMock()
        msg.subtype = None
        msg.content = ""
        msg.thinking = "Let me think about this..."
        msg.is_partial = False
        msg.session_id = None
        msg.tool_use = None
        msg.tool_result = None

        mock_claude_sdk.query.return_value = AsyncIteratorMock([msg])

        adapter = ClaudeAgentAdapter()
        messages = []
        async for m in adapter.execute("Think hard"):
            messages.append(m)

        assert len(messages) == 1
        assert messages[0].type == MessageType.THINKING
        assert messages[0].content == "Let me think about this..."

    @pytest.mark.asyncio()
    async def test_translate_tool_use_message(self, mock_claude_sdk: ModuleType) -> None:
        """Test translating tool use message."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        tool_use = MagicMock()
        tool_use.name = "search"
        tool_use.id = "call_123"
        tool_use.input = {"query": "test"}

        msg = MagicMock()
        msg.subtype = None
        msg.content = ""
        msg.tool_use = tool_use
        msg.tool_result = None
        msg.is_partial = False
        msg.session_id = None
        msg.thinking = None

        mock_claude_sdk.query.return_value = AsyncIteratorMock([msg])

        adapter = ClaudeAgentAdapter()
        messages = []
        async for m in adapter.execute("Search for something"):
            messages.append(m)

        assert len(messages) == 1
        assert messages[0].type == MessageType.TOOL_USE
        assert messages[0].role == MessageRole.TOOL_CALL
        assert messages[0].tool_name == "search"
        assert messages[0].tool_call_id == "call_123"
        assert messages[0].tool_input == {"query": "test"}

    @pytest.mark.asyncio()
    async def test_translate_error_message(self, mock_claude_sdk: ModuleType) -> None:
        """Test translating error message."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        msg = MagicMock()
        msg.subtype = "error"
        msg.error_message = "Something went wrong"
        msg.is_partial = False
        msg.session_id = None
        msg.tool_use = None
        msg.tool_result = None
        msg.thinking = None

        mock_claude_sdk.query.return_value = AsyncIteratorMock([msg])

        adapter = ClaudeAgentAdapter()
        messages = []
        async for m in adapter.execute("Fail"):
            messages.append(m)

        assert len(messages) == 1
        assert messages[0].type == MessageType.ERROR
        assert messages[0].role == MessageRole.ERROR


class TestClaudeAgentAdapterOptions:
    """Tests for option building."""

    def test_build_options_default_model(self, mock_claude_sdk: ModuleType) -> None:
        """Test building options uses default model."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        adapter = ClaudeAgentAdapter()
        adapter._build_options()  # Call to trigger ClaudeAgentOptions creation

        mock_claude_sdk.ClaudeAgentOptions.assert_called_once()
        call_kwargs = mock_claude_sdk.ClaudeAgentOptions.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    def test_build_options_custom_model(self, mock_claude_sdk: ModuleType) -> None:
        """Test building options with custom model."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        adapter = ClaudeAgentAdapter()
        adapter._build_options(model="claude-opus-4-20250514")

        call_kwargs = mock_claude_sdk.ClaudeAgentOptions.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-20250514"

    def test_build_options_with_system_prompt(self, mock_claude_sdk: ModuleType) -> None:
        """Test building options with system prompt."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        adapter = ClaudeAgentAdapter()
        adapter._build_options(system_prompt="You are a helpful assistant")

        call_kwargs = mock_claude_sdk.ClaudeAgentOptions.call_args.kwargs
        assert call_kwargs["system_prompt"] == "You are a helpful assistant"

    def test_build_options_with_session_id(self, mock_claude_sdk: ModuleType) -> None:
        """Test building options with session ID for resume."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        adapter = ClaudeAgentAdapter()
        adapter._build_options(session_id="existing-session")

        call_kwargs = mock_claude_sdk.ClaudeAgentOptions.call_args.kwargs
        assert call_kwargs["resume"] == "existing-session"

    def test_build_options_from_config(self, mock_claude_sdk: ModuleType) -> None:
        """Test building options from adapter config."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        adapter = ClaudeAgentAdapter(
            {
                "model": "claude-haiku-3-5-20241022",
                "system_prompt": "Be concise",
                "max_turns": 5,
                "permission_mode": "bypassPermissions",
            }
        )
        adapter._build_options()

        call_kwargs = mock_claude_sdk.ClaudeAgentOptions.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-3-5-20241022"
        assert call_kwargs["system_prompt"] == "Be concise"
        assert call_kwargs["max_turns"] == 5
        assert call_kwargs["permission_mode"] == "bypassPermissions"


class TestClaudeAgentAdapterContextInjection:
    """Tests for context injection."""

    def test_inject_context_none(self, mock_claude_sdk: ModuleType) -> None:
        """Test inject_context with no context."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        adapter = ClaudeAgentAdapter()
        result = adapter._inject_context("Hello", None)

        assert result == "Hello"

    def test_inject_context_with_collection(self, mock_claude_sdk: ModuleType) -> None:
        """Test inject_context with collection name."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        adapter = ClaudeAgentAdapter()
        context = AgentContext(
            request_id="test",
            collection_name="My Documents",
        )

        result = adapter._inject_context("Search for X", context)

        assert "<collection>My Documents</collection>" in result
        assert "Search for X" in result

    def test_inject_context_with_chunks(self, mock_claude_sdk: ModuleType) -> None:
        """Test inject_context with retrieved chunks."""
        from shared.agents.adapters.claude import ClaudeAgentAdapter

        adapter = ClaudeAgentAdapter()
        context = AgentContext(
            request_id="test",
            retrieved_chunks=[
                {"content": "Chunk 1", "score": 0.9, "source": "doc1.txt"},
                {"content": "Chunk 2", "score": 0.8, "source": "doc2.txt"},
            ],
        )

        result = adapter._inject_context("Question?", context)

        assert "<retrieved_context>" in result
        assert 'index="0"' in result
        assert 'score="0.9"' in result
        assert "Chunk 1" in result
        assert "Chunk 2" in result
        assert "</retrieved_context>" in result
        assert "Question?" in result
