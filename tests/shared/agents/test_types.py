"""Tests for agent type definitions."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from shared.agents.types import (
    AgentCapabilities,
    AgentContext,
    AgentMessage,
    AgentUseCase,
    MessageRole,
    MessageType,
    TokenUsage,
    UseCaseRequirements,
)


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_all_values_exist(self) -> None:
        """Test that all expected values exist."""
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.TOOL_CALL == "tool_call"
        assert MessageRole.TOOL_RESULT == "tool_result"
        assert MessageRole.ERROR == "error"

    def test_is_string_enum(self) -> None:
        """Test that values are strings for JSON serialization."""
        assert isinstance(MessageRole.USER.value, str)
        assert str(MessageRole.USER) == "MessageRole.USER"

    def test_from_string(self) -> None:
        """Test creating from string value."""
        role = MessageRole("user")
        assert role == MessageRole.USER


class TestMessageType:
    """Tests for MessageType enum."""

    def test_all_values_exist(self) -> None:
        """Test that all expected values exist."""
        assert MessageType.TEXT == "text"
        assert MessageType.THINKING == "thinking"
        assert MessageType.TOOL_USE == "tool_use"
        assert MessageType.TOOL_OUTPUT == "tool_output"
        assert MessageType.PARTIAL == "partial"
        assert MessageType.FINAL == "final"
        assert MessageType.ERROR == "error"
        assert MessageType.METADATA == "metadata"

    def test_is_string_enum(self) -> None:
        """Test that values are strings for JSON serialization."""
        assert isinstance(MessageType.TEXT.value, str)


class TestAgentUseCase:
    """Tests for AgentUseCase enum."""

    def test_search_enhancement_cases(self) -> None:
        """Test search enhancement use cases exist."""
        assert AgentUseCase.HYDE == "hyde"
        assert AgentUseCase.QUERY_EXPANSION == "query_expansion"
        assert AgentUseCase.QUERY_UNDERSTANDING == "query_understanding"

    def test_result_processing_cases(self) -> None:
        """Test result processing use cases exist."""
        assert AgentUseCase.SUMMARIZATION == "summarization"
        assert AgentUseCase.RERANKING == "reranking"
        assert AgentUseCase.ANSWER_SYNTHESIS == "answer_synthesis"

    def test_agentic_cases(self) -> None:
        """Test agentic use cases exist."""
        assert AgentUseCase.TOOL_USE == "tool_use"
        assert AgentUseCase.AGENTIC_SEARCH == "agentic_search"
        assert AgentUseCase.REASONING == "reasoning"

    def test_user_facing_cases(self) -> None:
        """Test user-facing use cases exist."""
        assert AgentUseCase.ASSISTANT == "assistant"

    def test_specialized_cases(self) -> None:
        """Test specialized use cases exist."""
        assert AgentUseCase.CODE_GENERATION == "code_generation"
        assert AgentUseCase.DATA_ANALYSIS == "data_analysis"


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_defaults(self) -> None:
        """Test default values are zero."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_read_tokens == 0
        assert usage.cache_write_tokens == 0
        assert usage.reasoning_tokens == 0

    def test_total_tokens_property(self) -> None:
        """Test total_tokens calculation."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=20,
            cache_write_tokens=10,
            reasoning_tokens=30,
        )
        data = usage.to_dict()

        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50
        assert data["cache_read_tokens"] == 20
        assert data["cache_write_tokens"] == 10
        assert data["reasoning_tokens"] == 30
        assert data["total_tokens"] == 150

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": 20,
        }
        usage = TokenUsage.from_dict(data)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_read_tokens == 20
        assert usage.cache_write_tokens == 0  # default
        assert usage.reasoning_tokens == 0  # default

    def test_from_dict_empty(self) -> None:
        """Test deserialization from empty dict uses defaults."""
        usage = TokenUsage.from_dict({})
        assert usage.input_tokens == 0
        assert usage.total_tokens == 0

    def test_immutable(self) -> None:
        """Test that TokenUsage is frozen (immutable)."""
        usage = TokenUsage(input_tokens=100)
        with pytest.raises(FrozenInstanceError):
            usage.input_tokens = 200  # type: ignore[misc]

    def test_round_trip_serialization(self) -> None:
        """Test serialization round-trip."""
        original = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=20,
            cache_write_tokens=10,
            reasoning_tokens=30,
        )
        data = original.to_dict()
        restored = TokenUsage.from_dict(data)

        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.cache_read_tokens == original.cache_read_tokens
        assert restored.cache_write_tokens == original.cache_write_tokens
        assert restored.reasoning_tokens == original.reasoning_tokens


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        msg = AgentMessage()

        assert msg.id is not None  # UUID generated
        assert msg.role == MessageRole.ASSISTANT
        assert msg.type == MessageType.TEXT
        assert msg.content == ""
        assert msg.tool_name is None
        assert msg.is_partial is False
        assert msg.sequence_number == 0

    def test_with_content(self) -> None:
        """Test creating message with content."""
        msg = AgentMessage(
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Hello, world!",
        )

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"

    def test_with_tool_data(self) -> None:
        """Test message with tool call data."""
        msg = AgentMessage(
            role=MessageRole.TOOL_CALL,
            type=MessageType.TOOL_USE,
            tool_name="search",
            tool_call_id="call_123",
            tool_input={"query": "test"},
        )

        assert msg.tool_name == "search"
        assert msg.tool_call_id == "call_123"
        assert msg.tool_input == {"query": "test"}

    def test_with_usage(self) -> None:
        """Test message with token usage."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        msg = AgentMessage(usage=usage, cost_usd=0.01)

        assert msg.usage is not None
        assert msg.usage.total_tokens == 150
        assert msg.cost_usd == 0.01

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        usage = TokenUsage(input_tokens=100)
        msg = AgentMessage(
            id="test-id",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Hello",
            model="claude-3",
            usage=usage,
        )
        data = msg.to_dict()

        assert data["id"] == "test-id"
        assert data["role"] == "assistant"
        assert data["type"] == "text"
        assert data["content"] == "Hello"
        assert data["model"] == "claude-3"
        assert data["usage"]["input_tokens"] == 100

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "id": "test-id",
            "role": "user",
            "type": "text",
            "content": "Hello",
            "timestamp": "2024-01-01T00:00:00+00:00",
        }
        msg = AgentMessage.from_dict(data)

        assert msg.id == "test-id"
        assert msg.role == MessageRole.USER
        assert msg.type == MessageType.TEXT
        assert msg.content == "Hello"

    def test_from_dict_with_usage(self) -> None:
        """Test deserialization with nested usage."""
        data = {
            "role": "assistant",
            "type": "text",
            "content": "Hi",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        msg = AgentMessage.from_dict(data)

        assert msg.usage is not None
        assert msg.usage.input_tokens == 100
        assert msg.usage.output_tokens == 50

    def test_immutable(self) -> None:
        """Test that AgentMessage is frozen."""
        msg = AgentMessage(content="Hello")
        with pytest.raises(FrozenInstanceError):
            msg.content = "World"  # type: ignore[misc]

    def test_round_trip_serialization(self) -> None:
        """Test serialization round-trip."""
        original = AgentMessage(
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Hello",
            tool_name="search",
            tool_call_id="call_123",
            tool_input={"query": "test"},
            model="claude-3",
            usage=TokenUsage(input_tokens=100, output_tokens=50),
            cost_usd=0.01,
            is_partial=True,
            sequence_number=5,
            error_code="TEST_ERROR",
            error_details={"foo": "bar"},
        )
        data = original.to_dict()
        restored = AgentMessage.from_dict(data)

        assert restored.role == original.role
        assert restored.type == original.type
        assert restored.content == original.content
        assert restored.tool_name == original.tool_name
        assert restored.tool_call_id == original.tool_call_id
        assert restored.tool_input == original.tool_input
        assert restored.model == original.model
        assert restored.usage is not None
        assert restored.usage.input_tokens == 100
        assert restored.cost_usd == original.cost_usd
        assert restored.is_partial == original.is_partial
        assert restored.sequence_number == original.sequence_number
        assert restored.error_code == original.error_code
        assert restored.error_details == original.error_details

    def test_timestamp_default(self) -> None:
        """Test timestamp is set to current time by default."""
        before = datetime.now(UTC)
        msg = AgentMessage()
        after = datetime.now(UTC)

        assert before <= msg.timestamp <= after


class TestAgentCapabilities:
    """Tests for AgentCapabilities dataclass."""

    def test_defaults(self) -> None:
        """Test sensible defaults."""
        caps = AgentCapabilities()

        assert caps.supports_streaming is True
        assert caps.supports_tools is True
        assert caps.supports_parallel_tools is True
        assert caps.supports_sessions is True
        assert caps.supports_session_fork is False
        assert caps.supports_interruption is True
        assert caps.supports_extended_thinking is False
        assert caps.supports_thinking_budget is False
        assert caps.supports_subagents is False
        assert caps.supports_handoffs is False
        assert caps.max_context_tokens is None
        assert caps.max_output_tokens is None
        assert caps.supported_models == ()
        assert caps.default_model is None
        assert caps.max_tools is None

    def test_with_models(self) -> None:
        """Test with model list."""
        caps = AgentCapabilities(
            supported_models=("claude-3", "claude-3.5"),
            default_model="claude-3",
        )

        assert "claude-3" in caps.supported_models
        assert caps.default_model == "claude-3"

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        caps = AgentCapabilities(
            supports_streaming=True,
            max_context_tokens=200000,
            supported_models=("claude-3",),
            default_model="claude-3",
        )
        data = caps.to_dict()

        assert data["supports_streaming"] is True
        assert data["max_context_tokens"] == 200000
        assert data["supported_models"] == ["claude-3"]  # tuple -> list
        assert data["default_model"] == "claude-3"

    def test_immutable(self) -> None:
        """Test that AgentCapabilities is frozen."""
        caps = AgentCapabilities()
        with pytest.raises(FrozenInstanceError):
            caps.supports_streaming = False  # type: ignore[misc]


class TestUseCaseRequirements:
    """Tests for UseCaseRequirements dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        reqs = UseCaseRequirements(use_case=AgentUseCase.ASSISTANT)

        assert reqs.use_case == AgentUseCase.ASSISTANT
        assert reqs.requires_streaming is False
        assert reqs.requires_tools is False
        assert reqs.requires_sessions is False
        assert reqs.typical_context_tokens == 4000
        assert reqs.typical_output_tokens == 1000
        assert reqs.latency_sensitive is False
        assert reqs.cost_sensitive is False

    def test_with_requirements(self) -> None:
        """Test with specific requirements."""
        reqs = UseCaseRequirements(
            use_case=AgentUseCase.AGENTIC_SEARCH,
            requires_streaming=True,
            requires_tools=True,
            requires_sessions=True,
            typical_context_tokens=50000,
            typical_output_tokens=2000,
        )

        assert reqs.requires_streaming is True
        assert reqs.requires_tools is True
        assert reqs.typical_context_tokens == 50000

    def test_immutable(self) -> None:
        """Test that UseCaseRequirements is frozen."""
        reqs = UseCaseRequirements(use_case=AgentUseCase.ASSISTANT)
        with pytest.raises(FrozenInstanceError):
            reqs.requires_streaming = True  # type: ignore[misc]


class TestAgentContext:
    """Tests for AgentContext dataclass."""

    def test_required_field(self) -> None:
        """Test request_id is required."""
        ctx = AgentContext(request_id="req-123")
        assert ctx.request_id == "req-123"

    def test_defaults(self) -> None:
        """Test default values are None."""
        ctx = AgentContext(request_id="req-123")

        assert ctx.user_id is None
        assert ctx.collection_id is None
        assert ctx.collection_name is None
        assert ctx.original_query is None
        assert ctx.retrieved_chunks is None
        assert ctx.session_id is None
        assert ctx.conversation_history is None
        assert ctx.available_tools is None
        assert ctx.tool_configs is None
        assert ctx.max_tokens is None
        assert ctx.timeout_seconds is None
        assert ctx.trace_id is None
        assert ctx.parent_span_id is None

    def test_with_all_fields(self) -> None:
        """Test with all fields populated."""
        ctx = AgentContext(
            request_id="req-123",
            user_id="user-456",
            collection_id="col-789",
            collection_name="My Collection",
            original_query="test query",
            retrieved_chunks=[{"content": "chunk1"}],
            session_id="sess-abc",
            available_tools=["search", "retrieve"],
            tool_configs={"search": {"top_k": 10}},
            max_tokens=1000,
            timeout_seconds=30.0,
            trace_id="trace-xyz",
            parent_span_id="span-123",
        )

        assert ctx.user_id == "user-456"
        assert ctx.collection_id == "col-789"
        assert ctx.available_tools == ["search", "retrieve"]

    def test_mutable(self) -> None:
        """Test that AgentContext is mutable (not frozen)."""
        ctx = AgentContext(request_id="req-123")

        # Should be able to modify
        ctx.session_id = "new-session"
        assert ctx.session_id == "new-session"

        ctx.user_id = "user-456"
        assert ctx.user_id == "user-456"
