"""Core type definitions for the agent plugin system.

This module defines the fundamental types used throughout the agent system:
- Enums for message roles, types, and use cases
- Dataclasses for token usage, messages, capabilities, and contexts
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.agents.types import AgentMessage as AgentMessageType


class MessageRole(str, Enum):
    """Role of the message sender in an agent conversation.

    Extends str to allow direct JSON serialization.
    """

    USER = "user"
    """Message from the user."""

    ASSISTANT = "assistant"
    """Message from the agent/assistant."""

    SYSTEM = "system"
    """System prompt or instruction."""

    TOOL_CALL = "tool_call"
    """Agent requesting to call a tool."""

    TOOL_RESULT = "tool_result"
    """Result returned from a tool execution."""

    ERROR = "error"
    """Error message."""


class MessageType(str, Enum):
    """Type of message content.

    Distinguishes between different kinds of content in agent responses.
    """

    TEXT = "text"
    """Plain text response."""

    THINKING = "thinking"
    """Extended thinking/reasoning (for models that support it)."""

    TOOL_USE = "tool_use"
    """Agent wants to use a tool."""

    TOOL_OUTPUT = "tool_output"
    """Result from tool execution."""

    PARTIAL = "partial"
    """Streaming partial content (incomplete)."""

    FINAL = "final"
    """Final response (complete)."""

    ERROR = "error"
    """Error information."""

    METADATA = "metadata"
    """Usage stats, costs, or other metadata."""


class AgentUseCase(str, Enum):
    """Use cases that agents can support.

    Agents declare which use cases they're suitable for,
    enabling intelligent agent selection based on task requirements.
    """

    # Search enhancement
    HYDE = "hyde"
    """Hypothetical Document Embeddings for query enhancement."""

    QUERY_EXPANSION = "query_expansion"
    """Generate alternative queries for improved recall."""

    QUERY_UNDERSTANDING = "query_understanding"
    """Parse intent, entities, and semantic meaning from queries."""

    # Result processing
    SUMMARIZATION = "summarization"
    """Compress and summarize retrieved content."""

    RERANKING = "reranking"
    """LLM-based reranking of search results."""

    ANSWER_SYNTHESIS = "answer_synthesis"
    """RAG answer generation from retrieved content."""

    # Agentic patterns
    TOOL_USE = "tool_use"
    """General tool-using agent capabilities."""

    AGENTIC_SEARCH = "agentic_search"
    """Multi-step retrieval with planning."""

    REASONING = "reasoning"
    """Chain-of-thought reasoning and planning."""

    # User-facing
    ASSISTANT = "assistant"
    """Conversational assistant interface."""

    # Specialized
    CODE_GENERATION = "code_generation"
    """Code writing and analysis."""

    DATA_ANALYSIS = "data_analysis"
    """Data exploration and analysis tasks."""


@dataclass(frozen=True)
class TokenUsage:
    """Token usage statistics for an agent execution.

    Tracks input, output, and specialized token counts for cost
    calculation and usage monitoring.

    Attributes:
        input_tokens: Tokens in the input prompt.
        output_tokens: Tokens in the generated output.
        cache_read_tokens: Tokens read from prompt cache.
        cache_write_tokens: Tokens written to prompt cache.
        reasoning_tokens: Tokens used for extended thinking/reasoning.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        """Serialize to dictionary for API responses and storage."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenUsage:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with token counts.

        Returns:
            TokenUsage instance.
        """
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            cache_write_tokens=data.get("cache_write_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens", 0),
        )


@dataclass(frozen=True)
class AgentMessage:
    """Unified message format for agent communication.

    Immutable to ensure message integrity across async boundaries.
    Supports all message types including text, tool calls, and errors.

    Attributes:
        id: Unique message identifier.
        role: Who sent the message (user, assistant, etc.).
        type: Type of content (text, tool_use, error, etc.).
        content: The message content.
        tool_name: Name of tool being called (for tool_use messages).
        tool_call_id: Unique ID for tool call/result correlation.
        tool_input: Input arguments for tool call.
        tool_output: Output from tool execution.
        timestamp: When the message was created.
        model: Model that generated the message (if applicable).
        usage: Token usage for this message.
        cost_usd: Estimated cost in USD.
        is_partial: Whether this is a streaming partial.
        sequence_number: Order in the response stream.
        error_code: Error code for error messages.
        error_details: Additional error context.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = MessageRole.ASSISTANT
    type: MessageType = MessageType.TEXT
    content: str = ""

    # Tool-related fields
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: dict[str, Any] | None = None

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    model: str | None = None
    usage: TokenUsage | None = None
    cost_usd: float | None = None

    # Streaming support
    is_partial: bool = False
    sequence_number: int = 0

    # Error details
    error_code: str | None = None
    error_details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API responses and storage.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "id": self.id,
            "role": self.role.value,
            "type": self.type.value,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "usage": self.usage.to_dict() if self.usage else None,
            "cost_usd": self.cost_usd,
            "is_partial": self.is_partial,
            "sequence_number": self.sequence_number,
            "error_code": self.error_code,
            "error_details": self.error_details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMessage:
        """Deserialize from storage or API.

        Args:
            data: Dictionary representation of a message.

        Returns:
            AgentMessage instance.
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(UTC)

        usage = data.get("usage")
        if usage is not None:
            usage = TokenUsage.from_dict(usage)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=MessageRole(data["role"]),
            type=MessageType(data["type"]),
            content=data.get("content", ""),
            tool_name=data.get("tool_name"),
            tool_call_id=data.get("tool_call_id"),
            tool_input=data.get("tool_input"),
            tool_output=data.get("tool_output"),
            timestamp=timestamp,
            model=data.get("model"),
            usage=usage,
            cost_usd=data.get("cost_usd"),
            is_partial=data.get("is_partial", False),
            sequence_number=data.get("sequence_number", 0),
            error_code=data.get("error_code"),
            error_details=data.get("error_details"),
        )


@dataclass(frozen=True)
class AgentCapabilities:
    """Declares capabilities of an agent implementation.

    Used for feature detection at runtime, UI adaptation,
    and graceful degradation across different SDK implementations.

    Attributes:
        supports_streaming: Can stream partial responses.
        supports_tools: Can use tools/functions.
        supports_parallel_tools: Can execute multiple tools concurrently.
        supports_sessions: Can maintain conversation history.
        supports_session_fork: Can branch conversation history.
        supports_interruption: Can cancel mid-execution.
        supports_extended_thinking: Supports reasoning/thinking mode.
        supports_thinking_budget: Can configure thinking token budget.
        supports_subagents: Can spawn child agents.
        supports_handoffs: Can hand off to other agents.
        max_context_tokens: Maximum input context size.
        max_output_tokens: Maximum output size.
        supported_models: List of supported model identifiers.
        default_model: Default model to use.
        max_tools: Maximum number of tools that can be registered.
    """

    # Execution capabilities
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_parallel_tools: bool = True
    supports_sessions: bool = True
    supports_session_fork: bool = False
    supports_interruption: bool = True

    # Thinking/reasoning
    supports_extended_thinking: bool = False
    supports_thinking_budget: bool = False

    # Multi-agent
    supports_subagents: bool = False
    supports_handoffs: bool = False

    # Context limits
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None

    # Model constraints
    supported_models: tuple[str, ...] = field(default_factory=tuple)
    default_model: str | None = None

    # Tool constraints
    max_tools: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API responses.

        Returns:
            Dictionary representation.
        """
        return {
            "supports_streaming": self.supports_streaming,
            "supports_tools": self.supports_tools,
            "supports_parallel_tools": self.supports_parallel_tools,
            "supports_sessions": self.supports_sessions,
            "supports_session_fork": self.supports_session_fork,
            "supports_interruption": self.supports_interruption,
            "supports_extended_thinking": self.supports_extended_thinking,
            "supports_thinking_budget": self.supports_thinking_budget,
            "supports_subagents": self.supports_subagents,
            "supports_handoffs": self.supports_handoffs,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "supported_models": list(self.supported_models),
            "default_model": self.default_model,
            "max_tools": self.max_tools,
        }


@dataclass(frozen=True)
class UseCaseRequirements:
    """Requirements for a specific agent use case.

    Helps match agents to tasks based on capabilities and constraints.

    Attributes:
        use_case: The use case these requirements are for.
        requires_streaming: Whether streaming is required.
        requires_tools: Whether tool use is required.
        requires_sessions: Whether session persistence is required.
        typical_context_tokens: Expected input context size.
        typical_output_tokens: Expected output size.
        latency_sensitive: Whether low latency is important.
        cost_sensitive: Whether cost optimization is important.
    """

    use_case: AgentUseCase
    requires_streaming: bool = False
    requires_tools: bool = False
    requires_sessions: bool = False
    typical_context_tokens: int = 4000
    typical_output_tokens: int = 1000
    latency_sensitive: bool = False
    cost_sensitive: bool = False


@dataclass
class AgentContext:
    """Runtime context for agent execution.

    Provides access to user/request context, collection/search context,
    and system configuration. This is mutable to allow updating during
    execution (e.g., setting session_id after creation).

    Attributes:
        request_id: Unique identifier for this request.
        user_id: ID of the user making the request.
        collection_id: Target collection for search operations.
        collection_name: Human-readable collection name.
        original_query: Original search query (for search use cases).
        retrieved_chunks: Previously retrieved content.
        session_id: Session ID for conversation continuity.
        conversation_history: Previous messages in this session.
        available_tools: Names of tools available for this execution.
        tool_configs: Per-tool configuration overrides.
        max_tokens: Maximum output tokens for this execution.
        timeout_seconds: Execution timeout.
        trace_id: Distributed tracing ID.
        parent_span_id: Parent span for tracing.
    """

    # Request context
    request_id: str
    user_id: str | None = None

    # Collection context (for search-related use cases)
    collection_id: str | None = None
    collection_name: str | None = None

    # Search context
    original_query: str | None = None
    retrieved_chunks: list[dict[str, Any]] | None = None

    # Session context
    session_id: str | None = None
    conversation_history: list[AgentMessageType] | None = None

    # Tool context
    available_tools: list[str] | None = None
    tool_configs: dict[str, dict[str, Any]] | None = None

    # Execution constraints
    max_tokens: int | None = None
    timeout_seconds: float | None = None

    # Metadata for logging/tracing
    trace_id: str | None = None
    parent_span_id: str | None = None
