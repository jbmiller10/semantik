"""Agent plugin system core types, exceptions, and adapters.

This package provides the foundational types for the agent plugin system:

Types:
    - MessageRole: Role of the message sender (user, assistant, etc.)
    - MessageType: Type of message content (text, tool_use, etc.)
    - AgentUseCase: Use cases agents can support (assistant, hyde, etc.)
    - TokenUsage: Token usage statistics
    - AgentMessage: Unified message format
    - AgentCapabilities: Agent capability declarations
    - UseCaseRequirements: Requirements for specific use cases
    - AgentContext: Runtime execution context

Base Classes:
    - AgentAdapter: SDK-agnostic abstract base for agent adapters

Tools:
    - AgentTool: Abstract base for agent-accessible tools
    - ToolDefinition: Tool metadata and parameter schema
    - ToolParameter: Individual parameter definition
    - ToolRegistry: Central registry for tool management
    - ToolRecord: Registry entry for a registered tool

Exceptions:
    - AgentError: Base exception for all agent errors
    - AgentInitializationError: Initialization failures
    - AgentExecutionError: Execution failures
    - AgentTimeoutError: Timeout errors
    - AgentInterruptedError: User interruption
    - ToolError: Base for tool errors
    - ToolNotFoundError: Tool not registered
    - ToolDisabledError: Tool is disabled
    - ToolExecutionError: Tool execution failed
    - SessionError: Base for session errors
    - SessionNotFoundError: Session not found
    - SessionExpiredError: Session expired

Example:
    >>> from shared.agents import AgentMessage, MessageRole, MessageType
    >>> msg = AgentMessage(
    ...     role=MessageRole.ASSISTANT,
    ...     type=MessageType.TEXT,
    ...     content="Hello, how can I help?"
    ... )
    >>> msg.to_dict()
    {'id': '...', 'role': 'assistant', 'type': 'text', ...}
"""

from shared.agents.base import AgentAdapter
from shared.agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentInitializationError,
    AgentInterruptedError,
    AgentTimeoutError,
    SessionError,
    SessionExpiredError,
    SessionNotFoundError,
    ToolDisabledError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
)
from shared.agents.metrics import (
    AGENT_ACTIVE_SESSIONS,
    AGENT_ERRORS_TOTAL,
    AGENT_EXECUTION_DURATION,
    AGENT_EXECUTIONS_TOTAL,
    AGENT_SESSIONS_CREATED_TOTAL,
    AGENT_TOKENS_TOTAL,
    AGENT_TOOL_CALLS_TOTAL,
    AGENT_TOOL_DURATION,
    record_error,
    record_execution,
    record_session_created,
    record_tokens,
    record_tool_call,
    timed_execution,
    timed_tool_call,
    update_active_sessions,
)
from shared.agents.tools import (
    AgentTool,
    ToolDefinition,
    ToolParameter,
    ToolRecord,
    ToolRegistry,
    get_tool_registry,
    reset_tool_registry,
)
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

__all__ = [
    # Types - Enums
    "MessageRole",
    "MessageType",
    "AgentUseCase",
    # Types - Dataclasses
    "TokenUsage",
    "AgentMessage",
    "AgentCapabilities",
    "UseCaseRequirements",
    "AgentContext",
    # Base Classes
    "AgentAdapter",
    # Tools
    "AgentTool",
    "ToolDefinition",
    "ToolParameter",
    "ToolRecord",
    "ToolRegistry",
    "get_tool_registry",
    "reset_tool_registry",
    # Metrics
    "AGENT_EXECUTIONS_TOTAL",
    "AGENT_EXECUTION_DURATION",
    "AGENT_TOKENS_TOTAL",
    "AGENT_TOOL_CALLS_TOTAL",
    "AGENT_TOOL_DURATION",
    "AGENT_ACTIVE_SESSIONS",
    "AGENT_SESSIONS_CREATED_TOTAL",
    "AGENT_ERRORS_TOTAL",
    "record_execution",
    "record_tokens",
    "record_tool_call",
    "record_session_created",
    "record_error",
    "update_active_sessions",
    "timed_execution",
    "timed_tool_call",
    # Exceptions - Base
    "AgentError",
    # Exceptions - Execution
    "AgentInitializationError",
    "AgentExecutionError",
    "AgentTimeoutError",
    "AgentInterruptedError",
    # Exceptions - Tools
    "ToolError",
    "ToolNotFoundError",
    "ToolDisabledError",
    "ToolExecutionError",
    # Exceptions - Sessions
    "SessionError",
    "SessionNotFoundError",
    "SessionExpiredError",
]
