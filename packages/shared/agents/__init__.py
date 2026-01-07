"""Agent plugin system core types and exceptions.

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
