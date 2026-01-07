"""Agent-specific exceptions.

This module defines the exception hierarchy for the agent plugin system.
All exceptions provide structured error information for API responses.

Exception Hierarchy:
    AgentError (base)
    ├── AgentInitializationError
    ├── AgentExecutionError
    ├── AgentTimeoutError
    ├── AgentInterruptedError
    ├── ToolError
    │   ├── ToolNotFoundError
    │   ├── ToolDisabledError
    │   └── ToolExecutionError
    └── SessionError
        ├── SessionNotFoundError
        └── SessionExpiredError
"""

from __future__ import annotations

import re
from typing import Any


def _class_name_to_error_code(class_name: str) -> str:
    """Convert CamelCase class name to SCREAMING_SNAKE_CASE error code.

    Example: AgentExecutionError -> AGENT_EXECUTION_ERROR
    """
    # Insert underscore before uppercase letters (except at start)
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", class_name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).upper()


class AgentError(Exception):
    """Base error for all agent-related issues.

    All agent exceptions inherit from this class and provide
    structured error information for API responses.

    Attributes:
        message: Human-readable error message.
        error_code: Machine-readable error code (auto-generated from class name).
        details: Additional structured context about the error.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        **details: Any,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable error message.
            error_code: Override for error code (defaults to class name conversion).
            **details: Additional context stored in details dict.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or _class_name_to_error_code(self.__class__.__name__)
        self.details: dict[str, Any] = details

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API responses.

        Returns:
            Dictionary with error information.
        """
        result: dict[str, Any] = {
            "error": self.error_code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result

    def __reduce__(self) -> tuple[type, tuple[str], dict[str, Any]]:
        """Support pickling for Celery task compatibility.

        Returns minimal constructor args and stores all state in __dict__.
        """
        return (
            self.__class__,
            (self.message,),
            self.__dict__.copy(),
        )

    def __setstate__(self, state: dict[str, Any] | None) -> None:
        """Restore state after unpickling."""
        if state:
            self.__dict__.update(state)


class AgentInitializationError(AgentError):
    """Raised when an agent or adapter fails to initialize.

    This typically occurs when:
    - SDK is not installed or unavailable
    - API credentials are invalid
    - Required dependencies are missing

    Attributes:
        adapter: Name of the adapter that failed to initialize.
    """

    def __init__(
        self,
        message: str,
        adapter: str | None = None,
        **details: Any,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable error message.
            adapter: Name of the adapter that failed.
            **details: Additional context.
        """
        super().__init__(message, **details)
        self.adapter = adapter
        if adapter:
            self.details["adapter"] = adapter


class AgentExecutionError(AgentError):
    """Raised when agent execution fails.

    This covers general execution failures that are not
    timeouts, interruptions, or tool-specific errors.

    Attributes:
        adapter: Name of the adapter that was executing.
        cause: Original error message or description.
    """

    def __init__(
        self,
        message: str,
        adapter: str | None = None,
        cause: str | None = None,
        **details: Any,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable error message.
            adapter: Name of the adapter that was executing.
            cause: Original error message.
            **details: Additional context.
        """
        super().__init__(message, **details)
        self.adapter = adapter
        self.cause = cause
        if adapter:
            self.details["adapter"] = adapter
        if cause:
            self.details["cause"] = cause


class AgentTimeoutError(AgentError):
    """Raised when agent execution exceeds the timeout.

    Attributes:
        timeout_seconds: The timeout that was exceeded.
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        **details: Any,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable error message.
            timeout_seconds: The timeout that was exceeded.
            **details: Additional context.
        """
        super().__init__(message, **details)
        self.timeout_seconds = timeout_seconds
        if timeout_seconds is not None:
            self.details["timeout_seconds"] = timeout_seconds


class AgentInterruptedError(AgentError):
    """Raised when agent execution is interrupted by the user.

    This is not an error per se, but signals that execution
    was cancelled at the user's request.
    """


class ToolError(AgentError):
    """Base error for tool-related issues.

    Attributes:
        tool_name: Name of the tool that caused the error.
    """

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        **details: Any,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable error message.
            tool_name: Name of the tool that caused the error.
            **details: Additional context.
        """
        super().__init__(message, **details)
        self.tool_name = tool_name
        if tool_name:
            self.details["tool_name"] = tool_name


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered.

    This occurs when:
    - Tool name is misspelled
    - Tool was never registered
    - Tool was unregistered
    """


class ToolDisabledError(ToolError):
    """Raised when a requested tool exists but is disabled.

    Tools can be disabled through configuration without
    being completely unregistered.
    """


class ToolExecutionError(ToolError):
    """Raised when tool execution fails.

    This covers errors that occur during tool invocation,
    such as invalid arguments or runtime failures.

    Attributes:
        cause: Original error message from the tool.
    """

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        cause: str | None = None,
        **details: Any,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable error message.
            tool_name: Name of the tool that failed.
            cause: Original error message from the tool.
            **details: Additional context.
        """
        super().__init__(message, tool_name=tool_name, **details)
        self.cause = cause
        if cause:
            self.details["cause"] = cause


class SessionError(AgentError):
    """Base error for session-related issues.

    Attributes:
        session_id: ID of the session that caused the error.
    """

    def __init__(
        self,
        message: str,
        session_id: str | None = None,
        **details: Any,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable error message.
            session_id: ID of the session that caused the error.
            **details: Additional context.
        """
        super().__init__(message, **details)
        self.session_id = session_id
        if session_id:
            self.details["session_id"] = session_id


class SessionNotFoundError(SessionError):
    """Raised when a session cannot be found.

    This occurs when:
    - Session ID is invalid
    - Session was deleted
    - Session never existed
    """


class SessionExpiredError(SessionError):
    """Raised when a session has expired.

    Sessions may expire due to:
    - Inactivity timeout
    - Maximum age limit
    - Manual archival
    """
