"""Exception hierarchy for the agent service.

This module defines all exceptions that can be raised by the agent service,
organized by layer (orchestrator, sub-agent, API).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from webui.services.agent.subagents.base import Uncertainty


class AgentError(Exception):
    """Base exception for all agent-related errors."""



class LLMNotConfiguredError(AgentError):
    """User hasn't configured an LLM provider for the required tier.

    Raised when attempting to start a conversation without having
    configured an LLM provider in settings.
    """

    def __init__(self, message: str = "LLM provider not configured"):
        super().__init__(message)


class SubAgentFailedError(AgentError):
    """Sub-agent failed to complete its task.

    This exception captures partial results so the orchestrator can
    decide how to handle the failure.

    Attributes:
        agent_id: Identifier of the sub-agent that failed
        reason: Human-readable explanation of the failure
        partial_result: Any partial data gathered before failure
    """

    def __init__(
        self,
        agent_id: str,
        reason: str,
        partial_result: dict[str, Any] | None = None,
    ):
        self.agent_id = agent_id
        self.reason = reason
        self.partial_result = partial_result
        super().__init__(f"Sub-agent {agent_id} failed: {reason}")


class ConversationNotActiveError(AgentError):
    """Cannot perform operation on a non-active conversation.

    Raised when attempting to send messages to a conversation that
    has already been applied or abandoned.
    """

    def __init__(self, conversation_id: str, status: str):
        self.conversation_id = conversation_id
        self.status = status
        super().__init__(
            f"Conversation {conversation_id} is not active (status: {status})"
        )


class BlockingUncertaintyError(AgentError):
    """Cannot proceed due to unresolved blocking issues.

    Raised when attempting to apply a pipeline while there are
    unresolved blocking uncertainties.

    Attributes:
        uncertainties: List of blocking uncertainties that must be resolved
    """

    def __init__(self, uncertainties: list[Uncertainty]):
        self.uncertainties = uncertainties
        messages = [u.message for u in uncertainties]
        super().__init__(
            f"Cannot proceed with {len(uncertainties)} blocking issues: {messages}"
        )


class ToolExecutionError(AgentError):
    """A tool failed to execute.

    Raised when a tool encounters an error during execution.
    The error is returned to the LLM so it can decide how to handle it.
    """

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool {tool_name} failed: {message}")


class MessageStoreError(AgentError):
    """Error interacting with the message store (Redis).

    Raised when there's a problem storing or retrieving messages
    from the Redis message store.
    """

