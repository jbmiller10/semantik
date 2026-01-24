"""Agent service for conversational pipeline building.

This module provides the core components for the agentic pipeline builder:
- AgentConversation: Persistent conversation state
- AgentOrchestrator: Main agent that handles conversation flow
- SubAgent: Base class for specialized sub-agents
- MessageStore: Redis-based message persistence
"""

from webui.services.agent.exceptions import (
    AgentError,
    BlockingUncertaintyError,
    ConversationNotActiveError,
    SubAgentFailedError,
)
from webui.services.agent.models import (
    AgentConversation,
    ConversationStatus,
    ConversationUncertainty,
    UncertaintySeverity,
)

__all__ = [
    # Models
    "AgentConversation",
    "ConversationUncertainty",
    "ConversationStatus",
    "UncertaintySeverity",
    # Exceptions
    "AgentError",
    "SubAgentFailedError",
    "ConversationNotActiveError",
    "BlockingUncertaintyError",
]
