"""Agent service for conversational pipeline building.

This module provides the core data models and exceptions for the agentic pipeline builder:
- AgentConversation: Persistent conversation state model
- ConversationUncertainty: Model for tracking uncertainties in conversations
- ConversationStatus: Enum for conversation lifecycle states
- UncertaintySeverity: Enum for uncertainty severity levels
- Agent exceptions: AgentError, SubAgentFailedError, ConversationNotActiveError, BlockingUncertaintyError
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
