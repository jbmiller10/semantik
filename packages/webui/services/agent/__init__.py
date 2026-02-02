"""Agent service for conversational pipeline building.

.. deprecated::
    This module is deprecated and will be removed in a future release.
    Use the new assisted_flow module instead:
    ``webui.services.assisted_flow``

    The assisted_flow module uses Claude Agent SDK for improved reliability,
    reduced code complexity, and better streaming support.

This module provides the core data models and exceptions for the agentic pipeline builder:
- AgentConversation: Persistent conversation state model
- ConversationUncertainty: Model for tracking uncertainties in conversations
- ConversationStatus: Enum for conversation lifecycle states
- UncertaintySeverity: Enum for uncertainty severity levels
- Agent exceptions: AgentError, SubAgentFailedError, ConversationNotActiveError, BlockingUncertaintyError
"""

import warnings

warnings.warn(
    "webui.services.agent is deprecated. Use webui.services.assisted_flow instead.",
    DeprecationWarning,
    stacklevel=2,
)

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
