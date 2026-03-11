"""Assisted flow service for pipeline configuration.

This module provides Claude Agent SDK-powered pipeline configuration assistant.
It replaces the older webui.services.agent module with a cleaner, more maintainable
implementation using the SDK's built-in agent loop and tool handling.

Key components:
- context.ToolContext: Shared context for all tools during a session
- server.create_mcp_server: Creates MCP server with pipeline tools
- sdk_service: Session lifecycle management (create, send, get_client, close)
- session_manager: In-memory TTL-based client storage
- callbacks: Question management and can_use_tool permission callback
- source_stats: Pre-session source statistics gathering
- subagents: Explorer and Validator subagent definitions
- prompts: System prompts and initial prompt builder

API endpoints are at /api/v2/assisted-flow/.
"""

from webui.services.assisted_flow.callbacks import (
    PendingQuestion,
    QuestionManager,
    compute_question_id,
    create_can_use_tool,
    get_question_manager,
)
from webui.services.assisted_flow.context import ToolContext
from webui.services.assisted_flow.prompts import SYSTEM_PROMPT, build_initial_prompt
from webui.services.assisted_flow.sdk_service import (
    SDKNotAvailableError,
    SDKServiceError,
    SDKSessionError,
    close_session,
    create_sdk_session,
    get_session_client,
    send_message,
)
from webui.services.assisted_flow.server import create_mcp_server
from webui.services.assisted_flow.session_manager import SessionManager, session_manager
from webui.services.assisted_flow.subagents import (
    EXPLORER_AGENT,
    VALIDATOR_AGENT,
    get_subagents,
)

__all__ = [
    # Callbacks
    "PendingQuestion",
    "QuestionManager",
    "create_can_use_tool",
    "compute_question_id",
    "get_question_manager",
    # Context
    "ToolContext",
    # Server
    "create_mcp_server",
    # Session management
    "SessionManager",
    "session_manager",
    # SDK service
    "create_sdk_session",
    "get_session_client",
    "send_message",
    "close_session",
    "SDKServiceError",
    "SDKNotAvailableError",
    "SDKSessionError",
    # Subagents
    "EXPLORER_AGENT",
    "VALIDATOR_AGENT",
    "get_subagents",
    # Prompts
    "SYSTEM_PROMPT",
    "build_initial_prompt",
]
