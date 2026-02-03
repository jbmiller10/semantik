"""SDK service for assisted flow sessions.

This module provides a service layer for creating and managing
ClaudeSDKClient sessions for the pipeline configuration assistant.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, cast

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    CLINotFoundError,
    ProcessError,
)

from webui.services.assisted_flow.callbacks import can_use_tool
from webui.services.assisted_flow.context import ToolContext
from webui.services.assisted_flow.prompts import SYSTEM_PROMPT
from webui.services.assisted_flow.server import create_mcp_server
from webui.services.assisted_flow.session_manager import session_manager
from webui.services.assisted_flow.subagents import get_subagents

logger = logging.getLogger(__name__)


class SDKServiceError(Exception):
    """Base exception for SDK service errors."""


class SDKNotAvailableError(SDKServiceError):
    """Raised when Claude Code CLI is not installed."""


class SDKSessionError(SDKServiceError):
    """Raised when SDK session creation or operation fails."""


def _build_system_prompt(source_stats: dict[str, Any]) -> str:
    """Build a per-session system prompt with non-sensitive source context."""
    safe_stats = dict(source_stats)
    safe_stats.pop("secrets", None)
    return f"{SYSTEM_PROMPT}\n\n## Source Context (non-sensitive)\n{safe_stats}\n"


async def create_sdk_session(
    user_id: int,
    source_id: int | None,
    source_stats: dict[str, Any],
) -> tuple[str, ClaudeSDKClient]:
    """Create a new SDK session for assisted flow.

    Args:
        user_id: Authenticated user's ID
        source_id: Collection source ID being configured (None for inline sources)
        source_stats: Source statistics for initial prompt

    Returns:
        Tuple of (session_id, client)

    Raises:
        SDKNotAvailableError: If Claude Code CLI is not installed
        SDKSessionError: If session creation fails
    """
    # Generate unique session ID
    session_id = f"af_{uuid.uuid4().hex[:16]}"

    # Create tool context
    ctx = ToolContext(
        user_id=user_id,
        source_id=source_id,
    )

    # Create MCP server with tools
    mcp_server = create_mcp_server(ctx)

    try:
        # Create SDK options
        # SECURITY: Use allowed_tools whitelist to restrict agent to only our MCP
        # tools plus AskUserQuestion. Setting tools=[] alone does NOT disable
        # default Claude Code tools (bash/edit/etc).
        options = ClaudeAgentOptions(
            system_prompt=_build_system_prompt(source_stats),
            allowed_tools=[
                "mcp__assisted-flow__list_plugins",
                "mcp__assisted-flow__get_plugin_details",
                "mcp__assisted-flow__build_pipeline",
                "mcp__assisted-flow__apply_pipeline",
                "mcp__assisted-flow__sample_files",
                "mcp__assisted-flow__preview_content",
                "mcp__assisted-flow__detect_patterns",
                "mcp__assisted-flow__validate_pipeline",
                "AskUserQuestion",
            ],
            permission_mode="default",
            mcp_servers={"assisted-flow": mcp_server},
            agents=get_subagents(),
            include_partial_messages=True,
            can_use_tool=can_use_tool,
        )

        # Create client
        client = ClaudeSDKClient(options=options)

        # Connect in streaming mode; do NOT pass a string prompt here, which
        # triggers one-shot CLI mode and exits.
        await client.connect()

        # Store in session manager
        await session_manager.store_client(session_id, client, user_id=user_id)

        logger.info(f"Created SDK session {session_id} for source {source_id}")
        return session_id, client

    except CLINotFoundError as e:
        logger.error("Claude Code CLI not installed", exc_info=True)
        raise SDKNotAvailableError(
            "Claude Code CLI is not installed. Please install it to use the assisted flow."
        ) from e

    except ProcessError as e:
        logger.error(f"SDK process failed: exit_code={e.exit_code}", exc_info=True)
        raise SDKSessionError(f"SDK session failed: {e}") from e

    except Exception as e:
        logger.error(f"Failed to create SDK session: {e}", exc_info=True)
        raise SDKSessionError(f"Failed to create session: {e}") from e


async def get_session_client(session_id: str, *, user_id: int | None = None) -> ClaudeSDKClient | None:
    """Get an existing SDK client by session ID.

    Args:
        session_id: Session ID from create_sdk_session

    Returns:
        ClaudeSDKClient or None if session not found/expired
    """
    if user_id is None:
        client = await session_manager.get_client(session_id)
    else:
        client = await session_manager.get_client(session_id, user_id=user_id)
    return cast("ClaudeSDKClient | None", client)


async def send_message(session_id: str, message: str, *, user_id: int | None = None) -> ClaudeSDKClient:
    """Send a message to an existing session.

    Args:
        session_id: Session ID
        message: User message to send

    Returns:
        ClaudeSDKClient for receiving response

    Raises:
        SDKSessionError: If session not found or message fails
    """
    if user_id is None:
        client = await session_manager.get_client(session_id)
    else:
        client = await session_manager.get_client(session_id, user_id=user_id)

    if not client:
        raise SDKSessionError(f"Session {session_id} not found or expired")

    try:
        sdk_client = cast("ClaudeSDKClient", client)
        await sdk_client.query(message)
        return sdk_client

    except ProcessError as e:
        logger.error(f"Message failed: exit_code={e.exit_code}", exc_info=True)
        raise SDKSessionError(f"Message failed: {e}") from e

    except Exception as e:
        logger.error(f"Failed to send message: {e}", exc_info=True)
        raise SDKSessionError(f"Failed to send message: {e}") from e


async def close_session(session_id: str) -> None:
    """Close and cleanup a session.

    Args:
        session_id: Session ID to close
    """
    await session_manager.remove_client(session_id)
    logger.info(f"Closed session {session_id}")
