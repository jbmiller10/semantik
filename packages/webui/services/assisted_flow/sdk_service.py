"""SDK service for assisted flow sessions.

This module provides a service layer for creating and managing
ClaudeSDKClient sessions for the pipeline configuration assistant.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    CLINotFoundError,
    ProcessError,
)

from webui.services.assisted_flow.context import ToolContext
from webui.services.assisted_flow.prompts import SYSTEM_PROMPT, build_initial_prompt
from webui.services.assisted_flow.server import create_mcp_server
from webui.services.assisted_flow.session_manager import session_manager

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class SDKServiceError(Exception):
    """Base exception for SDK service errors."""


class SDKNotAvailableError(SDKServiceError):
    """Raised when Claude Code CLI is not installed."""


class SDKSessionError(SDKServiceError):
    """Raised when SDK session creation or operation fails."""


async def create_sdk_session(
    db: AsyncSession,
    user_id: int,
    source_id: int,
    source_stats: dict[str, Any],
) -> tuple[str, ClaudeSDKClient]:
    """Create a new SDK session for assisted flow.

    Args:
        db: Database session
        user_id: Authenticated user's ID
        source_id: Collection source ID being configured
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
        session=db,
        user_id=user_id,
        source_id=source_id,
    )

    # Create MCP server with tools
    mcp_server = create_mcp_server(ctx)

    # Build initial prompt
    initial_prompt = build_initial_prompt(source_stats)

    try:
        # Create SDK options
        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            permission_mode="acceptEdits",
            mcp_servers={"assisted-flow": mcp_server},
        )

        # Create client
        client = ClaudeSDKClient(options=options)

        # Connect with initial prompt
        await client.connect(prompt=initial_prompt)

        # Store in session manager
        await session_manager.store_client(session_id, client)

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


async def get_session_client(session_id: str) -> ClaudeSDKClient | None:
    """Get an existing SDK client by session ID.

    Args:
        session_id: Session ID from create_sdk_session

    Returns:
        ClaudeSDKClient or None if session not found/expired
    """
    return await session_manager.get_client(session_id)


async def send_message(session_id: str, message: str) -> ClaudeSDKClient:
    """Send a message to an existing session.

    Args:
        session_id: Session ID
        message: User message to send

    Returns:
        ClaudeSDKClient for receiving response

    Raises:
        SDKSessionError: If session not found or message fails
    """
    client = await session_manager.get_client(session_id)

    if not client:
        raise SDKSessionError(f"Session {session_id} not found or expired")

    try:
        await client.query(message)
        return client

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
    client = await session_manager.get_client(session_id)

    if client:
        try:
            client.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting session {session_id}: {e}")

    await session_manager.remove_client(session_id)
    logger.info(f"Closed session {session_id}")
