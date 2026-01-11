"""Agent tool registration for the webui application.

This module registers built-in agent tools at application startup.
"""

from __future__ import annotations

import logging
import os

from shared.agents.tools.builtins import (
    DocumentRetrieveTool,
    GetChunkTool,
    ListCollectionsTool,
    SemanticSearchTool,
)
from shared.agents.tools.registry import get_tool_registry

logger = logging.getLogger(__name__)


async def register_builtin_agent_tools() -> None:
    """Register built-in agent tools with the tool registry.

    Tool configuration can be controlled via environment variables:
    - AGENT_SEARCH_USE_RERANKER: Enable reranking (default: false)
    - AGENT_SEARCH_RERANKER_ID: Reranker plugin ID (e.g., "qwen3-reranker")
    - AGENT_SEARCH_DEFAULT_TYPE: Default search type (default: semantic)
    - AGENT_SEARCH_HYBRID_ALPHA: Hybrid search alpha (default: 0.5)

    This function is idempotent - duplicate registrations are logged
    and silently ignored.
    """
    logger.info("Registering built-in agent tools...")

    registry = get_tool_registry()

    # Parse environment configuration
    use_reranker = os.getenv("AGENT_SEARCH_USE_RERANKER", "false").lower() == "true"
    reranker_id = os.getenv("AGENT_SEARCH_RERANKER_ID")
    default_search_type = os.getenv("AGENT_SEARCH_DEFAULT_TYPE", "semantic")

    try:
        hybrid_alpha = float(os.getenv("AGENT_SEARCH_HYBRID_ALPHA", "0.5"))
    except ValueError:
        hybrid_alpha = 0.5
        logger.warning("Invalid AGENT_SEARCH_HYBRID_ALPHA value, using default 0.5")

    # Configure and register search tool
    search_tool = SemanticSearchTool(
        use_reranker=use_reranker,
        reranker_id=reranker_id,
        hybrid_alpha=hybrid_alpha,
        default_search_type=default_search_type,
    )

    # Register all tools
    tools = [
        search_tool,
        DocumentRetrieveTool(),
        ListCollectionsTool(),
        GetChunkTool(),
    ]

    registered_count = 0
    for tool in tools:
        success = registry.register(tool, source="builtin")
        if success:
            logger.info("Registered builtin agent tool: %s", tool.name)
            registered_count += 1
        else:
            logger.debug("Agent tool already registered: %s", tool.name)

    if registered_count > 0:
        logger.info("Registered %d built-in agent tools", registered_count)
    else:
        logger.debug("All built-in agent tools were already registered")

    # Log configuration if reranker is enabled
    if use_reranker:
        logger.info(
            "Search tool configured with reranker: %s",
            reranker_id or "(default)",
        )
