"""Startup tasks for the webui application.

This module contains tasks that should be run when the application starts,
such as ensuring default data exists.
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from shared.chunking.plugin_loader import load_chunking_plugins
from shared.database.database import get_db
from shared.embedding.plugin_loader import ensure_providers_registered, load_embedding_plugins
from webui.services.chunking_strategy_service import ChunkingStrategyService

logger = logging.getLogger(__name__)


async def ensure_default_data() -> None:
    """Ensure all default data exists in the database.

    This function is idempotent and can be called multiple times safely.
    """
    logger.info("Running startup tasks to ensure default data...")

    # Ensure built-in embedding providers are registered
    ensure_providers_registered()
    logger.info("Built-in embedding providers registered")

    # Load any external embedding provider plugins
    registered_embedding_plugins = load_embedding_plugins()
    if registered_embedding_plugins:
        logger.info("Loaded embedding plugins: %s", ", ".join(registered_embedding_plugins))

    # Load any external chunking strategy plugins before interacting with metadata or DB.
    registered_plugins = load_chunking_plugins()
    if registered_plugins:
        logger.info("Loaded chunking plugins: %s", ", ".join(registered_plugins))

    session_gen = get_db()
    try:
        session = await anext(session_gen)
        await ensure_default_chunking_strategies(session)
    finally:
        await session_gen.aclose()

    logger.info("Startup tasks completed")


async def ensure_default_chunking_strategies(session: AsyncSession) -> None:
    """Ensure default chunking strategies exist.

    Args:
        session: Database session to use
    """
    try:
        service = ChunkingStrategyService(session)
        created_count = await service.ensure_default_strategies()

        if created_count > 0:
            logger.info(f"Created {created_count} default chunking strategies")
        else:
            logger.debug("All default chunking strategies already exist")

    except Exception as e:
        logger.error(f"Error ensuring default chunking strategies: {e}")
        # Don't fail the startup if default data can't be created
        # The data migration will handle it
