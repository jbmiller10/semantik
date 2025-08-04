"""Startup tasks for the webui application.

This module contains tasks that should be run when the application starts,
such as ensuring default data exists.
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.database import AsyncSessionLocal
from packages.webui.services.chunking_strategy_service import ChunkingStrategyService

logger = logging.getLogger(__name__)


async def ensure_default_data() -> None:
    """Ensure all default data exists in the database.

    This function is idempotent and can be called multiple times safely.
    """
    logger.info("Running startup tasks to ensure default data...")

    async with AsyncSessionLocal() as session:
        await ensure_default_chunking_strategies(session)

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
