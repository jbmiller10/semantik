"""Startup tasks for the webui application.

This module contains tasks that should be run when the application starts,
such as ensuring default data exists.
"""

import logging

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.database import get_db
from shared.database.models import PluginConfig
from shared.plugins.loader import load_plugins
from shared.plugins.registry import PluginSource
from webui.services.chunking_strategy_service import ChunkingStrategyService

logger = logging.getLogger(__name__)


async def ensure_default_data() -> None:
    """Ensure all default data exists in the database.

    This function is idempotent and can be called multiple times safely.
    """
    logger.info("Running startup tasks to ensure default data...")

    session_gen = get_db()
    disabled_plugin_ids: set[str] | None = None
    try:
        session = await anext(session_gen)
        try:
            result = await session.execute(select(PluginConfig.id).where(PluginConfig.enabled.is_(False)))
            disabled_plugin_ids = {row[0] for row in result.all()}
        except SQLAlchemyError as exc:
            logger.info("Plugin configs not available yet; loading all plugins (%s)", exc)

        registry = load_plugins(
            plugin_types={"embedding", "chunking", "connector"},
            disabled_plugin_ids=disabled_plugin_ids,
        )

        disabled_ids = disabled_plugin_ids or set()
        embedding_plugins = [
            plugin_id
            for plugin_id in registry.list_ids(plugin_type="embedding", source=PluginSource.EXTERNAL)
            if plugin_id not in disabled_ids
        ]
        if embedding_plugins:
            logger.info("Loaded embedding plugins: %s", ", ".join(embedding_plugins))

        chunking_plugins = [
            plugin_id
            for plugin_id in registry.list_ids(plugin_type="chunking", source=PluginSource.EXTERNAL)
            if plugin_id not in disabled_ids
        ]
        if chunking_plugins:
            logger.info("Loaded chunking plugins: %s", ", ".join(chunking_plugins))

        connector_plugins = [
            plugin_id
            for plugin_id in registry.list_ids(plugin_type="connector", source=PluginSource.EXTERNAL)
            if plugin_id not in disabled_ids
        ]
        if connector_plugins:
            logger.info("Loaded connector plugins: %s", ", ".join(connector_plugins))

        # Validate pipeline templates (fail fast on errors)
        validate_templates()

        await ensure_default_chunking_strategies(session)
    finally:
        await session_gen.aclose()

    logger.info("Startup tasks completed")


def validate_templates() -> None:
    """Validate all pipeline templates at startup.

    Loads and validates all templates from the templates module.
    Raises ValueError on invalid templates, causing fail-fast startup behavior.
    """
    from shared.pipeline.templates import list_templates

    try:
        templates = list_templates()  # Raises ValueError on invalid templates
        logger.info("Validated %d pipeline templates", len(templates))
    except ValueError as e:
        logger.error("Pipeline template validation failed: %s", e)
        raise


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
