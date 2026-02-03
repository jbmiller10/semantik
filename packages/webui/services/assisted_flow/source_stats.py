"""Source statistics gathering for assisted flow.

This module provides functions to gather information about a collection
source before starting the assisted configuration flow. The stats are
injected into the agent's initial prompt for context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from shared.database.exceptions import EntityNotFoundError
from shared.database.repositories.collection_source_repository import CollectionSourceRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def _get_display_path(source_type: str, config: dict[str, Any], source_path: str) -> str:
    """Derive human-readable display path from source config.

    Args:
        source_type: Type of source (directory, git, imap)
        config: Source-specific configuration
        source_path: The stored source_path

    Returns:
        Human-readable path/identifier
    """
    if source_type == "directory":
        return str(config.get("path", source_path))
    if source_type == "git":
        return str(config.get("repo_url", config.get("repository_url", source_path)))
    if source_type == "imap":
        username = str(config.get("username", ""))
        host = str(config.get("host", ""))
        return f"{username}@{host}" if username and host else (username or host or source_path)
    return source_path or str(config.get("path", config.get("url", str(config))))


async def get_source_stats(
    session: AsyncSession,
    source_id: int,
) -> dict[str, Any]:
    """Gather statistics about a collection source.

    This function retrieves the source configuration and any available
    metadata to provide context for the assisted flow agent.

    Args:
        session: Database session
        source_id: Integer ID of the collection source

    Returns:
        Dictionary with source stats:
        - source_name: Human-readable name (source_path)
        - source_type: Type of source (directory, git, imap)
        - source_path: Path or URL of the source
        - source_config: Full source configuration (secrets redacted)

    Raises:
        EntityNotFoundError: If source not found
    """
    repo = CollectionSourceRepository(session)
    source = await repo.get_by_id(source_id)

    if not source:
        raise EntityNotFoundError("collection_source", str(source_id))

    # Redact any secrets from config
    safe_config = {
        k: v
        for k, v in (source.source_config or {}).items()
        if "password" not in k.lower()
        and "secret" not in k.lower()
        and "token" not in k.lower()
        and "key" not in k.lower()
    }

    return {
        "source_name": source.source_path,
        "source_type": source.source_type,
        "source_path": _get_display_path(source.source_type, source.source_config or {}, source.source_path),
        "source_config": safe_config,
    }
