"""Source statistics gathering for assisted flow.

This module provides functions to gather information about a collection
source before starting the assisted configuration flow. The stats are
injected into the agent's initial prompt for context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from shared.database.models import Collection, CollectionSource

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Allowlist of safe config keys to include in agent prompt.
# Only include keys known to be non-sensitive. This prevents accidental
# exposure of credentials under unexpected key names.
_SAFE_SOURCE_CONFIG_KEYS = frozenset(
    {
        "path",
        "paths",
        "recursive",
        "file_extensions",
        "repo_url",
        "repository_url",
        "branch",
        "depth",
        "host",
        "port",
        "mailbox",
        "folder",
        "use_ssl",
        "username",  # username alone is not a secret
        "source_type",
        "name",
        "description",
    }
)


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
    user_id: int,
    source_id: int,
) -> dict[str, Any]:
    """Gather statistics about a collection source.

    This function retrieves the source configuration and any available
    metadata to provide context for the assisted flow agent.

    Args:
        session: Database session
        user_id: Authenticated user id (used to enforce ownership)
        source_id: Integer ID of the collection source

    Returns:
        Dictionary with source stats:
        - source_name: Human-readable name (source_path)
        - source_type: Type of source (directory, git, imap)
        - source_path: Path or URL of the source
        - source_config: Full source configuration (secrets redacted)

    Raises:
        EntityNotFoundError: If source not found
        AccessDeniedError: If the user does not own the source
    """
    stmt = (
        select(CollectionSource, Collection.owner_id)
        .join(Collection, Collection.id == CollectionSource.collection_id)
        .where(CollectionSource.id == source_id)
    )
    result = await session.execute(stmt)
    row = result.first()
    if not row:
        raise EntityNotFoundError("collection_source", str(source_id))
    source, owner_id = row
    if owner_id != user_id:
        raise AccessDeniedError(str(user_id), "collection_source", str(source_id))

    safe_config = {k: v for k, v in (source.source_config or {}).items() if k.lower() in _SAFE_SOURCE_CONFIG_KEYS}

    return {
        "source_name": source.source_path,
        "source_type": source.source_type,
        "source_path": _get_display_path(source.source_type, source.source_config or {}, source.source_path),
        "source_config": safe_config,
    }
