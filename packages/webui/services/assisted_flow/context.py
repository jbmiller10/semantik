"""Context for assisted flow tools.

This module provides a dataclass that holds the shared context
needed by all tools during an assisted flow session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class ToolContext:
    """Shared context for assisted flow tools.

    This context is created once per assisted flow session and passed
    to all tools via closure. It provides access to:
    - Database session for repository operations
    - User ID for permission checks
    - Source ID being configured
    - Current pipeline state (mutable during session)

    Attributes:
        session: Async SQLAlchemy session for database operations
        user_id: ID of the authenticated user
        source_id: Integer ID of the collection source being configured
        pipeline_state: Current pipeline DAG configuration (None if not yet built)
    """

    session: AsyncSession
    user_id: int
    source_id: int
    pipeline_state: dict[str, Any] | None = None
