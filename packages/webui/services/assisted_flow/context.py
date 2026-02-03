"""Context for assisted flow tools.

This module provides a dataclass that holds the shared context needed by all
tools during an assisted flow session.

Assisted-flow sessions can outlive a single HTTP request, so this context must
not hold request-scoped objects like an `AsyncSession`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

    # Type alias for session factory
    SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


@dataclass
class ToolContext:
    """Shared context for assisted flow tools.

    This context is created once per assisted flow session and passed
    to all tools via closure. It provides access to:
    - User ID for permission checks
    - Source ID being configured (optional for inline sources)
    - Current pipeline state (mutable during session)
    - Applied configuration (set when user confirms pipeline)
    - Inline source configuration (for inline source mode)
    - Database session factory (for persistence operations)

    Attributes:
        user_id: ID of the authenticated user
        source_id: Integer ID of the collection source being configured (if any)
        pipeline_state: Current pipeline DAG configuration (None if not yet built)
        applied_config: Final configuration after apply_pipeline (None until applied)
        inline_source_config: Source configuration for inline source mode (None for existing sources)
        inline_secrets: Encrypted secrets for inline source mode (None if no secrets)
        get_session: Factory function to create database sessions for persistence
    """

    user_id: int
    source_id: int | None
    pipeline_state: dict[str, Any] | None = None
    applied_config: dict[str, Any] | None = None

    # For inline source mode
    inline_source_config: dict[str, Any] | None = None
    inline_secrets: dict[str, str] | None = None

    # Session factory for DB operations (callable, not live session)
    # Type: Callable[[], AsyncContextManager[AsyncSession]] | None
    get_session: Any = field(default=None, repr=False)
