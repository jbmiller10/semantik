"""Context for assisted flow tools.

This module provides a dataclass that holds the shared context needed by all
tools during an assisted flow session.

Assisted-flow sessions can outlive a single HTTP request, so this context must
not hold request-scoped objects like an `AsyncSession`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolContext:
    """Shared context for assisted flow tools.

    This context is created once per assisted flow session and passed
    to all tools via closure. It provides access to:
    - User ID for permission checks
    - Source ID being configured (optional for inline sources)
    - Current pipeline state (mutable during session)
    - Applied configuration (set when user confirms pipeline)

    Attributes:
        user_id: ID of the authenticated user
        source_id: Integer ID of the collection source being configured (if any)
        pipeline_state: Current pipeline DAG configuration (None if not yet built)
        applied_config: Final configuration after apply_pipeline (None until applied)
    """

    user_id: int
    source_id: int | None
    pipeline_state: dict[str, Any] | None = None
    applied_config: dict[str, Any] | None = None
