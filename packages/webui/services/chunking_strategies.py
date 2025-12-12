"""Legacy compatibility wrapper for chunking strategy metadata.

Historically the service layer accessed strategy metadata through this module.
The canonical definitions now live in
``webui.services.chunking.strategy_registry``; this wrapper forwards
calls to the centralized helpers while preserving the original interface and
test surface.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from webui.services.chunking.strategy_registry import (
    build_metadata_by_enum,
    get_strategy_definition,
    recommend_strategy,
)

if TYPE_CHECKING:
    from webui.api.v2.chunking_schemas import ChunkingStrategy


class ChunkingStrategyRegistry:
    """Compatibility façade over the canonical strategy registry."""

    STRATEGY_DEFINITIONS: dict[ChunkingStrategy, dict[str, Any]] = build_metadata_by_enum()

    @classmethod
    def get_strategy_definition(cls, strategy: ChunkingStrategy) -> dict[str, Any]:
        """Return metadata for the requested strategy."""

        definition = get_strategy_definition(strategy)
        return definition.to_metadata_dict() if definition else {}

    @classmethod
    def get_all_definitions(cls) -> dict[ChunkingStrategy, dict[str, Any]]:
        """Return a defensive copy of all strategy metadata."""

        return copy.deepcopy(cls.STRATEGY_DEFINITIONS)

    @classmethod
    def get_recommended_strategy(cls, file_types: list[str]) -> ChunkingStrategy:
        """Recommend a strategy based on the provided file types."""

        return recommend_strategy(file_types)
