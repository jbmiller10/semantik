#!/usr/bin/env python3
"""
Character chunking strategy wrapper for domain interface.

This module provides a wrapper around the unified character chunking strategy
to make it compatible with domain tests.
"""

from typing import Any

from shared.chunking.unified.character_strategy import CharacterChunkingStrategy as UnifiedCharacterStrategy
from shared.chunking.unified.factory import DomainStrategyAdapter, UnifiedChunkingFactory


class CharacterChunkingStrategy(DomainStrategyAdapter):
    """
    Character-based chunking strategy that splits on character count.

    This wrapper provides compatibility with domain tests while delegating
    actual chunking to the unified strategy.
    """

    def __init__(self) -> None:
        """Initialize the character chunking strategy."""
        unified_strategy = UnifiedChunkingFactory.create_strategy("character", use_llama_index=False)
        super().__init__(unified_strategy)

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for plugin configuration."""
        return UnifiedCharacterStrategy.get_config_schema()
