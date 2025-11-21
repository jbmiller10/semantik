#!/usr/bin/env python3
"""
Factory for creating chunking strategies.

This module provides a factory pattern for instantiating different chunking
strategies based on configuration.
"""

import os
from typing import Any

from shared.text_processing.base_chunker import BaseChunker


class ChunkingFactory:
    """Factory for creating chunking strategies using LlamaIndex."""

    # Registry of available strategies
    _strategies: dict[str, type[BaseChunker]] = {}

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type[BaseChunker]) -> None:
        """Register a chunking strategy.

        Args:
            name: Name of the strategy
            strategy_class: Class implementing BaseChunker
        """
        cls._strategies[name] = strategy_class

    @classmethod
    def create_chunker(cls, config: dict[str, Any]) -> BaseChunker:
        """Create appropriate chunker based on configuration.

        Args:
            config: Configuration dictionary with 'strategy' and optional 'params'

        Returns:
            Instance of the requested chunking strategy

        Raises:
            ValueError: If unknown strategy is requested
        """
        strategy = config.get("strategy")
        if not strategy:
            raise ValueError("Configuration must include 'strategy' field")

        params = config.get("params", {})

        # Lazy import strategies to avoid circular imports
        if not cls._strategies:
            cls._initialize_strategies()

        strategy_class = cls._strategies.get(strategy)
        if not strategy_class:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        # Special handling for strategies that need embedding models
        if strategy in ["semantic", "hybrid"] and "embed_model" not in params:
            # Handle embedding model initialization
            if os.getenv("TESTING", "false").lower() == "true":
                # For testing, use mock embedding
                from llama_index.core.embeddings import MockEmbedding

                params["embed_model"] = MockEmbedding(embed_dim=384)
            else:
                # Check if we should use local embeddings
                use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

                if use_local:
                    # Use the local embedding service via a custom adapter
                    from shared.text_processing.embedding_adapter import LocalEmbeddingAdapter

                    params["embed_model"] = LocalEmbeddingAdapter()
                else:
                    # For production, use OpenAI embeddings
                    from llama_index.embeddings.openai import OpenAIEmbedding

                    params["embed_model"] = OpenAIEmbedding()

        return strategy_class(**params)

    @classmethod
    def _initialize_strategies(cls) -> None:
        """Initialize the strategy registry with available strategies."""
        # Import actual strategy wrapper classes for backward compatibility
        from shared.text_processing.strategies.character_chunker import CharacterChunker
        from shared.text_processing.strategies.hierarchical_chunker import HierarchicalChunker
        from shared.text_processing.strategies.hybrid_chunker import HybridChunker
        from shared.text_processing.strategies.markdown_chunker import MarkdownChunker
        from shared.text_processing.strategies.recursive_chunker import RecursiveChunker
        from shared.text_processing.strategies.semantic_chunker import SemanticChunker

        # Register the actual wrapper classes that use unified implementation internally
        cls.register_strategy("character", CharacterChunker)
        cls.register_strategy("recursive", RecursiveChunker)
        cls.register_strategy("semantic", SemanticChunker)
        cls.register_strategy("markdown", MarkdownChunker)
        cls.register_strategy("hierarchical", HierarchicalChunker)  # type: ignore[arg-type]
        cls.register_strategy("hybrid", HybridChunker)

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """Get list of available chunking strategies.

        Returns:
            List of strategy names
        """
        if not cls._strategies:
            cls._initialize_strategies()
        return list(cls._strategies.keys())
