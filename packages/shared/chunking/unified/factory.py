#!/usr/bin/env python3
"""
Factory and adapter patterns for unified chunking strategies.

This module provides factories and adapters to make the unified strategies
compatible with both the domain-based and text_processing interfaces.
"""

import logging
from enum import Enum
from typing import Any

from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.unified.base import UnifiedChunkingStrategy
from packages.shared.chunking.unified.character_strategy import (
    CharacterChunkingStrategy,
)
from packages.shared.chunking.unified.hierarchical_strategy import (
    HierarchicalChunkingStrategy,
)
from packages.shared.chunking.unified.hybrid_strategy import (
    HybridChunkingStrategy,
)
from packages.shared.chunking.unified.markdown_strategy import (
    MarkdownChunkingStrategy,
)
from packages.shared.chunking.unified.recursive_strategy import (
    RecursiveChunkingStrategy,
)
from packages.shared.chunking.unified.semantic_strategy import (
    SemanticChunkingStrategy,
)

logger = logging.getLogger(__name__)


class ChunkingStrategyType(str, Enum):
    """Enumeration of available chunking strategies."""

    CHARACTER = "character"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    MARKDOWN = "markdown"
    HYBRID = "hybrid"


class UnifiedChunkingFactory:
    """
    Factory for creating unified chunking strategies.

    This factory creates the appropriate unified strategy based on the requested type,
    with optional LlamaIndex support.
    """

    @staticmethod
    def create_strategy(
        strategy_type: str | ChunkingStrategyType,
        use_llama_index: bool = False,
        **kwargs: Any,
    ) -> UnifiedChunkingStrategy:
        """
        Create a unified chunking strategy.

        Args:
            strategy_type: Type of strategy to create
            use_llama_index: Whether to enable LlamaIndex support
            **kwargs: Additional strategy-specific parameters

        Returns:
            Unified chunking strategy instance

        Raises:
            ValueError: If strategy type is not supported
        """
        strategy_type = ChunkingStrategyType(strategy_type.lower())

        logger.info(f"Creating unified {strategy_type} strategy (LlamaIndex: {use_llama_index})")

        if strategy_type == ChunkingStrategyType.CHARACTER:
            return CharacterChunkingStrategy(use_llama_index=use_llama_index)
        elif strategy_type == ChunkingStrategyType.RECURSIVE:
            return RecursiveChunkingStrategy(use_llama_index=use_llama_index)
        elif strategy_type == ChunkingStrategyType.SEMANTIC:
            # Note: Semantic strategy may need embed_model
            embed_model = kwargs.get("embed_model")
            return SemanticChunkingStrategy(use_llama_index=use_llama_index, embed_model=embed_model)
        elif strategy_type == ChunkingStrategyType.HIERARCHICAL:
            return HierarchicalChunkingStrategy(use_llama_index=use_llama_index)
        elif strategy_type == ChunkingStrategyType.MARKDOWN:
            return MarkdownChunkingStrategy(use_llama_index=use_llama_index)
        elif strategy_type == ChunkingStrategyType.HYBRID:
            # Note: Hybrid strategy may need embed_model for semantic component
            embed_model = kwargs.get("embed_model")
            return HybridChunkingStrategy(use_llama_index=use_llama_index, embed_model=embed_model)
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")

    @staticmethod
    def get_available_strategies() -> list[str]:
        """
        Get list of available strategy types.

        Returns:
            List of strategy type names
        """
        return [s.value for s in ChunkingStrategyType]

    @staticmethod
    def create_from_config(
        config: ChunkConfig,
        use_llama_index: bool = False,
    ) -> UnifiedChunkingStrategy:
        """
        Create strategy from a ChunkConfig object.

        Args:
            config: Chunk configuration
            use_llama_index: Whether to enable LlamaIndex support

        Returns:
            Unified chunking strategy instance
        """
        return UnifiedChunkingFactory.create_strategy(
            strategy_type=config.strategy_name,
            use_llama_index=use_llama_index,
        )


class DomainStrategyAdapter:
    """
    Adapter to make unified strategies compatible with the domain interface.

    This adapter wraps a unified strategy and provides the exact interface
    expected by the domain-based chunking system.
    """

    def __init__(self, unified_strategy: UnifiedChunkingStrategy) -> None:
        """
        Initialize the adapter.

        Args:
            unified_strategy: The unified strategy to adapt
        """
        self.strategy = unified_strategy

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped strategy."""
        return getattr(self.strategy, name)


class TextProcessingStrategyAdapter:
    """
    Adapter to make unified strategies compatible with the text_processing interface.

    This adapter wraps a unified strategy and provides the exact interface
    expected by the text_processing chunking system.
    """

    def __init__(self, unified_strategy: UnifiedChunkingStrategy, **params) -> None:
        """
        Initialize the adapter.

        Args:
            unified_strategy: The unified strategy to adapt
            **params: Additional parameters for configuration
        """
        self.strategy = unified_strategy
        self.strategy_name = unified_strategy.name
        
        # Filter out known token-based parameters, ignore strategy-specific ones
        self.params = {}
        for key in ["max_tokens", "min_tokens", "overlap_tokens", "chunk_size", 
                    "chunk_overlap", "min_chunk_size", "custom_attributes"]:
            if key in params:
                self.params[key] = params[key]

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Adapt unified chunking to text_processing interface.

        Args:
            text: Text to chunk
            doc_id: Document ID
            metadata: Optional metadata

        Returns:
            List of chunk dictionaries
        """
        from packages.shared.chunking.domain.value_objects.chunk_config import (
            ChunkConfig,
        )

        # Use parameters passed from factory or defaults
        max_tokens = self.params.get("max_tokens", 1000)
        min_tokens = self.params.get("min_tokens", 100)
        overlap_tokens = self.params.get("overlap_tokens", 50)
        
        # Ensure overlap is valid
        if overlap_tokens >= min_tokens:
            overlap_tokens = min(overlap_tokens, min_tokens - 1)
        if overlap_tokens >= max_tokens:
            overlap_tokens = min(overlap_tokens, max_tokens - 1)
            
        # Create config using passed parameters
        config_params = {
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "overlap_tokens": overlap_tokens,
            "strategy_name": self.strategy.name,
        }
        
        # Only add custom_attributes if present
        if "custom_attributes" in self.params:
            config_params["custom_attributes"] = self.params["custom_attributes"]
            
        config = ChunkConfig(**config_params)

        # Use unified strategy to chunk
        chunks = self.strategy.chunk(text, config)

        # Convert to text_processing format with ChunkResult objects
        from packages.shared.text_processing.base_chunker import ChunkResult
        
        results = []
        for i, chunk in enumerate(chunks):
            # Generate chunk ID with doc_id prefix if needed
            chunk_id = chunk.metadata.chunk_id
            if doc_id and not chunk_id.startswith(doc_id):
                chunk_id = f"{doc_id}_{chunk.metadata.chunk_index:04d}"
            
            # Build metadata with all expected fields
            chunk_metadata = {
                "strategy": self.strategy.name,
                "chunk_index": chunk.metadata.chunk_index,
                "token_count": chunk.metadata.token_count,
                **(metadata or {}),
            }
            
            # Add hybrid-specific metadata if using hybrid strategy
            if self.strategy.name == "hybrid":
                chunk_metadata["hybrid_chunker"] = True
                chunk_metadata["selected_strategy"] = "hybrid"
                if i == 0:  # Add reasoning to first chunk
                    chunk_metadata["hybrid_strategy_used"] = "hybrid"
                    chunk_metadata["hybrid_strategy_reasoning"] = "Hybrid strategy selected"
            
            result = ChunkResult(
                chunk_id=chunk_id,
                text=chunk.content,
                start_offset=chunk.metadata.start_offset,
                end_offset=chunk.metadata.end_offset,
                metadata=chunk_metadata,
            )
            results.append(result)

        return results

    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Async adapter for text_processing interface.

        Args:
            text: Text to chunk
            doc_id: Document ID
            metadata: Optional metadata

        Returns:
            List of chunk dictionaries
        """
        from packages.shared.chunking.domain.value_objects.chunk_config import (
            ChunkConfig,
        )

        # Use parameters passed from factory or defaults
        max_tokens = self.params.get("max_tokens", 1000)
        min_tokens = self.params.get("min_tokens", 100)
        overlap_tokens = self.params.get("overlap_tokens", 50)
        
        # Ensure overlap is valid
        if overlap_tokens >= min_tokens:
            overlap_tokens = min(overlap_tokens, min_tokens - 1)
        if overlap_tokens >= max_tokens:
            overlap_tokens = min(overlap_tokens, max_tokens - 1)
            
        # Create config using passed parameters
        config_params = {
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "overlap_tokens": overlap_tokens,
            "strategy_name": self.strategy.name,
        }
        
        # Only add custom_attributes if present
        if "custom_attributes" in self.params:
            config_params["custom_attributes"] = self.params["custom_attributes"]
            
        config = ChunkConfig(**config_params)

        # Use unified strategy to chunk
        chunks = await self.strategy.chunk_async(text, config)

        # Convert to text_processing format with ChunkResult objects
        from packages.shared.text_processing.base_chunker import ChunkResult
        
        results = []
        for i, chunk in enumerate(chunks):
            # Generate chunk ID with doc_id prefix if needed
            chunk_id = chunk.metadata.chunk_id
            if doc_id and not chunk_id.startswith(doc_id):
                chunk_id = f"{doc_id}_{chunk.metadata.chunk_index:04d}"
            
            # Build metadata with all expected fields
            chunk_metadata = {
                "strategy": self.strategy.name,
                "chunk_index": chunk.metadata.chunk_index,
                "token_count": chunk.metadata.token_count,
                **(metadata or {}),
            }
            
            # Add hybrid-specific metadata if using hybrid strategy
            if self.strategy.name == "hybrid":
                chunk_metadata["hybrid_chunker"] = True
                chunk_metadata["selected_strategy"] = "hybrid"
                if i == 0:  # Add reasoning to first chunk
                    chunk_metadata["hybrid_strategy_used"] = "hybrid"
                    chunk_metadata["hybrid_strategy_reasoning"] = "Hybrid strategy selected"
            
            result = ChunkResult(
                chunk_id=chunk_id,
                text=chunk.content,
                start_offset=chunk.metadata.start_offset,
                end_offset=chunk.metadata.end_offset,
                metadata=chunk_metadata,
            )
            results.append(result)

        return results

    def validate_config(self, config: dict[str, Any]) -> bool:
        """
        Validate configuration for text_processing interface.

        Args:
            config: Configuration dictionary

        Returns:
            True if valid
        """
        # Convert to domain config for validation
        try:
            from packages.shared.chunking.domain.value_objects.chunk_config import (
                ChunkConfig,
            )
            
            # Handle both token-based and character-based parameters
            if "max_tokens" in config:
                # Token-based parameters
                max_tokens = config.get("max_tokens", 1000)
                min_tokens = config.get("min_tokens", 100)
                overlap_tokens = config.get("overlap_tokens", 50)
            else:
                # Character-based parameters (legacy)
                max_tokens = config.get("chunk_size", 1000)
                min_tokens = config.get("min_chunk_size", 100)
                overlap_tokens = config.get("chunk_overlap", 50)
            
            # Reject if overlap >= min_tokens
            if overlap_tokens >= min_tokens:
                return False
                
            # Reject if overlap >= max_tokens
            if overlap_tokens >= max_tokens:
                return False
            
            # Now create config with valid parameters
            ChunkConfig(
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                overlap_tokens=overlap_tokens,
                strategy_name=self.strategy.name,
            )
            return True
        except Exception:
            return False

    def estimate_chunks(self, text_length: int, config: dict[str, Any]) -> int:
        """
        Estimate chunks for text_processing interface.

        Args:
            text_length: Length of text
            config: Configuration dictionary

        Returns:
            Estimated number of chunks
        """
        from packages.shared.chunking.domain.value_objects.chunk_config import (
            ChunkConfig,
        )

        # Get config values
        chunk_size = config.get("chunk_size", 1000)
        min_tokens = config.get("min_chunk_size", 100)
        overlap_tokens = config.get("chunk_overlap", 50)
        
        # Don't try to fix invalid configs - let ChunkConfig validate
        domain_config = ChunkConfig(
            max_tokens=chunk_size,
            min_tokens=min_tokens,
            overlap_tokens=overlap_tokens,
            strategy_name=self.strategy.name,
        )

        return self.strategy.estimate_chunks(text_length, domain_config)
