"""
Chunking strategy definitions and registry.

This module contains the strategy definitions and configurations
that were previously hardcoded in the API router.
"""

from typing import Any

from packages.webui.api.v2.chunking_schemas import ChunkingStrategy


class ChunkingStrategyRegistry:
    """Registry for chunking strategies and their definitions."""

    # Strategy definitions with metadata
    STRATEGY_DEFINITIONS: dict[ChunkingStrategy, dict[str, Any]] = {
        ChunkingStrategy.FIXED_SIZE: {
            "name": "Fixed Size Chunking",
            "description": "Simple fixed-size chunking with consistent chunk sizes",
            "best_for": ["txt", "log", "csv", "json"],
            "pros": ["Predictable chunk sizes", "Fast processing", "Low memory usage", "Good for structured data"],
            "cons": ["May split sentences or paragraphs", "No semantic coherence", "Can break context"],
            "performance_characteristics": {"speed": "very_fast", "memory_usage": "low", "quality": "moderate"},
        },
        ChunkingStrategy.SEMANTIC: {
            "name": "Semantic",
            "description": "Uses embeddings to find natural semantic boundaries",
            "best_for": ["pdf", "docx", "md", "html", "tex"],
            "pros": [
                "Maintains semantic coherence",
                "Better search quality",
                "Context preservation",
                "Intelligent boundaries",
            ],
            "cons": ["Slower processing", "Higher memory usage", "Requires embedding model", "Variable chunk sizes"],
            "performance_characteristics": {"speed": "slow", "memory_usage": "high", "quality": "excellent"},
        },
        ChunkingStrategy.RECURSIVE: {
            "name": "Recursive",
            "description": "Recursively splits text using multiple separators",
            "best_for": ["md", "rst", "txt", "code files"],
            "pros": [
                "Respects document structure",
                "Good balance of speed and quality",
                "Handles nested content well",
                "Preserves formatting",
            ],
            "cons": ["May produce variable sizes", "Complex configuration", "Not ideal for unstructured text"],
            "performance_characteristics": {"speed": "fast", "memory_usage": "moderate", "quality": "good"},
        },
        ChunkingStrategy.SLIDING_WINDOW: {
            "name": "Sliding Window",
            "description": "Overlapping chunks with configurable window size",
            "best_for": ["txt", "log", "transcript", "chat"],
            "pros": [
                "No information loss at boundaries",
                "Good for continuous text",
                "Adjustable overlap",
                "Context preservation",
            ],
            "cons": ["Redundant information", "More storage required", "Slower search", "Higher costs"],
            "performance_characteristics": {"speed": "moderate", "memory_usage": "high", "quality": "good"},
        },
        ChunkingStrategy.DOCUMENT_STRUCTURE: {
            "name": "Document Structure",
            "description": "Splits based on document structure (headers, sections)",
            "best_for": ["pdf", "docx", "html", "epub", "tex"],
            "pros": [
                "Preserves document hierarchy",
                "Natural boundaries",
                "Maintains formatting",
                "Good for structured documents",
            ],
            "cons": [
                "Requires document parsing",
                "May produce very large chunks",
                "Not suitable for plain text",
                "Complex implementation",
            ],
            "performance_characteristics": {"speed": "moderate", "memory_usage": "moderate", "quality": "very_good"},
        },
        ChunkingStrategy.HYBRID: {
            "name": "Hybrid",
            "description": "Combines multiple strategies based on content analysis",
            "best_for": ["mixed content", "unknown formats", "large documents"],
            "pros": [
                "Adaptive to content",
                "Best of multiple strategies",
                "Handles diverse content",
                "Optimal quality",
            ],
            "cons": ["Complex configuration", "Slower processing", "Higher resource usage", "Unpredictable behavior"],
            "performance_characteristics": {"speed": "slow", "memory_usage": "very_high", "quality": "excellent"},
        },
    }

    @classmethod
    def get_strategy_definition(cls, strategy: ChunkingStrategy) -> dict[str, Any]:
        """Get the definition for a specific strategy.

        Args:
            strategy: The chunking strategy enum

        Returns:
            Dictionary containing strategy metadata
        """
        return cls.STRATEGY_DEFINITIONS.get(strategy, {})

    @classmethod
    def get_all_definitions(cls) -> dict[ChunkingStrategy, dict[str, Any]]:
        """Get all strategy definitions.

        Returns:
            Dictionary of all strategy definitions
        """
        return cls.STRATEGY_DEFINITIONS.copy()

    @classmethod
    def get_recommended_strategy(cls, file_types: list[str]) -> ChunkingStrategy:
        """Get recommended strategy based on file types.

        Args:
            file_types: List of file extensions

        Returns:
            Recommended chunking strategy
        """
        # Count which strategies are best for the given file types
        strategy_scores = {}
        for strategy, definition in cls.STRATEGY_DEFINITIONS.items():
            best_for = definition.get("best_for", [])
            score = sum(1 for ft in file_types if ft in best_for)
            strategy_scores[strategy] = score

        # Return the strategy with the highest score
        if strategy_scores:
            return max(strategy_scores, key=strategy_scores.get)

        # Default to recursive as a good general-purpose strategy
        return ChunkingStrategy.RECURSIVE
