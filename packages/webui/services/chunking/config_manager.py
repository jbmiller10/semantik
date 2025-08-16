"""
Chunking configuration manager service.

Handles strategy configuration, defaults, and recommendations.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ChunkingConfigManager:
    """Service responsible for chunking configuration management."""

    # Default configurations for each strategy
    DEFAULT_CONFIGS = {
        "fixed_size": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "separator": "\n",
        },
        "sliding_window": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "stride": 800,
        },
        "semantic": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "embedding_model": "sentence-transformers",
            "similarity_threshold": 0.8,
        },
        "recursive": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "separators": ["\n\n", "\n", " ", ""],
        },
        "document_structure": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "preserve_structure": True,
        },
        "markdown": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "split_by_headers": True,
            "min_header_level": 1,
            "max_header_level": 3,
        },
        "hierarchical": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_level": 3,
            "level_separator": "\n\n",
        },
        "hybrid": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "primary_strategy": "semantic",
            "fallback_strategy": "recursive",
        },
    }

    # Strategy metadata
    STRATEGY_INFO = {
        "fixed_size": {
            "name": "Fixed Size",
            "description": "Splits text into fixed-size chunks with optional overlap",
            "best_for": ["General text", "Consistent chunk sizes", "Simple documents"],
            "pros": ["Fast processing", "Predictable output", "Simple to configure"],
            "cons": ["May split sentences", "Ignores document structure"],
            "supported_file_types": [],  # All file types supported
        },
        "sliding_window": {
            "name": "Sliding Window",
            "description": "Uses a sliding window approach for overlapping chunks",
            "best_for": ["Sequential analysis", "Context preservation", "Time series text"],
            "pros": ["Better context preservation", "Smooth transitions"],
            "cons": ["More chunks produced", "Higher storage requirements"],
            "supported_file_types": [],  # All file types supported
        },
        "semantic": {
            "name": "Semantic",
            "description": "Creates chunks based on semantic similarity",
            "best_for": ["Technical documents", "Research papers", "Complex topics"],
            "pros": ["Preserves meaning", "Better for search", "Topic coherence"],
            "cons": ["Slower processing", "Requires embeddings", "Variable chunk sizes"],
            "supported_file_types": [],  # All file types supported
        },
        "recursive": {
            "name": "Recursive",
            "description": "Recursively splits text using multiple separators",
            "best_for": ["Mixed content", "Code files", "Structured documents"],
            "pros": ["Respects structure", "Flexible", "Good default choice"],
            "cons": ["May produce small chunks", "Configuration dependent"],
            "supported_file_types": [],  # All file types supported
        },
        "document_structure": {
            "name": "Document Structure",
            "description": "Splits documents based on structural elements",
            "best_for": ["Structured documents", "Reports", "Articles"],
            "pros": ["Preserves document structure", "Clean boundaries"],
            "cons": ["Requires structured input", "May miss context"],
            "supported_file_types": [],  # All file types supported
        },
        "markdown": {
            "name": "Markdown",
            "description": "Splits markdown documents preserving structure",
            "best_for": ["Markdown files", "Documentation", "README files"],
            "pros": ["Preserves formatting", "Header-aware", "Clean splits"],
            "cons": ["Only for markdown", "May create large chunks"],
            "supported_file_types": [],  # All file types supported - especially good for markdown
        },
        "hierarchical": {
            "name": "Hierarchical",
            "description": "Creates nested chunks at multiple levels",
            "best_for": ["Books", "Long documents", "Hierarchical content"],
            "pros": ["Multi-level context", "Good for navigation", "Preserves hierarchy"],
            "cons": ["Complex output", "More storage", "Harder to search"],
            "supported_file_types": [],  # All file types supported
        },
        "hybrid": {
            "name": "Hybrid",
            "description": "Combines multiple strategies with fallback",
            "best_for": ["Mixed content types", "Uncertain document structure"],
            "pros": ["Adaptable", "Best of both worlds", "Robust"],
            "cons": ["Slower", "Complex configuration", "Unpredictable behavior"],
            "supported_file_types": [],  # All file types supported
        },
    }

    def __init__(self):
        """Initialize the configuration manager."""
        self.custom_configs = {}

    def get_default_config(self, strategy: str) -> dict[str, Any]:
        """
        Get default configuration for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            Default configuration dictionary
        """
        return self.DEFAULT_CONFIGS.get(strategy, {}).copy()

    def get_strategy_info(self, strategy: str) -> dict[str, Any]:
        """
        Get metadata information for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            Strategy information dictionary
        """
        info = self.STRATEGY_INFO.get(strategy, {}).copy()
        info["id"] = strategy
        info["default_config"] = self.get_default_config(strategy)
        return info

    def get_all_strategies(self) -> list[dict[str, Any]]:
        """
        Get information for all available strategies.

        Returns:
            List of strategy information dictionaries
        """
        strategies = []
        for strategy_id in self.STRATEGY_INFO:
            strategies.append(self.get_strategy_info(strategy_id))
        return strategies

    def merge_configs(
        self,
        strategy: str,
        user_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Merge user configuration with defaults.

        Args:
            strategy: Strategy name
            user_config: User-provided configuration

        Returns:
            Merged configuration
        """
        default_config = self.get_default_config(strategy)

        if not user_config:
            return default_config

        # Merge configurations, user values override defaults
        merged = default_config.copy()
        merged.update(user_config)

        return merged

    def recommend_strategy(
        self,
        file_type: str | None = None,
        content_length: int | None = None,
        document_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Recommend a chunking strategy based on document characteristics.

        Args:
            file_type: File extension or MIME type
            content_length: Length of content
            document_type: Type of document (code, prose, technical, etc.)

        Returns:
            Recommendation with strategy and reasoning
        """
        recommendation = {
            "strategy": "recursive",  # Safe default
            "confidence": 0.5,
            "reasoning": [],
            "alternatives": [],
        }

        # File type based recommendations
        if file_type:
            file_type = file_type.lower()
            if file_type in [".md", "markdown", "text/markdown"]:
                recommendation["strategy"] = "markdown"
                recommendation["confidence"] = 0.9
                recommendation["reasoning"].append("Markdown file detected")
            elif file_type in [".py", ".js", ".java", ".cpp", ".c", ".go"]:
                recommendation["strategy"] = "recursive"
                recommendation["confidence"] = 0.8
                recommendation["reasoning"].append("Code file detected")
                recommendation["alternatives"].append("fixed_size")
            elif file_type in [".pdf", ".docx", ".doc"]:
                recommendation["strategy"] = "document_structure"
                recommendation["confidence"] = 0.7
                recommendation["reasoning"].append("Structured document format")

        # Content length based adjustments
        if content_length:
            if content_length < 1000:
                recommendation["strategy"] = "fixed_size"
                recommendation["confidence"] = max(recommendation["confidence"], 0.6)
                recommendation["reasoning"].append("Short content - simple chunking sufficient")
            elif content_length > 100000:
                if recommendation["strategy"] == "recursive":
                    recommendation["alternatives"].append("hierarchical")
                    recommendation["reasoning"].append("Large document - consider hierarchical")

        # Document type based adjustments
        if document_type:
            document_type = document_type.lower()
            if document_type in ["technical", "research", "academic"]:
                recommendation["strategy"] = "semantic"
                recommendation["confidence"] = 0.8
                recommendation["reasoning"].append("Technical content benefits from semantic chunking")
            elif document_type == "narrative":
                recommendation["strategy"] = "recursive"
                recommendation["confidence"] = 0.7
                recommendation["reasoning"].append("Narrative text works well with recursive splitting")

        # Add configuration suggestion
        recommendation["suggested_config"] = self.get_default_config(recommendation["strategy"])

        # Adjust chunk size based on content length
        if content_length and content_length < 5000:
            recommendation["suggested_config"]["chunk_size"] = 500
            recommendation["suggested_config"]["chunk_overlap"] = 100

        return recommendation

    def get_alternative_strategies(self, primary_strategy: str) -> list[dict[str, str]]:
        """
        Get alternative strategies for a given primary strategy.

        Args:
            primary_strategy: Primary strategy name

        Returns:
            List of alternative strategies with reasons
        """
        alternatives = []

        if primary_strategy == "semantic":
            alternatives.append(
                {
                    "strategy": "recursive",
                    "reason": "Faster processing without embeddings",
                }
            )
            alternatives.append(
                {
                    "strategy": "fixed_size",
                    "reason": "Simple and predictable",
                }
            )
        elif primary_strategy == "markdown":
            alternatives.append(
                {
                    "strategy": "recursive",
                    "reason": "Works with any text format",
                }
            )
            alternatives.append(
                {
                    "strategy": "document_structure",
                    "reason": "General document structure preservation",
                }
            )
        elif primary_strategy == "fixed_size":
            alternatives.append(
                {
                    "strategy": "recursive",
                    "reason": "Better structure preservation",
                }
            )
            alternatives.append(
                {
                    "strategy": "sliding_window",
                    "reason": "Better context overlap",
                }
            )
        else:
            # Default alternatives
            alternatives.append(
                {
                    "strategy": "recursive",
                    "reason": "Versatile default strategy",
                }
            )
            alternatives.append(
                {
                    "strategy": "fixed_size",
                    "reason": "Simple and fast",
                }
            )

        return alternatives

    def validate_config_compatibility(
        self,
        strategy: str,
        config: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """
        Check if configuration is compatible with strategy.

        Args:
            strategy: Strategy name
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        is_valid = True

        if strategy == "semantic" and config.get("chunk_size", 0) > 2000:
            warnings.append("Large chunk sizes may reduce semantic coherence")

        if strategy == "markdown" and not config.get("split_by_headers", True):
            warnings.append("Disabling header splitting defeats purpose of markdown strategy")

        if config.get("chunk_overlap", 0) > config.get("chunk_size", 1000) * 0.5:
            warnings.append("High overlap ratio will create many redundant chunks")

        if strategy == "hierarchical" and config.get("max_level", 3) > 5:
            warnings.append("Deep hierarchy levels may create complex output")
            is_valid = False

        return is_valid, warnings

    def store_custom_config(
        self,
        name: str,
        strategy: str,
        config: dict[str, Any],
    ) -> None:
        """
        Store a custom configuration for reuse.

        Args:
            name: Configuration name
            strategy: Associated strategy
            config: Configuration to store
        """
        self.custom_configs[name] = {
            "strategy": strategy,
            "config": config,
        }
        logger.info("Stored custom config '%s' for strategy '%s'", name, strategy)

    def get_custom_config(self, name: str) -> dict[str, Any] | None:
        """
        Retrieve a stored custom configuration.

        Args:
            name: Configuration name

        Returns:
            Stored configuration or None
        """
        return self.custom_configs.get(name)
