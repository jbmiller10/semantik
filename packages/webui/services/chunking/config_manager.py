"""
Chunking configuration manager service.

Handles strategy configuration, defaults, and recommendations.
"""

import logging
from typing import Any

from packages.webui.services.chunking.strategy_registry import (
    get_strategy_defaults,
    get_strategy_metadata,
    list_api_strategy_ids,
)

logger = logging.getLogger(__name__)


class ChunkingConfigManager:
    """Service responsible for chunking configuration management."""

    def __init__(self) -> None:
        """Initialize the configuration manager."""
        self.custom_configs: dict[str, dict[str, Any]] = {}

    def get_default_config(self, strategy: str) -> dict[str, Any]:
        """
        Get default configuration for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            Default configuration dictionary
        """
        return get_strategy_defaults(strategy, context="manager")

    def get_strategy_info(self, strategy: str) -> dict[str, Any]:
        """
        Get metadata information for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            Strategy information dictionary
        """
        metadata = get_strategy_metadata(strategy)
        info: dict[str, Any] = metadata.copy() if metadata else {}
        info["id"] = strategy
        info["default_config"] = self.get_default_config(strategy)
        return info

    def get_all_strategies(self) -> list[dict[str, Any]]:
        """
        Get information for all available strategies.

        Returns:
            List of strategy information dictionaries
        """
        return [self.get_strategy_info(strategy_id) for strategy_id in list_api_strategy_ids()]

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
        recommendation: dict[str, Any] = {
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
                reasoning_list = recommendation["reasoning"]
                if isinstance(reasoning_list, list):
                    reasoning_list.append("Markdown file detected")
            elif file_type in [".py", ".js", ".java", ".cpp", ".c", ".go"]:
                recommendation["strategy"] = "recursive"
                recommendation["confidence"] = 0.8
                reasoning_list = recommendation["reasoning"]
                if isinstance(reasoning_list, list):
                    reasoning_list.append("Code file detected")
                alternatives_list = recommendation["alternatives"]
                if isinstance(alternatives_list, list):
                    alternatives_list.append("fixed_size")
            elif file_type in [".pdf", ".docx", ".doc"]:
                recommendation["strategy"] = "document_structure"
                recommendation["confidence"] = 0.7
                reasoning_list = recommendation["reasoning"]
                if isinstance(reasoning_list, list):
                    reasoning_list.append("Structured document format")

        # Content length based adjustments
        if content_length:
            if content_length < 1000:
                recommendation["strategy"] = "fixed_size"
                confidence = recommendation.get("confidence", 0.5)
                if isinstance(confidence, int | float):
                    recommendation["confidence"] = max(confidence, 0.6)
                reasoning_list = recommendation["reasoning"]
                if isinstance(reasoning_list, list):
                    reasoning_list.append("Short content - simple chunking sufficient")
            elif content_length > 100000:
                if recommendation["strategy"] == "recursive":
                    alternatives_list = recommendation["alternatives"]
                    if isinstance(alternatives_list, list):
                        alternatives_list.append("hierarchical")
                    reasoning_list = recommendation["reasoning"]
                    if isinstance(reasoning_list, list):
                        reasoning_list.append("Large document - consider hierarchical")

        # Document type based adjustments
        if document_type:
            document_type = document_type.lower()
            if document_type in ["technical", "research", "academic"]:
                recommendation["strategy"] = "semantic"
                recommendation["confidence"] = 0.8
                reasoning_list = recommendation["reasoning"]
                if isinstance(reasoning_list, list):
                    reasoning_list.append("Technical content benefits from semantic chunking")
            elif document_type == "narrative":
                recommendation["strategy"] = "recursive"
                recommendation["confidence"] = 0.7
                reasoning_list = recommendation["reasoning"]
                if isinstance(reasoning_list, list):
                    reasoning_list.append("Narrative text works well with recursive splitting")

        # Add configuration suggestion
        strategy_str = recommendation.get("strategy", "recursive")
        if isinstance(strategy_str, str):
            recommendation["suggested_config"] = self.get_default_config(strategy_str)

        # Adjust chunk size based on content length
        if content_length and content_length < 5000:
            suggested_config = recommendation.get("suggested_config")
            if isinstance(suggested_config, dict):
                suggested_config["chunk_size"] = 500
                suggested_config["chunk_overlap"] = 100

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
