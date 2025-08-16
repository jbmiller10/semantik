"""
Chunking configuration builder for managing and validating configurations.

This module provides a centralized way to build, validate, and manage
chunking configurations, removing business logic from routers.
"""

from dataclasses import dataclass
from typing import Any

from packages.webui.api.v2.chunking_schemas import ChunkingStrategy as ChunkingStrategyEnum


class ChunkingConfigBuilder:
    """Builds and validates chunking configurations."""

    # Default configurations per strategy
    DEFAULT_CONFIGS: dict[ChunkingStrategyEnum, dict[str, Any]] = {
        ChunkingStrategyEnum.FIXED_SIZE: {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "separator": None,
            "keep_separator": False,
        },
        ChunkingStrategyEnum.RECURSIVE: {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "separators": ["\n\n", "\n", " ", ""],
            "keep_separator": True,
        },
        ChunkingStrategyEnum.DOCUMENT_STRUCTURE: {
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "preserve_headers": True,
            "preserve_code_blocks": True,
            "min_header_level": 1,
            "max_header_level": 6,
        },
        ChunkingStrategyEnum.SEMANTIC: {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "similarity_threshold": 0.7,
            "min_chunk_size": 100,
            "max_chunk_size": 1000,
            "embedding_model": "default",
        },
        ChunkingStrategyEnum.SLIDING_WINDOW: {
            "chunk_size": 500,
            "chunk_overlap": 200,
            "window_step": 300,
            "preserve_sentences": True,
        },
        ChunkingStrategyEnum.HYBRID: {
            "primary_strategy": "semantic",
            "fallback_strategy": "recursive",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "switch_threshold": 0.5,
        },
    }

    @dataclass
    class ChunkingConfigResult:
        """Result of configuration building."""

        strategy: ChunkingStrategyEnum
        config: dict[str, Any]
        validation_errors: list[str] | None = None
        warnings: list[str] | None = None

    def build_config(
        self,
        strategy: str | ChunkingStrategyEnum,
        user_config: dict[str, Any] | None = None,
    ) -> ChunkingConfigResult:
        """
        Build configuration for a chunking strategy.

        Args:
            strategy: Strategy name or enum
            user_config: User-provided configuration overrides

        Returns:
            ChunkingConfigResult with validated configuration
        """
        # Handle string or enum input
        if isinstance(strategy, str):
            try:
                strategy_enum = ChunkingStrategyEnum(strategy.lower())
            except ValueError:
                # Try to map common variations
                mapped_strategy = self._map_strategy_name(strategy)
                if not mapped_strategy:
                    return self.ChunkingConfigResult(
                        strategy=ChunkingStrategyEnum.RECURSIVE,  # Default fallback
                        config={},
                        validation_errors=[f"Unknown strategy: {strategy}"],
                    )
                strategy_enum = mapped_strategy
        else:
            strategy_enum = strategy

        # Get default config
        config = self.DEFAULT_CONFIGS.get(strategy_enum, {}).copy()

        # Apply user overrides
        if user_config:
            config = self._merge_configs(config, user_config)

        # Validate configuration
        errors = self._validate_config(strategy_enum, config)
        warnings = self._check_config_warnings(strategy_enum, config)

        if errors:
            return self.ChunkingConfigResult(
                strategy=strategy_enum,
                config=config,
                validation_errors=errors,
                warnings=warnings,
            )

        return self.ChunkingConfigResult(
            strategy=strategy_enum,
            config=config,
            warnings=warnings if warnings else None,
        )

    def _map_strategy_name(self, strategy: str) -> ChunkingStrategyEnum | None:
        """Map common strategy name variations to enum values."""
        strategy_mapping = {
            "fixed": ChunkingStrategyEnum.FIXED_SIZE,
            "fixed_size": ChunkingStrategyEnum.FIXED_SIZE,
            "character": ChunkingStrategyEnum.FIXED_SIZE,
            "recursive": ChunkingStrategyEnum.RECURSIVE,
            "recursive_text": ChunkingStrategyEnum.RECURSIVE,
            "markdown": ChunkingStrategyEnum.DOCUMENT_STRUCTURE,
            "document_structure": ChunkingStrategyEnum.DOCUMENT_STRUCTURE,
            "document": ChunkingStrategyEnum.DOCUMENT_STRUCTURE,
            "semantic": ChunkingStrategyEnum.SEMANTIC,
            "ai": ChunkingStrategyEnum.SEMANTIC,
            "sliding": ChunkingStrategyEnum.SLIDING_WINDOW,
            "sliding_window": ChunkingStrategyEnum.SLIDING_WINDOW,
            "window": ChunkingStrategyEnum.SLIDING_WINDOW,
            "hybrid": ChunkingStrategyEnum.HYBRID,
            "mixed": ChunkingStrategyEnum.HYBRID,
        }
        return strategy_mapping.get(strategy.lower())

    def _merge_configs(self, default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Merge user config with defaults."""
        result = default.copy()

        for key, value in override.items():
            if key in result:
                # Type checking and conversion
                if type(result[key]) is not type(value):
                    # Try to convert
                    try:
                        if isinstance(result[key], int):
                            value = int(value)
                        elif isinstance(result[key], float):
                            value = float(value)
                        elif isinstance(result[key], bool):
                            value = bool(value)
                        elif isinstance(result[key], list) and not isinstance(value, list):
                            value = [value]
                    except (ValueError, TypeError):
                        continue  # Skip invalid types

                result[key] = value
            else:
                # Allow additional parameters but track them
                result[key] = value

        return result

    def _validate_config(self, strategy: ChunkingStrategyEnum, config: dict[str, Any]) -> list[str]:
        """Validate configuration for a strategy."""
        errors = []

        # Common validations
        if "chunk_size" in config:
            chunk_size = config["chunk_size"]
            if not isinstance(chunk_size, int) or chunk_size < 10:
                errors.append("chunk_size must be at least 10")
            if chunk_size > 100000:
                errors.append("chunk_size cannot exceed 100000")

        if "chunk_overlap" in config:
            overlap = config["chunk_overlap"]
            if not isinstance(overlap, int) or overlap < 0:
                errors.append("chunk_overlap cannot be negative")
            if config.get("chunk_size") and overlap >= config["chunk_size"]:
                errors.append("chunk_overlap must be less than chunk_size")

        # Strategy-specific validations
        if strategy == ChunkingStrategyEnum.SEMANTIC:
            if config.get("similarity_threshold"):
                threshold = config["similarity_threshold"]
                if not 0 <= threshold <= 1:
                    errors.append("similarity_threshold must be between 0 and 1")

            if (
                config.get("min_chunk_size")
                and config.get("max_chunk_size")
                and config["min_chunk_size"] > config["max_chunk_size"]
            ):
                errors.append("min_chunk_size must be <= max_chunk_size")

        elif strategy == ChunkingStrategyEnum.SLIDING_WINDOW:
            if config.get("window_step"):
                step = config["window_step"]
                chunk_size = config.get("chunk_size", 500)
                if step <= 0:
                    errors.append("window_step must be positive")
                if step > chunk_size:
                    errors.append("window_step should not exceed chunk_size")

        elif strategy == ChunkingStrategyEnum.HYBRID:
            if config.get("switch_threshold"):
                threshold = config["switch_threshold"]
                if not 0 <= threshold <= 1:
                    errors.append("switch_threshold must be between 0 and 1")

        elif strategy == ChunkingStrategyEnum.DOCUMENT_STRUCTURE:
            min_level = config.get("min_header_level", 1)
            max_level = config.get("max_header_level", 6)
            if min_level > max_level:
                errors.append("min_header_level must be <= max_header_level")
            if min_level < 1 or max_level > 6:
                errors.append("header levels must be between 1 and 6")

        return errors

    def _check_config_warnings(self, strategy: ChunkingStrategyEnum, config: dict[str, Any]) -> list[str]:
        """Check for configuration issues that are warnings, not errors."""
        warnings = []

        # Check for very small chunks
        if config.get("chunk_size", 500) < 100:
            warnings.append("Very small chunk_size may impact search quality")

        # Check for very large overlap
        if (
            config.get("chunk_overlap")
            and config.get("chunk_size")
            and config["chunk_overlap"] / config["chunk_size"] > 0.5
        ):
            overlap_ratio = config["chunk_overlap"] / config["chunk_size"]
            warnings.append(f"Large overlap ratio ({overlap_ratio:.1%}) may cause redundancy")

        # Strategy-specific warnings
        if strategy == ChunkingStrategyEnum.SEMANTIC and not config.get("embedding_model"):
            warnings.append("No embedding model specified, using default")

        elif strategy == ChunkingStrategyEnum.HYBRID and config.get("primary_strategy") == config.get(
            "fallback_strategy"
        ):
            warnings.append("Primary and fallback strategies are the same")

        return warnings

    def get_default_config(self, strategy: ChunkingStrategyEnum) -> dict[str, Any]:
        """Get default configuration for a strategy."""
        return self.DEFAULT_CONFIGS.get(strategy, {}).copy()

    def validate_parameter(
        self,
        param_name: str,
        param_value: Any,
        strategy: ChunkingStrategyEnum,  # noqa: ARG002
    ) -> str | None:
        """Validate a single parameter.

        Args:
            param_name: Parameter name
            param_value: Parameter value
            strategy: Strategy context

        Returns:
            Error message if invalid, None if valid
        """
        # Common parameter validations
        if param_name == "chunk_size":
            if not isinstance(param_value, int) or param_value < 10:
                return "chunk_size must be an integer >= 10"
            if param_value > 100000:
                return "chunk_size cannot exceed 100000"

        elif param_name == "chunk_overlap":
            if not isinstance(param_value, int) or param_value < 0:
                return "chunk_overlap must be a non-negative integer"

        elif param_name == "similarity_threshold":
            if not isinstance(param_value, int | float) or not 0 <= param_value <= 1:
                return "similarity_threshold must be between 0 and 1"

        elif param_name == "separators":
            if not isinstance(param_value, list):
                return "separators must be a list"
            if not all(isinstance(s, str) for s in param_value):
                return "all separators must be strings"

        return None

    def suggest_config(
        self,
        file_type: str | None = None,
        content_size: int | None = None,
        use_case: str | None = None,
    ) -> ChunkingConfigResult:
        """Suggest optimal configuration based on context.

        Args:
            file_type: Type of file being processed
            content_size: Size of content in bytes
            use_case: Intended use case (search, analysis, etc.)

        Returns:
            Suggested configuration
        """
        # Determine best strategy based on context
        strategy: ChunkingStrategyEnum
        config: dict[str, Any]

        if file_type in [".md", ".markdown", ".rst", ".tex"]:
            strategy = ChunkingStrategyEnum.DOCUMENT_STRUCTURE
            config = {
                "chunk_size": 800 if content_size and content_size < 50000 else 1200,
                "chunk_overlap": 100,
                "preserve_headers": True,
            }
        elif file_type in [".pdf", ".docx", ".html"]:
            strategy = ChunkingStrategyEnum.SEMANTIC
            config = {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "similarity_threshold": 0.75,
            }
        elif use_case == "code_analysis" or file_type in [".py", ".js", ".java"]:
            strategy = ChunkingStrategyEnum.RECURSIVE
            config = {
                "chunk_size": 400,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", " "],
            }
        elif content_size and content_size > 1000000:  # Large files
            strategy = ChunkingStrategyEnum.SLIDING_WINDOW
            config = {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "window_step": 800,
            }
        else:
            # Default to recursive for general text
            strategy = ChunkingStrategyEnum.RECURSIVE
            config = {
                "chunk_size": 600,
                "chunk_overlap": 100,
            }

        return self.ChunkingConfigResult(
            strategy=strategy,
            config=config,
        )
