"""Canonical chunking strategy registry.

This module centralizes all chunking strategy metadata, alias mappings, and
default configuration values so that the service layer, configuration
utilities, and factories can share a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Iterable, Literal, Mapping, MutableMapping, Sequence

from copy import deepcopy

from packages.webui.api.v2.chunking_schemas import ChunkingStrategy

DefaultContext = Literal["manager", "builder", "factory"]


def _copy_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a defensive copy of the provided mapping."""

    if not mapping:
        return {}
    return deepcopy(dict(mapping))


@dataclass(frozen=True)
class StrategyDefinition:
    """Canonical description of an individual chunking strategy."""

    api_id: ChunkingStrategy
    internal_id: str
    display_name: str
    description: str
    best_for: tuple[str, ...]
    pros: tuple[str, ...]
    cons: tuple[str, ...]
    performance_characteristics: Mapping[str, Any]
    manager_defaults: Mapping[str, Any]
    builder_defaults: Mapping[str, Any] | None
    supported_file_types: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return strategy metadata in dictionary form."""

        return {
            "name": self.display_name,
            "description": self.description,
            "best_for": list(self.best_for),
            "pros": list(self.pros),
            "cons": list(self.cons),
            "performance_characteristics": _copy_mapping(self.performance_characteristics),
            "supported_file_types": list(self.supported_file_types),
        }


# Default configuration used by the shared chunking strategy implementations.
_FACTORY_DEFAULTS: dict[str, dict[str, Any]] = {
    "character": {"chunk_size": 1000, "chunk_overlap": 200},
    "recursive": {"chunk_size": 1000, "chunk_overlap": 200},
    "markdown": {"chunk_size": 1000, "chunk_overlap": 200},
    "semantic": {"buffer_size": 1, "breakpoint_percentile_threshold": 95},
    "hierarchical": {"chunk_sizes": [2048, 512], "chunk_overlap": 50},
    "hybrid": {"primary_strategy": "recursive", "fallback_strategy": "character"},
}


def _strategy_definition_data() -> dict[str, StrategyDefinition]:
    """Construct the canonical definitions used throughout the service."""

    return {
        ChunkingStrategy.FIXED_SIZE.value: StrategyDefinition(
            api_id=ChunkingStrategy.FIXED_SIZE,
            internal_id="character",
            display_name="Fixed Size Chunking",
            description="Simple fixed-size chunking with consistent chunk sizes",
            best_for=("txt", "log", "csv", "json"),
            pros=(
                "Predictable chunk sizes",
                "Fast processing",
                "Low memory usage",
                "Good for structured data",
            ),
            cons=(
                "May split sentences or paragraphs",
                "No semantic coherence",
                "Can break context",
            ),
            performance_characteristics={"speed": "very_fast", "memory_usage": "low", "quality": "moderate"},
            manager_defaults={
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separator": "\n",
            },
            builder_defaults={
                "chunk_size": 500,
                "chunk_overlap": 50,
                "separator": None,
                "keep_separator": False,
            },
            supported_file_types=(),
            aliases=("fixed", "character", "char"),
        ),
        ChunkingStrategy.SEMANTIC.value: StrategyDefinition(
            api_id=ChunkingStrategy.SEMANTIC,
            internal_id="semantic",
            display_name="Semantic",
            description="Uses embeddings to find natural semantic boundaries",
            best_for=("pdf", "docx", "md", "html", "tex"),
            pros=(
                "Maintains semantic coherence",
                "Better search quality",
                "Context preservation",
                "Intelligent boundaries",
            ),
            cons=(
                "Slower processing",
                "Higher memory usage",
                "Requires embedding model",
                "Variable chunk sizes",
            ),
            performance_characteristics={"speed": "slow", "memory_usage": "high", "quality": "excellent"},
            manager_defaults={
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "embedding_model": "sentence-transformers",
                "similarity_threshold": 0.8,
            },
            builder_defaults={
                "chunk_size": 512,
                "chunk_overlap": 50,
                "similarity_threshold": 0.7,
                "min_chunk_size": 100,
                "max_chunk_size": 1000,
                "embedding_model": "default",
            },
            supported_file_types=(),
            aliases=("semantic_chunking", "semantic"),
        ),
        ChunkingStrategy.RECURSIVE.value: StrategyDefinition(
            api_id=ChunkingStrategy.RECURSIVE,
            internal_id="recursive",
            display_name="Recursive",
            description="Recursively splits text using multiple separators",
            best_for=("md", "rst", "txt", "code files"),
            pros=(
                "Respects document structure",
                "Good balance of speed and quality",
                "Handles nested content well",
                "Preserves formatting",
            ),
            cons=(
                "May produce variable sizes",
                "Complex configuration",
                "Not ideal for unstructured text",
            ),
            performance_characteristics={"speed": "fast", "memory_usage": "moderate", "quality": "good"},
            manager_defaults={
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", " ", ""],
            },
            builder_defaults={
                "chunk_size": 500,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", " ", ""],
                "keep_separator": True,
            },
            supported_file_types=(),
            aliases=("recursive_text", "recursive"),
        ),
        ChunkingStrategy.SLIDING_WINDOW.value: StrategyDefinition(
            api_id=ChunkingStrategy.SLIDING_WINDOW,
            internal_id="character",
            display_name="Sliding Window",
            description="Overlapping chunks with configurable window size",
            best_for=("txt", "log", "transcript", "chat"),
            pros=(
                "No information loss at boundaries",
                "Good for continuous text",
                "Adjustable overlap",
                "Context preservation",
            ),
            cons=(
                "Redundant information",
                "More storage required",
                "Slower search",
                "Higher costs",
            ),
            performance_characteristics={"speed": "moderate", "memory_usage": "high", "quality": "good"},
            manager_defaults={
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "stride": 800,
            },
            builder_defaults={
                "chunk_size": 500,
                "chunk_overlap": 200,
                "window_step": 300,
                "preserve_sentences": True,
            },
            supported_file_types=(),
            aliases=("sliding", "window", "sliding_window"),
        ),
        ChunkingStrategy.DOCUMENT_STRUCTURE.value: StrategyDefinition(
            api_id=ChunkingStrategy.DOCUMENT_STRUCTURE,
            internal_id="markdown",
            display_name="Document Structure",
            description="Splits documents based on structural elements",
            best_for=("pdf", "docx", "html", "epub", "tex"),
            pros=(
                "Preserves document hierarchy",
                "Natural boundaries",
                "Maintains formatting",
                "Good for structured documents",
            ),
            cons=(
                "Requires document parsing",
                "May produce very large chunks",
                "Not suitable for plain text",
                "Complex implementation",
            ),
            performance_characteristics={"speed": "moderate", "memory_usage": "moderate", "quality": "very_good"},
            manager_defaults={
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "preserve_structure": True,
            },
            builder_defaults={
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "preserve_headers": True,
                "preserve_code_blocks": True,
                "min_header_level": 1,
                "max_header_level": 6,
            },
            supported_file_types=(),
            aliases=("document", "document_structure"),
        ),
        ChunkingStrategy.MARKDOWN.value: StrategyDefinition(
            api_id=ChunkingStrategy.MARKDOWN,
            internal_id="markdown",
            display_name="Markdown",
            description="Respects markdown structure and headings",
            best_for=("md", "markdown", "mdx"),
            pros=("Preserves structure", "Good for technical docs"),
            cons=("Only for markdown-like content",),
            performance_characteristics={"speed": "moderate", "memory_usage": "moderate", "quality": "very_good"},
            manager_defaults={
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "split_by_headers": True,
                "min_header_level": 1,
                "max_header_level": 3,
            },
            builder_defaults=None,
            supported_file_types=(),
            aliases=("markdown", "md"),
        ),
        ChunkingStrategy.HIERARCHICAL.value: StrategyDefinition(
            api_id=ChunkingStrategy.HIERARCHICAL,
            internal_id="hierarchical",
            display_name="Hierarchical",
            description="Creates parent-child chunks across multiple levels",
            best_for=("large documents", "books", "reports"),
            pros=("Multiple granularities", "Scalable for large docs"),
            cons=("Complex", "More storage"),
            performance_characteristics={"speed": "slow", "memory_usage": "high", "quality": "excellent"},
            manager_defaults={
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "max_level": 3,
                "level_separator": "\n\n",
            },
            builder_defaults=None,
            supported_file_types=(),
            aliases=("hierarchy", "hierarchical"),
        ),
        ChunkingStrategy.HYBRID.value: StrategyDefinition(
            api_id=ChunkingStrategy.HYBRID,
            internal_id="hybrid",
            display_name="Hybrid",
            description="Combines multiple strategies based on content analysis",
            best_for=("mixed content", "unknown formats", "large documents"),
            pros=(
                "Adaptive to content",
                "Best of multiple strategies",
                "Handles diverse content",
                "Optimal quality",
            ),
            cons=(
                "Complex configuration",
                "Slower processing",
                "Higher resource usage",
                "Unpredictable behavior",
            ),
            performance_characteristics={"speed": "slow", "memory_usage": "very_high", "quality": "excellent"},
            manager_defaults={
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "primary_strategy": "semantic",
                "fallback_strategy": "recursive",
            },
            builder_defaults={
                "primary_strategy": "semantic",
                "fallback_strategy": "recursive",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "switch_threshold": 0.5,
            },
            supported_file_types=(),
            aliases=("mixed", "hybrid"),
        ),
    }


_STRATEGIES: dict[str, StrategyDefinition] = _strategy_definition_data()


@lru_cache(maxsize=1)
def get_api_to_internal_map() -> dict[str, str]:
    """Return mapping from API strategy identifiers to internal strategy names."""

    return {key: definition.internal_id for key, definition in _STRATEGIES.items()}


@lru_cache(maxsize=1)
def get_internal_to_primary_api_map() -> dict[str, str]:
    """Return mapping from internal strategy name to canonical API identifier."""

    mapping: dict[str, str] = {}
    for api_id, definition in _STRATEGIES.items():
        mapping.setdefault(definition.internal_id, api_id)
    return mapping


@lru_cache(maxsize=1)
def get_alias_to_api_map() -> dict[str, str]:
    """Return mapping for resolving user-provided aliases to API identifiers."""

    alias_map: dict[str, str] = {}
    for api_id, definition in _STRATEGIES.items():
        alias_map[api_id] = api_id
        alias_map[definition.api_id.name.lower()] = api_id
        for alias in definition.aliases:
            alias_map[alias.lower()] = api_id
        # Allow lookup by internal strategy name as a convenience
        alias_map.setdefault(definition.internal_id.lower(), api_id)
    return alias_map


def _normalize_identifier(identifier: str | Enum | ChunkingStrategy) -> str:
    """Normalise any identifier into a lower-case string."""

    if isinstance(identifier, Enum):
        value = getattr(identifier, "value", identifier.name)
        if isinstance(value, str):
            return value.lower()
        return str(value).lower()
    return str(identifier).lower()


def _resolve_strategy_definition(identifier: str | Enum | ChunkingStrategy) -> StrategyDefinition | None:
    """Resolve arbitrary identifier to a canonical strategy definition."""

    normalized = _normalize_identifier(identifier)
    alias_map = get_alias_to_api_map()
    api_id = alias_map.get(normalized)

    if api_id is None and normalized in get_api_to_internal_map():
        api_id = normalized

    if api_id is None:
        # Attempt reverse lookup via internal mapping when identifier is internal
        primary_api = get_internal_to_primary_api_map().get(normalized)
        if primary_api:
            api_id = primary_api

    if api_id is None:
        return None

    return _STRATEGIES.get(api_id)


def get_strategy_definition(identifier: str | Enum | ChunkingStrategy) -> StrategyDefinition | None:
    """Expose canonical strategy definition for the given identifier."""

    return _resolve_strategy_definition(identifier)


def list_strategy_definitions() -> Iterable[StrategyDefinition]:
    """Iterate over all canonical strategy definitions."""

    return _STRATEGIES.values()


def list_strategy_metadata() -> list[dict[str, Any]]:
    """Return metadata for all strategies formatted for API consumers."""

    results: list[dict[str, Any]] = []
    for definition in list_strategy_definitions():
        metadata = definition.to_metadata_dict()
        metadata["id"] = definition.api_id.value
        metadata["default_config"] = get_strategy_defaults(definition.api_id, context="manager")
        results.append(metadata)
    return results


def get_strategy_defaults(identifier: str | Enum | ChunkingStrategy, *, context: DefaultContext) -> dict[str, Any]:
    """Return default configuration dictionary for the given strategy/context."""

    context_key = context.lower()

    if context_key == "factory":
        internal_name = resolve_internal_strategy_name(identifier)
        if not internal_name:
            return {}
        return _copy_mapping(_FACTORY_DEFAULTS.get(internal_name))

    definition = _resolve_strategy_definition(identifier)
    if not definition:
        return {}

    if context_key == "builder":
        return _copy_mapping(definition.builder_defaults)

    return _copy_mapping(definition.manager_defaults)


def resolve_internal_strategy_name(identifier: str | Enum | ChunkingStrategy) -> str | None:
    """Resolve identifier to its internal strategy name used by shared services."""

    definition = _resolve_strategy_definition(identifier)
    if definition:
        return definition.internal_id

    normalized = _normalize_identifier(identifier)
    if normalized in _FACTORY_DEFAULTS:
        return normalized

    return None


def resolve_api_identifier(identifier: str | Enum | ChunkingStrategy) -> str | None:
    """Resolve identifier to canonical API strategy identifier."""

    definition = _resolve_strategy_definition(identifier)
    if definition:
        return definition.api_id.value

    normalized = _normalize_identifier(identifier)
    alias_map = get_alias_to_api_map()
    if normalized in alias_map:
        return alias_map[normalized]

    return None


def get_strategy_metadata(identifier: str | Enum | ChunkingStrategy) -> dict[str, Any]:
    """Return metadata dictionary for a strategy identifier."""

    definition = _resolve_strategy_definition(identifier)
    if not definition:
        return {}
    return definition.to_metadata_dict()


def get_internal_strategy_aliases() -> dict[str, set[str]]:
    """Return aliases grouped by internal strategy name."""

    grouped: dict[str, set[str]] = {}
    alias_map = get_alias_to_api_map()
    for alias, api_id in alias_map.items():
        internal = _STRATEGIES[api_id].internal_id
        grouped.setdefault(internal, set()).add(alias)
    return grouped


def recommend_strategy(file_types: Sequence[str] | None) -> ChunkingStrategy:
    """Recommend a strategy using the canonical metadata."""

    if not file_types:
        return ChunkingStrategy.RECURSIVE

    scores: MutableMapping[str, int] = {definition.api_id.value: 0 for definition in list_strategy_definitions()}

    for raw_file_type in file_types:
        file_type = (raw_file_type or "").strip().lower().lstrip(".")
        if not file_type:
            continue

        for definition in list_strategy_definitions():
            if file_type in definition.best_for:
                scores[definition.api_id.value] = scores.get(definition.api_id.value, 0) + 1

    if not scores:
        return ChunkingStrategy.RECURSIVE

    top_api_id = max(scores, key=lambda key: scores[key])
    if scores[top_api_id] == 0:
        return ChunkingStrategy.RECURSIVE

    return _STRATEGIES[top_api_id].api_id


def get_factory_default_map() -> dict[str, dict[str, Any]]:
    """Expose a defensive copy of factory defaults keyed by internal strategy name."""

    return {key: _copy_mapping(value) for key, value in _FACTORY_DEFAULTS.items()}


def build_metadata_by_enum() -> dict[ChunkingStrategy, dict[str, Any]]:
    """Return metadata keyed by API enum for backwards compatibility helpers."""

    return {definition.api_id: definition.to_metadata_dict() for definition in list_strategy_definitions()}


def list_api_strategy_ids() -> list[str]:
    """Return the ordered list of API strategy identifiers."""

    return [definition.api_id.value for definition in list_strategy_definitions()]

