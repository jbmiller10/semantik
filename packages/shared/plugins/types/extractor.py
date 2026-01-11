"""Extractor plugin base class for metadata extraction from text."""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from shared.plugins.base import SemanticPlugin
from shared.plugins.manifest import PluginManifest


class ExtractionType(Enum):
    """Types of metadata that can be extracted from text."""

    ENTITIES = "entities"
    """Named entities (PERSON, ORG, LOC, DATE, etc.)."""
    KEYWORDS = "keywords"
    """Keywords and keyphrases."""
    LANGUAGE = "language"
    """Language detection."""
    TOPICS = "topics"
    """Topic classification."""
    SENTIMENT = "sentiment"
    """Sentiment analysis (-1.0 to 1.0)."""
    SUMMARY = "summary"
    """Auto-summarization."""
    CUSTOM = "custom"
    """Plugin-specific extractions."""


@dataclass
class Entity:
    """A named entity extracted from text."""

    text: str
    """The entity text as it appears in the document."""
    type: str
    """Entity type (PERSON, ORG, LOC, DATE, MONEY, etc.)."""
    start: int
    """Start character offset in the source text."""
    end: int
    """End character offset in the source text."""
    confidence: float = 1.0
    """Confidence score (0.0 to 1.0)."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional entity-specific metadata."""


@dataclass
class ExtractionResult:
    """Result of metadata extraction from text."""

    entities: list[Entity] = field(default_factory=list)
    """Extracted named entities."""
    keywords: list[str] = field(default_factory=list)
    """Extracted keywords/keyphrases, ordered by relevance."""
    language: str | None = None
    """Detected language code (ISO 639-1, e.g., 'en', 'fr')."""
    language_confidence: float | None = None
    """Language detection confidence (0.0 to 1.0)."""
    topics: list[str] = field(default_factory=list)
    """Classified topics."""
    sentiment: float | None = None
    """Sentiment score (-1.0 = negative, 0.0 = neutral, 1.0 = positive)."""
    summary: str | None = None
    """Auto-generated summary."""
    custom: dict[str, Any] = field(default_factory=dict)
    """Plugin-specific extraction results."""

    def to_searchable_dict(self) -> dict[str, Any]:
        """Convert to dict suitable for search filtering and indexing.

        Returns a flattened structure optimized for Qdrant payload filtering.
        """
        result: dict[str, Any] = {}

        if self.entities:
            # Group entities by type for efficient filtering
            entities_by_type: dict[str, list[str]] = {}
            for entity in self.entities:
                if entity.type not in entities_by_type:
                    entities_by_type[entity.type] = []
                entities_by_type[entity.type].append(entity.text)
            result["entities"] = entities_by_type
            result["entity_types"] = list(entities_by_type.keys())

        if self.keywords:
            result["keywords"] = self.keywords

        if self.language:
            result["language"] = self.language

        if self.topics:
            result["topics"] = self.topics

        if self.sentiment is not None:
            result["sentiment"] = self.sentiment

        if self.custom:
            result["custom"] = self.custom

        return result

    def merge(self, other: ExtractionResult) -> ExtractionResult:
        """Merge another extraction result into this one.

        Useful for combining results from multiple extractors.
        """
        # Merge entities (deduplicate by text+type)
        seen_entities = {(e.text, e.type) for e in self.entities}
        merged_entities = list(self.entities)
        for entity in other.entities:
            key = (entity.text, entity.type)
            if key not in seen_entities:
                merged_entities.append(entity)
                seen_entities.add(key)

        # Merge keywords (deduplicate, preserve order)
        seen_keywords = set(self.keywords)
        merged_keywords = list(self.keywords)
        for kw in other.keywords:
            if kw not in seen_keywords:
                merged_keywords.append(kw)
                seen_keywords.add(kw)

        # Merge topics (deduplicate)
        merged_topics = list(dict.fromkeys(self.topics + other.topics))

        # Use first non-None values for scalars
        merged_language = self.language or other.language
        merged_lang_conf = self.language_confidence or other.language_confidence
        merged_sentiment = self.sentiment if self.sentiment is not None else other.sentiment
        merged_summary = self.summary or other.summary

        # Merge custom dicts
        merged_custom = {**other.custom, **self.custom}

        return ExtractionResult(
            entities=merged_entities,
            keywords=merged_keywords,
            language=merged_language,
            language_confidence=merged_lang_conf,
            topics=merged_topics,
            sentiment=merged_sentiment,
            summary=merged_summary,
            custom=merged_custom,
        )


class ExtractorPlugin(SemanticPlugin, ABC):
    """Base class for metadata extraction plugins.

    Extractors analyze text content to extract structured metadata like
    named entities, keywords, language, topics, and sentiment.

    Example usage:
        extractor = MyExtractorPlugin(config={...})
        await extractor.initialize()

        result = await extractor.extract(
            "Apple Inc. announced new products in Cupertino today.",
            extraction_types=[ExtractionType.ENTITIES, ExtractionType.KEYWORDS]
        )
        # result.entities = [Entity(text="Apple Inc.", type="ORG", ...)]
        # result.keywords = ["Apple", "products", "Cupertino"]
    """

    PLUGIN_TYPE = "extractor"

    @classmethod
    @abstractmethod
    def supported_extractions(cls) -> list[ExtractionType]:
        """Return list of extraction types this plugin supports.

        The plugin should only be asked to perform extractions it supports.
        """

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin manifest for discovery and UI.

        Builds a PluginManifest from the plugin's class variables and capabilities.
        Subclasses may override for custom manifest generation.

        Returns:
            PluginManifest with extractor metadata.
        """
        metadata = getattr(cls, "METADATA", {})

        # Include supported extractions if available
        extractions: list[str] = []
        with contextlib.suppress(TypeError, NotImplementedError):
            extractions = [e.value for e in cls.supported_extractions()]

        capabilities = {"supported_extractions": extractions} if extractions else None

        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name=metadata.get("display_name", cls.PLUGIN_ID),
            description=metadata.get("description", ""),
            author=metadata.get("author"),
            homepage=metadata.get("homepage"),
            capabilities=capabilities,
        )

    @abstractmethod
    async def extract(
        self,
        text: str,
        extraction_types: list[ExtractionType] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Extract metadata from text.

        Args:
            text: Text content to extract from.
            extraction_types: Which extractions to perform. If None, perform
                all supported extractions.
            options: Plugin-specific options (e.g., top_k for keywords).

        Returns:
            ExtractionResult with extracted metadata.
        """

    async def extract_batch(
        self,
        texts: list[str],
        extraction_types: list[ExtractionType] | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[ExtractionResult]:
        """Batch extract metadata from multiple texts.

        Override for optimized batch processing. Default implementation
        calls extract() for each text sequentially.

        Args:
            texts: List of text contents to extract from.
            extraction_types: Which extractions to perform.
            options: Plugin-specific options.

        Returns:
            List of ExtractionResult, one per input text.
        """
        return [await self.extract(text, extraction_types, options) for text in texts]
