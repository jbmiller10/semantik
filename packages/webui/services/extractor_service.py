"""Extractor service for running metadata extraction plugins."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from shared.plugins.loader import load_plugins
from shared.plugins.registry import PluginRecord, plugin_registry
from shared.plugins.types.extractor import (
    ExtractionResult,
    ExtractionType,
    ExtractorPlugin,
)

logger = logging.getLogger(__name__)


class ExtractorService:
    """Service for running metadata extraction on text content.

    This service manages extractor plugins and provides methods to:
    - List available extractors
    - Run extractors on text content
    - Merge results from multiple extractors
    """

    def __init__(self) -> None:
        """Initialize the extractor service."""
        self._extractor_instances: dict[str, ExtractorPlugin] = {}

    def get_available_extractors(self) -> list[PluginRecord]:
        """Get list of available extractor plugins.

        Returns:
            List of PluginRecord for registered extractor plugins.
        """
        # Ensure plugins are loaded
        load_plugins(plugin_types={"extractor"})

        return plugin_registry.list_records(plugin_type="extractor")

    def get_extractor(self, extractor_id: str) -> PluginRecord | None:
        """Get a specific extractor by ID.

        Args:
            extractor_id: The plugin ID of the extractor.

        Returns:
            PluginRecord if found, None otherwise.
        """
        load_plugins(plugin_types={"extractor"})
        return plugin_registry.get("extractor", extractor_id)

    async def _get_extractor_instance(
        self,
        extractor_id: str,
        config: dict[str, Any] | None = None,
    ) -> ExtractorPlugin | None:
        """Get or create an extractor plugin instance.

        Args:
            extractor_id: The plugin ID.
            config: Optional configuration for the plugin.

        Returns:
            Initialized ExtractorPlugin instance or None if not found.
        """
        # Check cache first - use JSON for stable hashing of nested structures
        config_hash = hashlib.md5(
            json.dumps(config or {}, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        cache_key = f"{extractor_id}:{config_hash}"
        if cache_key in self._extractor_instances:
            return self._extractor_instances[cache_key]

        # Get plugin record
        record = self.get_extractor(extractor_id)
        if record is None:
            logger.warning("Extractor plugin not found: %s", extractor_id)
            return None

        # Create instance
        plugin_cls = record.plugin_class
        try:
            instance = plugin_cls(config=config)
            await instance.initialize(config)
            self._extractor_instances[cache_key] = instance
            return instance
        except Exception as e:
            logger.error("Failed to create extractor instance %s: %s", extractor_id, e)
            return None

    async def run_extractors(
        self,
        text: str,
        extractor_ids: list[str],
        extraction_types: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Run multiple extractors on text and merge results.

        Args:
            text: Text content to extract from.
            extractor_ids: List of extractor plugin IDs to run.
            extraction_types: Which extraction types to perform (as strings).
            options: Options passed to extractors.

        Returns:
            Merged ExtractionResult from all extractors.
        """
        if not text or not text.strip():
            return ExtractionResult()

        if not extractor_ids:
            return ExtractionResult()

        # Convert extraction type strings to enum
        types_enum: list[ExtractionType] | None = None
        if extraction_types:
            types_enum = []
            for type_str in extraction_types:
                try:
                    types_enum.append(ExtractionType(type_str))
                except ValueError:
                    logger.warning("Unknown extraction type: %s", type_str)

        # Run each extractor
        results: list[ExtractionResult] = []
        for extractor_id in extractor_ids:
            extractor = await self._get_extractor_instance(extractor_id, options)
            if extractor is None:
                continue

            try:
                result = await extractor.extract(text, types_enum, options)
                results.append(result)
                logger.debug(
                    "Extractor %s extracted: %d keywords, %d entities",
                    extractor_id,
                    len(result.keywords),
                    len(result.entities),
                )
            except Exception as e:
                logger.error("Extractor %s failed: %s", extractor_id, e)
                continue

        # Merge results
        return self.merge_results(results)

    def merge_results(self, results: list[ExtractionResult]) -> ExtractionResult:
        """Merge multiple extraction results into one.

        Args:
            results: List of ExtractionResult to merge.

        Returns:
            Single merged ExtractionResult.
        """
        if not results:
            return ExtractionResult()

        if len(results) == 1:
            return results[0]

        # Start with first result and merge others
        merged = results[0]
        for result in results[1:]:
            merged = merged.merge(result)

        return merged

    async def extract_for_collection(
        self,
        text: str,
        extraction_config: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Run extraction based on collection's extraction_config.

        This is the main entry point for ingestion integration.

        Args:
            text: Text content to extract from.
            extraction_config: Collection's extraction_config field.
                Expected schema: {
                    "enabled": bool,
                    "extractor_ids": ["keyword-extractor", ...],
                    "types": ["keywords", "entities", ...],
                    "options": {...}
                }

        Returns:
            Searchable dict from ExtractionResult, or None if disabled/empty.
        """
        if not extraction_config:
            return None

        if not extraction_config.get("enabled", False):
            return None

        extractor_ids = extraction_config.get("extractor_ids", [])
        if not extractor_ids:
            return None

        extraction_types = extraction_config.get("types")
        options = extraction_config.get("options")

        result = await self.run_extractors(
            text=text,
            extractor_ids=extractor_ids,
            extraction_types=extraction_types,
            options=options,
        )

        # Convert to searchable dict
        searchable = result.to_searchable_dict()
        if not searchable:
            return None

        return searchable

    async def cleanup(self) -> None:
        """Clean up all extractor instances."""
        for extractor_id, instance in self._extractor_instances.items():
            try:
                await instance.cleanup()
            except Exception as e:
                logger.warning("Failed to cleanup extractor %s: %s", extractor_id, e)

        self._extractor_instances.clear()


# Module-level service instance for convenience
_extractor_service: ExtractorService | None = None


def get_extractor_service() -> ExtractorService:
    """Get or create the singleton ExtractorService instance."""
    global _extractor_service
    if _extractor_service is None:
        _extractor_service = ExtractorService()
    return _extractor_service
