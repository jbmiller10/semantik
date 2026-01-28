"""Adapters for legacy plugin classes."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from .manifest import PluginManifest

logger = logging.getLogger(__name__)


def _metadata_value(metadata: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if not metadata:
        return default
    return metadata.get(key, default)


def manifest_from_embedding_plugin(
    plugin_cls: type,
    definition: Any,
) -> PluginManifest:
    """Build a plugin manifest for an embedding provider."""
    if isinstance(definition, Mapping):
        from .dto_adapters import ValidationError, dict_to_embedding_provider_definition

        try:
            definition = dict_to_embedding_provider_definition(dict(definition))
        except ValidationError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid embedding provider definition dict: {exc}") from exc

    metadata = getattr(plugin_cls, "METADATA", {}) or {}
    display_name = _metadata_value(metadata, "display_name", definition.display_name or definition.api_id)
    description = _metadata_value(metadata, "description", definition.description or "")

    # Get AgentHints if available
    agent_hints = None
    if hasattr(plugin_cls, "AGENT_HINTS"):
        agent_hints = plugin_cls.AGENT_HINTS

    return PluginManifest(
        id=definition.api_id,
        type="embedding",
        version=getattr(plugin_cls, "PLUGIN_VERSION", "0.0.0"),
        display_name=str(display_name),
        description=str(description),
        author=_metadata_value(metadata, "author"),
        license=_metadata_value(metadata, "license"),
        homepage=_metadata_value(metadata, "homepage"),
        requires=list(_metadata_value(metadata, "requires", [])),
        semantik_version=_metadata_value(metadata, "semantik_version"),
        capabilities={
            "internal_id": definition.internal_id,
            "provider_type": definition.provider_type,
            "supports_quantization": definition.supports_quantization,
            "supports_instruction": definition.supports_instruction,
            "supports_batch_processing": definition.supports_batch_processing,
            "supports_asymmetric": definition.supports_asymmetric,
            "supported_models": list(definition.supported_models),
            "default_config": dict(definition.default_config),
            "performance": dict(definition.performance_characteristics),
        },
        agent_hints=agent_hints,
    )


def manifest_from_chunking_plugin(
    plugin_cls: type,
    *,
    api_id: str,
    internal_id: str,
) -> PluginManifest:
    """Build a plugin manifest for a chunking strategy."""
    metadata = getattr(plugin_cls, "METADATA", {}) or {}
    display_name = _metadata_value(metadata, "display_name", api_id.replace("_", " ").title())
    description = _metadata_value(metadata, "description", "")

    # Get AgentHints if available
    agent_hints = None
    if hasattr(plugin_cls, "AGENT_HINTS"):
        agent_hints = plugin_cls.AGENT_HINTS

    return PluginManifest(
        id=api_id,
        type="chunking",
        version=getattr(plugin_cls, "PLUGIN_VERSION", "0.0.0"),
        display_name=str(display_name),
        description=str(description),
        author=_metadata_value(metadata, "author"),
        license=_metadata_value(metadata, "license"),
        homepage=_metadata_value(metadata, "homepage"),
        requires=list(_metadata_value(metadata, "requires", [])),
        semantik_version=_metadata_value(metadata, "semantik_version"),
        capabilities={
            "internal_id": internal_id,
            "best_for": list(_metadata_value(metadata, "best_for", [])),
            "pros": list(_metadata_value(metadata, "pros", [])),
            "cons": list(_metadata_value(metadata, "cons", [])),
            "performance_characteristics": dict(_metadata_value(metadata, "performance_characteristics", {})),
            "manager_defaults": dict(_metadata_value(metadata, "manager_defaults", {})),
            "builder_defaults": _metadata_value(metadata, "builder_defaults"),
            "factory_defaults": _metadata_value(metadata, "factory_defaults"),
            "supported_file_types": list(_metadata_value(metadata, "supported_file_types", [])),
            "aliases": list(_metadata_value(metadata, "aliases", [])),
            "visual_example": _metadata_value(metadata, "visual_example"),
        },
        agent_hints=agent_hints,
    )


def manifest_from_connector_plugin(plugin_cls: type, plugin_id: str) -> PluginManifest:
    """Build a plugin manifest for a connector."""
    metadata = getattr(plugin_cls, "METADATA", {}) or {}
    display_name = _metadata_value(metadata, "name", plugin_id.replace("_", " ").title())
    description = _metadata_value(metadata, "description", "")
    return PluginManifest(
        id=plugin_id,
        type="connector",
        version=getattr(plugin_cls, "PLUGIN_VERSION", "0.0.0"),
        display_name=str(display_name),
        description=str(description),
        author=_metadata_value(metadata, "author"),
        license=_metadata_value(metadata, "license"),
        homepage=_metadata_value(metadata, "homepage"),
        requires=list(_metadata_value(metadata, "requires", [])),
        semantik_version=_metadata_value(metadata, "semantik_version"),
        capabilities={
            "icon": _metadata_value(metadata, "icon"),
            "supports_sync": _metadata_value(metadata, "supports_sync"),
            "preview_endpoint": _metadata_value(metadata, "preview_endpoint"),
        },
    )


def manifest_from_reranker_plugin(plugin_cls: type, plugin_id: str) -> PluginManifest:
    """Build a plugin manifest for a reranker."""
    metadata = getattr(plugin_cls, "METADATA", {}) or {}
    display_name = _metadata_value(metadata, "display_name", plugin_id.replace("_", " ").title())
    description = _metadata_value(metadata, "description", "")

    # Get capabilities from the class method if available
    capabilities_data: dict[str, Any] = {}
    if hasattr(plugin_cls, "get_capabilities") and callable(plugin_cls.get_capabilities):
        try:
            caps = plugin_cls.get_capabilities()
            capabilities_data = {
                "max_documents": caps.max_documents,
                "max_query_length": caps.max_query_length,
                "max_doc_length": caps.max_doc_length,
                "supports_batching": caps.supports_batching,
                "models": list(caps.models),
            }
        except Exception as exc:
            logger.warning("Failed to get capabilities for reranker plugin '%s': %s", plugin_id, exc)

    return PluginManifest(
        id=plugin_id,
        type="reranker",
        version=getattr(plugin_cls, "PLUGIN_VERSION", "0.0.0"),
        display_name=str(display_name),
        description=str(description),
        author=_metadata_value(metadata, "author"),
        license=_metadata_value(metadata, "license"),
        homepage=_metadata_value(metadata, "homepage"),
        requires=list(_metadata_value(metadata, "requires", [])),
        semantik_version=_metadata_value(metadata, "semantik_version"),
        capabilities=capabilities_data,
    )


def manifest_from_extractor_plugin(plugin_cls: type, plugin_id: str) -> PluginManifest:
    """Build a plugin manifest for an extractor."""
    metadata = getattr(plugin_cls, "METADATA", {}) or {}
    display_name = _metadata_value(metadata, "display_name", plugin_id.replace("_", " ").title())
    description = _metadata_value(metadata, "description", "")

    # Get supported extractions from class method if available
    supported_types: list[str] = []
    if hasattr(plugin_cls, "supported_extractions") and callable(plugin_cls.supported_extractions):
        try:
            types = plugin_cls.supported_extractions()
            supported_types = [t.value if hasattr(t, "value") else str(t) for t in types]
        except Exception as exc:
            logger.warning("Failed to get supported extractions for extractor plugin '%s': %s", plugin_id, exc)

    # Get AgentHints if available
    agent_hints = None
    if hasattr(plugin_cls, "AGENT_HINTS"):
        agent_hints = plugin_cls.AGENT_HINTS

    return PluginManifest(
        id=plugin_id,
        type="extractor",
        version=getattr(plugin_cls, "PLUGIN_VERSION", "0.0.0"),
        display_name=str(display_name),
        description=str(description),
        author=_metadata_value(metadata, "author"),
        license=_metadata_value(metadata, "license"),
        homepage=_metadata_value(metadata, "homepage"),
        requires=list(_metadata_value(metadata, "requires", [])),
        semantik_version=_metadata_value(metadata, "semantik_version"),
        capabilities={
            "supported_extractions": supported_types,
        },
        agent_hints=agent_hints,
    )


def manifest_from_parser_plugin(plugin_cls: type, plugin_id: str) -> PluginManifest:
    """Build a plugin manifest for a parser.

    Args:
        plugin_cls: The parser plugin class.
        plugin_id: The plugin ID.

    Returns:
        PluginManifest for the parser plugin.
    """
    metadata = getattr(plugin_cls, "METADATA", {}) or {}
    display_name = _metadata_value(metadata, "display_name", plugin_id.replace("-", " ").title())
    description = _metadata_value(metadata, "description", "")

    # Get supported extensions from class method if available
    extensions: list[str] = []
    if hasattr(plugin_cls, "supported_extensions") and callable(plugin_cls.supported_extensions):
        try:
            ext_set = plugin_cls.supported_extensions()
            extensions = sorted(ext_set) if ext_set else []
        except Exception as exc:
            logger.warning("Failed to get supported extensions for parser plugin '%s': %s", plugin_id, exc)

    # Get supported MIME types from class method if available
    mime_types: list[str] = []
    if hasattr(plugin_cls, "supported_mime_types") and callable(plugin_cls.supported_mime_types):
        try:
            mime_set = plugin_cls.supported_mime_types()
            mime_types = sorted(mime_set) if mime_set else []
        except Exception as exc:
            logger.warning("Failed to get supported MIME types for parser plugin '%s': %s", plugin_id, exc)

    # Get config options from class method if available
    config_options: list[dict[str, Any]] = []
    if hasattr(plugin_cls, "get_config_options") and callable(plugin_cls.get_config_options):
        try:
            config_options = plugin_cls.get_config_options()
        except Exception as exc:
            logger.warning("Failed to get config options for parser plugin '%s': %s", plugin_id, exc)

    # Build capabilities
    capabilities: dict[str, Any] = {}
    if extensions:
        capabilities["supported_extensions"] = extensions
    if mime_types:
        capabilities["supported_mime_types"] = mime_types
    if config_options:
        capabilities["config_options"] = config_options

    # Get AgentHints if available
    agent_hints = None
    if hasattr(plugin_cls, "AGENT_HINTS"):
        agent_hints = plugin_cls.AGENT_HINTS
    elif hasattr(plugin_cls, "get_manifest") and callable(plugin_cls.get_manifest):
        # Try to get agent_hints from the class's own get_manifest implementation
        try:
            manifest = plugin_cls.get_manifest()
            if hasattr(manifest, "agent_hints"):
                agent_hints = manifest.agent_hints
        except Exception:
            pass

    return PluginManifest(
        id=plugin_id,
        type="parser",
        version=getattr(plugin_cls, "PLUGIN_VERSION", "0.0.0"),
        display_name=str(display_name),
        description=str(description),
        author=_metadata_value(metadata, "author"),
        license=_metadata_value(metadata, "license"),
        homepage=_metadata_value(metadata, "homepage"),
        requires=list(_metadata_value(metadata, "requires", [])),
        semantik_version=_metadata_value(metadata, "semantik_version"),
        capabilities=capabilities,
        agent_hints=agent_hints,
    )


def get_config_schema(plugin_cls: type) -> dict[str, Any] | None:
    """Return config schema for a plugin class if declared."""
    schema = None
    getter = getattr(plugin_cls, "get_config_schema", None)
    if callable(getter):
        try:
            schema = getter()
        except Exception:
            schema = None
    if schema is None:
        schema = getattr(plugin_cls, "CONFIG_SCHEMA", None)
    if isinstance(schema, dict):
        return schema
    return None


def manifest_from_sparse_indexer_plugin(plugin_cls: type, plugin_id: str) -> PluginManifest:
    """Build a plugin manifest for a sparse indexer plugin.

    Args:
        plugin_cls: The sparse indexer plugin class.
        plugin_id: The plugin ID.

    Returns:
        PluginManifest for the sparse indexer plugin.
    """
    metadata = getattr(plugin_cls, "METADATA", {}) or {}
    display_name = _metadata_value(metadata, "display_name", plugin_id.replace("-", " ").title())
    description = _metadata_value(metadata, "description", "")

    # Extract sparse_type from class variable
    sparse_type = getattr(plugin_cls, "SPARSE_TYPE", None)

    # Get capabilities from the class method if available
    capabilities_data: dict[str, Any] = {}
    if hasattr(plugin_cls, "get_capabilities") and callable(plugin_cls.get_capabilities):
        try:
            caps = plugin_cls.get_capabilities()
            # Convert dataclass to dict via asdict or direct attribute access
            if hasattr(caps, "__dataclass_fields__"):
                from dataclasses import asdict

                capabilities_data = asdict(caps)
            elif isinstance(caps, dict):
                capabilities_data = dict(caps)
        except Exception as exc:
            logger.warning("Failed to get capabilities for sparse indexer plugin '%s': %s", plugin_id, exc)

    # Ensure sparse_type is in capabilities
    if sparse_type and "sparse_type" not in capabilities_data:
        capabilities_data["sparse_type"] = sparse_type

    return PluginManifest(
        id=plugin_id,
        type="sparse_indexer",
        version=getattr(plugin_cls, "PLUGIN_VERSION", "0.0.0"),
        display_name=str(display_name),
        description=str(description),
        author=_metadata_value(metadata, "author"),
        license=_metadata_value(metadata, "license"),
        homepage=_metadata_value(metadata, "homepage"),
        requires=list(_metadata_value(metadata, "requires", [])),
        semantik_version=_metadata_value(metadata, "semantik_version"),
        capabilities=capabilities_data,
    )
