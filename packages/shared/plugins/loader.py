"""Unified plugin loader for Semantik."""

from __future__ import annotations

import logging
import os
from importlib import metadata
from threading import Lock
from typing import TYPE_CHECKING, Any

from .adapters import (
    get_config_schema,
    manifest_from_chunking_plugin,
    manifest_from_connector_plugin,
    manifest_from_embedding_plugin,
    manifest_from_extractor_plugin,
    manifest_from_reranker_plugin,
)
from .base import SemanticPlugin
from .registry import PluginRecord, PluginSource, plugin_registry

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .manifest import PluginManifest
    from .registry import PluginRegistry

logger = logging.getLogger(__name__)

ENTRYPOINT_GROUP = "semantik.plugins"

_ENV_FLAG_GLOBAL = "SEMANTIK_ENABLE_PLUGINS"
_ENV_FLAG_BY_TYPE = {
    "embedding": "SEMANTIK_ENABLE_EMBEDDING_PLUGINS",
    "chunking": "SEMANTIK_ENABLE_CHUNKING_PLUGINS",
    "connector": "SEMANTIK_ENABLE_CONNECTOR_PLUGINS",
    "reranker": "SEMANTIK_ENABLE_RERANKER_PLUGINS",
    "extractor": "SEMANTIK_ENABLE_EXTRACTOR_PLUGINS",
}

_DEFAULT_PLUGIN_TYPES = {"embedding", "chunking", "connector", "reranker", "extractor"}

_PLUGIN_LOAD_LOCK = Lock()


def _flag_enabled(flag: str | None, default: str = "true") -> bool:
    if not flag:
        return True
    value = os.getenv(flag, default).lower()
    return value not in {"0", "false", "no", "off"}


def _plugin_type_enabled(plugin_type: str) -> bool:
    if not _flag_enabled(_ENV_FLAG_GLOBAL):
        return False
    return _flag_enabled(_ENV_FLAG_BY_TYPE.get(plugin_type))


def _coerce_class(obj: Any) -> type | None:
    if isinstance(obj, type):
        return obj
    if callable(obj):
        maybe_cls = obj()
        if isinstance(maybe_cls, type):
            return maybe_cls
        if maybe_cls is not None:
            return maybe_cls.__class__
    return None


def _is_internal_module(module_name: str) -> bool:
    return module_name.startswith(("shared.", "webui.", "vecpipe."))


def load_plugins(
    *,
    plugin_types: Iterable[str] | None = None,
    include_builtins: bool = True,
    include_external: bool = True,
    entry_point_group: str = ENTRYPOINT_GROUP,
    disabled_plugin_ids: set[str] | None = None,
) -> PluginRegistry:
    """Load plugins for the requested types.

    Built-ins load first, followed by external entry points. This function is
    idempotent per plugin type.
    """
    requested = set(plugin_types or _DEFAULT_PLUGIN_TYPES)

    with _PLUGIN_LOAD_LOCK:
        if disabled_plugin_ids is not None:
            plugin_registry.set_disabled(disabled_plugin_ids)

        if plugin_registry.is_loaded(requested):
            return plugin_registry

        missing = requested - plugin_registry.loaded_types()

        if include_builtins and missing:
            _load_builtin_plugins(missing)

        if include_external and missing:
            _load_external_plugins(missing, entry_point_group, disabled_plugin_ids)

        plugin_registry.mark_loaded(missing)

    return plugin_registry


def _load_builtin_plugins(plugin_types: set[str]) -> None:
    """Register built-in plugins for the specified types."""
    if "embedding" in plugin_types:
        _register_builtin_embedding_plugins()
    if "chunking" in plugin_types:
        _register_builtin_chunking_plugins()
    if "connector" in plugin_types:
        _register_builtin_connector_plugins()
    if "reranker" in plugin_types:
        _register_builtin_reranker_plugins()
    if "extractor" in plugin_types:
        _register_builtin_extractor_plugins()


def _register_builtin_embedding_plugins() -> None:
    from shared.embedding import providers as provider_module
    from shared.embedding.factory import EmbeddingProviderFactory
    from shared.embedding.provider_registry import list_provider_definitions

    if (not list_provider_definitions() or not EmbeddingProviderFactory.list_available_providers()) and hasattr(
        provider_module, "_register_builtin_providers"
    ):
        provider_module._register_builtin_providers()

    for definition in list_provider_definitions():
        if definition.is_plugin:
            continue
        provider_cls = EmbeddingProviderFactory.get_provider_class(definition.internal_id)
        if provider_cls is None:
            logger.warning("No provider class found for embedding provider '%s'", definition.internal_id)
            continue
        manifest = manifest_from_embedding_plugin(provider_cls, definition)
        _register_plugin_record(
            plugin_type="embedding",
            plugin_id=definition.api_id,
            plugin_cls=provider_cls,
            manifest=manifest,
            source=PluginSource.BUILTIN,
        )


def _register_builtin_chunking_plugins() -> None:
    from shared.chunking.domain.services.chunking_strategies import STRATEGY_REGISTRY
    from webui.services.chunking import strategy_registry

    # Ensure default strategy definitions exist
    _ = strategy_registry.list_strategy_definitions()

    for definition in strategy_registry.list_strategy_definitions():
        if definition.is_plugin:
            continue
        strategy_cls = STRATEGY_REGISTRY.get(definition.internal_id)
        if strategy_cls is None:
            logger.warning("No strategy class found for chunking strategy '%s'", definition.internal_id)
            continue
        manifest = manifest_from_chunking_plugin(
            strategy_cls,
            api_id=definition.api_id,
            internal_id=definition.internal_id,
        )
        _register_plugin_record(
            plugin_type="chunking",
            plugin_id=definition.api_id,
            plugin_cls=strategy_cls,
            manifest=manifest,
            source=PluginSource.BUILTIN,
        )


def _register_builtin_connector_plugins() -> None:
    from shared.connectors.git import GitConnector
    from shared.connectors.imap import ImapConnector
    from shared.connectors.local import LocalFileConnector

    for connector_cls in (LocalFileConnector, GitConnector, ImapConnector):
        plugin_id = getattr(connector_cls, "PLUGIN_ID", "") or ""
        if not plugin_id:
            logger.warning("Skipping connector without PLUGIN_ID: %s", connector_cls)
            continue
        manifest = manifest_from_connector_plugin(connector_cls, plugin_id)
        _register_plugin_record(
            plugin_type="connector",
            plugin_id=plugin_id,
            plugin_cls=connector_cls,
            manifest=manifest,
            source=PluginSource.BUILTIN,
        )


def _register_builtin_reranker_plugins() -> None:
    """Register built-in reranker plugins."""
    try:
        from shared.plugins.builtins.qwen3_reranker import Qwen3RerankerPlugin

        plugin_id = Qwen3RerankerPlugin.PLUGIN_ID
        manifest = manifest_from_reranker_plugin(Qwen3RerankerPlugin, plugin_id)
        _register_plugin_record(
            plugin_type="reranker",
            plugin_id=plugin_id,
            plugin_cls=Qwen3RerankerPlugin,
            manifest=manifest,
            source=PluginSource.BUILTIN,
        )
    except ImportError:
        logger.debug("Qwen3 reranker plugin not available (vecpipe not installed)")


def _register_builtin_extractor_plugins() -> None:
    """Register built-in extractor plugins."""
    try:
        from shared.plugins.builtins.keyword_extractor import KeywordExtractorPlugin

        plugin_id = KeywordExtractorPlugin.PLUGIN_ID
        manifest = manifest_from_extractor_plugin(KeywordExtractorPlugin, plugin_id)
        _register_plugin_record(
            plugin_type="extractor",
            plugin_id=plugin_id,
            plugin_cls=KeywordExtractorPlugin,
            manifest=manifest,
            source=PluginSource.BUILTIN,
        )
    except ImportError:
        logger.debug("Keyword extractor plugin not available")


def _load_external_plugins(
    plugin_types: set[str],
    entry_point_group: str,
    disabled_plugin_ids: set[str] | None,
) -> None:
    if not _flag_enabled(_ENV_FLAG_GLOBAL):
        logger.info("Plugin loading disabled via %s", _ENV_FLAG_GLOBAL)
        return

    try:
        eps = metadata.entry_points()
        if hasattr(eps, "select"):
            ep_group = list(eps.select(group=entry_point_group))
        else:
            ep_group = list(eps.get(entry_point_group, []))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unable to query entry points for plugins: %s", exc)
        return

    for ep in ep_group:
        ep_name = getattr(ep, "name", "unknown")
        try:
            loaded = ep.load()
            plugin_cls = _coerce_class(loaded)
            if plugin_cls is None:
                raise TypeError(f"Entry point {ep_name} did not resolve to a class")

            plugin_type = _resolve_plugin_type(plugin_cls)
            if plugin_type is None:
                logger.warning("Skipping unknown plugin type for entry point '%s'", ep_name)
                continue

            if plugin_type not in plugin_types:
                continue

            if not _plugin_type_enabled(plugin_type):
                logger.info("Skipping %s plugins disabled via env", plugin_type)
                continue

            source = PluginSource.EXTERNAL
            if _is_internal_module(getattr(plugin_cls, "__module__", "")):
                source = PluginSource.BUILTIN

            _register_plugin_class(
                plugin_cls,
                plugin_type,
                source,
                entry_point=ep_name,
                disabled_plugin_ids=disabled_plugin_ids,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load plugin entry point %s: %s", ep_name, exc)
            continue


def _resolve_plugin_type(plugin_cls: type) -> str | None:
    try:
        import shared.embedding.plugin_base as embedding_plugin_base
    except Exception:
        embedding_plugin_base = None

    try:
        import shared.chunking.domain.services.chunking_strategies.base as chunking_strategy_base
    except Exception:
        chunking_strategy_base = None

    try:
        import shared.connectors.base as connector_base
    except Exception:
        connector_base = None

    if embedding_plugin_base is not None and issubclass(plugin_cls, embedding_plugin_base.BaseEmbeddingPlugin):
        return "embedding"
    if chunking_strategy_base is not None and issubclass(plugin_cls, chunking_strategy_base.ChunkingStrategy):
        return "chunking"
    if connector_base is not None and issubclass(plugin_cls, connector_base.BaseConnector):
        return "connector"

    if issubclass(plugin_cls, SemanticPlugin):
        return getattr(plugin_cls, "PLUGIN_TYPE", None)

    return None


def _register_plugin_class(
    plugin_cls: type,
    plugin_type: str,
    source: PluginSource,
    *,
    entry_point: str | None,
    disabled_plugin_ids: set[str] | None,
) -> None:
    if plugin_type == "embedding":
        _register_embedding_plugin(plugin_cls, source, entry_point, disabled_plugin_ids)
    elif plugin_type == "chunking":
        _register_chunking_plugin(plugin_cls, source, entry_point, disabled_plugin_ids)
    elif plugin_type == "connector":
        _register_connector_plugin(plugin_cls, source, entry_point, disabled_plugin_ids)
    elif plugin_type == "reranker":
        _register_reranker_plugin(plugin_cls, source, entry_point, disabled_plugin_ids)
    elif plugin_type == "extractor":
        _register_extractor_plugin(plugin_cls, source, entry_point, disabled_plugin_ids)
    else:
        if issubclass(plugin_cls, SemanticPlugin):
            manifest = plugin_cls.get_manifest()
            _register_plugin_record(
                plugin_type=manifest.type,
                plugin_id=manifest.id,
                plugin_cls=plugin_cls,
                manifest=manifest,
                source=source,
                entry_point=entry_point,
            )


def _register_embedding_plugin(
    plugin_cls: type,
    source: PluginSource,
    entry_point: str | None,
    disabled_plugin_ids: set[str] | None,
) -> None:
    from shared.embedding.factory import EmbeddingProviderFactory
    from shared.embedding.plugin_base import BaseEmbeddingPlugin, EmbeddingProviderDefinition
    from shared.embedding.provider_registry import register_provider_definition

    if not hasattr(plugin_cls, "get_definition") or not callable(plugin_cls.get_definition):
        logger.warning("Embedding plugin missing get_definition(): %s", plugin_cls)
        return

    if issubclass(plugin_cls, BaseEmbeddingPlugin):
        is_valid, error = plugin_cls.validate_plugin_contract()
        if not is_valid:
            logger.warning("Skipping invalid embedding plugin '%s': %s", plugin_cls, error)
            return

    try:
        definition = plugin_cls.get_definition()
    except Exception as exc:
        logger.warning("Skipping embedding plugin '%s': get_definition() failed: %s", plugin_cls, exc)
        return

    if source == PluginSource.EXTERNAL and not definition.is_plugin:
        definition = EmbeddingProviderDefinition(
            api_id=definition.api_id,
            internal_id=definition.internal_id,
            display_name=definition.display_name,
            description=definition.description,
            provider_type=definition.provider_type,
            supports_quantization=definition.supports_quantization,
            supports_instruction=definition.supports_instruction,
            supports_batch_processing=definition.supports_batch_processing,
            supports_asymmetric=definition.supports_asymmetric,
            supported_models=definition.supported_models,
            default_config=dict(definition.default_config),
            performance_characteristics=dict(definition.performance_characteristics),
            is_plugin=True,
        )

    manifest = manifest_from_embedding_plugin(plugin_cls, definition)
    record_registered = _register_plugin_record(
        plugin_type="embedding",
        plugin_id=definition.api_id,
        plugin_cls=plugin_cls,
        manifest=manifest,
        source=source,
        entry_point=entry_point,
    )

    if not record_registered:
        return

    if source == PluginSource.EXTERNAL and disabled_plugin_ids and definition.api_id in disabled_plugin_ids:
        logger.info("Embedding plugin '%s' disabled; skipping activation", definition.api_id)
        return

    EmbeddingProviderFactory.register_provider(definition.internal_id, plugin_cls)
    register_provider_definition(definition)


def _register_chunking_plugin(
    plugin_cls: type,
    source: PluginSource,
    entry_point: str | None,
    disabled_plugin_ids: set[str] | None,
) -> None:
    from webui.services.chunking.strategy_registry import register_strategy_definition
    from webui.services.chunking_strategy_factory import ChunkingStrategyFactory

    internal_name = (
        getattr(plugin_cls, "INTERNAL_NAME", None)
        or getattr(plugin_cls, "name", None)
        or getattr(plugin_cls, "__name__", "")
    )
    api_id = getattr(plugin_cls, "API_ID", None) or internal_name
    metadata_dict: dict[str, Any] = getattr(plugin_cls, "METADATA", {}) or {}

    if not internal_name:
        logger.warning("Skipping chunking plugin with missing INTERNAL_NAME: %s", plugin_cls)
        return

    if source == PluginSource.EXTERNAL:
        visual_example = metadata_dict.get("visual_example")
        if not visual_example or not isinstance(visual_example, dict):
            logger.warning(
                "Skipping chunking plugin '%s': missing required visual_example (url + optional caption)",
                api_id,
            )
            return
        url = visual_example.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            logger.warning("Skipping chunking plugin '%s': visual_example.url must be https://", api_id)
            return

    manifest = manifest_from_chunking_plugin(plugin_cls, api_id=str(api_id), internal_id=str(internal_name))
    record_registered = _register_plugin_record(
        plugin_type="chunking",
        plugin_id=str(api_id),
        plugin_cls=plugin_cls,
        manifest=manifest,
        source=source,
        entry_point=entry_point,
    )

    if not record_registered:
        return

    if source == PluginSource.EXTERNAL and disabled_plugin_ids and str(api_id) in disabled_plugin_ids:
        logger.info("Chunking plugin '%s' disabled; skipping activation", api_id)
        return

    ChunkingStrategyFactory.register_strategy(str(internal_name), plugin_cls, api_enum=None)

    register_strategy_definition(
        api_id=str(api_id),
        internal_id=str(internal_name),
        display_name=metadata_dict.get("display_name", str(api_id).replace("_", " ").title()),
        description=metadata_dict.get("description", f"Plugin strategy {api_id}"),
        best_for=tuple(metadata_dict.get("best_for", ())),
        pros=tuple(metadata_dict.get("pros", ())),
        cons=tuple(metadata_dict.get("cons", ())),
        performance_characteristics=metadata_dict.get("performance_characteristics", {}),
        manager_defaults=metadata_dict.get("manager_defaults"),
        builder_defaults=metadata_dict.get("builder_defaults"),
        supported_file_types=tuple(metadata_dict.get("supported_file_types", ())),
        aliases=tuple(metadata_dict.get("aliases", ())),
        factory_defaults=metadata_dict.get("factory_defaults"),
        is_plugin=source == PluginSource.EXTERNAL,
        visual_example=metadata_dict.get("visual_example"),
    )


def _register_connector_plugin(
    plugin_cls: type,
    source: PluginSource,
    entry_point: str | None,
    disabled_plugin_ids: set[str] | None,
) -> None:
    plugin_id = getattr(plugin_cls, "PLUGIN_ID", None) or ""
    if not plugin_id:
        logger.warning("Skipping connector without PLUGIN_ID: %s", plugin_cls)
        return

    manifest = manifest_from_connector_plugin(plugin_cls, plugin_id)
    record_registered = _register_plugin_record(
        plugin_type="connector",
        plugin_id=plugin_id,
        plugin_cls=plugin_cls,
        manifest=manifest,
        source=source,
        entry_point=entry_point,
    )

    if not record_registered:
        return

    if source == PluginSource.EXTERNAL and disabled_plugin_ids and plugin_id in disabled_plugin_ids:
        logger.info("Connector plugin '%s' disabled; skipping activation", plugin_id)
        return

    # Connector activation uses plugin registry; no extra registration required.


def _register_reranker_plugin(
    plugin_cls: type,
    source: PluginSource,
    entry_point: str | None,
    disabled_plugin_ids: set[str] | None,
) -> None:
    """Register a reranker plugin."""
    plugin_id = getattr(plugin_cls, "PLUGIN_ID", None) or ""
    if not plugin_id:
        logger.warning("Skipping reranker without PLUGIN_ID: %s", plugin_cls)
        return

    manifest = manifest_from_reranker_plugin(plugin_cls, plugin_id)
    record_registered = _register_plugin_record(
        plugin_type="reranker",
        plugin_id=plugin_id,
        plugin_cls=plugin_cls,
        manifest=manifest,
        source=source,
        entry_point=entry_point,
    )

    if not record_registered:
        return

    if source == PluginSource.EXTERNAL and disabled_plugin_ids and plugin_id in disabled_plugin_ids:
        logger.info("Reranker plugin '%s' disabled; skipping activation", plugin_id)
        return

    # Reranker activation uses plugin registry; no extra registration required.


def _register_extractor_plugin(
    plugin_cls: type,
    source: PluginSource,
    entry_point: str | None,
    disabled_plugin_ids: set[str] | None,
) -> None:
    """Register an extractor plugin."""
    plugin_id = getattr(plugin_cls, "PLUGIN_ID", None) or ""
    if not plugin_id:
        logger.warning("Skipping extractor without PLUGIN_ID: %s", plugin_cls)
        return

    manifest = manifest_from_extractor_plugin(plugin_cls, plugin_id)
    record_registered = _register_plugin_record(
        plugin_type="extractor",
        plugin_id=plugin_id,
        plugin_cls=plugin_cls,
        manifest=manifest,
        source=source,
        entry_point=entry_point,
    )

    if not record_registered:
        return

    if source == PluginSource.EXTERNAL and disabled_plugin_ids and plugin_id in disabled_plugin_ids:
        logger.info("Extractor plugin '%s' disabled; skipping activation", plugin_id)
        return

    # Extractor activation uses plugin registry; no extra registration required.


def _register_plugin_record(
    *,
    plugin_type: str,
    plugin_id: str,
    plugin_cls: type,
    manifest: PluginManifest,
    source: PluginSource,
    entry_point: str | None = None,
) -> bool:
    record = PluginRecord(
        plugin_type=plugin_type,
        plugin_id=plugin_id,
        plugin_version=getattr(plugin_cls, "PLUGIN_VERSION", getattr(manifest, "version", "0.0.0")),
        manifest=manifest,
        plugin_class=plugin_cls,
        source=source,
        entry_point=entry_point,
    )
    return plugin_registry.register(record)


def get_plugin_config_schema(plugin_id: str) -> dict[str, Any] | None:
    """Return config schema for a plugin id if available."""
    record = _find_external_plugin(plugin_id)
    if record is None:
        return None
    return get_config_schema(record.plugin_class)


def _find_external_plugin(plugin_id: str) -> PluginRecord | None:
    for record in plugin_registry.list_records(source=PluginSource.EXTERNAL):
        if record.plugin_id == plugin_id:
            return record
    return None
