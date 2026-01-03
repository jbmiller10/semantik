"""Contract test base classes for plugin testing.

These classes provide automatic contract verification for plugin implementations.
The contracts reflect the interfaces Semantik actually loads at runtime:

- Embedding providers: `shared.embedding.plugin_base.BaseEmbeddingPlugin`
- Chunking strategies: classes registered via `ChunkingStrategyFactory`
  (must be instantiable with no required args and implement `chunk()` methods)
- Connectors: `shared.connectors.base.BaseConnector`
- Rerankers/Extractors: `shared.plugins.base.SemanticPlugin`

Plugin authors can inherit from these classes to ensure their plugins match the
expected contracts.
"""

from __future__ import annotations

import inspect
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

if TYPE_CHECKING:
    from shared.connectors.base import BaseConnector
    from shared.embedding.plugin_base import BaseEmbeddingPlugin
    from shared.plugins.base import SemanticPlugin


# =============================================================================
# SemanticPlugin-based contracts (rerankers, extractors, etc.)
# =============================================================================


class PluginContractTest(ABC):
    """Contract tests for SemanticPlugin-based plugins."""

    plugin_class: ClassVar[type[SemanticPlugin]]
    """The plugin class to test. Must be set by subclasses."""

    @pytest.fixture()
    def plugin_config(self) -> dict[str, Any] | None:
        """Override to provide configuration for plugin tests."""
        return None

    @pytest.fixture()
    def plugin_instance(self, plugin_config: dict[str, Any] | None) -> SemanticPlugin:
        """Instantiate the plugin class with optional config."""
        return self.plugin_class(config=plugin_config)

    # =========================================================================
    # Required Class Attributes
    # =========================================================================

    def test_has_plugin_type(self) -> None:
        assert hasattr(self.plugin_class, "PLUGIN_TYPE"), f"{self.plugin_class.__name__} missing PLUGIN_TYPE"
        assert isinstance(self.plugin_class.PLUGIN_TYPE, str)
        assert self.plugin_class.PLUGIN_TYPE

    def test_has_plugin_id(self) -> None:
        assert hasattr(self.plugin_class, "PLUGIN_ID"), f"{self.plugin_class.__name__} missing PLUGIN_ID"
        assert isinstance(self.plugin_class.PLUGIN_ID, str)
        assert self.plugin_class.PLUGIN_ID

    def test_has_plugin_version(self) -> None:
        assert hasattr(self.plugin_class, "PLUGIN_VERSION"), f"{self.plugin_class.__name__} missing PLUGIN_VERSION"
        assert isinstance(self.plugin_class.PLUGIN_VERSION, str)
        parts = self.plugin_class.PLUGIN_VERSION.split(".")
        assert len(parts) >= 2, "Version should have at least major.minor format"

    # =========================================================================
    # Manifest
    # =========================================================================

    def test_get_manifest_returns_valid_manifest(self) -> None:
        from shared.plugins.manifest import PluginManifest

        manifest = self.plugin_class.get_manifest()
        assert isinstance(manifest, PluginManifest), f"get_manifest() must return PluginManifest, got {type(manifest)}"
        assert manifest.id == self.plugin_class.PLUGIN_ID
        assert manifest.type == self.plugin_class.PLUGIN_TYPE
        assert manifest.display_name
        assert manifest.description

    # =========================================================================
    # Configuration Schema
    # =========================================================================

    def test_get_config_schema_returns_valid_schema_or_none(self) -> None:
        schema = self.plugin_class.get_config_schema()
        if schema is not None:
            assert isinstance(schema, dict), "Config schema must be a dict"
            assert schema.get("type") == "object", "Config schema must have type: object"

    # =========================================================================
    # Health Check
    # =========================================================================

    @pytest.mark.asyncio()
    async def test_health_check_returns_bool(self, plugin_config: dict[str, Any] | None) -> None:
        health_fn = getattr(self.plugin_class, "health_check", None)
        assert callable(health_fn), "Plugin must define health_check()"

        try:
            result = health_fn(config=plugin_config)
        except TypeError:
            # Allow legacy signatures without `config` parameter.
            result = health_fn()

        if inspect.isawaitable(result):
            result = await result
        assert isinstance(result, bool), f"health_check must return bool, got {type(result)}"

    # =========================================================================
    # Instance Lifecycle
    # =========================================================================

    @pytest.mark.asyncio()
    async def test_initialize_and_cleanup(self, plugin_instance: SemanticPlugin) -> None:
        await plugin_instance.initialize()
        await plugin_instance.cleanup()

    def test_config_property(self, plugin_instance: SemanticPlugin, plugin_config: dict[str, Any] | None) -> None:
        config = plugin_instance.config
        assert isinstance(config, dict)
        if plugin_config:
            for key, value in plugin_config.items():
                assert config.get(key) == value


# =============================================================================
# Embedding providers
# =============================================================================


class EmbeddingPluginContractTest(ABC):
    """Contract tests for embedding provider plugins."""

    plugin_class: ClassVar[type[BaseEmbeddingPlugin]]

    @pytest.fixture()
    def plugin_config(self) -> dict[str, Any] | None:
        """Override to provide provider config passed to the constructor."""
        return None

    @pytest.fixture()
    def plugin_instance(self, plugin_config: dict[str, Any] | None) -> BaseEmbeddingPlugin:
        return self.plugin_class(config=plugin_config)

    def test_is_base_embedding_plugin(self) -> None:
        from shared.embedding.plugin_base import BaseEmbeddingPlugin as _BaseEmbeddingPlugin

        assert issubclass(self.plugin_class, _BaseEmbeddingPlugin)

    def test_has_required_class_attributes(self) -> None:
        assert isinstance(getattr(self.plugin_class, "INTERNAL_NAME", None), str)
        assert self.plugin_class.INTERNAL_NAME
        assert isinstance(getattr(self.plugin_class, "API_ID", None), str)
        assert self.plugin_class.API_ID
        assert getattr(self.plugin_class, "PROVIDER_TYPE", None) in {"local", "remote", "hybrid"}

    def test_has_required_classmethods(self) -> None:
        assert hasattr(self.plugin_class, "get_definition")
        assert callable(self.plugin_class.get_definition)
        assert hasattr(self.plugin_class, "supports_model")
        assert callable(self.plugin_class.supports_model)

    def test_get_definition_returns_valid_definition(self) -> None:
        from shared.embedding.plugin_base import EmbeddingProviderDefinition

        definition = self.plugin_class.get_definition()
        assert isinstance(definition, EmbeddingProviderDefinition)
        assert definition.api_id == self.plugin_class.API_ID
        assert definition.internal_id == self.plugin_class.INTERNAL_NAME
        assert definition.provider_type == self.plugin_class.PROVIDER_TYPE

    def test_supports_model_returns_bool(self) -> None:
        result = self.plugin_class.supports_model("test-model")
        assert isinstance(result, bool)

    def test_has_required_instance_methods(self, plugin_instance: BaseEmbeddingPlugin) -> None:
        for name in ("initialize", "embed_single", "embed_texts", "get_dimension", "get_model_info", "cleanup"):
            assert hasattr(plugin_instance, name), f"Missing method: {name}"
            assert callable(getattr(plugin_instance, name))

        assert hasattr(plugin_instance, "is_initialized"), "Missing is_initialized property"


# =============================================================================
# Chunking strategies
# =============================================================================


class ChunkingPluginContractTest(ABC):
    """Contract tests for chunking strategy plugins."""

    plugin_class: ClassVar[type[Any]]

    @pytest.fixture()
    def plugin_instance(self) -> Any:
        return self.plugin_class()

    def test_has_internal_name(self) -> None:
        internal_name = (
            getattr(self.plugin_class, "INTERNAL_NAME", None)
            or getattr(self.plugin_class, "name", None)
            or getattr(self.plugin_class, "__name__", "")
        )
        assert isinstance(internal_name, str)
        assert internal_name

    def test_metadata_has_visual_example(self) -> None:
        metadata = getattr(self.plugin_class, "METADATA", None)
        assert isinstance(metadata, dict), "Chunking plugin must define METADATA dict"
        visual_example = metadata.get("visual_example")
        assert isinstance(visual_example, dict), "METADATA.visual_example must be a dict"
        url = visual_example.get("url")
        assert isinstance(url, str)
        assert url.startswith("https://"), "visual_example.url must be https://"

    def test_has_chunk_method(self, plugin_instance: Any) -> None:
        assert hasattr(plugin_instance, "chunk")
        assert callable(plugin_instance.chunk)

    def test_has_validate_content_method(self, plugin_instance: Any) -> None:
        assert hasattr(plugin_instance, "validate_content")
        assert callable(plugin_instance.validate_content)

    def test_has_estimate_chunks_method(self, plugin_instance: Any) -> None:
        assert hasattr(plugin_instance, "estimate_chunks")
        assert callable(plugin_instance.estimate_chunks)


# =============================================================================
# Connectors
# =============================================================================


class ConnectorPluginContractTest(ABC):
    """Contract tests for connector plugins."""

    plugin_class: ClassVar[type[BaseConnector]]

    @pytest.fixture()
    def connector_config(self) -> dict[str, Any]:
        """Override to provide a valid connector config."""
        return {}

    @pytest.fixture()
    def connector_instance(self, connector_config: dict[str, Any]) -> BaseConnector:
        return self.plugin_class(connector_config)

    def test_is_base_connector(self) -> None:
        from shared.connectors.base import BaseConnector as _BaseConnector

        assert issubclass(self.plugin_class, _BaseConnector)

    def test_has_plugin_id(self) -> None:
        assert isinstance(getattr(self.plugin_class, "PLUGIN_ID", None), str)
        assert self.plugin_class.PLUGIN_ID

    def test_has_authenticate_method(self) -> None:
        assert hasattr(self.plugin_class, "authenticate")
        assert callable(self.plugin_class.authenticate)

    def test_has_load_documents_method(self) -> None:
        assert hasattr(self.plugin_class, "load_documents")
        assert callable(self.plugin_class.load_documents)

    def test_has_get_config_fields_classmethod(self) -> None:
        assert hasattr(self.plugin_class, "get_config_fields")
        fields = self.plugin_class.get_config_fields()
        assert isinstance(fields, list)

    def test_get_config_fields_returns_valid_fields(self) -> None:
        fields = self.plugin_class.get_config_fields()
        for field in fields:
            assert isinstance(field, dict), "Each field must be a dict"
            assert "name" in field, "Field must have 'name'"
            assert "type" in field, "Field must have 'type'"
            assert "label" in field, "Field must have 'label'"


# =============================================================================
# SemanticPlugin specializations
# =============================================================================


class RerankerPluginContractTest(PluginContractTest):
    """Contract tests for reranker plugins."""

    def test_plugin_type_is_reranker(self) -> None:
        assert self.plugin_class.PLUGIN_TYPE == "reranker"

    def test_has_rerank_method(self) -> None:
        assert hasattr(self.plugin_class, "rerank")
        assert callable(self.plugin_class.rerank)

    def test_has_get_capabilities_classmethod(self) -> None:
        assert hasattr(self.plugin_class, "get_capabilities")

    def test_get_capabilities_returns_valid_capabilities(self) -> None:
        from shared.plugins.types.reranker import RerankerCapabilities

        caps = self.plugin_class.get_capabilities()
        assert isinstance(caps, RerankerCapabilities)
        assert caps.max_documents > 0
        assert caps.max_query_length > 0
        assert caps.max_doc_length > 0


class ExtractorPluginContractTest(PluginContractTest):
    """Contract tests for extractor plugins."""

    def test_plugin_type_is_extractor(self) -> None:
        assert self.plugin_class.PLUGIN_TYPE == "extractor"

    def test_has_supported_extractions_classmethod(self) -> None:
        assert hasattr(self.plugin_class, "supported_extractions")

    def test_supported_extractions_returns_list(self) -> None:
        from shared.plugins.types.extractor import ExtractionType

        extractions = self.plugin_class.supported_extractions()
        assert isinstance(extractions, list)
        assert len(extractions) > 0
        for ext in extractions:
            assert isinstance(ext, ExtractionType)

    def test_has_extract_method(self) -> None:
        assert hasattr(self.plugin_class, "extract")
        assert callable(self.plugin_class.extract)

    @pytest.mark.asyncio()
    async def test_extract_returns_extraction_result(self, plugin_instance: SemanticPlugin) -> None:
        from shared.plugins.types.extractor import ExtractionResult

        await plugin_instance.initialize()
        try:
            result = await plugin_instance.extract("Test text for extraction.")  # type: ignore[attr-defined]
            assert isinstance(result, ExtractionResult)
        finally:
            await plugin_instance.cleanup()
