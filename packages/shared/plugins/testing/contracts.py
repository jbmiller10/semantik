"""Contract test base classes for plugin testing.

These classes provide automatic contract verification for plugin implementations.
Plugin authors can inherit from these classes to ensure their plugins meet
the required interface contracts.

Example usage:

    from shared.plugins.testing import EmbeddingPluginContractTest
    from my_plugin import MyEmbeddingPlugin

    class TestMyEmbeddingPlugin(EmbeddingPluginContractTest):
        plugin_class = MyEmbeddingPlugin

        # Optional: provide custom config for tests
        @pytest.fixture
        def plugin_config(self):
            return {"model_name": "my-model"}

        # Contract tests run automatically:
        # - test_has_required_class_attributes
        # - test_get_manifest_returns_valid_manifest
        # - etc.

        # Add custom tests below
        def test_my_custom_feature(self):
            ...
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

if TYPE_CHECKING:
    from shared.plugins.base import SemanticPlugin


class PluginContractTest(ABC):
    """Base contract tests for all Semantik plugins.

    Subclasses must set the `plugin_class` attribute to the plugin class
    being tested.
    """

    plugin_class: ClassVar[type[SemanticPlugin]]
    """The plugin class to test. Must be set by subclasses."""

    @pytest.fixture()
    def plugin_config(self) -> dict[str, Any] | None:
        """Override to provide configuration for plugin tests.

        Returns:
            Plugin configuration dict, or None for no config.
        """
        return None

    @pytest.fixture()
    def plugin_instance(self, plugin_config: dict[str, Any] | None) -> SemanticPlugin:
        """Create a plugin instance for testing.

        Args:
            plugin_config: Configuration from plugin_config fixture.

        Returns:
            Instantiated plugin.
        """
        return self.plugin_class(config=plugin_config)

    # =========================================================================
    # Required Class Attributes
    # =========================================================================

    def test_has_plugin_type(self) -> None:
        """Plugin must have PLUGIN_TYPE class attribute."""
        assert hasattr(self.plugin_class, "PLUGIN_TYPE"), (
            f"{self.plugin_class.__name__} must have PLUGIN_TYPE class attribute"
        )
        assert isinstance(self.plugin_class.PLUGIN_TYPE, str)
        assert len(self.plugin_class.PLUGIN_TYPE) > 0

    def test_has_plugin_id(self) -> None:
        """Plugin must have PLUGIN_ID class attribute."""
        assert hasattr(self.plugin_class, "PLUGIN_ID"), (
            f"{self.plugin_class.__name__} must have PLUGIN_ID class attribute"
        )
        assert isinstance(self.plugin_class.PLUGIN_ID, str)
        assert len(self.plugin_class.PLUGIN_ID) > 0

    def test_has_plugin_version(self) -> None:
        """Plugin must have PLUGIN_VERSION class attribute."""
        assert hasattr(self.plugin_class, "PLUGIN_VERSION"), (
            f"{self.plugin_class.__name__} must have PLUGIN_VERSION class attribute"
        )
        assert isinstance(self.plugin_class.PLUGIN_VERSION, str)
        # Version should follow semver pattern (at least x.y.z)
        parts = self.plugin_class.PLUGIN_VERSION.split(".")
        assert len(parts) >= 2, "Version should have at least major.minor format"

    # =========================================================================
    # Manifest
    # =========================================================================

    def test_get_manifest_returns_valid_manifest(self) -> None:
        """get_manifest() must return a valid PluginManifest."""
        from shared.plugins.manifest import PluginManifest

        manifest = self.plugin_class.get_manifest()

        assert isinstance(manifest, PluginManifest), (
            f"get_manifest() must return PluginManifest, got {type(manifest)}"
        )

        # Manifest ID should match class PLUGIN_ID
        assert manifest.id == self.plugin_class.PLUGIN_ID, (
            f"Manifest id '{manifest.id}' must match PLUGIN_ID '{self.plugin_class.PLUGIN_ID}'"
        )

        # Manifest type should match class PLUGIN_TYPE
        assert manifest.type == self.plugin_class.PLUGIN_TYPE, (
            f"Manifest type '{manifest.type}' must match PLUGIN_TYPE '{self.plugin_class.PLUGIN_TYPE}'"
        )

        # Required fields
        assert manifest.display_name, "Manifest must have display_name"
        assert manifest.description, "Manifest must have description"

    # =========================================================================
    # Configuration Schema
    # =========================================================================

    def test_get_config_schema_returns_valid_schema_or_none(self) -> None:
        """get_config_schema() must return valid JSON Schema or None."""
        schema = self.plugin_class.get_config_schema()

        if schema is not None:
            assert isinstance(schema, dict), "Config schema must be a dict"
            assert schema.get("type") == "object", "Config schema must have type: object"

    # =========================================================================
    # Health Check
    # =========================================================================

    @pytest.mark.asyncio()
    async def test_health_check_returns_bool(self, plugin_config: dict[str, Any] | None) -> None:
        """health_check() must return a boolean."""
        result = await self.plugin_class.health_check(config=plugin_config)
        assert isinstance(result, bool), f"health_check must return bool, got {type(result)}"

    # =========================================================================
    # Instance Lifecycle
    # =========================================================================

    @pytest.mark.asyncio()
    async def test_initialize_and_cleanup(
        self, plugin_instance: SemanticPlugin
    ) -> None:
        """Plugin must support initialize() and cleanup() lifecycle."""
        # Initialize should not raise
        await plugin_instance.initialize()

        # Cleanup should not raise
        await plugin_instance.cleanup()

    def test_config_property(
        self, plugin_instance: SemanticPlugin, plugin_config: dict[str, Any] | None
    ) -> None:
        """Plugin must have config property."""
        config = plugin_instance.config
        assert isinstance(config, dict)

        if plugin_config:
            # Config should contain provided values
            for key, value in plugin_config.items():
                assert config.get(key) == value


class EmbeddingPluginContractTest(PluginContractTest):
    """Contract tests for embedding plugins.

    Embedding plugins must implement:
    - embed_single(text) -> list[float]
    - embed_texts(texts) -> list[list[float]]
    - get_dimension() -> int
    """

    def test_plugin_type_is_embedding(self) -> None:
        """Embedding plugin must have PLUGIN_TYPE = 'embedding'."""
        assert self.plugin_class.PLUGIN_TYPE == "embedding"

    def test_has_embed_single_method(self) -> None:
        """Plugin must have embed_single method."""
        assert hasattr(self.plugin_class, "embed_single")
        assert callable(self.plugin_class.embed_single)

    def test_has_embed_texts_method(self) -> None:
        """Plugin must have embed_texts method."""
        assert hasattr(self.plugin_class, "embed_texts")
        assert callable(self.plugin_class.embed_texts)

    def test_has_get_dimension_method(self) -> None:
        """Plugin must have get_dimension method."""
        assert hasattr(self.plugin_class, "get_dimension")
        assert callable(self.plugin_class.get_dimension)


class ChunkingPluginContractTest(PluginContractTest):
    """Contract tests for chunking strategy plugins.

    Chunking plugins must implement:
    - chunk(content, config) -> list[Chunk]
    - validate_content(content) -> tuple[bool, str | None]
    - estimate_chunks(content_length, config) -> int
    """

    def test_plugin_type_is_chunking(self) -> None:
        """Chunking plugin must have PLUGIN_TYPE = 'chunking'."""
        assert self.plugin_class.PLUGIN_TYPE == "chunking"

    def test_has_chunk_method(self) -> None:
        """Plugin must have chunk method."""
        assert hasattr(self.plugin_class, "chunk")
        assert callable(self.plugin_class.chunk)

    def test_has_validate_content_method(self) -> None:
        """Plugin must have validate_content method."""
        assert hasattr(self.plugin_class, "validate_content")
        assert callable(self.plugin_class.validate_content)

    def test_has_estimate_chunks_method(self) -> None:
        """Plugin must have estimate_chunks method."""
        assert hasattr(self.plugin_class, "estimate_chunks")
        assert callable(self.plugin_class.estimate_chunks)


class ConnectorPluginContractTest(PluginContractTest):
    """Contract tests for connector plugins.

    Connector plugins must implement:
    - authenticate() -> bool
    - load_documents(source_id) -> AsyncIterator[IngestedDocument]
    - get_config_fields() -> list[dict]
    """

    def test_plugin_type_is_connector(self) -> None:
        """Connector plugin must have PLUGIN_TYPE = 'connector'."""
        assert self.plugin_class.PLUGIN_TYPE == "connector"

    def test_has_authenticate_method(self) -> None:
        """Plugin must have authenticate method."""
        assert hasattr(self.plugin_class, "authenticate")
        assert callable(self.plugin_class.authenticate)

    def test_has_load_documents_method(self) -> None:
        """Plugin must have load_documents method."""
        assert hasattr(self.plugin_class, "load_documents")
        assert callable(self.plugin_class.load_documents)

    def test_has_get_config_fields_classmethod(self) -> None:
        """Plugin must have get_config_fields classmethod."""
        assert hasattr(self.plugin_class, "get_config_fields")
        fields = self.plugin_class.get_config_fields()
        assert isinstance(fields, list)

    def test_get_config_fields_returns_valid_fields(self) -> None:
        """get_config_fields must return valid field definitions."""
        fields = self.plugin_class.get_config_fields()

        for field in fields:
            assert isinstance(field, dict), "Each field must be a dict"
            assert "name" in field, "Field must have 'name'"
            assert "type" in field, "Field must have 'type'"
            assert "label" in field, "Field must have 'label'"


class RerankerPluginContractTest(PluginContractTest):
    """Contract tests for reranker plugins.

    Reranker plugins must implement:
    - rerank(query, documents, top_k, metadata) -> list[RerankResult]
    - get_capabilities() -> RerankerCapabilities
    """

    def test_plugin_type_is_reranker(self) -> None:
        """Reranker plugin must have PLUGIN_TYPE = 'reranker'."""
        assert self.plugin_class.PLUGIN_TYPE == "reranker"

    def test_has_rerank_method(self) -> None:
        """Plugin must have rerank method."""
        assert hasattr(self.plugin_class, "rerank")
        assert callable(self.plugin_class.rerank)

    def test_has_get_capabilities_classmethod(self) -> None:
        """Plugin must have get_capabilities classmethod."""
        assert hasattr(self.plugin_class, "get_capabilities")

    def test_get_capabilities_returns_valid_capabilities(self) -> None:
        """get_capabilities must return valid RerankerCapabilities."""
        from shared.plugins.types.reranker import RerankerCapabilities

        caps = self.plugin_class.get_capabilities()

        assert isinstance(caps, RerankerCapabilities), (
            f"get_capabilities() must return RerankerCapabilities, got {type(caps)}"
        )
        assert caps.max_documents > 0, "max_documents must be positive"
        assert caps.max_query_length > 0, "max_query_length must be positive"
        assert caps.max_doc_length > 0, "max_doc_length must be positive"


class ExtractorPluginContractTest(PluginContractTest):
    """Contract tests for extractor plugins.

    Extractor plugins must implement:
    - supported_extractions() -> list[ExtractionType]
    - extract(text, extraction_types, options) -> ExtractionResult
    """

    def test_plugin_type_is_extractor(self) -> None:
        """Extractor plugin must have PLUGIN_TYPE = 'extractor'."""
        assert self.plugin_class.PLUGIN_TYPE == "extractor"

    def test_has_supported_extractions_classmethod(self) -> None:
        """Plugin must have supported_extractions classmethod."""
        assert hasattr(self.plugin_class, "supported_extractions")

    def test_supported_extractions_returns_list(self) -> None:
        """supported_extractions must return a list of ExtractionType."""
        from shared.plugins.types.extractor import ExtractionType

        extractions = self.plugin_class.supported_extractions()

        assert isinstance(extractions, list), "supported_extractions must return a list"
        assert len(extractions) > 0, "Plugin must support at least one extraction type"

        for ext in extractions:
            assert isinstance(ext, ExtractionType), (
                f"Each extraction must be ExtractionType, got {type(ext)}"
            )

    def test_has_extract_method(self) -> None:
        """Plugin must have extract method."""
        assert hasattr(self.plugin_class, "extract")
        assert callable(self.plugin_class.extract)

    @pytest.mark.asyncio()
    async def test_extract_returns_extraction_result(
        self, plugin_instance: SemanticPlugin
    ) -> None:
        """extract() must return ExtractionResult."""
        from shared.plugins.types.extractor import ExtractionResult

        await plugin_instance.initialize()

        try:
            result = await plugin_instance.extract("Test text for extraction.")  # type: ignore[attr-defined]

            assert isinstance(result, ExtractionResult), (
                f"extract() must return ExtractionResult, got {type(result)}"
            )
        finally:
            await plugin_instance.cleanup()
