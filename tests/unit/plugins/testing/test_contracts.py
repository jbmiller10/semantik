"""Tests for plugin contract test base classes."""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from shared.plugins.manifest import PluginManifest
from shared.plugins.testing.contracts import (
    ChunkingPluginContractTest,
    ConnectorPluginContractTest,
    EmbeddingPluginContractTest,
    ExtractorPluginContractTest,
    PluginContractTest,
    RerankerPluginContractTest,
)
from shared.plugins.types.extractor import ExtractionResult, ExtractionType
from shared.plugins.types.reranker import RerankerCapabilities

# =============================================================================
# Mock Plugin Base Classes
# =============================================================================


class MockBasePlugin:
    """Base mock plugin for testing contracts."""

    PLUGIN_TYPE: ClassVar[str] = "mock"
    PLUGIN_ID: ClassVar[str] = "mock-plugin"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name="Mock Plugin",
            description="A mock plugin for testing",
            requires=[],
            capabilities={},
        )

    @classmethod
    def get_config_schema(cls) -> dict[str, Any] | None:
        return {
            "type": "object",
            "properties": {"key": {"type": "string"}},
        }

    @classmethod
    async def health_check(cls, config: dict[str, Any] | None = None) -> bool:  # noqa: ARG003
        return True

    async def initialize(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass


# =============================================================================
# Mock Plugin Implementations
# =============================================================================


class MockEmbeddingPlugin(MockBasePlugin):
    """Mock embedding plugin for testing."""

    PLUGIN_TYPE = "embedding"
    PLUGIN_ID = "mock-embedding"

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name="Mock Embedding",
            description="Mock embedding plugin",
            requires=[],
            capabilities={},
        )

    async def embed_single(self, text: str) -> list[float]:
        return [0.1] * 384

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    def get_dimension(self) -> int:
        return 384


class MockChunkingPlugin(MockBasePlugin):
    """Mock chunking plugin for testing."""

    PLUGIN_TYPE = "chunking"
    PLUGIN_ID = "mock-chunking"

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name="Mock Chunking",
            description="Mock chunking plugin",
            requires=[],
            capabilities={},
        )

    def chunk(self, content: str, config: dict | None = None) -> list:
        return []

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        return (True, None)

    def estimate_chunks(self, content_length: int, config: dict | None = None) -> int:
        return content_length // 100


class MockConnectorPlugin(MockBasePlugin):
    """Mock connector plugin for testing."""

    PLUGIN_TYPE = "connector"
    PLUGIN_ID = "mock-connector"

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name="Mock Connector",
            description="Mock connector plugin",
            requires=[],
            capabilities={},
        )

    async def authenticate(self) -> bool:
        return True

    async def load_documents(self, source_id: str):
        yield MagicMock()

    @classmethod
    def get_config_fields(cls) -> list[dict]:
        return [
            {"name": "path", "type": "string", "label": "Path"},
        ]


class MockRerankerPlugin(MockBasePlugin):
    """Mock reranker plugin for testing."""

    PLUGIN_TYPE = "reranker"
    PLUGIN_ID = "mock-reranker"

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name="Mock Reranker",
            description="Mock reranker plugin",
            requires=[],
            capabilities={},
        )

    async def rerank(self, query: str, documents: list, top_k: int | None = None, metadata: dict | None = None) -> list:
        return []

    @classmethod
    def get_capabilities(cls) -> RerankerCapabilities:
        return RerankerCapabilities(
            max_documents=100,
            max_query_length=512,
            max_doc_length=2048,
            supports_batching=False,
        )


class MockExtractorPlugin(MockBasePlugin):
    """Mock extractor plugin for testing."""

    PLUGIN_TYPE = "extractor"
    PLUGIN_ID = "mock-extractor"

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name="Mock Extractor",
            description="Mock extractor plugin",
            requires=[],
            capabilities={},
        )

    @classmethod
    def supported_extractions(cls) -> list[ExtractionType]:
        return [ExtractionType.ENTITIES, ExtractionType.KEYWORDS]

    async def extract(
        self,
        text: str,
        extraction_types: list[ExtractionType] | None = None,
        options: dict | None = None,
    ) -> ExtractionResult:
        return ExtractionResult(
            entities=[],
            keywords=[],
        )


# =============================================================================
# Contract Test Implementations
# =============================================================================


class MockPluginTest(PluginContractTest):
    """Test implementation for base contract tests."""

    plugin_class = MockBasePlugin


class MockEmbeddingTest(EmbeddingPluginContractTest):
    """Test implementation for embedding contract tests."""

    plugin_class = MockEmbeddingPlugin


class MockChunkingTest(ChunkingPluginContractTest):
    """Test implementation for chunking contract tests."""

    plugin_class = MockChunkingPlugin


class MockConnectorTest(ConnectorPluginContractTest):
    """Test implementation for connector contract tests."""

    plugin_class = MockConnectorPlugin


class MockRerankerTest(RerankerPluginContractTest):
    """Test implementation for reranker contract tests."""

    plugin_class = MockRerankerPlugin


class MockExtractorTest(ExtractorPluginContractTest):
    """Test implementation for extractor contract tests."""

    plugin_class = MockExtractorPlugin


# =============================================================================
# Tests for Base PluginContractTest
# =============================================================================


class TestPluginContractTestBase:
    """Tests for the base PluginContractTest class."""

    def test_has_plugin_type_passes(self) -> None:
        """test_has_plugin_type should pass for valid plugin."""
        test = MockPluginTest()
        test.test_has_plugin_type()

    def test_has_plugin_type_fails_without_attribute(self) -> None:
        """test_has_plugin_type should fail when PLUGIN_TYPE is missing."""

        class BadPlugin:
            pass

        class BadPluginTest(PluginContractTest):
            plugin_class = BadPlugin  # type: ignore[assignment]

        test = BadPluginTest()
        with pytest.raises(AssertionError):
            test.test_has_plugin_type()

    def test_has_plugin_id_passes(self) -> None:
        """test_has_plugin_id should pass for valid plugin."""
        test = MockPluginTest()
        test.test_has_plugin_id()

    def test_has_plugin_version_passes(self) -> None:
        """test_has_plugin_version should pass for valid plugin."""
        test = MockPluginTest()
        test.test_has_plugin_version()

    def test_has_plugin_version_fails_invalid_format(self) -> None:
        """test_has_plugin_version should fail for invalid version format."""

        class BadVersionPlugin(MockBasePlugin):
            PLUGIN_VERSION = "1"  # Missing minor version

        class BadVersionTest(PluginContractTest):
            plugin_class = BadVersionPlugin

        test = BadVersionTest()
        with pytest.raises(AssertionError):
            test.test_has_plugin_version()

    def test_get_manifest_passes(self) -> None:
        """test_get_manifest_returns_valid_manifest should pass for valid plugin."""
        test = MockPluginTest()
        test.test_get_manifest_returns_valid_manifest()

    def test_get_manifest_fails_wrong_id(self) -> None:
        """test_get_manifest should fail when manifest ID doesn't match PLUGIN_ID."""

        class WrongIdPlugin(MockBasePlugin):
            PLUGIN_ID = "correct-id"

            @classmethod
            def get_manifest(cls) -> PluginManifest:
                return PluginManifest(
                    id="wrong-id",
                    type=cls.PLUGIN_TYPE,
                    version=cls.PLUGIN_VERSION,
                    display_name="Test",
                    description="Test",
                    requires=[],
                    capabilities={},
                )

        class WrongIdTest(PluginContractTest):
            plugin_class = WrongIdPlugin

        test = WrongIdTest()
        with pytest.raises(AssertionError):
            test.test_get_manifest_returns_valid_manifest()

    def test_get_config_schema_passes(self) -> None:
        """test_get_config_schema should pass for valid schema."""
        test = MockPluginTest()
        test.test_get_config_schema_returns_valid_schema_or_none()

    def test_get_config_schema_passes_with_none(self) -> None:
        """test_get_config_schema should pass when schema is None."""

        class NoSchemaPlugin(MockBasePlugin):
            @classmethod
            def get_config_schema(cls) -> None:
                return None

        class NoSchemaTest(PluginContractTest):
            plugin_class = NoSchemaPlugin

        test = NoSchemaTest()
        test.test_get_config_schema_returns_valid_schema_or_none()

    @pytest.mark.asyncio()
    async def test_health_check_passes(self) -> None:
        """test_health_check_returns_bool should pass for valid plugin."""
        test = MockPluginTest()
        await test.test_health_check_returns_bool(None)

    @pytest.mark.asyncio()
    async def test_initialize_and_cleanup_passes(self) -> None:
        """test_initialize_and_cleanup should pass for valid plugin."""
        test = MockPluginTest()
        instance = test.plugin_class(config=None)
        await test.test_initialize_and_cleanup(instance)

    def test_config_property_passes(self) -> None:
        """test_config_property should pass for valid plugin."""
        test = MockPluginTest()
        config = {"key": "value"}
        instance = test.plugin_class(config=config)
        test.test_config_property(instance, config)


# =============================================================================
# Tests for EmbeddingPluginContractTest
# =============================================================================


class TestEmbeddingPluginContractTest:
    """Tests for EmbeddingPluginContractTest."""

    def test_plugin_type_is_embedding_passes(self) -> None:
        """test_plugin_type_is_embedding should pass for embedding plugin."""
        test = MockEmbeddingTest()
        test.test_plugin_type_is_embedding()

    def test_plugin_type_fails_wrong_type(self) -> None:
        """test_plugin_type_is_embedding should fail for non-embedding plugin."""

        class WrongTypePlugin(MockEmbeddingPlugin):
            PLUGIN_TYPE = "wrong"

        class WrongTypeTest(EmbeddingPluginContractTest):
            plugin_class = WrongTypePlugin

        test = WrongTypeTest()
        with pytest.raises(AssertionError):
            test.test_plugin_type_is_embedding()

    def test_has_embed_single_method(self) -> None:
        """test_has_embed_single_method should pass for valid plugin."""
        test = MockEmbeddingTest()
        test.test_has_embed_single_method()

    def test_has_embed_texts_method(self) -> None:
        """test_has_embed_texts_method should pass for valid plugin."""
        test = MockEmbeddingTest()
        test.test_has_embed_texts_method()

    def test_has_get_dimension_method(self) -> None:
        """test_has_get_dimension_method should pass for valid plugin."""
        test = MockEmbeddingTest()
        test.test_has_get_dimension_method()


# =============================================================================
# Tests for ChunkingPluginContractTest
# =============================================================================


class TestChunkingPluginContractTest:
    """Tests for ChunkingPluginContractTest."""

    def test_plugin_type_is_chunking_passes(self) -> None:
        """test_plugin_type_is_chunking should pass for chunking plugin."""
        test = MockChunkingTest()
        test.test_plugin_type_is_chunking()

    def test_has_chunk_method(self) -> None:
        """test_has_chunk_method should pass for valid plugin."""
        test = MockChunkingTest()
        test.test_has_chunk_method()

    def test_has_validate_content_method(self) -> None:
        """test_has_validate_content_method should pass for valid plugin."""
        test = MockChunkingTest()
        test.test_has_validate_content_method()

    def test_has_estimate_chunks_method(self) -> None:
        """test_has_estimate_chunks_method should pass for valid plugin."""
        test = MockChunkingTest()
        test.test_has_estimate_chunks_method()


# =============================================================================
# Tests for ConnectorPluginContractTest
# =============================================================================


class TestConnectorPluginContractTest:
    """Tests for ConnectorPluginContractTest."""

    def test_plugin_type_is_connector_passes(self) -> None:
        """test_plugin_type_is_connector should pass for connector plugin."""
        test = MockConnectorTest()
        test.test_plugin_type_is_connector()

    def test_has_authenticate_method(self) -> None:
        """test_has_authenticate_method should pass for valid plugin."""
        test = MockConnectorTest()
        test.test_has_authenticate_method()

    def test_has_load_documents_method(self) -> None:
        """test_has_load_documents_method should pass for valid plugin."""
        test = MockConnectorTest()
        test.test_has_load_documents_method()

    def test_has_get_config_fields(self) -> None:
        """test_has_get_config_fields_classmethod should pass for valid plugin."""
        test = MockConnectorTest()
        test.test_has_get_config_fields_classmethod()

    def test_get_config_fields_returns_valid_fields(self) -> None:
        """test_get_config_fields_returns_valid_fields should pass for valid plugin."""
        test = MockConnectorTest()
        test.test_get_config_fields_returns_valid_fields()

    def test_get_config_fields_fails_missing_name(self) -> None:
        """test_get_config_fields should fail when field is missing name."""

        class BadFieldPlugin(MockConnectorPlugin):
            @classmethod
            def get_config_fields(cls) -> list[dict]:
                return [{"type": "string", "label": "Path"}]

        class BadFieldTest(ConnectorPluginContractTest):
            plugin_class = BadFieldPlugin

        test = BadFieldTest()
        with pytest.raises(AssertionError):
            test.test_get_config_fields_returns_valid_fields()


# =============================================================================
# Tests for RerankerPluginContractTest
# =============================================================================


class TestRerankerPluginContractTest:
    """Tests for RerankerPluginContractTest."""

    def test_plugin_type_is_reranker_passes(self) -> None:
        """test_plugin_type_is_reranker should pass for reranker plugin."""
        test = MockRerankerTest()
        test.test_plugin_type_is_reranker()

    def test_has_rerank_method(self) -> None:
        """test_has_rerank_method should pass for valid plugin."""
        test = MockRerankerTest()
        test.test_has_rerank_method()

    def test_has_get_capabilities_classmethod(self) -> None:
        """test_has_get_capabilities_classmethod should pass for valid plugin."""
        test = MockRerankerTest()
        test.test_has_get_capabilities_classmethod()

    def test_get_capabilities_returns_valid_capabilities(self) -> None:
        """test_get_capabilities_returns_valid_capabilities should pass for valid plugin."""
        test = MockRerankerTest()
        test.test_get_capabilities_returns_valid_capabilities()

    def test_get_capabilities_fails_invalid_max_documents(self) -> None:
        """test_get_capabilities should fail when max_documents is zero."""

        class BadCapsPlugin(MockRerankerPlugin):
            @classmethod
            def get_capabilities(cls) -> RerankerCapabilities:
                return RerankerCapabilities(
                    max_documents=0,
                    max_query_length=512,
                    max_doc_length=2048,
                    supports_batching=False,
                )

        class BadCapsTest(RerankerPluginContractTest):
            plugin_class = BadCapsPlugin

        test = BadCapsTest()
        with pytest.raises(AssertionError):
            test.test_get_capabilities_returns_valid_capabilities()


# =============================================================================
# Tests for ExtractorPluginContractTest
# =============================================================================


class TestExtractorPluginContractTest:
    """Tests for ExtractorPluginContractTest."""

    def test_plugin_type_is_extractor_passes(self) -> None:
        """test_plugin_type_is_extractor should pass for extractor plugin."""
        test = MockExtractorTest()
        test.test_plugin_type_is_extractor()

    def test_has_supported_extractions_classmethod(self) -> None:
        """test_has_supported_extractions_classmethod should pass for valid plugin."""
        test = MockExtractorTest()
        test.test_has_supported_extractions_classmethod()

    def test_supported_extractions_returns_list(self) -> None:
        """test_supported_extractions_returns_list should pass for valid plugin."""
        test = MockExtractorTest()
        test.test_supported_extractions_returns_list()

    def test_has_extract_method(self) -> None:
        """test_has_extract_method should pass for valid plugin."""
        test = MockExtractorTest()
        test.test_has_extract_method()

    @pytest.mark.asyncio()
    async def test_extract_returns_extraction_result(self) -> None:
        """test_extract_returns_extraction_result should pass for valid plugin."""
        test = MockExtractorTest()
        instance = test.plugin_class(config=None)
        await test.test_extract_returns_extraction_result(instance)

    def test_supported_extractions_fails_empty(self) -> None:
        """test_supported_extractions should fail when list is empty."""

        class EmptyExtractorPlugin(MockExtractorPlugin):
            @classmethod
            def supported_extractions(cls) -> list[ExtractionType]:
                return []

        class EmptyExtractorTest(ExtractorPluginContractTest):
            plugin_class = EmptyExtractorPlugin

        test = EmptyExtractorTest()
        with pytest.raises(AssertionError):
            test.test_supported_extractions_returns_list()
