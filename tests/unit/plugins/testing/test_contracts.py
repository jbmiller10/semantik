"""Tests for plugin contract test base classes."""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest

from shared.connectors.base import BaseConnector
from shared.embedding.plugin_base import BaseEmbeddingPlugin, EmbeddingProviderDefinition
from shared.plugins.manifest import PluginManifest
from shared.plugins.testing.contracts import (
    ChunkingPluginContractTest,
    ConnectorPluginContractTest,
    EmbeddingPluginContractTest,
    ExtractorPluginContractTest,
    PluginContractTest,
    RerankerPluginContractTest,
)
from shared.plugins.types.extractor import Entity, ExtractionResult, ExtractionType
from shared.plugins.types.reranker import RerankerCapabilities, RerankResult

# =============================================================================
# Mock plugin implementations used by the contract tests
# =============================================================================


class MockSemanticPlugin:
    """Minimal SemanticPlugin-like class for contract tests."""

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
        )

    @classmethod
    def get_config_schema(cls) -> dict[str, Any] | None:
        return {"type": "object", "properties": {"key": {"type": "string"}}}

    @classmethod
    async def health_check(cls, config: dict[str, Any] | None = None) -> bool:  # noqa: ARG003
        return True

    async def initialize(self) -> None:
        return

    async def cleanup(self) -> None:
        return


class MockEmbeddingProvider(BaseEmbeddingPlugin):
    """Minimal embedding provider plugin for contract tests."""

    INTERNAL_NAME: ClassVar[str] = "mock_embeddings"
    API_ID: ClassVar[str] = "mock-embeddings"
    PROVIDER_TYPE: ClassVar[str] = "local"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"
    METADATA: ClassVar[dict[str, Any]] = {
        "display_name": "Mock Embeddings",
        "description": "Mock embeddings provider",
    }

    def __init__(self, config: Any | None = None, **kwargs: Any) -> None:  # noqa: ARG002
        super().__init__(config)
        self._initialized = False
        self._model_name: str | None = None
        self._dim = 8

    @classmethod
    def get_definition(cls) -> EmbeddingProviderDefinition:
        return EmbeddingProviderDefinition(
            api_id=cls.API_ID,
            internal_id=cls.INTERNAL_NAME,
            display_name="Mock Embeddings",
            description="Mock embeddings provider",
            provider_type=cls.PROVIDER_TYPE,
            supported_models=("mock",),
            is_plugin=True,
        )

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return bool(model_name)

    async def initialize(self, model_name: str, **kwargs: Any) -> None:  # noqa: ARG002
        self._initialized = True
        self._model_name = model_name

    async def embed_texts(self, texts: list[str], batch_size: int = 32, *, mode: Any | None = None, **kwargs: Any):  # type: ignore[override]  # noqa: ARG002,E501
        return np.zeros((len(texts), self._dim), dtype=np.float32)

    async def embed_single(self, text: str, *, mode: Any | None = None, **kwargs: Any):  # type: ignore[override]  # noqa: ARG002,E501
        return np.zeros((self._dim,), dtype=np.float32)

    def get_dimension(self) -> int:
        return self._dim

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name or "mock",
            "dimension": self._dim,
            "device": "cpu",
            "max_sequence_length": 1,
        }

    async def cleanup(self) -> None:
        self._initialized = False
        self._model_name = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized


class MockChunkingPlugin:
    """Minimal chunking plugin for contract tests."""

    INTERNAL_NAME: ClassVar[str] = "mock_chunking"
    API_ID: ClassVar[str] = "mock-chunking"
    METADATA: ClassVar[dict[str, Any]] = {
        "visual_example": {"url": "https://example.com/example.png", "caption": "Example"},
        "display_name": "Mock Chunking",
        "description": "Mock chunking strategy",
    }

    def chunk(self, content: str, config: dict | None = None) -> list:  # noqa: ARG002
        return [content]

    def validate_content(self, content: str) -> tuple[bool, str | None]:  # noqa: ARG002
        return True, None

    def estimate_chunks(self, content_length: int, config: dict | None = None) -> int:  # noqa: ARG002
        return max(1, content_length // 100)


class MockConnector(BaseConnector):
    """Minimal connector plugin for contract tests."""

    PLUGIN_ID: ClassVar[str] = "mock-connector"
    METADATA: ClassVar[dict[str, Any]] = {"name": "Mock Connector", "description": "Mock connector"}

    async def authenticate(self) -> bool:
        return True

    async def load_documents(self, source_id: int | None = None):  # noqa: ARG002
        yield MagicMock()

    @classmethod
    def get_config_fields(cls) -> list[dict[str, Any]]:
        return [{"name": "path", "type": "string", "label": "Path"}]


class MockRerankerPlugin(MockSemanticPlugin):
    """Mock reranker plugin for contract tests."""

    PLUGIN_TYPE = "reranker"
    PLUGIN_ID = "mock-reranker"

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,  # noqa: ARG002
        metadata: list[dict[str, Any]] | None = None,  # noqa: ARG002
    ) -> list[RerankResult]:
        return [RerankResult(index=0, score=1.0, document=documents[0] if documents else "", metadata={})]

    @classmethod
    def get_capabilities(cls) -> RerankerCapabilities:
        return RerankerCapabilities(
            max_documents=100,
            max_query_length=512,
            max_doc_length=2048,
            supports_batching=False,
            models=["mock"],
        )


class MockExtractorPlugin(MockSemanticPlugin):
    """Mock extractor plugin for contract tests."""

    PLUGIN_TYPE = "extractor"
    PLUGIN_ID = "mock-extractor"

    @classmethod
    def supported_extractions(cls) -> list[ExtractionType]:
        return [ExtractionType.ENTITIES, ExtractionType.KEYWORDS]

    async def extract(
        self,
        text: str,
        extraction_types: list[ExtractionType] | None = None,  # noqa: ARG002
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> ExtractionResult:
        return ExtractionResult(
            entities=[Entity(text="Mock", type="MOCK", start=0, end=4)],
            keywords=["mock"],
        )


# =============================================================================
# Contract test harness implementations
# =============================================================================


class MockPluginTest(PluginContractTest):
    plugin_class = MockSemanticPlugin  # type: ignore[assignment]


class MockEmbeddingTest(EmbeddingPluginContractTest):
    plugin_class = MockEmbeddingProvider


class MockChunkingTest(ChunkingPluginContractTest):
    plugin_class = MockChunkingPlugin


class MockConnectorTest(ConnectorPluginContractTest):
    plugin_class = MockConnector


class MockRerankerTest(RerankerPluginContractTest):
    plugin_class = MockRerankerPlugin  # type: ignore[assignment]


class MockExtractorTest(ExtractorPluginContractTest):
    plugin_class = MockExtractorPlugin  # type: ignore[assignment]


# =============================================================================
# Tests for PluginContractTest (SemanticPlugin-based)
# =============================================================================


class TestPluginContractTestBase:
    def test_has_plugin_type_passes(self) -> None:
        MockPluginTest().test_has_plugin_type()

    def test_has_plugin_id_passes(self) -> None:
        MockPluginTest().test_has_plugin_id()

    def test_has_plugin_version_passes(self) -> None:
        MockPluginTest().test_has_plugin_version()

    def test_get_manifest_passes(self) -> None:
        MockPluginTest().test_get_manifest_returns_valid_manifest()

    def test_get_manifest_fails_wrong_id(self) -> None:
        class WrongIdPlugin(MockSemanticPlugin):
            PLUGIN_ID = "correct-id"

            @classmethod
            def get_manifest(cls) -> PluginManifest:
                return PluginManifest(
                    id="wrong-id",
                    type=cls.PLUGIN_TYPE,
                    version=cls.PLUGIN_VERSION,
                    display_name="Test",
                    description="Test",
                )

        class WrongIdTest(PluginContractTest):
            plugin_class = WrongIdPlugin  # type: ignore[assignment]

        with pytest.raises(AssertionError):
            WrongIdTest().test_get_manifest_returns_valid_manifest()


# =============================================================================
# Tests for EmbeddingPluginContractTest
# =============================================================================


class TestEmbeddingPluginContractTest:
    def test_embedding_contract_passes(self) -> None:
        test = MockEmbeddingTest()
        test.test_is_base_embedding_plugin()
        test.test_has_required_class_attributes()
        test.test_has_required_classmethods()
        test.test_get_definition_returns_valid_definition()
        test.test_supports_model_returns_bool()

    def test_embedding_contract_fails_missing_internal_name(self) -> None:
        class BadEmbedding(MockEmbeddingProvider):
            INTERNAL_NAME = ""

        class BadTest(EmbeddingPluginContractTest):
            plugin_class = BadEmbedding

        with pytest.raises(AssertionError):
            BadTest().test_has_required_class_attributes()


# =============================================================================
# Tests for ChunkingPluginContractTest
# =============================================================================


class TestChunkingPluginContractTest:
    def test_chunking_contract_passes(self) -> None:
        test = MockChunkingTest()
        test.test_has_internal_name()
        test.test_metadata_has_visual_example()

    def test_chunking_contract_fails_without_visual_example(self) -> None:
        class BadChunking(MockChunkingPlugin):
            METADATA = {}

        class BadTest(ChunkingPluginContractTest):
            plugin_class = BadChunking

        with pytest.raises(AssertionError):
            BadTest().test_metadata_has_visual_example()


# =============================================================================
# Tests for ConnectorPluginContractTest
# =============================================================================


class TestConnectorPluginContractTest:
    def test_connector_contract_passes(self) -> None:
        test = MockConnectorTest()
        test.test_is_base_connector()
        test.test_has_plugin_id()
        test.test_has_get_config_fields_classmethod()
        test.test_get_config_fields_returns_valid_fields()

    def test_get_config_fields_fails_missing_name(self) -> None:
        class BadConnector(MockConnector):
            @classmethod
            def get_config_fields(cls) -> list[dict[str, Any]]:
                return [{"type": "string", "label": "Path"}]

        class BadTest(ConnectorPluginContractTest):
            plugin_class = BadConnector

        with pytest.raises(AssertionError):
            BadTest().test_get_config_fields_returns_valid_fields()


# =============================================================================
# Tests for RerankerPluginContractTest
# =============================================================================


class TestRerankerPluginContractTest:
    def test_reranker_contract_passes(self) -> None:
        test = MockRerankerTest()
        test.test_plugin_type_is_reranker()
        test.test_has_rerank_method()
        test.test_get_capabilities_returns_valid_capabilities()

    def test_get_capabilities_fails_invalid_max_documents(self) -> None:
        class BadCaps(MockRerankerPlugin):
            @classmethod
            def get_capabilities(cls) -> RerankerCapabilities:
                return RerankerCapabilities(
                    max_documents=0,
                    max_query_length=512,
                    max_doc_length=2048,
                    supports_batching=False,
                )

        class BadTest(RerankerPluginContractTest):
            plugin_class = BadCaps  # type: ignore[assignment]

        with pytest.raises(AssertionError):
            BadTest().test_get_capabilities_returns_valid_capabilities()


# =============================================================================
# Tests for ExtractorPluginContractTest
# =============================================================================


class TestExtractorPluginContractTest:
    def test_extractor_contract_passes(self) -> None:
        test = MockExtractorTest()
        test.test_plugin_type_is_extractor()
        test.test_supported_extractions_returns_list()

    def test_supported_extractions_fails_empty(self) -> None:
        class EmptyExtractor(MockExtractorPlugin):
            @classmethod
            def supported_extractions(cls) -> list[ExtractionType]:
                return []

        class BadTest(ExtractorPluginContractTest):
            plugin_class = EmptyExtractor  # type: ignore[assignment]

        with pytest.raises(AssertionError):
            BadTest().test_supported_extractions_returns_list()
