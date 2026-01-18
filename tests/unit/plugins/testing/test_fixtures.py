"""Tests for plugin testing fixtures.

Note: Fixtures are loaded via conftest.py which uses pytest_plugins to load
shared.plugins.testing.fixtures. We only import the mock types for type hints.
"""

from __future__ import annotations

import pytest

from shared.plugins.testing.mocks import MockChunker, MockDocument, MockEmbeddingService, MockExtractor, MockReranker


class TestSampleDataFixtures:
    """Tests for sample data fixtures."""

    def test_sample_text_is_string(self, sample_text: str) -> None:
        """sample_text should return a non-empty string."""
        assert isinstance(sample_text, str)
        assert len(sample_text) > 0

    def test_sample_text_has_entities(self, sample_text: str) -> None:
        """sample_text should contain named entities for NER testing."""
        assert "Apple" in sample_text
        assert "Steve Jobs" in sample_text
        assert "California" in sample_text

    def test_sample_short_text_is_short(self, sample_short_text: str) -> None:
        """sample_short_text should be relatively short."""
        assert len(sample_short_text) < 100

    def test_sample_documents_is_list(self, sample_documents: list[str]) -> None:
        """sample_documents should return a list of strings."""
        assert isinstance(sample_documents, list)
        assert len(sample_documents) > 0
        assert all(isinstance(doc, str) for doc in sample_documents)

    def test_sample_documents_varied_content(self, sample_documents: list[str]) -> None:
        """sample_documents should have varied content for reranking tests."""
        # Should have multiple distinct documents
        assert len(sample_documents) >= 3
        # Documents should have different content
        unique_docs = set(sample_documents)
        assert len(unique_docs) == len(sample_documents)

    def test_sample_query_is_string(self, sample_query: str) -> None:
        """sample_query should return a non-empty string."""
        assert isinstance(sample_query, str)
        assert len(sample_query) > 0

    def test_sample_long_document_is_long(self, sample_long_document: str) -> None:
        """sample_long_document should be substantially longer than sample_text."""
        assert len(sample_long_document) > 500

    def test_sample_long_document_has_paragraphs(self, sample_long_document: str) -> None:
        """sample_long_document should contain multiple paragraphs."""
        paragraphs = sample_long_document.split("\n\n")
        assert len(paragraphs) >= 5

    def test_sample_metadata_is_dict(self, sample_metadata: dict) -> None:
        """sample_metadata should return a dictionary."""
        assert isinstance(sample_metadata, dict)
        assert len(sample_metadata) > 0

    def test_sample_metadata_has_common_fields(self, sample_metadata: dict) -> None:
        """sample_metadata should have common metadata fields."""
        assert "source" in sample_metadata
        assert "author" in sample_metadata
        assert "date" in sample_metadata


class TestMockDocumentFixture:
    """Tests for sample_mock_document fixture."""

    def test_sample_mock_document_type(self, sample_mock_document: MockDocument) -> None:
        """sample_mock_document should be a MockDocument instance."""
        assert isinstance(sample_mock_document, MockDocument)

    def test_sample_mock_document_has_content(self, sample_mock_document: MockDocument, sample_text: str) -> None:
        """sample_mock_document should have sample_text content."""
        assert sample_mock_document.content == sample_text

    def test_sample_mock_document_has_metadata(self, sample_mock_document: MockDocument, sample_metadata: dict) -> None:
        """sample_mock_document should have sample_metadata."""
        assert sample_mock_document.metadata == sample_metadata


class TestMockServiceFixtures:
    """Tests for mock service fixtures."""

    def test_mock_embedding_service_type(self, mock_embedding_service: MockEmbeddingService) -> None:
        """mock_embedding_service should be a MockEmbeddingService instance."""
        assert isinstance(mock_embedding_service, MockEmbeddingService)

    def test_mock_embedding_service_dimension(self, mock_embedding_service: MockEmbeddingService) -> None:
        """mock_embedding_service should have 384 dimension."""
        assert mock_embedding_service.dimension == 384

    def test_mock_embedding_service_768_dimension(self, mock_embedding_service_768: MockEmbeddingService) -> None:
        """mock_embedding_service_768 should have 768 dimension."""
        assert mock_embedding_service_768.dimension == 768

    def test_mock_reranker_type(self, mock_reranker: MockReranker) -> None:
        """mock_reranker should be a MockReranker instance."""
        assert isinstance(mock_reranker, MockReranker)

    def test_mock_extractor_type(self, mock_extractor: MockExtractor) -> None:
        """mock_extractor should be a MockExtractor instance."""
        assert isinstance(mock_extractor, MockExtractor)

    def test_mock_chunker_type(self, mock_chunker: MockChunker) -> None:
        """mock_chunker should be a MockChunker instance."""
        assert isinstance(mock_chunker, MockChunker)

    def test_mock_chunker_settings(self, mock_chunker: MockChunker) -> None:
        """mock_chunker should have default settings."""
        assert mock_chunker.chunk_size == 100
        assert mock_chunker.overlap == 20

    def test_mock_chunker_large_settings(self, mock_chunker_large: MockChunker) -> None:
        """mock_chunker_large should have larger chunk settings."""
        assert mock_chunker_large.chunk_size == 500
        assert mock_chunker_large.overlap == 50


class TestConfigurationFixtures:
    """Tests for configuration fixtures."""

    def test_sample_plugin_config_is_dict(self, sample_plugin_config: dict) -> None:
        """sample_plugin_config should be a dictionary."""
        assert isinstance(sample_plugin_config, dict)

    def test_sample_plugin_config_has_common_fields(self, sample_plugin_config: dict) -> None:
        """sample_plugin_config should have common plugin config fields."""
        assert "api_key_env" in sample_plugin_config
        assert "model" in sample_plugin_config

    def test_sample_embedding_config_has_model(self, sample_embedding_config: dict) -> None:
        """sample_embedding_config should have model_name."""
        assert "model_name" in sample_embedding_config
        assert isinstance(sample_embedding_config["model_name"], str)

    def test_sample_embedding_config_has_device(self, sample_embedding_config: dict) -> None:
        """sample_embedding_config should have device setting."""
        assert "device" in sample_embedding_config

    def test_sample_reranker_config_has_model(self, sample_reranker_config: dict) -> None:
        """sample_reranker_config should have model_name."""
        assert "model_name" in sample_reranker_config

    def test_sample_reranker_config_has_max_documents(self, sample_reranker_config: dict) -> None:
        """sample_reranker_config should have max_documents."""
        assert "max_documents" in sample_reranker_config
        assert sample_reranker_config["max_documents"] > 0

    def test_sample_extractor_config_has_model(self, sample_extractor_config: dict) -> None:
        """sample_extractor_config should have model_name."""
        assert "model_name" in sample_extractor_config

    def test_sample_chunk_config_has_size(self, sample_chunk_config: dict) -> None:
        """sample_chunk_config should have chunk_size."""
        assert "chunk_size" in sample_chunk_config
        assert sample_chunk_config["chunk_size"] > 0

    def test_sample_chunk_config_has_overlap(self, sample_chunk_config: dict) -> None:
        """sample_chunk_config should have chunk_overlap."""
        assert "chunk_overlap" in sample_chunk_config
        assert sample_chunk_config["chunk_overlap"] >= 0


class TestIntegration:
    """Integration tests for fixtures working together."""

    @pytest.mark.asyncio()
    async def test_embedding_service_with_sample_documents(
        self,
        mock_embedding_service: MockEmbeddingService,
        sample_documents: list[str],
    ) -> None:
        """MockEmbeddingService should work with sample_documents."""
        embeddings = await mock_embedding_service.embed_texts(sample_documents)

        assert len(embeddings) == len(sample_documents)
        for emb in embeddings:
            assert len(emb) == mock_embedding_service.dimension

    @pytest.mark.asyncio()
    async def test_reranker_with_sample_query_and_documents(
        self,
        mock_reranker: MockReranker,
        sample_query: str,
        sample_documents: list[str],
    ) -> None:
        """MockReranker should work with sample_query and sample_documents."""
        results = await mock_reranker.rerank(sample_query, sample_documents)

        assert len(results) == len(sample_documents)

    @pytest.mark.asyncio()
    async def test_extractor_with_sample_text(
        self,
        mock_extractor: MockExtractor,
        sample_text: str,
    ) -> None:
        """MockExtractor should work with sample_text."""
        result = await mock_extractor.extract(sample_text)

        assert result.entities
        assert result.keywords

    def test_chunker_with_long_document(
        self,
        mock_chunker: MockChunker,
        sample_long_document: str,
    ) -> None:
        """MockChunker should work with sample_long_document."""
        chunks = mock_chunker.chunk(sample_long_document)

        # Long document should produce multiple chunks
        assert len(chunks) > 1

        # All content should be covered
        combined_length = sum(len(c["content"]) for c in chunks)
        # With overlap, combined length may be greater than original
        assert combined_length >= len(sample_long_document)
