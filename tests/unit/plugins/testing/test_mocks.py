"""Tests for plugin testing mock implementations."""

from __future__ import annotations

import pytest

from shared.plugins.testing.mocks import MockChunker, MockDocument, MockEmbeddingService, MockExtractor, MockReranker
from shared.plugins.types.extractor import ExtractionResult


class TestMockDocument:
    """Tests for MockDocument dataclass."""

    def test_creates_with_defaults(self) -> None:
        """Should create document with default values."""
        doc = MockDocument(content="Test content")

        assert doc.content == "Test content"
        assert doc.source_id == "test-source"
        assert doc.file_path == "test/document.txt"
        assert doc.metadata == {}

    def test_creates_with_custom_values(self) -> None:
        """Should create document with custom values."""
        metadata = {"key": "value", "count": 42}
        doc = MockDocument(
            content="Custom content",
            source_id="custom-source",
            file_path="/path/to/file.md",
            metadata=metadata,
        )

        assert doc.content == "Custom content"
        assert doc.source_id == "custom-source"
        assert doc.file_path == "/path/to/file.md"
        assert doc.metadata == metadata


class TestMockEmbeddingService:
    """Tests for MockEmbeddingService."""

    def test_creates_with_default_dimension(self) -> None:
        """Should create with default 384 dimension."""
        service = MockEmbeddingService()
        assert service.dimension == 384

    def test_creates_with_custom_dimension(self) -> None:
        """Should create with custom dimension."""
        service = MockEmbeddingService(dimension=768)
        assert service.dimension == 768

    @pytest.mark.asyncio()
    async def test_embed_single_returns_correct_dimension(self) -> None:
        """embed_single should return vector of correct dimension."""
        service = MockEmbeddingService(dimension=384)
        embedding = await service.embed_single("Test text")

        assert len(embedding) == 384
        assert all(isinstance(v, float) for v in embedding)

    @pytest.mark.asyncio()
    async def test_embed_single_tracks_calls(self) -> None:
        """embed_single should track call history."""
        service = MockEmbeddingService()

        await service.embed_single("First")
        await service.embed_single("Second")

        assert service.embed_calls == ["First", "Second"]

    @pytest.mark.asyncio()
    async def test_embed_single_is_deterministic(self) -> None:
        """Same text should produce same embedding."""
        service = MockEmbeddingService()

        embedding1 = await service.embed_single("Test text")
        embedding2 = await service.embed_single("Test text")

        assert embedding1 == embedding2

    @pytest.mark.asyncio()
    async def test_embed_single_different_texts_different_embeddings(self) -> None:
        """Different texts should produce different embeddings."""
        service = MockEmbeddingService()

        embedding1 = await service.embed_single("First text")
        embedding2 = await service.embed_single("Second text")

        assert embedding1 != embedding2

    @pytest.mark.asyncio()
    async def test_embed_texts_returns_batch(self) -> None:
        """embed_texts should return embeddings for all texts."""
        service = MockEmbeddingService(dimension=384)
        texts = ["First", "Second", "Third"]

        embeddings = await service.embed_texts(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384

    @pytest.mark.asyncio()
    async def test_embed_texts_tracks_calls(self) -> None:
        """embed_texts should track batch call history."""
        service = MockEmbeddingService()

        await service.embed_texts(["A", "B"])
        await service.embed_texts(["C"])

        assert service.batch_calls == [["A", "B"], ["C"]]

    def test_get_dimension(self) -> None:
        """get_dimension should return configured dimension."""
        service = MockEmbeddingService(dimension=1024)
        assert service.get_dimension() == 1024


class TestMockReranker:
    """Tests for MockReranker."""

    def test_creates_empty_call_history(self) -> None:
        """Should create with empty call history."""
        reranker = MockReranker()
        assert reranker.rerank_calls == []

    @pytest.mark.asyncio()
    async def test_rerank_returns_all_documents(self) -> None:
        """rerank should return all documents when no top_k."""
        reranker = MockReranker()
        docs = ["Doc A", "Doc B", "Doc C"]

        results = await reranker.rerank("query", docs)

        assert len(results) == 3
        returned_docs = [r.document for r in results]
        assert set(returned_docs) == set(docs)

    @pytest.mark.asyncio()
    async def test_rerank_respects_top_k(self) -> None:
        """rerank should limit results to top_k."""
        reranker = MockReranker()
        docs = ["Doc A", "Doc B", "Doc C", "Doc D", "Doc E"]

        results = await reranker.rerank("query", docs, top_k=3)

        assert len(results) == 3

    @pytest.mark.asyncio()
    async def test_rerank_scores_by_common_words(self) -> None:
        """rerank should score by common words with query."""
        reranker = MockReranker()
        query = "machine learning"
        docs = [
            "Machine learning is powerful",  # Should rank high - contains both words
            "Deep learning uses neural networks",  # Contains "learning"
            "Data science is growing",  # Contains neither
        ]

        results = await reranker.rerank(query, docs)

        # First result should be the doc with "machine learning"
        first_doc = results[0].document.lower()
        assert "machine" in first_doc
        assert "learning" in first_doc

    @pytest.mark.asyncio()
    async def test_rerank_tracks_calls(self) -> None:
        """rerank should track call history."""
        reranker = MockReranker()

        await reranker.rerank("query1", ["doc1"])
        await reranker.rerank("query2", ["doc2", "doc3"])

        assert len(reranker.rerank_calls) == 2
        assert reranker.rerank_calls[0] == ("query1", ["doc1"])
        assert reranker.rerank_calls[1] == ("query2", ["doc2", "doc3"])

    @pytest.mark.asyncio()
    async def test_rerank_returns_original_index(self) -> None:
        """rerank should include original document index."""
        reranker = MockReranker()
        docs = ["First", "Second", "Third"]

        results = await reranker.rerank("query", docs)

        indices = [r.index for r in results]
        assert set(indices) == {0, 1, 2}


class TestMockExtractor:
    """Tests for MockExtractor."""

    def test_creates_empty_call_history(self) -> None:
        """Should create with empty call history."""
        extractor = MockExtractor()
        assert extractor.extract_calls == []

    @pytest.mark.asyncio()
    async def test_extract_returns_extraction_result(self) -> None:
        """extract should return an ExtractionResult."""
        extractor = MockExtractor()

        result = await extractor.extract("Test text for extraction")

        assert isinstance(result, ExtractionResult)

    @pytest.mark.asyncio()
    async def test_extract_includes_entities(self) -> None:
        """extract should include entities from first words."""
        extractor = MockExtractor()

        result = await extractor.extract("Apple Inc headquartered in Cupertino")

        assert len(result.entities) <= 3

    @pytest.mark.asyncio()
    async def test_extract_includes_keywords(self) -> None:
        """extract should include keywords from first words."""
        extractor = MockExtractor()

        result = await extractor.extract("machine learning artificial intelligence")

        assert len(result.keywords) <= 5

    @pytest.mark.asyncio()
    async def test_extract_includes_language(self) -> None:
        """extract should include language detection."""
        extractor = MockExtractor()

        result = await extractor.extract("Some text")

        assert result.language == "en"
        assert result.language_confidence == 0.95

    @pytest.mark.asyncio()
    async def test_extract_tracks_calls(self) -> None:
        """extract should track call history."""
        extractor = MockExtractor()

        await extractor.extract("First text")
        await extractor.extract("Second text")

        assert extractor.extract_calls == ["First text", "Second text"]


class TestMockChunker:
    """Tests for MockChunker."""

    def test_creates_with_default_settings(self) -> None:
        """Should create with default chunk size and overlap."""
        chunker = MockChunker()
        assert chunker.chunk_size == 100
        assert chunker.overlap == 20

    def test_creates_with_custom_settings(self) -> None:
        """Should create with custom settings."""
        chunker = MockChunker(chunk_size=500, overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.overlap == 50

    def test_chunk_short_content(self) -> None:
        """chunk should handle content shorter than chunk_size."""
        chunker = MockChunker(chunk_size=100)
        content = "Short content"

        chunks = chunker.chunk(content)

        assert len(chunks) == 1
        assert chunks[0]["content"] == content

    def test_chunk_long_content(self) -> None:
        """chunk should split long content into multiple chunks."""
        chunker = MockChunker(chunk_size=50, overlap=10)
        content = "A" * 120  # 120 characters

        chunks = chunker.chunk(content)

        assert len(chunks) > 1

    def test_chunk_includes_index(self) -> None:
        """chunks should include sequential index."""
        chunker = MockChunker(chunk_size=50, overlap=10)
        content = "A" * 120

        chunks = chunker.chunk(content)

        for i, chunk in enumerate(chunks):
            assert chunk["index"] == i

    def test_chunk_includes_offsets(self) -> None:
        """chunks should include start and end offsets."""
        chunker = MockChunker(chunk_size=50, overlap=10)
        content = "A" * 120

        chunks = chunker.chunk(content)

        for chunk in chunks:
            assert "start_offset" in chunk
            assert "end_offset" in chunk
            assert chunk["start_offset"] < chunk["end_offset"]

    def test_chunk_includes_metadata(self) -> None:
        """chunks should include metadata dict."""
        chunker = MockChunker()
        content = "Some content"

        chunks = chunker.chunk(content)

        assert all("metadata" in chunk for chunk in chunks)
        assert all(isinstance(chunk["metadata"], dict) for chunk in chunks)

    def test_chunk_tracks_calls(self) -> None:
        """chunk should track call history."""
        chunker = MockChunker()

        chunker.chunk("First content")
        chunker.chunk("Second content")

        assert chunker.chunk_calls == ["First content", "Second content"]

    def test_chunk_overlap_works_correctly(self) -> None:
        """chunks should overlap by configured amount."""
        chunker = MockChunker(chunk_size=50, overlap=10)
        content = "A" * 100

        chunks = chunker.chunk(content)

        # Second chunk should start at (50 - 10) = 40
        if len(chunks) > 1:
            assert chunks[1]["start_offset"] == 40
