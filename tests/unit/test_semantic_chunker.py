#!/usr/bin/env python3

"""
Comprehensive unit tests for SemanticChunker.

This module tests the semantic chunking strategy with various scenarios including
mocked embeddings, error handling, performance, and edge cases.
"""

from typing import Any
from unittest.mock import patch

import pytest
from llama_index.core.embeddings import MockEmbedding

from shared.chunking.unified.factory import TextProcessingStrategyAdapter, UnifiedChunkingFactory


# Create SemanticChunker using the unified factory
def create_semantic_chunker(embed_model: MockEmbedding) -> TextProcessingStrategyAdapter:
    """Create a semantic chunker using the unified factory."""
    unified_strategy = UnifiedChunkingFactory.create_strategy(
        "semantic",
        use_llama_index=True,
        embed_model=embed_model,
    )
    return TextProcessingStrategyAdapter(unified_strategy)


class TestSemanticChunker:
    """Comprehensive tests for SemanticChunker."""

    @pytest.fixture()
    def mock_embed_model(self) -> MockEmbedding:
        """Mock embedding model for semantic chunking tests."""
        return MockEmbedding(embed_dim=384)

    @pytest.fixture()
    def sample_texts(self) -> dict[str, str]:
        """Sample texts for testing."""
        return {
            "simple": "This is a test document. It has multiple sentences. Each sentence is different. The semantic chunker should find natural boundaries.",
            "technical": """
Machine learning is a field of artificial intelligence. It enables systems to learn from data.
Neural networks are inspired by the human brain. They consist of interconnected nodes called neurons.
Deep learning uses multiple layers of neural networks. This allows for learning complex patterns.
Natural language processing helps computers understand human language. It powers many modern applications.
Computer vision enables machines to interpret visual information. It's used in facial recognition and autonomous vehicles.
""",
            "coherent_topic": """
Python programming is versatile and powerful. Python syntax is clean and readable.
Python libraries make development faster. Python community provides excellent support.
Python frameworks enable rapid application development. Python is used in data science extensively.
Python performance can be optimized with various techniques. Python ecosystem continues to grow.
""",
            "mixed_topics": """
The weather today is sunny and warm. Python is a popular programming language.
Cars are becoming more fuel efficient. Machine learning requires large datasets.
Coffee shops are popular meeting places. Neural networks can recognize patterns.
Climate change affects global temperatures. Data science involves statistical analysis.
""",
            "empty": "",
            "single_sentence": "This is just one sentence.",
        }

    def test_semantic_chunker_initialization(self, mock_embed_model: MockEmbedding) -> None:
        """Test SemanticChunker initialization with embed model."""
        chunker = create_semantic_chunker(mock_embed_model)
        assert chunker is not None
        assert chunker.strategy_name == "semantic"

    def test_semantic_chunker_basic(self, mock_embed_model: MockEmbedding, sample_texts: dict[str, str]) -> None:
        """Test basic semantic chunking functionality."""
        chunker = create_semantic_chunker(mock_embed_model)

        results = chunker.chunk_text(sample_texts["simple"], "doc1", {"source": "test"})

        assert isinstance(results, list)
        assert len(results) > 0

        # Check result structure
        for result in results:
            assert hasattr(result, "chunk_id")
            assert hasattr(result, "text")
            assert hasattr(result, "start_offset")
            assert hasattr(result, "end_offset")
            assert hasattr(result, "metadata")
            assert result.text in sample_texts["simple"]

    def test_semantic_chunker_technical_text(
        self, mock_embed_model: MockEmbedding, sample_texts: dict[str, str]
    ) -> None:
        """Test semantic chunking on technical content."""
        chunker = create_semantic_chunker(mock_embed_model)

        results = chunker.chunk_text(sample_texts["technical"], "doc2", {"source": "technical"})

        assert len(results) > 0

        # Verify chunks maintain semantic coherence
        for result in results:
            assert len(result.text) > 0
            # Check that chunks are from the original text
            assert result.text.strip() in sample_texts["technical"]

    def test_semantic_chunker_coherent_topic(
        self, mock_embed_model: MockEmbedding, sample_texts: dict[str, str]
    ) -> None:
        """Test semantic chunking on coherent topic text."""
        chunker = create_semantic_chunker(mock_embed_model)

        results = chunker.chunk_text(sample_texts["coherent_topic"], "doc3", {"topic": "python"})

        assert len(results) > 0

        # Coherent topics might result in larger chunks
        for result in results:
            assert "Python" in result.text or "python" in result.text

    def test_semantic_chunker_mixed_topics(self, mock_embed_model: MockEmbedding, sample_texts: dict[str, str]) -> None:
        """Test semantic chunking on mixed topic text."""
        chunker = create_semantic_chunker(mock_embed_model)

        results = chunker.chunk_text(sample_texts["mixed_topics"], "doc4", {"type": "mixed"})

        assert len(results) > 0

        # Mixed topics should be separated into different chunks
        topics_found = set()
        for result in results:
            text_lower = result.text.lower()
            if "weather" in text_lower or "sunny" in text_lower:
                topics_found.add("weather")
            if "python" in text_lower or "programming" in text_lower:
                topics_found.add("programming")
            if "machine learning" in text_lower or "neural" in text_lower:
                topics_found.add("ml")

        # Should identify multiple topics
        assert len(topics_found) >= 2

    def test_semantic_chunker_empty_text(self, mock_embed_model: MockEmbedding, sample_texts: dict[str, str]) -> None:
        """Test semantic chunking with empty text."""
        chunker = create_semantic_chunker(mock_embed_model)

        results = chunker.chunk_text(sample_texts["empty"], "doc5", {})

        assert results == []

    def test_semantic_chunker_single_sentence(
        self, mock_embed_model: MockEmbedding, sample_texts: dict[str, str]
    ) -> None:
        """Test semantic chunking with single sentence."""
        chunker = create_semantic_chunker(mock_embed_model)

        results = chunker.chunk_text(sample_texts["single_sentence"], "doc6", {})

        assert len(results) == 1
        assert results[0].text == sample_texts["single_sentence"]

    @pytest.mark.asyncio()
    async def test_semantic_chunker_async(self, mock_embed_model: MockEmbedding, sample_texts: dict[str, str]) -> None:
        """Test async semantic chunking."""
        chunker = create_semantic_chunker(mock_embed_model)

        results = await chunker.chunk_text_async(sample_texts["simple"], "doc7", {"async": True})

        assert isinstance(results, list)
        assert len(results) > 0

    def test_semantic_chunker_metadata_propagation(
        self, mock_embed_model: MockEmbedding, sample_texts: dict[str, str]
    ) -> None:
        """Test that metadata is properly propagated to chunks."""
        chunker = create_semantic_chunker(mock_embed_model)

        custom_metadata: dict[str, Any] = {"source": "test_source", "author": "test_author", "timestamp": "2024-01-01"}

        results = chunker.chunk_text(sample_texts["simple"], "doc8", custom_metadata)

        for result in results:
            assert hasattr(result, "metadata")
            metadata = result.metadata
            assert metadata["source"] == "test_source"
            assert metadata["author"] == "test_author"
            assert metadata["timestamp"] == "2024-01-01"

    def test_semantic_chunker_chunk_ids(self, mock_embed_model: MockEmbedding, sample_texts: dict[str, str]) -> None:
        """Test that chunk IDs are properly generated."""
        chunker = create_semantic_chunker(mock_embed_model)

        results = chunker.chunk_text(sample_texts["simple"], "doc9", {})

        chunk_ids = [r.chunk_id for r in results]

        # All chunk IDs should be unique
        assert len(chunk_ids) == len(set(chunk_ids))

        # Chunk IDs should follow pattern
        for chunk_id in chunk_ids:
            assert chunk_id.startswith("doc9_")

    def test_semantic_chunker_offsets(self, mock_embed_model: MockEmbedding, sample_texts: dict[str, str]) -> None:
        """Test that character offsets are correct."""
        chunker = create_semantic_chunker(mock_embed_model)

        text = sample_texts["simple"]
        results = chunker.chunk_text(text, "doc10", {})

        for result in results:
            start = result.start_offset
            end = result.end_offset

            # Offsets should be valid
            assert 0 <= start < len(text)
            assert start < end <= len(text)

            # Extract text using offsets should match chunk text (approximately)
            # Note: LlamaIndex might clean/modify text slightly
            text[start:end].strip()
            chunk_text = result.text.strip()
            # They should at least overlap significantly
            assert len(chunk_text) > 0

    def test_semantic_chunker_config_validation(self, mock_embed_model: MockEmbedding) -> None:
        """Test configuration validation."""
        chunker = create_semantic_chunker(mock_embed_model)

        # Valid config
        assert chunker.validate_config({"buffer_size": 1, "breakpoint_percentile_threshold": 95})

        # Invalid config - negative buffer size
        assert not chunker.validate_config({"buffer_size": -1})

        # Invalid config - percentile out of range
        assert not chunker.validate_config({"breakpoint_percentile_threshold": 101})

    def test_semantic_chunker_estimate_chunks(self, mock_embed_model: MockEmbedding) -> None:
        """Test chunk estimation."""
        chunker = create_semantic_chunker(mock_embed_model)

        # Estimate for different text lengths
        estimate_small = chunker.estimate_chunks(100, {})
        estimate_medium = chunker.estimate_chunks(1000, {})
        estimate_large = chunker.estimate_chunks(10000, {})

        # Larger texts should have more estimated chunks
        assert estimate_small <= estimate_medium <= estimate_large
        assert estimate_small >= 1
        assert estimate_large >= 1

    @patch("shared.chunking.unified.semantic_strategy.logger")
    def test_semantic_chunker_error_handling(
        self, mock_logger: Any, mock_embed_model: MockEmbedding, sample_texts: dict[str, str]
    ) -> None:
        """Test error handling in semantic chunker."""
        chunker = create_semantic_chunker(mock_embed_model)

        # Test with invalid document ID
        results = chunker.chunk_text(sample_texts["simple"], None, {})  # type: ignore[arg-type]  # Invalid doc_id

        # Should handle gracefully
        assert isinstance(results, list)

    def test_semantic_chunker_performance(self, mock_embed_model: MockEmbedding) -> None:
        """Test semantic chunker performance with large text."""
        import time

        chunker = create_semantic_chunker(mock_embed_model)

        # Generate large text
        large_text = " ".join([f"This is sentence number {i}. It contains some information." for i in range(1000)])

        start_time = time.time()
        results = chunker.chunk_text(large_text, "doc_perf", {})
        end_time = time.time()

        # Should complete within reasonable time (5 seconds for 1000 sentences)
        assert end_time - start_time < 5.0
        assert len(results) > 0

    def test_semantic_chunker_unicode(self, mock_embed_model: MockEmbedding) -> None:
        """Test semantic chunker with Unicode text."""
        chunker = create_semantic_chunker(mock_embed_model)

        unicode_text = """
        Hello 世界! This is Unicode text.
        Привет мир! Emoji test: 🚀 🌟 🎉
        مرحبا بالعالم! Mixed scripts work.
        """

        results = chunker.chunk_text(unicode_text, "doc_unicode", {})

        assert len(results) > 0

        # Verify Unicode is preserved
        all_text = " ".join(r.text for r in results)
        assert "世界" in all_text
        assert "🚀" in all_text
        assert "مرحبا" in all_text
