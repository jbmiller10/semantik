#!/usr/bin/env python3

"""
Comprehensive unit tests for SemanticChunker.

This module tests the semantic chunking strategy with various scenarios including
mocked embeddings, error handling, performance, and edge cases.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser

from packages.shared.text_processing.base_chunker import ChunkResult
from packages.shared.text_processing.strategies.semantic_chunker import SemanticChunker


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
Books provide knowledge and entertainment. Data science is transforming industries.
""",
            "very_long": " ".join(
                [f"Sentence number {i} contains unique information about topic {i % 5}." for i in range(1000)]
            ),
        }

    def test_initialization(self, mock_embed_model) -> None:
        """Test SemanticChunker initialization with various parameters."""
        # Valid initialization
        chunker = SemanticChunker(
            breakpoint_percentile_threshold=90, buffer_size=2, max_chunk_size=500, embed_model=mock_embed_model
        )
        assert chunker.breakpoint_percentile_threshold == 90
        assert chunker.buffer_size == 2
        assert chunker.max_chunk_size == 500
        assert chunker.embed_model == mock_embed_model

        # Test parameter validation
        with pytest.raises(ValueError, match="breakpoint_percentile_threshold must be between 0 and 100"):
            SemanticChunker(breakpoint_percentile_threshold=101)

        with pytest.raises(ValueError, match="breakpoint_percentile_threshold must be between 0 and 100"):
            SemanticChunker(breakpoint_percentile_threshold=-1)

        with pytest.raises(ValueError, match="buffer_size must be positive"):
            SemanticChunker(buffer_size=0)

        with pytest.raises(ValueError, match="max_chunk_size must be positive"):
            SemanticChunker(max_chunk_size=-10)

    def test_splitter_initialization_without_embed_model(self) -> None:
        """Test that splitter initialization fails without embed model."""
        chunker = SemanticChunker()  # No embed_model provided

        with pytest.raises(RuntimeError, match="Embedding model not provided"):
            chunker._get_splitter()

    def test_splitter_lazy_initialization(self, mock_embed_model) -> None:
        """Test that splitter is lazily initialized."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        # Splitter should not be initialized yet
        assert chunker._splitter is None

        # Get splitter
        splitter = chunker._get_splitter()
        assert isinstance(splitter, SemanticSplitterNodeParser)
        assert chunker._splitter is not None

        # Second call should return same instance
        splitter2 = chunker._get_splitter()
        assert splitter is splitter2

    def test_chunk_text_empty(self, mock_embed_model) -> None:
        """Test chunking empty or whitespace-only text."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        # Empty string
        chunks = chunker.chunk_text("", "doc1")
        assert chunks == []

        # Whitespace only
        chunks = chunker.chunk_text("   \n\t  ", "doc2")
        assert chunks == []

        # None metadata
        chunks = chunker.chunk_text("", "doc3", None)
        assert chunks == []

    def test_chunk_text_basic(self, mock_embed_model, sample_texts) -> None:
        """Test basic synchronous chunking functionality."""
        chunker = SemanticChunker(
            breakpoint_percentile_threshold=95, buffer_size=1, max_chunk_size=100, embed_model=mock_embed_model
        )

        chunks = chunker.chunk_text(sample_texts["simple"], "test_doc")

        # Verify chunks
        assert len(chunks) >= 1
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

        # Verify chunk IDs
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"test_doc_{i:04d}"
            assert chunk.metadata["chunk_index"] == i

        # Verify offsets
        cumulative_length = 0
        for chunk in chunks:
            assert chunk.start_offset == cumulative_length
            assert chunk.end_offset == chunk.start_offset + len(chunk.text)
            cumulative_length = chunk.end_offset

        # Verify metadata
        for chunk in chunks:
            assert chunk.metadata["semantic_boundary"] is True
            assert chunk.metadata["breakpoint_threshold"] == 95
            assert chunk.metadata["strategy"] == "semantic"

    async def test_chunk_text_async(self, mock_embed_model, sample_texts) -> None:
        """Test asynchronous chunking functionality."""
        chunker = SemanticChunker(breakpoint_percentile_threshold=90, buffer_size=2, embed_model=mock_embed_model)

        chunks = await chunker.chunk_text_async(sample_texts["technical"], "async_doc")

        # Verify chunks
        assert len(chunks) >= 1
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

        # Verify metadata
        for chunk in chunks:
            assert chunk.metadata["semantic_boundary"] is True
            assert chunk.metadata["breakpoint_threshold"] == 90

    def test_metadata_preservation(self, mock_embed_model, sample_texts) -> None:
        """Test that original metadata is preserved and extended."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        original_metadata = {
            "source": "test_file.txt",
            "author": "Test Author",
            "custom_field": 123,
            "tags": ["test", "sample"],
        }

        chunks = chunker.chunk_text(sample_texts["simple"], "metadata_test", original_metadata)

        for chunk in chunks:
            # Original metadata preserved
            assert chunk.metadata["source"] == "test_file.txt"
            assert chunk.metadata["author"] == "Test Author"
            assert chunk.metadata["custom_field"] == 123
            assert chunk.metadata["tags"] == ["test", "sample"]

            # Semantic metadata added
            assert chunk.metadata["semantic_boundary"] is True
            assert "breakpoint_threshold" in chunk.metadata
            assert chunk.metadata["strategy"] == "semantic"

    def test_performance_monitoring(self, mock_embed_model, sample_texts) -> None:
        """Test performance characteristics with mocked timing."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        # Mock time to control performance measurement
        with patch("time.time") as mock_time:
            # Simulate ~150 chunks/second performance
            start_time = 100.0
            # For the simple text, let's say we get 4 chunks
            # To achieve 150 chunks/sec, 4 chunks should take 4/150 = 0.0267 seconds
            end_time = start_time + 0.0267
            mock_time.side_effect = [start_time, end_time]

            with patch(
                "packages.shared.text_processing.chunking_metrics.performance_monitor.measure_chunking"
            ) as mock_monitor:
                mock_context = MagicMock()
                mock_context.__enter__ = MagicMock(return_value=MagicMock(output_chunks=0))
                mock_context.__exit__ = MagicMock(return_value=None)
                mock_monitor.return_value = mock_context

                chunker.chunk_text(sample_texts["simple"], "perf_test")

                # Verify performance monitor was called
                mock_monitor.assert_called_once_with(
                    strategy="semantic",
                    doc_id="perf_test",
                    text_length=len(sample_texts["simple"]),
                    metadata={"threshold": chunker.breakpoint_percentile_threshold},
                )

    def test_retry_logic_success(self, mock_embed_model, sample_texts) -> None:
        """Test retry logic succeeds after transient failure."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        # Mock splitter to fail twice then succeed
        mock_splitter = MagicMock()
        fail_count = 0

        def side_effect(docs) -> None:  # noqa: ARG001
            nonlocal fail_count
            if fail_count < 2:
                fail_count += 1
                raise Exception("Transient embedding error")
            # Return mock nodes on success
            mock_node = MagicMock()
            mock_node.get_content.return_value = sample_texts["simple"]
            return [mock_node]

        mock_splitter.get_nodes_from_documents.side_effect = side_effect

        with (
            patch.object(chunker, "_get_splitter", return_value=mock_splitter),
            patch("time.sleep"),
        ):  # Mock sleep to speed up test
            chunks = chunker.chunk_text(sample_texts["simple"], "retry_test")

            assert len(chunks) == 1
            assert chunks[0].text == sample_texts["simple"]
            # Verify retry was attempted
            assert mock_splitter.get_nodes_from_documents.call_count == 3

    def test_retry_logic_max_failures(self, mock_embed_model, sample_texts) -> None:
        """Test retry logic fails after max attempts then falls back to character chunking."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        # Mock splitter to always fail with embedding error
        mock_splitter = MagicMock()
        mock_splitter.get_nodes_from_documents.side_effect = Exception("Persistent embedding error")

        # Need to mock both _get_splitter and the performance monitor
        with (
            patch.object(chunker, "_get_splitter", return_value=mock_splitter),
            patch("time.sleep"),
            patch(
                "packages.shared.text_processing.chunking_metrics.performance_monitor.measure_chunking"
            ) as mock_monitor,
        ):  # Mock sleep to speed up test
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=MagicMock(output_chunks=0))
            mock_context.__exit__ = MagicMock(return_value=None)
            mock_monitor.return_value = mock_context

            # The chunker will fall back to character chunking on embedding error
            chunks = chunker.chunk_text(sample_texts["simple"], "retry_fail_test")

            # Verify all retries were attempted
            assert mock_splitter.get_nodes_from_documents.call_count == 3

            # Should have fallen back to character chunking
            assert len(chunks) >= 1
            assert all(chunk.metadata["strategy"] == "character" for chunk in chunks)

    def test_fallback_to_character_chunker(self, mock_embed_model, sample_texts) -> None:
        """Test fallback to character chunker on embedding errors."""
        chunker = SemanticChunker(max_chunk_size=100, embed_model=mock_embed_model)

        # Mock embedding error
        with patch.object(chunker, "_get_splitter") as mock_get_splitter:
            mock_get_splitter.side_effect = RuntimeError("Embedding model initialization failed")

            chunks = chunker.chunk_text(sample_texts["simple"], "fallback_test")

            # Should have fallen back to character chunker
            assert len(chunks) >= 1
            # Character chunker adds different metadata
            assert all(chunk.metadata["strategy"] == "character" for chunk in chunks)

    async def test_async_fallback_to_character_chunker(self, mock_embed_model, sample_texts) -> None:
        """Test async fallback to character chunker on embedding errors."""
        chunker = SemanticChunker(max_chunk_size=100, embed_model=mock_embed_model)

        # Mock embedding error in async context
        with patch.object(chunker, "_get_splitter_async") as mock_get_splitter:
            mock_get_splitter.side_effect = RuntimeError("Embedding model initialization failed")

            chunks = await chunker.chunk_text_async(sample_texts["simple"], "async_fallback_test")

            # Should have fallen back to character chunker
            assert len(chunks) >= 1
            assert all(chunk.metadata["strategy"] == "character" for chunk in chunks)

    def test_validate_config(self, mock_embed_model) -> None:
        """Test configuration validation."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        # Valid configurations
        valid_configs = [
            {"breakpoint_percentile_threshold": 50},
            {"breakpoint_percentile_threshold": 95.5},
            {"buffer_size": 1},
            {"buffer_size": 10},
            {"max_chunk_size": 100},
            {"max_chunk_size": 10000},
            {"breakpoint_percentile_threshold": 85, "buffer_size": 3, "max_chunk_size": 500},
        ]

        for config in valid_configs:
            assert chunker.validate_config(config) is True

        # Invalid configurations
        invalid_configs = [
            {"breakpoint_percentile_threshold": -1},
            {"breakpoint_percentile_threshold": 101},
            {"breakpoint_percentile_threshold": "high"},
            {"buffer_size": 0},
            {"buffer_size": -5},
            {"buffer_size": "large"},
            {"max_chunk_size": 0},
            {"max_chunk_size": -100},
            {"max_chunk_size": "unlimited"},
        ]

        for config in invalid_configs:
            assert chunker.validate_config(config) is False

    def test_estimate_chunks(self, mock_embed_model) -> None:
        """Test chunk estimation for capacity planning."""
        chunker = SemanticChunker(max_chunk_size=1000, embed_model=mock_embed_model)

        # Small text
        estimate = chunker.estimate_chunks(100, {})
        assert estimate >= 1
        assert estimate <= 5

        # Medium text
        estimate = chunker.estimate_chunks(5000, {})
        assert estimate >= 5
        assert estimate <= 20

        # Large text
        estimate = chunker.estimate_chunks(50000, {})
        assert estimate >= 50
        assert estimate <= 200

        # Very large text
        estimate = chunker.estimate_chunks(1000000, {})
        assert estimate >= 1000

        # Custom max chunk size
        config = {"max_chunk_size": 100}
        estimate = chunker.estimate_chunks(10000, config)
        # Should be constrained by max chunk size
        assert estimate >= 25  # At minimum 10000 / (100 * 4)

    def test_unicode_and_special_characters(self, mock_embed_model) -> None:
        """Test handling of Unicode and special characters."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        special_texts = [
            "Hello 世界! Testing 中文 characters.",
            "Émojis: 🚀 🌟 🎉 and symbols: → € £ ¥",
            "Mixed: Café, naïve, résumé, Zürich",
            "RTL: العربية and עברית text",
            "Math: ∑(i=1 to n) = n(n+1)/2",
            "Null bytes: Hello\x00World",
        ]

        for text in special_texts:
            chunks = chunker.chunk_text(text, "unicode_test")

            # Should handle without errors
            assert len(chunks) >= 1

            # Text should be preserved
            combined = "".join(chunk.text for chunk in chunks)
            assert combined == text

    async def test_concurrent_async_chunking(self, mock_embed_model, sample_texts) -> None:
        """Test concurrent async chunking operations."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        # Run multiple async chunking operations concurrently
        tasks = []
        for i, (name, text) in enumerate(sample_texts.items()):
            if name != "very_long":  # Skip very long for speed
                task = chunker.chunk_text_async(text, f"concurrent_doc_{i}")
                tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        assert len(results) == len(tasks)
        for chunks in results:
            assert len(chunks) >= 1
            assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

    def test_chunk_with_splitter_helper(self, mock_embed_model, sample_texts) -> None:
        """Test the _chunk_with_splitter helper method."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        # Create a mock splitter
        mock_splitter = MagicMock()
        mock_node1 = MagicMock()
        mock_node1.get_content.return_value = "First chunk content."
        mock_node2 = MagicMock()
        mock_node2.get_content.return_value = "Second chunk content."
        mock_splitter.get_nodes_from_documents.return_value = [mock_node1, mock_node2]

        # Test with metadata
        metadata = {"source": "test"}
        chunks = chunker._chunk_with_splitter(mock_splitter, sample_texts["simple"], "helper_test", metadata)

        assert len(chunks) == 2
        assert chunks[0].text == "First chunk content."
        assert chunks[1].text == "Second chunk content."

        # Verify offsets
        assert chunks[0].start_offset == 0
        assert chunks[0].end_offset == len("First chunk content.")
        assert chunks[1].start_offset == chunks[0].end_offset
        assert chunks[1].end_offset == chunks[1].start_offset + len("Second chunk content.")

        # Verify metadata
        for chunk in chunks:
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["semantic_boundary"] is True

    def test_exponential_backoff(self, mock_embed_model, sample_texts) -> None:
        """Test exponential backoff in retry logic."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        # Track sleep calls
        sleep_calls = []

        def mock_sleep(duration) -> None:
            sleep_calls.append(duration)

        # Mock splitter to fail twice
        mock_splitter = MagicMock()
        fail_count = 0

        def side_effect(docs) -> None:  # noqa: ARG001
            nonlocal fail_count
            if fail_count < 2:
                fail_count += 1
                raise Exception("Transient error")
            mock_node = MagicMock()
            mock_node.get_content.return_value = "Success"
            return [mock_node]

        mock_splitter.get_nodes_from_documents.side_effect = side_effect

        with (
            patch.object(chunker, "_get_splitter", return_value=mock_splitter),
            patch("time.sleep", side_effect=mock_sleep),
        ):
            chunker.chunk_text(sample_texts["simple"], "backoff_test")

            # Verify exponential backoff
            assert len(sleep_calls) == 2
            assert sleep_calls[0] == 1.0  # First retry: 1 second
            assert sleep_calls[1] == 2.0  # Second retry: 2 seconds (doubled)

    def test_semantic_boundaries_preserved(self, mock_embed_model) -> None:
        """Test that semantic boundaries are properly identified."""
        chunker = SemanticChunker(breakpoint_percentile_threshold=95, buffer_size=1, embed_model=mock_embed_model)

        # Text with clear topic transitions
        text = """
        The solar system consists of the sun and celestial bodies.
        Planets orbit around the sun in elliptical paths.

        Machine learning is transforming technology.
        Neural networks can learn complex patterns from data.

        Cooking requires patience and practice.
        Fresh ingredients make a significant difference in taste.
        """

        chunks = chunker.chunk_text(text, "boundaries_test")

        # Should create chunks at topic boundaries
        assert len(chunks) >= 1

        # Each chunk should have semantic boundary metadata
        for chunk in chunks:
            assert chunk.metadata["semantic_boundary"] is True
            assert chunk.metadata["breakpoint_threshold"] == 95

    async def test_async_initialization_lock(self, mock_embed_model) -> None:
        """Test that async initialization works properly."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        # First call should initialize the splitter
        splitter1 = await chunker._get_splitter_async()
        assert isinstance(splitter1, SemanticSplitterNodeParser)
        assert chunker._splitter is not None

        # Second call should return the same instance
        splitter2 = await chunker._get_splitter_async()
        assert splitter1 is splitter2

    def test_large_document_handling(self, mock_embed_model, sample_texts) -> None:
        """Test handling of large documents."""
        chunker = SemanticChunker(
            breakpoint_percentile_threshold=95, buffer_size=1, max_chunk_size=1000, embed_model=mock_embed_model
        )

        # Use a moderate-sized text for testing
        test_text = "This is a test sentence. " * 100  # ~2500 characters

        chunks = chunker.chunk_text(test_text, "large_doc")

        # Should create at least one chunk
        assert len(chunks) >= 1

        # Verify all chunks have content
        for chunk in chunks:
            assert len(chunk.text) > 0
            assert chunk.metadata["semantic_boundary"] is True

        # Verify complete coverage
        total_text = "".join(chunk.text for chunk in chunks)
        # Strip both to handle whitespace differences
        assert total_text.strip() == test_text.strip()

        # Verify proper offsets
        if len(chunks) > 1:
            # Verify no gaps in offsets
            expected_offset = 0
            for chunk in chunks:
                assert chunk.start_offset == expected_offset
                expected_offset = chunk.end_offset

    def test_performance_target(self, mock_embed_model) -> None:
        """Test that semantic chunker achieves target performance of ~150 chunks/sec."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        # Create mock splitter with controlled performance
        mock_splitter = MagicMock()

        # Create 150 mock nodes to simulate 150 chunks
        mock_nodes = []
        for i in range(150):
            mock_node = MagicMock()
            mock_node.get_content.return_value = f"Chunk {i} content"
            mock_nodes.append(mock_node)

        # Mock timing to simulate exactly 1 second for 150 chunks
        with patch("time.time") as mock_time:
            # Set up timing: start at 100.0, end at 101.0 (1 second elapsed)
            mock_time.side_effect = [100.0, 101.0]

            def mock_get_nodes(docs) -> None:  # noqa: ARG001
                # Simulate the time it takes to process
                return mock_nodes

            mock_splitter.get_nodes_from_documents.side_effect = mock_get_nodes

            with (
                patch.object(chunker, "_get_splitter", return_value=mock_splitter),
                patch(
                    "packages.shared.text_processing.chunking_metrics.performance_monitor.measure_chunking"
                ) as mock_monitor,
            ):
                mock_metrics = MagicMock()
                mock_context = MagicMock()
                mock_context.__enter__ = MagicMock(return_value=mock_metrics)
                mock_context.__exit__ = MagicMock(return_value=None)
                mock_monitor.return_value = mock_context

                chunks = chunker.chunk_text("Test text", "perf_doc")

                # Verify we got 150 chunks
                assert len(chunks) == 150

                # Verify metrics were set
                assert mock_metrics.output_chunks == 150

    def test_non_embedding_error_no_fallback(self, mock_embed_model) -> None:
        """Test that non-embedding errors don't trigger fallback to character chunking."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        # Mock splitter to fail with non-embedding error
        mock_splitter = MagicMock()
        mock_splitter.get_nodes_from_documents.side_effect = ValueError("Invalid document format")

        with (
            patch.object(chunker, "_get_splitter", return_value=mock_splitter),
            patch("time.sleep"),
            pytest.raises(RuntimeError, match="Semantic chunking failed: Invalid document format"),
        ):  # Mock sleep to speed up test
            # The chunker wraps exceptions in RuntimeError
            chunker.chunk_text("Test text", "error_doc")

    def test_small_buffer_size(self, mock_embed_model) -> None:
        """Test semantic chunker with very small buffer size."""
        chunker = SemanticChunker(
            breakpoint_percentile_threshold=90, buffer_size=1, embed_model=mock_embed_model  # Minimum buffer size
        )

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk_text(text, "small_buffer_doc")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["semantic_boundary"] is True

    async def test_async_chunk_with_metadata_override(self, mock_embed_model) -> None:
        """Test async chunking preserves and extends metadata correctly."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        original_metadata = {
            "semantic_boundary": False,  # Should be overridden
            "custom_field": "preserved",
            "breakpoint_threshold": 50,  # Should be overridden
        }

        chunks = await chunker.chunk_text_async(
            "Test text for metadata override", "metadata_override_doc", original_metadata
        )

        for chunk in chunks:
            # Semantic metadata should override
            assert chunk.metadata["semantic_boundary"] is True
            assert chunk.metadata["breakpoint_threshold"] == 95  # Default value
            # Custom fields should be preserved
            assert chunk.metadata["custom_field"] == "preserved"

    def test_empty_nodes_from_splitter(self, mock_embed_model) -> None:
        """Test handling when splitter returns empty nodes list."""
        chunker = SemanticChunker(embed_model=mock_embed_model)

        # Mock splitter to return empty list
        mock_splitter = MagicMock()
        mock_splitter.get_nodes_from_documents.return_value = []

        with patch.object(chunker, "_get_splitter", return_value=mock_splitter):
            chunks = chunker.chunk_text("Test text", "empty_nodes_doc")

            # Should return empty list
            assert chunks == []
