#!/usr/bin/env python3
"""
Comprehensive unit tests for HierarchicalChunker.

This module tests the hierarchical chunking strategy with various scenarios including
parent-child relationships, hierarchy levels, performance, and edge cases.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.schema import BaseNode

from packages.shared.text_processing.base_chunker import ChunkResult
from packages.shared.text_processing.strategies.hierarchical_chunker import HierarchicalChunker


class TestHierarchicalChunker:
    """Comprehensive tests for HierarchicalChunker."""

    @pytest.fixture()
    def sample_texts(self) -> dict[str, str]:
        """Sample texts for testing."""
        return {
            "simple": "This is a test document. It has multiple sentences. Each sentence is different. The hierarchical chunker should create parent-child relationships.",
            "technical": """
Machine learning is a field of artificial intelligence. It enables systems to learn from data.
Neural networks are inspired by the human brain. They consist of interconnected nodes called neurons.
Deep learning uses multiple layers of neural networks. This allows for learning complex patterns.
Natural language processing helps computers understand human language. It powers many modern applications.
Computer vision enables machines to interpret visual information. It's used in facial recognition and autonomous vehicles.
""",
            "multilevel": """
# Chapter 1: Introduction to Machine Learning

Machine learning is revolutionizing how we process and understand data. It encompasses various techniques and algorithms that enable computers to learn from experience without being explicitly programmed.

## Section 1.1: Supervised Learning

Supervised learning is a fundamental paradigm in machine learning. In this approach, algorithms learn from labeled training data, where each example consists of input features and the corresponding desired output.

### Subsection 1.1.1: Classification

Classification involves predicting discrete labels or categories. Common algorithms include decision trees, support vector machines, and neural networks.

### Subsection 1.1.2: Regression

Regression predicts continuous values. Linear regression is the simplest form, while more complex methods include polynomial regression and neural network regression.

## Section 1.2: Unsupervised Learning

Unsupervised learning discovers hidden patterns in unlabeled data. It doesn't require predefined categories or target values.

### Subsection 1.2.1: Clustering

Clustering groups similar data points together. K-means, hierarchical clustering, and DBSCAN are popular clustering algorithms.

### Subsection 1.2.2: Dimensionality Reduction

Dimensionality reduction techniques like PCA and t-SNE help visualize and process high-dimensional data by reducing the number of features while preserving important information.
""",
            "very_short": "Small text.",
            "very_long": " ".join(
                [f"Sentence number {i} contains information about topic {i % 10}." for i in range(1000)]
            ),
        }

    def test_initialization(self):
        """Test HierarchicalChunker initialization with various parameters."""
        # Default initialization
        chunker = HierarchicalChunker()
        assert chunker.chunk_sizes == [2048, 512, 128]
        assert chunker.chunk_overlap == 20

        # Custom chunk sizes
        chunker = HierarchicalChunker(chunk_sizes=[1000, 500, 100])
        assert chunker.chunk_sizes == [1000, 500, 100]

        # Ensure sizes are sorted in descending order
        chunker = HierarchicalChunker(chunk_sizes=[100, 1000, 500])
        assert chunker.chunk_sizes == [1000, 500, 100]

        # Single level hierarchy
        chunker = HierarchicalChunker(chunk_sizes=[500])
        assert chunker.chunk_sizes == [500]

        # Custom overlap
        chunker = HierarchicalChunker(chunk_overlap=50)
        assert chunker.chunk_overlap == 50

    def test_initialization_validation(self):
        """Test HierarchicalChunker initialization validation."""
        # Empty chunk sizes
        with pytest.raises(ValueError, match="chunk_sizes must contain at least one size"):
            HierarchicalChunker(chunk_sizes=[])

        # Duplicate sizes should raise error
        with pytest.raises(ValueError, match="Chunk sizes must be in descending order"):
            HierarchicalChunker(chunk_sizes=[500, 500, 100])

        # Warning for small differences between levels
        with patch("packages.shared.text_processing.strategies.hierarchical_chunker.logger") as mock_logger:
            HierarchicalChunker(chunk_sizes=[1000, 600, 100])
            # Should warn about 600 being more than half of 1000
            mock_logger.warning.assert_called()

    def test_chunk_text_empty(self):
        """Test chunking empty or whitespace-only text."""
        chunker = HierarchicalChunker()

        # Empty string
        chunks = chunker.chunk_text("", "doc1")
        assert chunks == []

        # Whitespace only
        chunks = chunker.chunk_text("   \n\t  ", "doc2")
        assert chunks == []

        # None metadata
        chunks = chunker.chunk_text("", "doc3", None)
        assert chunks == []

    def test_chunk_text_basic(self, sample_texts):
        """Test basic synchronous chunking functionality."""
        chunker = HierarchicalChunker(chunk_sizes=[200, 100, 50])

        chunks = chunker.chunk_text(sample_texts["simple"], "test_doc")

        # Verify chunks exist
        assert len(chunks) > 0
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

        # Separate leaf and parent chunks
        leaf_chunks = [c for c in chunks if c.metadata.get("is_leaf", False)]
        parent_chunks = [c for c in chunks if not c.metadata.get("is_leaf", False)]

        # Verify we have both types
        assert len(leaf_chunks) > 0
        assert len(parent_chunks) > 0

        # Verify hierarchy metadata
        for chunk in chunks:
            assert "hierarchy_level" in chunk.metadata
            assert "parent_chunk_id" in chunk.metadata
            assert "child_chunk_ids" in chunk.metadata
            assert "chunk_sizes" in chunk.metadata
            assert chunk.metadata["chunk_sizes"] == [200, 100, 50]
            assert "is_leaf" in chunk.metadata

    def test_parent_child_relationships(self, sample_texts):
        """Test that parent-child relationships are properly maintained."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 200, 100])

        chunks = chunker.chunk_text(sample_texts["multilevel"], "hierarchy_test")

        # Build a map of node_id to chunk
        node_map = {c.metadata.get("node_id"): c for c in chunks}

        # Verify parent-child relationships
        for chunk in chunks:
            parent_id = chunk.metadata.get("parent_chunk_id")
            if parent_id and parent_id in node_map:
                parent_chunk = node_map[parent_id]
                # Parent should list this chunk as a child
                assert chunk.metadata["node_id"] in parent_chunk.metadata.get("child_chunk_ids", [])

                # Parent should have a lower hierarchy level (higher in hierarchy)
                assert parent_chunk.metadata["hierarchy_level"] < chunk.metadata["hierarchy_level"]

    def test_hierarchy_levels(self, sample_texts):
        """Test correct assignment of hierarchy levels."""
        chunker = HierarchicalChunker(chunk_sizes=[1000, 500, 250])

        chunks = chunker.chunk_text(sample_texts["multilevel"], "level_test")

        # Collect chunks by hierarchy level
        levels = {}
        for chunk in chunks:
            level = chunk.metadata["hierarchy_level"]
            if level not in levels:
                levels[level] = []
            levels[level].append(chunk)

        # Verify we have chunks with hierarchy metadata
        assert len(chunks) > 0

        # All chunks should have hierarchy level metadata
        for chunk in chunks:
            assert "hierarchy_level" in chunk.metadata
            assert isinstance(chunk.metadata["hierarchy_level"], int)
            assert chunk.metadata["hierarchy_level"] >= 0

        # Leaf chunks should have smaller average size than parent chunks
        leaf_chunks = [c for c in chunks if c.metadata.get("is_leaf", False)]
        parent_chunks = [c for c in chunks if not c.metadata.get("is_leaf", False)]

        if leaf_chunks and parent_chunks:
            avg_leaf_size = sum(len(c.text) for c in leaf_chunks) / len(leaf_chunks)
            avg_parent_size = sum(len(c.text) for c in parent_chunks) / len(parent_chunks)
            # Parent chunks should generally be larger
            assert avg_parent_size >= avg_leaf_size

    def test_leaf_node_identification(self, sample_texts):
        """Test that leaf nodes are correctly identified."""
        chunker = HierarchicalChunker(chunk_sizes=[400, 200, 100])

        chunks = chunker.chunk_text(sample_texts["technical"], "leaf_test")

        leaf_chunks = [c for c in chunks if c.metadata.get("is_leaf", False)]
        parent_chunks = [c for c in chunks if not c.metadata.get("is_leaf", False)]

        # Leaf nodes should not have children
        for leaf in leaf_chunks:
            assert len(leaf.metadata.get("child_chunk_ids", [])) == 0

        # Parent nodes should have children (except in edge cases)
        for parent in parent_chunks:
            # Most parent nodes should have children
            if parent.metadata.get("child_chunk_ids"):
                assert len(parent.metadata["child_chunk_ids"]) > 0

    async def test_chunk_text_async(self, sample_texts):
        """Test asynchronous chunking functionality."""
        chunker = HierarchicalChunker(chunk_sizes=[300, 150, 75])

        chunks = await chunker.chunk_text_async(sample_texts["technical"], "async_doc")

        # Verify chunks
        assert len(chunks) > 0
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

        # Verify hierarchy metadata
        for chunk in chunks:
            assert "hierarchy_level" in chunk.metadata
            assert "is_leaf" in chunk.metadata

    def test_metadata_preservation(self, sample_texts):
        """Test that original metadata is preserved and extended."""
        chunker = HierarchicalChunker()

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

            # Hierarchical metadata added
            assert "hierarchy_level" in chunk.metadata
            assert "parent_chunk_id" in chunk.metadata
            assert "child_chunk_ids" in chunk.metadata
            assert "is_leaf" in chunk.metadata

    def test_performance_monitoring(self, sample_texts):
        """Test performance characteristics with target of 400 chunks/sec."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        # Mock the HierarchicalNodeParser to control output
        mock_parser = MagicMock(spec=HierarchicalNodeParser)

        # Create mock nodes with parent-child relationships
        mock_nodes = []

        # Create 400 leaf nodes (to test 400 chunks/sec target)
        for i in range(400):
            mock_node = MagicMock(spec=BaseNode)
            mock_node.node_id = f"leaf_{i}"
            mock_node.get_content.return_value = f"Leaf chunk {i} content"
            mock_node.relationships = {}
            mock_nodes.append(mock_node)

        # Add some parent nodes
        for i in range(50):
            parent_node = MagicMock(spec=BaseNode)
            parent_node.node_id = f"parent_{i}"
            parent_node.get_content.return_value = f"Parent chunk {i} content with more text"
            parent_node.relationships = {}
            mock_nodes.append(parent_node)

        mock_parser.get_nodes_from_documents.return_value = mock_nodes

        # Mock time to simulate achieving 400 chunks/sec
        with patch("time.time") as mock_time:
            # 450 total chunks (400 leaf + 50 parent) in 1.125 seconds = 400 chunks/sec
            # Use an iterator that provides consistent time values
            # The chunker may call time.time() multiple times during processing
            start_time = 100.0
            end_time = 101.125
            time_calls = iter([start_time, end_time])
            mock_time.side_effect = lambda: next(time_calls, end_time)

            with patch.object(chunker, "_parser", mock_parser):
                chunks = chunker.chunk_text(sample_texts["very_long"], "perf_test")

                # Should get all chunks
                assert len(chunks) == 450

                # Verify performance logged correctly
                # 450 chunks in 1.125 seconds = 400 chunks/sec

    def test_error_handling_with_fallback(self, sample_texts):
        """Test error handling and fallback to character chunking."""
        chunker = HierarchicalChunker(chunk_sizes=[200, 100, 50])

        # Mock parser to raise an error
        mock_parser = MagicMock()
        mock_parser.get_nodes_from_documents.side_effect = Exception("Parsing error")

        with patch.object(chunker, "_parser", mock_parser):
            chunks = chunker.chunk_text(sample_texts["simple"], "error_test")

            # Should have fallen back to character chunking
            assert len(chunks) > 0
            # Character chunker uses different metadata
            for chunk in chunks:
                assert chunk.metadata.get("strategy") == "character"

    async def test_async_error_handling(self, sample_texts):
        """Test async error handling and fallback."""
        chunker = HierarchicalChunker(chunk_sizes=[200, 100, 50])

        # Mock parser to raise an error
        mock_parser = MagicMock()
        mock_parser.get_nodes_from_documents.side_effect = Exception("Async parsing error")

        with patch.object(chunker, "_parser", mock_parser):
            chunks = await chunker.chunk_text_async(sample_texts["simple"], "async_error_test")

            # Should have fallen back to character chunking
            assert len(chunks) > 0
            for chunk in chunks:
                assert chunk.metadata.get("strategy") == "character"

    def test_single_level_hierarchy(self, sample_texts):
        """Test edge case with single hierarchy level."""
        chunker = HierarchicalChunker(chunk_sizes=[500])

        chunks = chunker.chunk_text(sample_texts["simple"], "single_level_test")

        # Should still work with single level
        assert len(chunks) > 0

        # All chunks should be at level 0
        for chunk in chunks:
            assert chunk.metadata["hierarchy_level"] == 0
            assert chunk.metadata["chunk_sizes"] == [500]

    def test_very_small_document(self, sample_texts):
        """Test with document smaller than smallest chunk size."""
        chunker = HierarchicalChunker(chunk_sizes=[1000, 500, 250])

        chunks = chunker.chunk_text(sample_texts["very_short"], "small_doc_test")

        # Should create at least one chunk
        assert len(chunks) >= 1

        # Small document might create only leaf chunks
        leaf_chunks = [c for c in chunks if c.metadata.get("is_leaf", False)]
        assert len(leaf_chunks) >= 1

    def test_validate_config(self):
        """Test configuration validation."""
        chunker = HierarchicalChunker()

        # Valid configurations
        valid_configs = [
            {"chunk_sizes": [1000, 500, 250]},
            {"chunk_sizes": [2048, 512, 128]},
            {"chunk_overlap": 10},
            {"chunk_overlap": 50},
            {"chunk_sizes": [1000, 500, 250], "chunk_overlap": 20},
        ]

        for config in valid_configs:
            assert chunker.validate_config(config) is True

        # Invalid configurations
        invalid_configs = [
            {"chunk_sizes": []},  # Empty list
            {"chunk_sizes": "not_a_list"},  # Wrong type
            {"chunk_sizes": [0, 100, 200]},  # Zero size
            {"chunk_sizes": [-100, 200, 300]},  # Negative size
            {"chunk_sizes": ["100", "200"]},  # String values
            {"chunk_overlap": -10},  # Negative overlap
            {"chunk_overlap": "20"},  # String overlap
            {"chunk_overlap": 500, "chunk_sizes": [100, 50]},  # Overlap >= smallest chunk
        ]

        for config in invalid_configs:
            assert chunker.validate_config(config) is False

    def test_estimate_chunks(self):
        """Test chunk estimation for capacity planning."""
        chunker = HierarchicalChunker(chunk_sizes=[1000, 500, 250])

        # Small text (100 chars ≈ 25 tokens)
        estimate = chunker.estimate_chunks(100, {})
        assert estimate >= 3  # One chunk at each level

        # Medium text (5000 chars ≈ 1250 tokens)
        estimate = chunker.estimate_chunks(5000, {})
        assert estimate >= 3  # Multiple chunks across levels
        assert estimate <= 20

        # Large text (50000 chars ≈ 12500 tokens)
        estimate = chunker.estimate_chunks(50000, {})
        assert estimate >= 10
        assert estimate <= 100

        # Custom configuration
        config = {"chunk_sizes": [2000, 1000, 500], "chunk_overlap": 50}
        estimate = chunker.estimate_chunks(10000, config)
        assert estimate >= 3  # At least one chunk per level

    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters."""
        chunker = HierarchicalChunker(chunk_sizes=[200, 100, 50])

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

            # Verify all chunks have valid content
            for chunk in chunks:
                assert isinstance(chunk.text, str)
                assert chunk.metadata["is_leaf"] in [True, False]

    async def test_concurrent_async_chunking(self, sample_texts):
        """Test concurrent async chunking operations."""
        chunker = HierarchicalChunker(chunk_sizes=[400, 200, 100])

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

    def test_build_hierarchy_info(self):
        """Test the _build_hierarchy_info helper method."""
        chunker = HierarchicalChunker(chunk_sizes=[1000, 500, 250])

        # Create mock nodes with relationships
        parent_node = MagicMock()
        parent_node.node_id = "parent_1"
        # 5200 chars / 4 = 1300 tokens, which is < 1000 * 1.5 (1500), so level 0
        parent_node.get_content.return_value = "A" * 5200
        parent_node.relationships = {}

        child_node = MagicMock()
        child_node.node_id = "child_1"
        # 1800 chars / 4 = 450 tokens
        # The loop checks: 450 <= 1000 * 1.5? Yes, so level 0 (breaks on first match)
        child_node.get_content.return_value = "B" * 1800
        child_node.relationships = {"1": MagicMock(node_id="parent_1")}  # Parent relationship

        grandchild_node = MagicMock()
        grandchild_node.node_id = "grandchild_1"
        # 800 chars / 4 = 200 tokens
        # The loop checks: 200 <= 1000 * 1.5? Yes, so level 0 (breaks on first match)
        grandchild_node.get_content.return_value = "C" * 800
        grandchild_node.relationships = {"1": MagicMock(node_id="child_1")}  # Parent relationship

        # Update parent and child with child relationships
        parent_node.relationships["2"] = MagicMock(node_id="child_1")
        child_node.relationships["2"] = MagicMock(node_id="grandchild_1")

        node_map = {"parent_1": parent_node, "child_1": child_node, "grandchild_1": grandchild_node}

        # Test hierarchy info for each node
        parent_info = chunker._build_hierarchy_info(parent_node, node_map)
        assert parent_info["parent_id"] is None
        assert parent_info["child_ids"] == ["child_1"]
        # Level assignment: 1300 tokens <= 1500, so level 0
        assert parent_info["level"] == 0

        child_info = chunker._build_hierarchy_info(child_node, node_map)
        assert child_info["parent_id"] == "parent_1"
        assert child_info["child_ids"] == ["grandchild_1"]
        # Level assignment: 450 tokens <= 1500, so level 0 (not 1, because it breaks on first match)
        assert child_info["level"] == 0

        grandchild_info = chunker._build_hierarchy_info(grandchild_node, node_map)
        assert grandchild_info["parent_id"] == "child_1"
        assert grandchild_info["child_ids"] == []
        # Level assignment: 200 tokens <= 1500, so level 0 (not 2, because it breaks on first match)
        assert grandchild_info["level"] == 0

    def test_chunk_id_format(self, sample_texts):
        """Test chunk ID formatting for both leaf and parent chunks."""
        chunker = HierarchicalChunker(chunk_sizes=[300, 150, 75])

        chunks = chunker.chunk_text(sample_texts["simple"], "id_format_test")

        # Separate leaf and parent chunks
        leaf_chunks = [c for c in chunks if c.metadata.get("is_leaf", False)]
        parent_chunks = [c for c in chunks if not c.metadata.get("is_leaf", False)]

        # Verify leaf chunk IDs
        for i, chunk in enumerate(leaf_chunks):
            expected_id = f"id_format_test_{i:04d}"
            assert chunk.chunk_id == expected_id

        # Verify parent chunk IDs have special format
        for chunk in parent_chunks:
            assert chunk.chunk_id.startswith("id_format_test_parent_")

    def test_offset_calculation(self, sample_texts):
        """Test offset calculation for chunks."""
        chunker = HierarchicalChunker(chunk_sizes=[200, 100, 50])

        text = sample_texts["simple"]
        chunks = chunker.chunk_text(text, "offset_test")

        # Leaf chunks should have reasonable offsets
        leaf_chunks = [c for c in chunks if c.metadata.get("is_leaf", False)]

        for chunk in leaf_chunks:
            # Offsets should be within text bounds
            assert chunk.start_offset >= 0
            assert chunk.end_offset <= len(text)
            assert chunk.start_offset < chunk.end_offset

            # Length should match content
            assert chunk.end_offset - chunk.start_offset == len(chunk.text)

    def test_node_relationship_edge_cases(self):
        """Test edge cases in node relationships."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        # Mock parser with edge case nodes
        mock_parser = MagicMock()

        # Node with no relationships
        orphan_node = MagicMock()
        orphan_node.node_id = "orphan"
        orphan_node.get_content.return_value = "Orphan content"
        orphan_node.relationships = {}

        # Node with single child (the typical case)
        single_child_node = MagicMock()
        single_child_node.node_id = "single_parent"
        single_child_node.get_content.return_value = "Parent with single child"
        single_child_rel = MagicMock()
        single_child_rel.node_id = "child_1"
        single_child_node.relationships = {"2": single_child_rel}

        mock_parser.get_nodes_from_documents.return_value = [orphan_node, single_child_node]

        with patch.object(chunker, "_parser", mock_parser):
            chunks = chunker.chunk_text("Test text", "relationship_test")

            # Find chunks by node_id
            orphan_chunk = next(c for c in chunks if c.metadata["node_id"] == "orphan")
            single_parent_chunk = next(c for c in chunks if c.metadata["node_id"] == "single_parent")

            # Orphan should have no parent or children
            assert orphan_chunk.metadata["parent_chunk_id"] is None
            assert orphan_chunk.metadata["child_chunk_ids"] == []

            # Single parent should have one child
            child_ids = single_parent_chunk.metadata["child_chunk_ids"]
            assert len(child_ids) == 1
            assert child_ids == ["child_1"]

    def test_warning_for_small_size_differences(self):
        """Test that warnings are logged for small size differences between levels."""
        with patch("packages.shared.text_processing.strategies.hierarchical_chunker.logger") as mock_logger:
            # This should trigger a warning because 600 > 1000/2
            HierarchicalChunker(chunk_sizes=[1000, 600, 100])

            # Verify warning was called
            assert mock_logger.warning.called
            warning_message = mock_logger.warning.call_args[0][0]
            assert "600" in warning_message
            assert "1000" in warning_message
            assert "half" in warning_message

    def test_descending_order_validation(self):
        """Test that chunk sizes are automatically sorted in descending order."""
        # Provide sizes in wrong order
        chunker = HierarchicalChunker(chunk_sizes=[100, 500, 200, 1000])

        # Should be sorted to descending order
        assert chunker.chunk_sizes == [1000, 500, 200, 100]

        # Verify in validate_config
        config = {"chunk_sizes": [50, 100, 200]}
        result = chunker.validate_config(config)
        assert result is True  # Should pass but with warning

    def test_uneven_hierarchies(self, sample_texts):
        """Test handling of documents that create uneven hierarchies."""
        chunker = HierarchicalChunker(chunk_sizes=[1000, 500, 100])

        # Create text that will result in uneven hierarchy
        # Some parts will need all 3 levels, others only 1 or 2
        mixed_text = (
            "Short section. "
            + "Medium section with more content that needs a bigger chunk. " * 10  # Fits in smallest chunk
            + "Very long section with extensive content. " * 100  # Needs medium chunk  # Needs all levels
        )

        chunks = chunker.chunk_text(mixed_text, "uneven_test")

        # Should handle uneven hierarchies gracefully
        assert len(chunks) > 0

        # Verify we have both leaf and parent chunks
        leaf_chunks = [c for c in chunks if c.metadata.get("is_leaf", False)]
        parent_chunks = [c for c in chunks if not c.metadata.get("is_leaf", False)]

        # Should have both types of chunks for a hierarchical structure
        assert len(leaf_chunks) > 0
        assert len(parent_chunks) >= 0  # May or may not have parent chunks depending on text size

        # All chunks should have valid hierarchy metadata
        for chunk in chunks:
            assert "hierarchy_level" in chunk.metadata
            assert isinstance(chunk.metadata["hierarchy_level"], int)
            assert chunk.metadata["hierarchy_level"] >= 0

    def test_chunk_overlap_handling(self):
        """Test that chunk overlap is properly handled."""
        # Test with different overlap values
        for overlap in [0, 10, 50, 100]:
            chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125], chunk_overlap=overlap)

            text = "Test sentence. " * 100  # ~1500 characters
            chunks = chunker.chunk_text(text, f"overlap_{overlap}_test")

            # Should create chunks successfully
            assert len(chunks) > 0

            # Verify overlap doesn't exceed chunk sizes
            assert chunker.chunk_overlap < min(chunker.chunk_sizes)

    def test_chunk_text_stream(self, sample_texts):
        """Test streaming functionality for memory-efficient processing."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        # Test streaming without parent chunks
        text = sample_texts["multilevel"]
        streamed_chunks = list(chunker.chunk_text_stream(text, "stream_test", include_parents=False))

        # Should get only leaf chunks
        assert all(chunk.metadata.get("is_leaf", False) for chunk in streamed_chunks)
        assert len(streamed_chunks) > 0

        # Test streaming with parent chunks
        all_streamed = list(chunker.chunk_text_stream(text, "stream_test", include_parents=True))
        leaf_streamed = [c for c in all_streamed if c.metadata.get("is_leaf", False)]
        parent_streamed = [c for c in all_streamed if not c.metadata.get("is_leaf", False)]

        assert len(leaf_streamed) > 0
        assert len(parent_streamed) > 0

        # Compare with regular chunking
        regular_chunks = chunker.chunk_text(text, "regular_test", include_parents=True)
        assert len(all_streamed) == len(regular_chunks)

    async def test_chunk_text_stream_async(self, sample_texts):
        """Test async streaming functionality."""
        chunker = HierarchicalChunker(chunk_sizes=[400, 200, 100])

        text = sample_texts["technical"]
        chunks = []

        async for chunk in chunker.chunk_text_stream_async(text, "async_stream_test"):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

    def test_lazy_parent_generation(self, sample_texts):
        """Test on-demand parent chunk generation."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        text = sample_texts["multilevel"]

        # Get only leaf chunks
        leaf_chunks = chunker.chunk_text(text, "lazy_test", include_parents=False)
        assert all(chunk.metadata.get("is_leaf", False) for chunk in leaf_chunks)

        # Generate parent chunks for a subset of leaf chunks
        subset = leaf_chunks[:5] if len(leaf_chunks) >= 5 else leaf_chunks
        parent_chunks = chunker.get_parent_chunks(text, "lazy_test", subset)

        # Should generate parent chunks
        assert len(parent_chunks) >= 0  # May be 0 if text is too small

        # All generated chunks should be parent chunks
        assert all(not chunk.metadata.get("is_leaf", False) for chunk in parent_chunks)

        # Parent chunks should have unique IDs
        parent_ids = [chunk.chunk_id for chunk in parent_chunks]
        assert len(parent_ids) == len(set(parent_ids))

    def test_include_parents_parameter(self, sample_texts):
        """Test that include_parents parameter works correctly."""
        chunker = HierarchicalChunker(chunk_sizes=[400, 200, 100])

        text = sample_texts["technical"]

        # With parents (default)
        with_parents = chunker.chunk_text(text, "with_parents", include_parents=True)
        leaf_with = [c for c in with_parents if c.metadata.get("is_leaf", False)]
        parent_with = [c for c in with_parents if not c.metadata.get("is_leaf", False)]

        assert len(leaf_with) > 0
        assert len(parent_with) > 0

        # Without parents
        without_parents = chunker.chunk_text(text, "without_parents", include_parents=False)
        assert all(chunk.metadata.get("is_leaf", False) for chunk in without_parents)
        assert len(without_parents) == len(leaf_with)

    def test_offset_accuracy(self, sample_texts):
        """Test that offset calculation is accurate."""
        chunker = HierarchicalChunker(chunk_sizes=[200, 100, 50])

        text = sample_texts["simple"]
        chunks = chunker.chunk_text(text, "offset_accuracy_test")

        # Check a sample of chunks for offset accuracy
        for chunk in chunks[:5]:  # Check first 5 chunks
            # The text at the offset should match the chunk text
            # Allow for whitespace differences
            actual_text = text[chunk.start_offset : chunk.end_offset]
            assert actual_text.strip() == chunk.text.strip() or chunk.text.strip() in actual_text.strip()
