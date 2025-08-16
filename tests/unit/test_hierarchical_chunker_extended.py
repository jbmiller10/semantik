#!/usr/bin/env python3

"""
Extended tests for HierarchicalChunker to improve code coverage.

This module provides additional test cases focusing on edge cases,
security validations, and error handling scenarios.
"""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo

from packages.shared.text_processing.base_chunker import ChunkResult
from packages.shared.text_processing.strategies.hierarchical_chunker import (
    MAX_CHUNK_SIZE,
    MAX_HIERARCHY_DEPTH,
    MAX_TEXT_LENGTH,
    STREAMING_CHUNK_SIZE,
    HierarchicalChunker,
)


class TestHierarchicalChunkerExtended:
    """Extended test suite for HierarchicalChunker edge cases and security."""

    def test_security_max_hierarchy_depth(self) -> None:
        """Test that excessive hierarchy depth is rejected."""
        # Create chunk sizes exceeding MAX_HIERARCHY_DEPTH
        excessive_sizes = [1000 - (i * 100) for i in range(MAX_HIERARCHY_DEPTH + 2)]

        with pytest.raises(ValueError, match="Too many hierarchy levels"):
            HierarchicalChunker(chunk_sizes=excessive_sizes)

    def test_security_max_chunk_size(self) -> None:
        """Test that excessive chunk sizes are rejected."""
        # Try to create chunker with size exceeding MAX_CHUNK_SIZE
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            HierarchicalChunker(chunk_sizes=[MAX_CHUNK_SIZE + 1, 1000, 500])

    def test_security_negative_chunk_size(self) -> None:
        """Test that negative chunk sizes are rejected."""
        with pytest.raises(ValueError, match="Invalid chunk size.*Must be positive"):
            HierarchicalChunker(chunk_sizes=[1000, -500, 250])

    def test_security_zero_chunk_size(self) -> None:
        """Test that zero chunk sizes are rejected."""
        with pytest.raises(ValueError, match="Invalid chunk size.*Must be positive"):
            HierarchicalChunker(chunk_sizes=[1000, 0, 250])

    def test_text_exceeds_max_length(self) -> None:
        """Test handling of text exceeding MAX_TEXT_LENGTH."""
        chunker = HierarchicalChunker(chunk_sizes=[1000, 500, 250])

        # Create text exceeding MAX_TEXT_LENGTH
        oversized_text = "a" * (MAX_TEXT_LENGTH + 1)

        with pytest.raises(ValueError, match="Text too large to process"):
            chunker.chunk_text(oversized_text, "oversized_doc")

    def test_text_exceeds_max_length_streaming(self) -> None:
        """Test streaming with text exceeding MAX_TEXT_LENGTH."""
        chunker = HierarchicalChunker(chunk_sizes=[1000, 500, 250])

        oversized_text = "a" * (MAX_TEXT_LENGTH + 1)

        with pytest.raises(ValueError, match="Text too large to process"):
            list(chunker.chunk_text_stream(oversized_text, "oversized_stream"))

    async def test_text_exceeds_max_length_async(self) -> None:
        """Test async chunking with text exceeding MAX_TEXT_LENGTH."""
        chunker = HierarchicalChunker(chunk_sizes=[1000, 500, 250])

        oversized_text = "a" * (MAX_TEXT_LENGTH + 1)

        with pytest.raises(ValueError, match="Text too large to process"):
            await chunker.chunk_text_async(oversized_text, "oversized_async")

    def test_large_text_segmentation(self) -> None:
        """Test that large texts are processed in segments."""
        chunker = HierarchicalChunker(chunk_sizes=[2000, 1000, 500])

        # Create text larger than STREAMING_CHUNK_SIZE but within MAX_TEXT_LENGTH
        large_text = "Test sentence. " * (STREAMING_CHUNK_SIZE // 14 + 100)

        # Mock the parser to verify segmentation
        mock_parser = MagicMock(spec=HierarchicalNodeParser)
        mock_nodes = []

        # Create mock nodes for segmented processing
        for i in range(5):
            node = MagicMock(spec=BaseNode)
            node.node_id = f"segment_node_{i}"
            node.get_content.return_value = f"Segment {i} content"
            node.relationships = {}
            node.metadata = {}
            mock_nodes.append(node)

        mock_parser.get_nodes_from_documents.return_value = mock_nodes

        with patch.object(chunker, "_parser", mock_parser):
            chunks = list(chunker.chunk_text_stream(large_text, "large_doc"))

            # Verify segmentation occurred
            assert mock_parser.get_nodes_from_documents.call_count > 1
            assert len(chunks) > 0

    def test_estimate_node_offset_with_children(self) -> None:
        """Test offset estimation for parent nodes with children."""
        chunker = HierarchicalChunker(chunk_sizes=[1000, 500, 250])

        # Create nodes with parent-child relationships
        parent_node = MagicMock(spec=BaseNode)
        parent_node.node_id = "parent"
        parent_node.get_content.return_value = "Parent content"

        child_nodes = []
        child_infos = []
        for i in range(3):
            child = MagicMock(spec=BaseNode)
            child.node_id = f"child_{i}"
            child.get_content.return_value = f"Child {i} content"
            child.relationships = {}
            child_nodes.append(child)

            # Create RelatedNodeInfo for child
            child_info = MagicMock(spec=RelatedNodeInfo)
            child_info.node_id = f"child_{i}"
            child_infos.append(child_info)

        # Set up parent relationships with list of children
        parent_node.relationships = {NodeRelationship.CHILD: child_infos}

        all_nodes = [parent_node] + child_nodes
        text = "Parent content Child 0 content Child 1 content Child 2 content"

        # Pre-populate some offsets
        existing_offsets = {
            "child_0": (15, 30),
            "child_1": (31, 46),
            "child_2": (47, 62),
        }

        offset = chunker._estimate_node_offset(parent_node, all_nodes, text, existing_offsets)

        # Parent offset should span from min child start to max child end
        assert offset == (15, 62)

    def test_estimate_node_offset_single_child(self) -> None:
        """Test offset estimation for parent with single child."""
        chunker = HierarchicalChunker(chunk_sizes=[1000, 500, 250])

        parent_node = MagicMock(spec=BaseNode)
        parent_node.node_id = "parent"
        parent_node.get_content.return_value = "Parent content"

        # Single child as RelatedNodeInfo (not list)
        child_info = MagicMock(spec=RelatedNodeInfo)
        child_info.node_id = "child_single"

        parent_node.relationships = {NodeRelationship.CHILD: child_info}

        child_node = MagicMock(spec=BaseNode)
        child_node.node_id = "child_single"
        child_node.get_content.return_value = "Single child content"

        all_nodes = [parent_node, child_node]
        text = "Parent content Single child content"

        existing_offsets = {"child_single": (15, 35)}

        offset = chunker._estimate_node_offset(parent_node, all_nodes, text, existing_offsets)

        # Should use the single child's offset
        assert offset == (15, 35)

    def test_build_offset_map_with_exact_match_failure(self) -> None:
        """Test offset map building when exact match fails."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        # Create nodes with content that needs stripping
        nodes = []
        for i in range(3):
            node = MagicMock(spec=BaseNode)
            node.node_id = f"node_{i}"
            # Add whitespace that will require stripping
            node.get_content.return_value = f"  Content {i}  "
            node.relationships = {}
            nodes.append(node)

        # Text without the extra whitespace
        text = "Content 0 Content 1 Content 2"

        offset_map = chunker._build_offset_map(text, nodes)

        # Should find the stripped content
        assert "node_0" in offset_map
        assert "node_1" in offset_map
        assert "node_2" in offset_map

    def test_build_offset_map_overlapping_content(self) -> None:
        """Test offset map with overlapping content."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        # Create nodes with overlapping content
        node1 = MagicMock(spec=BaseNode)
        node1.node_id = "node1"
        node1.get_content.return_value = "Test content with overlap"
        node1.relationships = {}

        node2 = MagicMock(spec=BaseNode)
        node2.node_id = "node2"
        node2.get_content.return_value = "content with"  # Substring of node1
        node2.relationships = {}

        nodes = [node1, node2]
        text = "Test content with overlap and more content with extra"

        offset_map = chunker._build_offset_map(text, nodes)

        # Both nodes should be mapped to non-overlapping positions
        assert "node1" in offset_map
        assert "node2" in offset_map

        # Verify non-overlapping
        start1, end1 = offset_map["node1"]
        start2, end2 = offset_map["node2"]
        assert end1 <= start2 or end2 <= start1  # No overlap

    def test_empty_stream_with_whitespace(self) -> None:
        """Test streaming with whitespace-only text."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        chunks = list(chunker.chunk_text_stream("   \n\t  ", "whitespace_doc"))
        assert chunks == []

    async def test_empty_async_stream_with_whitespace(self) -> None:
        """Test async streaming with whitespace-only text."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        chunks = []
        async for chunk in chunker.chunk_text_stream_async("   \n\t  ", "whitespace_async"):
            chunks.append(chunk)

        assert chunks == []

    def test_validate_config_excessive_hierarchy(self) -> None:
        """Test config validation with excessive hierarchy depth."""
        chunker = HierarchicalChunker()

        # Create config with too many hierarchy levels
        config = {"chunk_sizes": [1000 - (i * 50) for i in range(MAX_HIERARCHY_DEPTH + 1)]}

        assert chunker.validate_config(config) is False

    def test_validate_config_excessive_chunk_size(self) -> None:
        """Test config validation with chunk size exceeding maximum."""
        chunker = HierarchicalChunker()

        config = {"chunk_sizes": [MAX_CHUNK_SIZE + 1, 1000, 500]}

        assert chunker.validate_config(config) is False

    def test_validate_config_exception_handling(self) -> None:
        """Test config validation exception handling."""
        chunker = HierarchicalChunker()

        # Invalid config that will cause exception
        config = {"chunk_sizes": None}  # This will cause an exception

        with patch("packages.shared.text_processing.strategies.hierarchical_chunker.logger") as mock_logger:
            result = chunker.validate_config(config)
            assert result is False
            # The actual error message varies, just check that error was called
            mock_logger.error.assert_called()

    def test_get_parent_chunks_no_node_ids(self) -> None:
        """Test parent chunk generation when leaf chunks lack node IDs."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        # Create leaf chunks without node_id in metadata
        leaf_chunks = []
        for i in range(3):
            chunk = ChunkResult(
                chunk_id=f"leaf_{i}",
                text=f"Leaf content {i}",
                start_offset=i * 20,
                end_offset=(i + 1) * 20,
                metadata={"is_leaf": True},  # No node_id
            )
            leaf_chunks.append(chunk)

        text = "Test text for parent generation"

        with patch("packages.shared.text_processing.strategies.hierarchical_chunker.logger") as mock_logger:
            parent_chunks = chunker.get_parent_chunks(text, "test_doc", leaf_chunks)

            # Should log warning and return empty list
            mock_logger.warning.assert_called_with("No valid node IDs found in leaf chunks")
            assert parent_chunks == []

    def test_get_parent_chunks_with_node_children_list(self) -> None:
        """Test parent chunk generation with children as list."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        # Create mock parser
        mock_parser = MagicMock(spec=HierarchicalNodeParser)

        # Create parent node with multiple children
        parent_node = MagicMock(spec=BaseNode)
        parent_node.node_id = "parent"
        parent_node.get_content.return_value = "Parent content"

        # Create child infos as list
        child_infos = []
        for i in range(2):
            child_info = MagicMock(spec=RelatedNodeInfo)
            child_info.node_id = f"leaf_{i}"
            child_infos.append(child_info)

        parent_node.relationships = {NodeRelationship.CHILD: child_infos}

        # Create leaf nodes
        leaf_nodes = []
        for i in range(2):
            leaf = MagicMock(spec=BaseNode)
            leaf.node_id = f"leaf_{i}"
            leaf.get_content.return_value = f"Leaf {i}"
            leaf.relationships = {}
            leaf_nodes.append(leaf)

        mock_parser.get_nodes_from_documents.return_value = [parent_node] + leaf_nodes

        # Create leaf chunks with node_ids
        leaf_chunks = []
        for i in range(2):
            chunk = ChunkResult(
                chunk_id=f"chunk_{i}",
                text=f"Leaf {i}",
                start_offset=i * 10,
                end_offset=(i + 1) * 10,
                metadata={"node_id": f"leaf_{i}", "is_leaf": True},
            )
            leaf_chunks.append(chunk)

        text = "Parent content Leaf 0 Leaf 1"

        with patch.object(chunker, "_parser", mock_parser):
            parent_chunks = chunker.get_parent_chunks(text, "test_doc", leaf_chunks)

            # Should find and return parent chunk
            assert len(parent_chunks) == 1
            assert parent_chunks[0].metadata["node_id"] == "parent"
            assert not parent_chunks[0].metadata["is_leaf"]

    def test_get_parent_chunks_with_single_child(self) -> None:
        """Test parent chunk generation with single child (not list)."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        mock_parser = MagicMock(spec=HierarchicalNodeParser)

        parent_node = MagicMock(spec=BaseNode)
        parent_node.node_id = "parent"
        parent_node.get_content.return_value = "Parent content"

        # Single child (not a list)
        child_info = MagicMock(spec=RelatedNodeInfo)
        child_info.node_id = "leaf_0"

        parent_node.relationships = {NodeRelationship.CHILD: child_info}

        leaf_node = MagicMock(spec=BaseNode)
        leaf_node.node_id = "leaf_0"
        leaf_node.get_content.return_value = "Single leaf"
        leaf_node.relationships = {}

        mock_parser.get_nodes_from_documents.return_value = [parent_node, leaf_node]

        leaf_chunk = ChunkResult(
            chunk_id="chunk_0",
            text="Single leaf",
            start_offset=0,
            end_offset=11,
            metadata={"node_id": "leaf_0", "is_leaf": True},
        )

        text = "Parent content Single leaf"

        with patch.object(chunker, "_parser", mock_parser):
            parent_chunks = chunker.get_parent_chunks(text, "test_doc", [leaf_chunk])

            assert len(parent_chunks) == 1
            assert parent_chunks[0].metadata["node_id"] == "parent"

    def test_get_parent_chunks_exception_handling(self) -> None:
        """Test parent chunk generation with exception."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        mock_parser = MagicMock(spec=HierarchicalNodeParser)
        mock_parser.get_nodes_from_documents.side_effect = Exception("Parser error")

        leaf_chunk = ChunkResult(
            chunk_id="chunk_0",
            text="Leaf",
            start_offset=0,
            end_offset=4,
            metadata={"node_id": "leaf_0", "is_leaf": True},
        )

        with (
            patch.object(chunker, "_parser", mock_parser),
            patch("packages.shared.text_processing.strategies.hierarchical_chunker.logger") as mock_logger,
        ):
            parent_chunks = chunker.get_parent_chunks("Test text", "test_doc", [leaf_chunk])

            # Should log error and return empty list
            mock_logger.error.assert_called()
            assert parent_chunks == []

    def test_build_hierarchy_info_infinite_loop_prevention(self) -> None:
        """Test that hierarchy info building prevents infinite loops."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        # Create nodes with circular parent relationships
        node1 = MagicMock(spec=BaseNode)
        node1.node_id = "node1"
        node1.get_content.return_value = "Node 1"

        node2 = MagicMock(spec=BaseNode)
        node2.node_id = "node2"
        node2.get_content.return_value = "Node 2"

        # Create circular reference
        parent_rel1 = MagicMock()
        parent_rel1.node_id = "node2"
        node1.relationships = {NodeRelationship.PARENT: parent_rel1}

        parent_rel2 = MagicMock()
        parent_rel2.node_id = "node1"
        node2.relationships = {NodeRelationship.PARENT: parent_rel2}

        node_map = {"node1": node1, "node2": node2}

        # Should not infinite loop
        hierarchy_info = chunker._build_hierarchy_info(node1, node_map)

        # Should have detected the cycle and stopped
        assert hierarchy_info["level"] >= 0  # Should have a valid level

    def test_node_without_get_content_method(self) -> None:
        """Test handling of nodes without get_content method."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        # Create a node without get_content method
        bad_node = MagicMock(spec=BaseNode)
        bad_node.node_id = "bad_node"
        # Remove the get_content method
        del bad_node.get_content

        good_node = MagicMock(spec=BaseNode)
        good_node.node_id = "good_node"
        good_node.get_content.return_value = "Good content"

        nodes = [bad_node, good_node]
        text = "Good content"

        # Should handle gracefully
        with pytest.raises(AttributeError):
            chunker._build_offset_map(text, nodes)

    def test_streaming_error_handling_fallback(self) -> None:
        """Test streaming fallback when hierarchical parsing fails."""
        chunker = HierarchicalChunker(chunk_sizes=[200, 100, 50])

        mock_parser = MagicMock()
        mock_parser.get_nodes_from_documents.side_effect = Exception("Streaming error")

        with (
            patch.object(chunker, "_parser", mock_parser),
            patch("packages.shared.text_processing.strategies.hierarchical_chunker.logger") as mock_logger,
        ):
            chunks = list(chunker.chunk_text_stream("Test text for streaming", "stream_error"))

            # Should fall back to character chunking
            assert len(chunks) > 0
            mock_logger.error.assert_called()
            mock_logger.warning.assert_called_with("Using fallback chunking strategy")

            # Verify fallback chunks
            for chunk in chunks:
                assert chunk.metadata.get("strategy") == "character"

    def test_build_hierarchy_info_node_not_in_map(self) -> None:
        """Test hierarchy info when parent node is not in node map."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        node = MagicMock(spec=BaseNode)
        node.node_id = "child"
        node.get_content.return_value = "Child content"

        # Parent relationship pointing to non-existent node
        parent_rel = MagicMock()
        parent_rel.node_id = "missing_parent"
        node.relationships = {NodeRelationship.PARENT: parent_rel}

        node_map = {"child": node}  # Parent not in map

        hierarchy_info = chunker._build_hierarchy_info(node, node_map)

        # Should handle gracefully
        assert hierarchy_info["parent_id"] == "missing_parent"
        assert hierarchy_info["level"] == 1  # One level up even though parent is missing

    def test_build_offset_map_empty_content(self) -> None:
        """Test offset map with nodes having empty content."""
        chunker = HierarchicalChunker(chunk_sizes=[500, 250, 125])

        # Create nodes with empty content
        empty_node = MagicMock(spec=BaseNode)
        empty_node.node_id = "empty"
        empty_node.get_content.return_value = ""
        empty_node.relationships = {}

        normal_node = MagicMock(spec=BaseNode)
        normal_node.node_id = "normal"
        normal_node.get_content.return_value = "Normal content"
        normal_node.relationships = {}

        nodes = [empty_node, normal_node]
        text = "Normal content"

        offset_map = chunker._build_offset_map(text, nodes)

        # Empty node should be skipped or handled gracefully
        assert "normal" in offset_map
        # Empty node might not be in map or have default offset
        if "empty" in offset_map:
            start, end = offset_map["empty"]
            assert end - start == 0  # Empty content

    async def test_async_streaming_executor_handling(self) -> None:
        """Test async streaming with executor."""
        chunker = HierarchicalChunker(chunk_sizes=[200, 100, 50])

        text = "Test text for async streaming"

        chunks = []
        async for chunk in chunker.chunk_text_stream_async(text, "async_executor_test"):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
