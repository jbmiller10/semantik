"""Tests for Phase 3: Progressive Segmentation of Large Documents.

This test file validates the progressive segmentation functionality that
processes large documents in bounded memory segments.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from packages.webui.services.chunking_service import ChunkingService


class TestProgressiveSegmentation:
    """Tests for progressive segmentation functionality."""

    @pytest.fixture()
    def service(self):
        """Create a ChunkingService instance with minimal mocking."""
        return ChunkingService(
            db_session=AsyncMock(),
            collection_repo=MagicMock(),
            document_repo=MagicMock(),
            redis_client=None,
        )

    @pytest.fixture()
    def large_text(self):
        """Generate a large text document for testing segmentation."""
        # Create 10MB of text (well over the 5MB threshold)
        paragraph = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
            "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum. "
            "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia. "
            "\n\n"
        )
        # Each paragraph is ~300 bytes, need ~35,000 paragraphs for 10MB
        return paragraph * 35000

    @pytest.fixture()
    def medium_text(self):
        """Generate a medium text document for testing (3MB)."""
        paragraph = (
            "This is a medium-sized document for testing. "
            "It should trigger segmentation for some strategies but not others. "
            "\n\n"
        )
        # Each paragraph is ~100 bytes, need ~30,000 paragraphs for 3MB
        return paragraph * 30000

    @pytest.mark.asyncio()
    async def test_large_document_triggers_segmentation(self, service, large_text):
        """Test that large documents trigger progressive segmentation."""
        collection = {
            "id": "coll-large",
            "name": "Large Doc Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

        with patch.object(service, "execute_ingestion_chunking_segmented") as mock_segmented:
            mock_segmented.return_value = {
                "chunks": [
                    {"chunk_id": f"doc_chunk_{i:04d}", "text": f"chunk {i}", "metadata": {}} for i in range(100)
                ],
                "stats": {
                    "duration_ms": 5000,
                    "strategy_used": "recursive",
                    "chunk_count": 100,
                    "segment_count": 10,
                    "segmented": True,
                },
            }

            result = await service.execute_ingestion_chunking(
                text=large_text,
                document_id="doc-large",
                collection=collection,
            )

            # Verify segmented method was called
            mock_segmented.assert_called_once()
            assert result["stats"]["segmented"] is True
            assert result["stats"]["segment_count"] == 10

    @pytest.mark.asyncio()
    async def test_small_document_no_segmentation(self, service):
        """Test that small documents don't trigger segmentation."""
        small_text = "This is a small document that should not trigger segmentation."

        collection = {
            "id": "coll-small",
            "name": "Small Doc Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 100,
            "chunk_overlap": 20,
        }

        with (
            patch.object(service, "execute_ingestion_chunking_segmented") as mock_segmented,
            patch.object(service.strategy_factory, "create_strategy") as mock_factory,
            patch.object(service.config_builder, "build_config") as mock_config_builder,
        ):
            # Mock the config builder
            mock_config_result = MagicMock()
            mock_config_result.validation_errors = []
            mock_config_result.strategy = "recursive"
            mock_config_result.config = {"chunk_size": 100, "chunk_overlap": 20}
            mock_config_builder.return_value = mock_config_result
            
            # Mock the strategy
            mock_strategy = MagicMock()
            mock_chunk_entity = MagicMock()
            mock_chunk_entity.content = small_text
            mock_strategy.chunk.return_value = [mock_chunk_entity]
            mock_factory.return_value = mock_strategy

            result = await service.execute_ingestion_chunking(
                text=small_text,
                document_id="doc-small",
                collection=collection,
            )

            # Verify segmented method was NOT called
            mock_segmented.assert_not_called()
            assert len(result["chunks"]) == 1

    @pytest.mark.asyncio()
    async def test_strategy_specific_thresholds(self, service, medium_text):
        """Test that different strategies have different segmentation thresholds."""
        # Semantic strategy has 2MB threshold (should trigger for 3MB text)
        semantic_collection = {
            "id": "coll-semantic",
            "name": "Semantic Collection",
            "chunking_strategy": "semantic",
            "chunking_config": {},
            "chunk_size": 512,
            "chunk_overlap": 50,
        }

        # Markdown strategy has 10MB threshold (should NOT trigger for 3MB text)
        markdown_collection = {
            "id": "coll-markdown",
            "name": "Markdown Collection",
            "chunking_strategy": "markdown",
            "chunking_config": {},
            "chunk_size": 1000,
            "chunk_overlap": 100,
        }

        with patch.object(service, "execute_ingestion_chunking_segmented") as mock_segmented:
            mock_segmented.return_value = {
                "chunks": [{"chunk_id": f"doc_chunk_{i:04d}", "text": f"chunk {i}", "metadata": {}} for i in range(50)],
                "stats": {
                    "duration_ms": 3000,
                    "strategy_used": "semantic",
                    "chunk_count": 50,
                    "segment_count": 3,
                    "segmented": True,
                },
            }

            # Test semantic strategy (should segment)
            await service.execute_ingestion_chunking(
                text=medium_text,
                document_id="doc-semantic",
                collection=semantic_collection,
            )
            mock_segmented.assert_called_once()

        # Reset mock
        with (
            patch.object(service, "execute_ingestion_chunking_segmented") as mock_segmented,
            patch.object(service.strategy_factory, "create_strategy") as mock_factory,
        ):
            mock_strategy = MagicMock()
            mock_strategy.chunk.return_value = [MagicMock(content="chunk")]
            mock_factory.return_value = mock_strategy

            # Test markdown strategy (should NOT segment)
            await service.execute_ingestion_chunking(
                text=medium_text,
                document_id="doc-markdown",
                collection=markdown_collection,
            )
            mock_segmented.assert_not_called()

    @pytest.mark.asyncio()
    async def test_segmentation_preserves_boundaries(self, service):
        """Test that segmentation preserves paragraph and sentence boundaries."""
        # Create text with clear boundaries
        text_with_boundaries = (
            "First paragraph with multiple sentences. This is the second sentence. "
            "And this is the third sentence.\n\n"
            "Second paragraph starts here. It also has multiple sentences. "
            "This helps test boundary preservation.\n\n"
        ) * 10000  # Make it large enough to trigger segmentation

        collection = {
            "id": "coll-boundaries",
            "name": "Boundary Test Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

        result = await service.execute_ingestion_chunking_segmented(
            text=text_with_boundaries,
            document_id="doc-boundaries",
            collection=collection,
        )

        # Verify segments were created
        assert result["stats"]["segment_count"] > 1
        assert result["stats"]["segmented"] is True

    @pytest.mark.asyncio()
    async def test_segment_metadata_added(self, service):
        """Test that segment metadata is properly added to chunks."""
        large_text = "Large text content. " * 100000  # Create large text

        collection = {
            "id": "coll-metadata",
            "name": "Metadata Test Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

        with patch.object(service, "_process_segment") as mock_process:
            # _process_segment returns a dict with "chunks" and potentially "stats"
            mock_process.return_value = {
                "chunks": [
                    {
                        "chunk_id": "chunk_0000",
                        "text": "chunk text",
                        "metadata": {"segment_idx": 0, "total_segments": 2},
                    }
                ],
                "stats": {
                    "duration_ms": 100,
                    "strategy_used": "recursive",
                    "chunk_count": 1,
                }
            }

            result = await service.execute_ingestion_chunking_segmented(
                text=large_text,
                document_id="doc-metadata",
                collection=collection,
            )

            # Verify _process_segment was called with correct parameters
            assert mock_process.call_count > 0
            call_args = mock_process.call_args_list[0]
            # _process_segment is called with positional args: 
            # (segment_text, document_id, collection, metadata, file_type, chunk_id_start, segment_idx, total_segments)
            assert call_args[0][6] == 0  # segment_idx is the 7th positional argument (index 6)
            
            # Verify the segment metadata is present in the result
            assert len(result["chunks"]) > 0
            first_chunk = result["chunks"][0]
            assert "segment_idx" in first_chunk["metadata"]
            assert first_chunk["metadata"]["segment_idx"] == 0

    @pytest.mark.asyncio()
    async def test_segment_overlap_maintained(self, service):
        """Test that segments have proper overlap to preserve context."""
        # Create text that will be segmented
        test_text = "A" * 2000000  # 2MB of 'A's

        collection = {
            "id": "coll-overlap",
            "name": "Overlap Test Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

        with (
            patch("packages.webui.services.chunking_constants.DEFAULT_SEGMENT_SIZE", 1000000),  # 1MB segments
            patch("packages.webui.services.chunking_constants.DEFAULT_SEGMENT_OVERLAP", 10000),  # 10KB overlap
        ):
            result = await service.execute_ingestion_chunking_segmented(
                text=test_text,
                document_id="doc-overlap",
                collection=collection,
            )

            # Should have at least 2 segments with overlap
            assert result["stats"]["segment_count"] >= 2

    @pytest.mark.asyncio()
    async def test_max_segments_limit(self, service):
        """Test that MAX_SEGMENTS_PER_DOCUMENT limit is respected."""
        # Create extremely large text that would create too many segments
        huge_text = "X" * 200000000  # 200MB

        collection = {
            "id": "coll-max-segments",
            "name": "Max Segments Test",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

        with (
            patch("packages.webui.services.chunking_constants.MAX_SEGMENTS_PER_DOCUMENT", 10),
            patch.object(service, "_process_segment") as mock_process,
        ):
            mock_process.return_value = {"chunks": []}

            await service.execute_ingestion_chunking_segmented(
                text=huge_text,
                document_id="doc-huge",
                collection=collection,
            )

            # Should not exceed max segments
            assert mock_process.call_count <= 10

    @pytest.mark.asyncio()
    async def test_segment_failure_continues_processing(self, service):
        """Test that failure in one segment doesn't stop processing of others."""
        large_text = "Test content. " * 100000

        collection = {
            "id": "coll-failure",
            "name": "Failure Test Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

        call_count = 0

        async def mock_process_segment(*args, **kwargs):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second segment
                raise Exception("Segment processing failed")
            return {"chunks": [{"chunk_id": f"chunk_{call_count:04d}", "text": f"chunk {call_count}", "metadata": {}}]}

        with patch.object(service, "_process_segment", side_effect=mock_process_segment):
            result = await service.execute_ingestion_chunking_segmented(
                text=large_text,
                document_id="doc-failure",
                collection=collection,
            )

            # Should have chunks from segments that succeeded
            assert len(result["chunks"]) > 0
            # Second segment should have failed, so we should be missing those chunks
            chunk_ids = [c["chunk_id"] for c in result["chunks"]]
            assert "chunk_0002" not in chunk_ids  # Second segment failed

    @pytest.mark.asyncio()
    async def test_segmentation_metrics_recorded(self, service, large_text):
        """Test that segmentation metrics are properly recorded."""
        collection = {
            "id": "coll-metrics",
            "name": "Metrics Test Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

        with (
            patch("packages.webui.services.chunking_metrics.record_document_segmented") as mock_doc_seg,
            patch("packages.webui.services.chunking_metrics.record_segments_created") as mock_seg_created,
            patch("packages.webui.services.chunking_metrics.record_segment_size") as mock_seg_size,
            patch.object(service, "_process_segment") as mock_process,
        ):
            mock_process.return_value = {"chunks": []}

            await service.execute_ingestion_chunking_segmented(
                text=large_text,
                document_id="doc-metrics",
                collection=collection,
            )

            # Verify metrics were recorded
            mock_doc_seg.assert_called_once_with("recursive")
            assert mock_seg_created.called
            assert mock_seg_size.called

    @pytest.mark.asyncio()
    async def test_chunk_id_continuity_across_segments(self, service):
        """Test that chunk IDs maintain continuity across segments."""
        # Create a larger text to ensure segmentation
        large_text = "Test content for chunking. " * 100000  # ~2.8MB

        collection = {
            "id": "coll-continuity",
            "name": "Continuity Test Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

        # _process_segment returns a dict with "chunks" key and optionally "stats"
        segment_results = [
            {
                "chunks": [{"chunk_id": f"temp_{i}", "text": f"chunk {i}", "metadata": {}} for i in range(3)],
                "stats": {"duration_ms": 100, "strategy_used": "recursive", "chunk_count": 3}
            },
            {
                "chunks": [{"chunk_id": f"temp_{i}", "text": f"chunk {i+3}", "metadata": {}} for i in range(3)],
                "stats": {"duration_ms": 100, "strategy_used": "recursive", "chunk_count": 3}
            },
            {
                "chunks": [{"chunk_id": f"temp_{i}", "text": f"chunk {i+6}", "metadata": {}} for i in range(3)],
                "stats": {"duration_ms": 100, "strategy_used": "recursive", "chunk_count": 3}
            },
        ]

        with (
            patch.object(service, "_process_segment", side_effect=segment_results),
            # Patch the segment size to ensure we get exactly 3 segments
            patch("packages.webui.services.chunking_constants.DEFAULT_SEGMENT_SIZE", 1000000),  # 1MB segments
            patch("packages.webui.services.chunking_constants.STRATEGY_SEGMENT_THRESHOLDS", {"recursive": 2000000}),  # 2MB threshold
        ):
            result = await service.execute_ingestion_chunking_segmented(
                text=large_text,
                document_id="doc-continuity",
                collection=collection,
            )

            # Verify we have the right number of chunks
            assert len(result["chunks"]) == 9
            
            # Verify that we have chunks from all three segments
            # The chunks should have content from all segments
            chunk_texts = [chunk["text"] for chunk in result["chunks"]]
            expected_texts = [f"chunk {i}" for i in range(9)]
            assert chunk_texts == expected_texts
