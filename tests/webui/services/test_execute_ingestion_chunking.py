"""Unit tests specifically for ChunkingService.execute_ingestion_chunking method.

This test file provides focused unit tests for the execute_ingestion_chunking method,
complementing the integration tests by focusing on specific implementation details
and edge cases.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.webui.services.chunking_service import ChunkingService


class TestExecuteIngestionChunkingUnit:
    """Focused unit tests for execute_ingestion_chunking method."""

    @pytest.fixture
    def service(self):
        """Create a ChunkingService instance with minimal mocking."""
        return ChunkingService(
            db_session=AsyncMock(),
            collection_repo=MagicMock(),
            document_repo=MagicMock(),
            redis_client=None,
        )

    @pytest.mark.asyncio
    async def test_hierarchical_strategy_configuration(self, service):
        """Test hierarchical strategy with multi-level configuration."""
        collection = {
            "id": "coll-hierarchical",
            "name": "Hierarchical Collection",
            "chunking_strategy": "hierarchical",
            "chunking_config": {
                "chunk_sizes": [2048, 512, 128],
                "chunk_overlap": 50,
            },
            "chunk_size": 1000,
            "chunk_overlap": 100,
        }
        
        mock_strategy = MagicMock()
        mock_chunks = [
            MagicMock(content=f"Level {i // 3} - Chunk {i % 3}")
            for i in range(9)  # 3 levels x 3 chunks
        ]
        mock_strategy.chunk.return_value = mock_chunks
        
        with patch.object(service.strategy_factory, 'create_strategy', return_value=mock_strategy):
            result = await service.execute_ingestion_chunking(
                text="Hierarchical document content",
                document_id="doc-hier",
                collection=collection,
            )
        
        assert result["stats"]["strategy_used"] in ["hierarchical", "ChunkingStrategy.HIERARCHICAL"]
        assert result["stats"]["chunk_count"] == 9
        
        # Verify ChunkConfig was built correctly
        call_args = mock_strategy.chunk.call_args
        chunk_config = call_args[1]["config"]
        assert isinstance(chunk_config, ChunkConfig)
        assert chunk_config.strategy_name == "hierarchical"

    @pytest.mark.asyncio
    async def test_hybrid_strategy_configuration(self, service):
        """Test hybrid strategy with primary and fallback strategies."""
        collection = {
            "id": "coll-hybrid",
            "name": "Hybrid Collection",
            "chunking_strategy": "hybrid",
            "chunking_config": {
                "primary_strategy": "semantic",
                "fallback_strategy": "recursive",
                "semantic_threshold": 0.8,
            },
            "chunk_size": 500,
            "chunk_overlap": 50,
        }
        
        mock_strategy = MagicMock()
        mock_chunks = [
            MagicMock(content="Hybrid chunk 1"),
            MagicMock(content="Hybrid chunk 2"),
        ]
        mock_strategy.chunk.return_value = mock_chunks
        
        with patch.object(service.strategy_factory, 'create_strategy', return_value=mock_strategy):
            result = await service.execute_ingestion_chunking(
                text="Text for hybrid chunking",
                document_id="doc-hybrid",
                collection=collection,
            )
        
        assert result["stats"]["strategy_used"] in ["hybrid", "ChunkingStrategy.HYBRID"]
        assert len(result["chunks"]) == 2

    @pytest.mark.asyncio
    async def test_config_builder_validation_errors_trigger_fallback(self, service):
        """Test that validation errors from config builder trigger fallback."""
        collection = {
            "id": "coll-invalid",
            "name": "Invalid Config Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {
                "chunk_size": -100,  # Invalid negative size
                "chunk_overlap": 500,  # Overlap larger than size
            },
            "chunk_size": 100,
            "chunk_overlap": 20,
        }
        
        # Mock config builder to return validation errors
        mock_config_result = MagicMock()
        mock_config_result.validation_errors = [
            "chunk_size must be positive",
            "chunk_overlap cannot exceed chunk_size",
        ]
        mock_config_result.strategy = "recursive"
        mock_config_result.config = {}
        
        with patch.object(service.config_builder, 'build_config', return_value=mock_config_result):
            with patch('shared.text_processing.chunking.TokenChunker') as MockTokenChunker:
                mock_chunker = MagicMock()
                MockTokenChunker.return_value = mock_chunker
                mock_chunker.chunk_text.return_value = [
                    {"chunk_id": "doc-invalid_chunk_0000", "text": "Fallback chunk", "metadata": {}}
                ]
                
                result = await service.execute_ingestion_chunking(
                    text="Text with invalid config",
                    document_id="doc-invalid",
                    collection=collection,
                )
        
        # Verify fallback was used
        assert result["stats"]["strategy_used"] == "TokenChunker"
        assert result["stats"]["fallback"] is True
        MockTokenChunker.assert_called_once_with(chunk_size=100, chunk_overlap=20)

    @pytest.mark.asyncio
    async def test_strategy_factory_creation_failure_triggers_fallback(self, service):
        """Test that strategy factory failures trigger fallback."""
        collection = {
            "id": "coll-factory-fail",
            "name": "Factory Fail Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 200,
            "chunk_overlap": 40,
        }
        
        # Mock successful config building
        mock_config_result = MagicMock()
        mock_config_result.validation_errors = []
        mock_config_result.strategy = "recursive"
        mock_config_result.config = {"chunk_size": 200, "chunk_overlap": 40}
        
        with patch.object(service.config_builder, 'build_config', return_value=mock_config_result):
            # Mock strategy factory to raise exception
            with patch.object(service.strategy_factory, 'create_strategy', side_effect=RuntimeError("Strategy creation failed")):
                with patch('shared.text_processing.chunking.TokenChunker') as MockTokenChunker:
                    mock_chunker = MagicMock()
                    MockTokenChunker.return_value = mock_chunker
                    mock_chunker.chunk_text.return_value = [
                        {"chunk_id": "fallback_chunk_0000", "text": "Fallback after factory fail", "metadata": {}}
                    ]
                    
                    result = await service.execute_ingestion_chunking(
                        text="Text causing factory failure",
                        document_id="doc-factory-fail",
                        collection=collection,
                    )
        
        assert result["stats"]["strategy_used"] == "TokenChunker"
        assert result["stats"]["fallback"] is True

    @pytest.mark.asyncio
    async def test_chunk_config_min_max_token_calculation(self, service):
        """Test that ChunkConfig min/max tokens are calculated correctly."""
        test_cases = [
            # (chunk_size, expected_min, expected_max)
            (100, 50, 100),
            (50, 25, 50),
            (200, 100, 200),
            (10, 5, 10),
            (1, 0, 1),  # Edge case: very small chunk
        ]
        
        for chunk_size, expected_min, expected_max in test_cases:
            collection = {
                "id": f"coll-size-{chunk_size}",
                "name": f"Collection Size {chunk_size}",
                "chunking_strategy": "recursive",
                "chunking_config": {"chunk_size": chunk_size, "chunk_overlap": min(10, chunk_size // 2)},
                "chunk_size": chunk_size,
                "chunk_overlap": min(10, chunk_size // 2),
            }
            
            mock_strategy = MagicMock()
            mock_strategy.chunk.return_value = [MagicMock(content="test chunk")]
            
            with patch.object(service.strategy_factory, 'create_strategy', return_value=mock_strategy):
                await service.execute_ingestion_chunking(
                    text="Test text",
                    document_id=f"doc-{chunk_size}",
                    collection=collection,
                )
                
                # Verify ChunkConfig parameters
                call_args = mock_strategy.chunk.call_args
                chunk_config = call_args[1]["config"]
                
                assert chunk_config.min_tokens == expected_min
                assert chunk_config.max_tokens == expected_max

    @pytest.mark.asyncio
    async def test_overlap_tokens_boundary_validation(self, service):
        """Test that overlap tokens are properly bounded."""
        collection = {
            "id": "coll-overlap",
            "name": "Overlap Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {
                "chunk_size": 100,
                "chunk_overlap": 200,  # Overlap larger than chunk size
            },
            "chunk_size": 100,
            "chunk_overlap": 200,
        }
        
        mock_strategy = MagicMock()
        mock_strategy.chunk.return_value = [MagicMock(content="chunk")]
        
        with patch.object(service.strategy_factory, 'create_strategy', return_value=mock_strategy):
            await service.execute_ingestion_chunking(
                text="Test text",
                document_id="doc-overlap",
                collection=collection,
            )
            
            # Verify overlap was capped appropriately
            call_args = mock_strategy.chunk.call_args
            chunk_config = call_args[1]["config"]
            
            # Overlap should be less than min_tokens
            assert chunk_config.overlap_tokens < chunk_config.min_tokens
            assert chunk_config.overlap_tokens >= 0

    @pytest.mark.asyncio
    async def test_file_type_passed_correctly(self, service):
        """Test that file_type is properly handled."""
        file_types = ["txt", "pdf", "md", "py", "json", None]
        
        for file_type in file_types:
            collection = {
                "id": f"coll-{file_type or 'none'}",
                "name": f"Collection {file_type or 'none'}",
                "chunking_strategy": "recursive",
                "chunking_config": {},
                "chunk_size": 100,
                "chunk_overlap": 20,
            }
            
            mock_strategy = MagicMock()
            mock_strategy.chunk.return_value = [MagicMock(content="chunk")]
            
            with patch.object(service.strategy_factory, 'create_strategy', return_value=mock_strategy):
                result = await service.execute_ingestion_chunking(
                    text="Test content",
                    document_id=f"doc-{file_type or 'none'}",
                    collection=collection,
                    file_type=file_type,
                )
                
                # File type doesn't affect the chunking directly, but should be available for logging
                assert result["stats"]["strategy_used"] in ["recursive", "ChunkingStrategy.RECURSIVE"]

    @pytest.mark.asyncio
    async def test_metadata_merge_with_chunk_metadata(self, service):
        """Test that provided metadata is properly merged with chunk metadata."""
        input_metadata = {
            "source": "test_source",
            "author": "test_author",
            "custom_field": "custom_value",
            "index": 999,  # This should be overridden by chunk index
        }
        
        collection = {
            "id": "coll-metadata",
            "name": "Metadata Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 100,
            "chunk_overlap": 20,
        }
        
        mock_strategy = MagicMock()
        mock_chunks = [
            MagicMock(content=f"Chunk {i}") for i in range(3)
        ]
        mock_strategy.chunk.return_value = mock_chunks
        
        with patch.object(service.strategy_factory, 'create_strategy', return_value=mock_strategy):
            result = await service.execute_ingestion_chunking(
                text="Text with metadata",
                document_id="doc-metadata",
                collection=collection,
                metadata=input_metadata,
            )
        
        # Check metadata merging
        for idx, chunk in enumerate(result["chunks"]):
            assert chunk["metadata"]["source"] == "test_source"
            assert chunk["metadata"]["author"] == "test_author"
            assert chunk["metadata"]["custom_field"] == "custom_value"
            assert chunk["metadata"]["index"] == idx  # Should be chunk index, not 999
            assert chunk["metadata"]["strategy"] in ["recursive", "ChunkingStrategy.RECURSIVE"]

    @pytest.mark.asyncio
    async def test_empty_metadata_handling(self, service):
        """Test handling of None and empty metadata."""
        collection = {
            "id": "coll-no-metadata",
            "name": "No Metadata Collection",
            "chunk_size": 100,
            "chunk_overlap": 20,
        }
        
        with patch('packages.webui.services.chunking_service.TokenChunker') as MockTokenChunker:
            mock_chunker = MagicMock()
            MockTokenChunker.return_value = mock_chunker
            mock_chunker.chunk_text.return_value = [
                {"chunk_id": "chunk_0000", "text": "chunk", "metadata": {"index": 0}}
            ]
            
            # Test with None metadata
            result = await service.execute_ingestion_chunking(
                text="Text",
                document_id="doc-1",
                collection=collection,
                metadata=None,
            )
            
            assert result["chunks"][0]["metadata"]["index"] == 0
            
            # Test with empty dict metadata
            result = await service.execute_ingestion_chunking(
                text="Text",
                document_id="doc-2",
                collection=collection,
                metadata={},
            )
            
            assert result["chunks"][0]["metadata"]["index"] == 0

    @pytest.mark.asyncio
    async def test_logging_on_successful_chunking(self, service):
        """Test that appropriate logging occurs on successful chunking."""
        collection = {
            "id": "coll-logging",
            "name": "Logging Collection",
            "chunking_strategy": "semantic",
            "chunking_config": {},
            "chunk_size": 100,
            "chunk_overlap": 20,
        }
        
        mock_strategy = MagicMock()
        mock_chunks = [MagicMock(content=f"Chunk {i}") for i in range(5)]
        mock_strategy.chunk.return_value = mock_chunks
        
        with patch.object(service.strategy_factory, 'create_strategy', return_value=mock_strategy):
            with patch('packages.webui.services.chunking_service.logger') as mock_logger:
                result = await service.execute_ingestion_chunking(
                    text="Text for logging test",
                    document_id="doc-log",
                    collection=collection,
                )
                
                # Verify info log for successful chunking
                mock_logger.info.assert_called()
                log_message = mock_logger.info.call_args[0][0]
                assert "doc-log" in log_message
                assert "semantic" in log_message
                assert "5 chunks" in log_message

    @pytest.mark.asyncio
    async def test_logging_on_fallback(self, service):
        """Test that warning logs are generated on fallback."""
        collection = {
            "id": "coll-fallback-log",
            "name": "Fallback Log Collection",
            "chunking_strategy": "invalid_strategy",
            "chunking_config": {},
            "chunk_size": 100,
            "chunk_overlap": 20,
        }
        
        with patch.object(service.config_builder, 'build_config', side_effect=Exception("Config error")):
            with patch('shared.text_processing.chunking.TokenChunker') as MockTokenChunker:
                mock_chunker = MagicMock()
                MockTokenChunker.return_value = mock_chunker
                mock_chunker.chunk_text.return_value = [
                    {"chunk_id": "chunk_0000", "text": "fallback", "metadata": {}}
                ]
                
                with patch('packages.webui.services.chunking_service.logger') as mock_logger:
                    result = await service.execute_ingestion_chunking(
                        text="Text",
                        document_id="doc-fallback",
                        collection=collection,
                    )
                    
                    # Should have warning logs for fallback
                    assert mock_logger.warning.call_count >= 2
                    
                    # Check final fallback warning
                    fallback_warning_found = False
                    for call in mock_logger.warning.call_args_list:
                        if "Chunking fallback occurred" in call[0][0]:
                            fallback_warning_found = True
                            assert "doc-fallback" in call[0][0]
                            assert "coll-fallback-log" in call[0][0]
                            break
                    
                    assert fallback_warning_found

    @pytest.mark.asyncio
    async def test_performance_stats_accuracy(self, service):
        """Test that performance statistics are accurate."""
        collection = {
            "id": "coll-perf",
            "name": "Performance Collection",
            "chunk_size": 100,
            "chunk_overlap": 20,
        }
        
        with patch('packages.webui.services.chunking_service.TokenChunker') as MockTokenChunker:
            mock_chunker = MagicMock()
            MockTokenChunker.return_value = mock_chunker
            
            # Add artificial delay to test timing
            def slow_chunk(*args, **kwargs):
                time.sleep(0.05)  # 50ms delay
                return [
                    {"chunk_id": f"chunk_{i:04d}", "text": f"chunk {i}", "metadata": {}}
                    for i in range(10)
                ]
            
            mock_chunker.chunk_text = slow_chunk
            
            result = await service.execute_ingestion_chunking(
                text="Performance test text",
                document_id="doc-perf",
                collection=collection,
            )
            
            stats = result["stats"]
            
            # Check all required stats are present
            assert "duration_ms" in stats
            assert "strategy_used" in stats
            assert "fallback" in stats
            assert "chunk_count" in stats
            
            # Verify values
            assert stats["duration_ms"] >= 50  # At least 50ms due to sleep
            assert stats["strategy_used"] == "TokenChunker"
            assert stats["fallback"] is False
            assert stats["chunk_count"] == 10

    @pytest.mark.asyncio
    async def test_concurrent_chunking_requests(self, service):
        """Test that multiple concurrent chunking requests work correctly."""
        collection = {
            "id": "coll-concurrent",
            "name": "Concurrent Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 100,
            "chunk_overlap": 20,
        }
        
        mock_strategy = MagicMock()
        
        # Different results for different documents
        def create_chunks(content, config):
            doc_id = content[:10]  # Extract doc ID from content
            return [MagicMock(content=f"{doc_id} chunk {i}") for i in range(3)]
        
        mock_strategy.chunk = create_chunks
        
        with patch.object(service.strategy_factory, 'create_strategy', return_value=mock_strategy):
            # Run multiple concurrent chunking operations
            tasks = [
                service.execute_ingestion_chunking(
                    text=f"doc-{i} content for concurrent test",
                    document_id=f"doc-{i}",
                    collection=collection,
                )
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verify each result is correct
            for i, result in enumerate(results):
                assert result["stats"]["chunk_count"] == 3
                assert result["chunks"][0]["chunk_id"] == f"doc-{i}_chunk_0000"
                assert f"doc-{i}" in result["chunks"][0]["text"]

    @pytest.mark.asyncio
    async def test_unicode_and_special_characters_in_chunk_ids(self, service):
        """Test that document IDs with special characters are handled correctly."""
        special_doc_ids = [
            "doc-with-dash",
            "doc_with_underscore",
            "doc.with.dots",
            "doc/with/slashes",
            "doc\\with\\backslashes",
            "doc with spaces",
            "doc-123-αβγ",  # Greek letters
            "doc-中文",  # Chinese characters
        ]
        
        collection = {
            "id": "coll-special-ids",
            "name": "Special IDs Collection",
            "chunk_size": 100,
            "chunk_overlap": 20,
        }
        
        for doc_id in special_doc_ids:
            with patch('shared.text_processing.chunking.TokenChunker') as MockTokenChunker:
                mock_chunker = MagicMock()
                MockTokenChunker.return_value = mock_chunker
                mock_chunker.chunk_text.return_value = [
                    {"chunk_id": f"{doc_id}_chunk_0000", "text": "chunk", "metadata": {}}
                ]
                
                result = await service.execute_ingestion_chunking(
                    text="Test text",
                    document_id=doc_id,
                    collection=collection,
                )
                
                # Chunk ID should preserve the original document ID
                assert result["chunks"][0]["chunk_id"] == f"{doc_id}_chunk_0000"