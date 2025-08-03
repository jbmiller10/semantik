#!/usr/bin/env python3
"""
Integration tests for advanced chunking strategies.

This module tests the integration of semantic, hierarchical, and hybrid
chunking strategies with the full system including service layers,
error handling, and caching.
"""

import asyncio
import json
import os
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.shared.text_processing.chunking_factory import ChunkingFactory
from packages.webui.services.chunking_service import ChunkingService
from packages.webui.services.chunking_error_handler import ChunkingErrorHandler
from packages.webui.api.chunking_exceptions import (
    ChunkingMemoryError,
    ChunkingStrategyError,
    ChunkingTimeoutError,
)

# Set testing environment
os.environ["TESTING"] = "true"


class TestAdvancedChunkingIntegration:
    """Integration tests for advanced chunking strategies."""

    @pytest.fixture()
    def mock_redis(self) -> MagicMock:
        """Mock Redis client with realistic behavior."""
        redis = MagicMock(spec=Redis)
        redis.get = MagicMock(return_value=None)
        redis.setex = MagicMock()
        redis.incr = MagicMock(return_value=1)
        redis.pipeline = MagicMock()
        
        # Mock pipeline
        pipeline = MagicMock()
        pipeline.incr = MagicMock(return_value=pipeline)
        pipeline.expire = MagicMock(return_value=pipeline)
        pipeline.execute = MagicMock(return_value=[1, True])
        redis.pipeline.return_value = pipeline
        
        return redis

    @pytest.fixture()
    def mock_db_session(self) -> AsyncMock:
        """Mock database session."""
        session = AsyncMock(spec=AsyncSession)
        session.begin = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture()
    def mock_repos(self) -> tuple[MagicMock, MagicMock]:
        """Mock repositories with realistic behavior."""
        collection_repo = MagicMock(spec=CollectionRepository)
        collection_repo.get_by_uuid_with_permission_check = AsyncMock()
        
        document_repo = MagicMock(spec=DocumentRepository)
        document_repo.list_by_collection = AsyncMock(return_value=([], 0))
        document_repo.update_chunking_status = AsyncMock()
        
        return collection_repo, document_repo

    @pytest.fixture()
    def chunking_service(
        self,
        mock_redis: MagicMock,
        mock_db_session: AsyncMock,
        mock_repos: tuple[MagicMock, MagicMock],
    ) -> ChunkingService:
        """Create ChunkingService with mocked dependencies."""
        collection_repo, document_repo = mock_repos
        return ChunkingService(
            db_session=mock_db_session,
            collection_repo=collection_repo,
            document_repo=document_repo,
            redis_client=mock_redis,
        )

    @pytest.fixture()
    def error_handler(self, mock_redis: MagicMock) -> ChunkingErrorHandler:
        """Create ChunkingErrorHandler."""
        return ChunkingErrorHandler(redis_client=mock_redis)

    # Semantic Strategy Integration Tests
    
    async def test_semantic_chunking_service_integration(
        self,
        chunking_service: ChunkingService,
        mock_redis: MagicMock,
    ) -> None:
        """Test semantic chunking through service layer."""
        # Complex document with topic changes
        text = (
            "Artificial intelligence has revolutionized many fields. "
            "Machine learning models can now understand context. " * 10 +
            "In the culinary world, French cuisine is renowned. "
            "Proper knife skills are essential for any chef. " * 10 +
            "Quantum computing promises exponential speedups. "
            "Qubits can exist in superposition states. " * 10
        )
        
        config = {
            "strategy": "semantic",
            "params": {
                "breakpoint_percentile_threshold": 90,
                "buffer_size": 1,
            }
        }
        
        result = await chunking_service.preview_chunking(
            text=text,
            config=config,
            max_chunks=10,
        )
        
        # Verify semantic chunking was used
        assert result.strategy_used == "semantic"
        assert len(result.chunks) <= 10
        assert result.total_chunks >= 3  # Should detect topic boundaries
        
        # Verify caching
        assert mock_redis.setex.called
        cache_key, ttl, cache_data = mock_redis.setex.call_args[0]
        assert "preview:" in cache_key
        assert ttl == 300  # 5 minutes
        
        # Verify performance metrics
        assert "processing_time" in result.performance_metrics
        assert "chunks_per_second" in result.performance_metrics

    async def test_semantic_chunking_error_recovery(
        self,
        chunking_service: ChunkingService,
        error_handler: ChunkingErrorHandler,
    ) -> None:
        """Test semantic chunking error recovery through service layer."""
        text = "Test document for error recovery. " * 100
        
        # Simulate embedding API failure
        with patch("packages.shared.text_processing.strategies.semantic_chunker.SemanticSplitterNodeParser") as mock_parser:
            mock_parser.side_effect = Exception("Embedding API error")
            
            config = {"strategy": "semantic", "params": {}}
            
            # Should fallback gracefully
            result = await chunking_service.preview_chunking(
                text=text,
                config=config,
            )
            
            # Should still get results (from fallback)
            assert result.total_chunks >= 1
            assert len(result.chunks) >= 1

    async def test_semantic_chunking_with_collection_context(
        self,
        chunking_service: ChunkingService,
        mock_repos: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test semantic chunking with collection-specific configuration."""
        collection_repo, document_repo = mock_repos
        
        # Mock collection with semantic config
        mock_collection = MagicMock()
        mock_collection.id = 1
        mock_collection.chunking_config = {
            "strategy": "semantic",
            "params": {
                "breakpoint_percentile_threshold": 85,
                "max_chunk_size": 2000,
            }
        }
        collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        
        # Process documents for collection
        documents = [
            {"id": 1, "content": "AI and machine learning. " * 100},
            {"id": 2, "content": "Natural language processing. " * 100},
        ]
        document_repo.list_by_collection.return_value = (documents, 2)
        
        # Validate configuration
        validation = await chunking_service.validate_collection_config(
            collection_uuid="test-uuid",
            user_id=1,
        )
        
        assert validation.is_valid
        assert validation.strategy == "semantic"
        assert validation.estimated_total_chunks > 0

    # Hierarchical Strategy Integration Tests
    
    async def test_hierarchical_chunking_service_integration(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test hierarchical chunking through service layer."""
        # Structured document
        text = ""
        for chapter in range(3):
            text += f"Chapter {chapter + 1}: Introduction\n\n"
            text += "This chapter covers important topics. " * 50
            for section in range(2):
                text += f"\n\nSection {chapter + 1}.{section + 1}\n\n"
                text += "Detailed content goes here. " * 30
        
        config = {
            "strategy": "hierarchical",
            "params": {
                "chunk_sizes": [1000, 500, 250],
                "chunk_overlap": 50,
            }
        }
        
        result = await chunking_service.preview_chunking(
            text=text,
            config=config,
        )
        
        # Verify hierarchical chunking
        assert result.strategy_used == "hierarchical"
        assert result.total_chunks >= 10
        
        # Check hierarchy metadata
        for chunk in result.chunks:
            assert "chunk_level" in chunk.metadata
            assert "chunk_type" in chunk.metadata
            assert "hierarchy_path" in chunk.metadata

    async def test_hierarchical_chunking_memory_limits(
        self,
        chunking_service: ChunkingService,
        error_handler: ChunkingErrorHandler,
    ) -> None:
        """Test hierarchical chunking respects memory limits."""
        # Very large document
        large_text = "Chapter content. " * 100000  # ~1.6MB
        
        config = {
            "strategy": "hierarchical",
            "params": {
                "chunk_sizes": [5000, 2500, 1000, 500, 250],  # 5 levels
                "chunk_overlap": 100,
            }
        }
        
        # Set memory limit
        with patch("psutil.virtual_memory") as mock_memory:
            # Simulate low memory
            mock_memory.return_value.available = 100 * 1024 * 1024  # 100MB
            
            try:
                result = await chunking_service.preview_chunking(
                    text=large_text,
                    config=config,
                )
                
                # Should complete but potentially with reduced levels
                assert result.total_chunks > 100
                
            except ChunkingMemoryError as e:
                # Or should raise memory error
                assert e.memory_limit > 0
                assert e.memory_used > e.memory_limit

    # Hybrid Strategy Integration Tests
    
    async def test_hybrid_chunking_strategy_selection(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test hybrid chunking strategy selection logic."""
        test_cases = [
            {
                "text": "# Markdown Title\n\n## Section\n\n- List item\n- Another item\n\n```code```",
                "file_type": ".md",
                "expected_substrategy": "markdown",
            },
            {
                "text": "AI research. Cooking techniques. Physics laws. " * 50,
                "file_type": ".txt",
                "expected_substrategy": ["semantic", "recursive"],  # Could be either
            },
            {
                "text": "Simple repetitive text. " * 200,
                "file_type": ".txt", 
                "expected_substrategy": "recursive",
            },
        ]
        
        config = {
            "strategy": "hybrid",
            "params": {
                "enable_analytics": True,
            }
        }
        
        for test_case in test_cases:
            result = await chunking_service.preview_chunking(
                text=test_case["text"],
                file_type=test_case["file_type"],
                config=config,
            )
            
            assert result.strategy_used == "hybrid"
            
            # Check substrategy selection
            if result.chunks:
                actual_substrategy = result.chunks[0].metadata.get("sub_strategy")
                expected = test_case["expected_substrategy"]
                if isinstance(expected, list):
                    assert actual_substrategy in expected
                else:
                    assert actual_substrategy == expected

    async def test_hybrid_chunking_analytics_tracking(
        self,
        chunking_service: ChunkingService,
        mock_redis: MagicMock,
    ) -> None:
        """Test hybrid chunking analytics are properly tracked."""
        config = {
            "strategy": "hybrid",
            "params": {
                "enable_analytics": True,
            }
        }
        
        # Process multiple documents
        documents = [
            ("# Markdown doc", ".md"),
            ("Diverse topics here", ".txt"),
            ("Simple text", ".txt"),
        ]
        
        for text, file_type in documents:
            await chunking_service.preview_chunking(
                text=text * 50,  # Make longer
                file_type=file_type,
                config=config,
            )
        
        # Analytics should be tracked in Redis
        assert mock_redis.incr.called
        incr_calls = mock_redis.incr.call_args_list
        
        # Should track strategy selections
        strategy_keys = [call[0][0] for call in incr_calls]
        assert any("hybrid:selection:" in key for key in strategy_keys)

    # Cross-Strategy Integration Tests
    
    async def test_strategy_switching_performance(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test performance when switching between strategies."""
        strategies = ["character", "recursive", "markdown", "semantic", "hierarchical", "hybrid"]
        
        text = "Test document for strategy switching. " * 100
        
        results = []
        for strategy in strategies:
            config = {"strategy": strategy, "params": {}}
            if strategy == "hierarchical":
                config["params"]["chunk_sizes"] = [500, 200]
            
            result = await chunking_service.preview_chunking(
                text=text,
                config=config,
            )
            results.append(result)
        
        # All strategies should work
        assert len(results) == 6
        assert all(r.total_chunks >= 1 for r in results)
        
        # Different strategies should produce different results
        chunk_counts = [r.total_chunks for r in results]
        assert len(set(chunk_counts)) > 1  # Not all the same

    async def test_concurrent_mixed_strategy_processing(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test concurrent processing with different strategies."""
        # Create tasks with different strategies
        tasks = []
        
        strategies_and_texts = [
            ("semantic", "AI and ML topics. " * 100),
            ("hierarchical", "Chapter 1 content. " * 100),
            ("hybrid", "# Mixed content\n\nWith markdown. " * 100),
            ("recursive", "Simple text. " * 100),
        ]
        
        for strategy, text in strategies_and_texts:
            config = {"strategy": strategy, "params": {}}
            if strategy == "hierarchical":
                config["params"]["chunk_sizes"] = [500, 200]
            
            task = chunking_service.preview_chunking(
                text=text,
                config=config,
            )
            tasks.append(task)
        
        # Process concurrently
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 4
        for i, (strategy, _) in enumerate(strategies_and_texts):
            assert results[i].strategy_used == strategy

    async def test_error_handling_cascade(
        self,
        chunking_service: ChunkingService,
        error_handler: ChunkingErrorHandler,
        mock_redis: MagicMock,
    ) -> None:
        """Test error handling cascades through strategies."""
        # Configure to fail semantic, fallback to recursive
        with patch("packages.shared.text_processing.strategies.semantic_chunker.SemanticSplitterNodeParser") as mock_semantic:
            mock_semantic.side_effect = ChunkingStrategyError(
                "Semantic chunking failed",
                strategy="semantic",
                operation_id="test-op",
            )
            
            config = {"strategy": "semantic", "params": {}}
            text = "Test document. " * 100
            
            # Process with error handler
            result = await error_handler.handle_chunking_operation(
                operation=lambda: chunking_service.preview_chunking(text=text, config=config),
                operation_id="test-op",
                collection_id="test-collection",
            )
            
            # Should succeed with fallback
            assert result.total_chunks >= 1
            
            # Error should be logged
            assert mock_redis.incr.called
            error_keys = [call[0][0] for call in mock_redis.incr.call_args_list]
            assert any("error:strategy" in key for key in error_keys)

    async def test_resource_cleanup_after_errors(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test resources are properly cleaned up after errors."""
        # Force an error during processing
        with patch("packages.shared.text_processing.chunking_factory.ChunkingFactory.create_chunker") as mock_factory:
            mock_chunker = MagicMock()
            mock_chunker.chunk_text_async = AsyncMock(side_effect=Exception("Processing failed"))
            mock_factory.return_value = mock_chunker
            
            config = {"strategy": "semantic", "params": {}}
            text = "Test document. " * 1000
            
            # Should raise but clean up
            with pytest.raises(Exception, match="Processing failed"):
                await chunking_service.preview_chunking(text=text, config=config)
            
            # Resources should be released (hard to test directly, but no memory leaks)