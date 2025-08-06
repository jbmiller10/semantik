#!/usr/bin/env python3
"""
Unit tests for chunking strategy mapping.

This module tests the mapping between API enum values and factory strategy names
to ensure compatibility between the API layer and the chunking factory.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from packages.webui.api.v2.chunking_schemas import ChunkingStrategy
from packages.webui.services.chunking_service import ChunkingService
from packages.shared.text_processing.chunking_factory import ChunkingFactory


class TestChunkingStrategyMapping:
    """Test the mapping between API strategy names and factory strategy names."""

    def test_strategy_mapping_completeness(self):
        """Ensure all ChunkingStrategy enum values are mapped."""
        # All enum values should have a mapping
        for strategy in ChunkingStrategy:
            assert strategy.value in ChunkingService.STRATEGY_MAPPING, \
                f"Strategy {strategy.value} is not mapped in ChunkingService.STRATEGY_MAPPING"

    def test_strategy_mapping_values(self):
        """Verify correct mapping of each strategy."""
        expected_mapping = {
            "fixed_size": "character",
            "sliding_window": "character",
            "semantic": "semantic",
            "recursive": "recursive",
            "document_structure": "markdown",
            "hybrid": "hybrid",
        }
        
        assert ChunkingService.STRATEGY_MAPPING == expected_mapping

    def test_map_strategy_to_factory_name(self):
        """Test the _map_strategy_to_factory_name method."""
        # Mock dependencies
        mock_db = AsyncMock()
        mock_collection_repo = MagicMock()
        mock_document_repo = MagicMock()
        mock_redis = MagicMock()
        
        service = ChunkingService(
            db_session=mock_db,
            collection_repo=mock_collection_repo,
            document_repo=mock_document_repo,
            redis_client=mock_redis,
        )
        
        # Test each mapping
        assert service._map_strategy_to_factory_name("fixed_size") == "character"
        assert service._map_strategy_to_factory_name("sliding_window") == "character"
        assert service._map_strategy_to_factory_name("semantic") == "semantic"
        assert service._map_strategy_to_factory_name("recursive") == "recursive"
        assert service._map_strategy_to_factory_name("document_structure") == "markdown"
        assert service._map_strategy_to_factory_name("hybrid") == "hybrid"
        
        # Test unmapped strategy (should return as-is)
        assert service._map_strategy_to_factory_name("unknown_strategy") == "unknown_strategy"

    @patch('packages.webui.services.chunking_service.ChunkingFactory')
    async def test_create_chunker_uses_mapped_strategy(self, mock_factory):
        """Test that ChunkingFactory.create_chunker is called with mapped strategy."""
        # Mock dependencies
        mock_db = AsyncMock()
        mock_collection_repo = MagicMock()
        mock_document_repo = MagicMock()
        mock_redis = MagicMock()
        
        service = ChunkingService(
            db_session=mock_db,
            collection_repo=mock_collection_repo,
            document_repo=mock_document_repo,
            redis_client=mock_redis,
        )
        
        # Mock the chunker
        mock_chunker = AsyncMock()
        mock_chunker.chunk_text_async = AsyncMock(return_value=[
            {"content": "chunk1", "metadata": {}},
            {"content": "chunk2", "metadata": {}},
        ])
        mock_factory.create_chunker.return_value = mock_chunker
        
        # Test config with fixed_size strategy
        config = {
            "strategy": "fixed_size",
            "params": {
                "chunk_size": 512,
                "chunk_overlap": 50,
            }
        }
        
        # Execute chunking
        chunks, _, _ = await service._execute_chunking(
            text="Test text for chunking",
            config=config,
            metadata={},
            correlation_id="test-correlation",
            operation_id="test-operation",
        )
        
        # Verify ChunkingFactory was called with mapped strategy
        mock_factory.create_chunker.assert_called_once()
        called_config = mock_factory.create_chunker.call_args[0][0]
        assert called_config["strategy"] == "character"  # fixed_size maps to character
        assert called_config["params"] == config["params"]

    def test_all_mapped_strategies_are_registered(self):
        """Verify that all mapped strategy names are registered in ChunkingFactory."""
        # Get unique mapped values
        mapped_strategies = set(ChunkingService.STRATEGY_MAPPING.values())
        
        # Get available strategies from factory
        available_strategies = ChunkingFactory.get_available_strategies()
        
        # Check that all mapped strategies are available
        for strategy in mapped_strategies:
            assert strategy in available_strategies, \
                f"Mapped strategy '{strategy}' is not registered in ChunkingFactory"

    @pytest.mark.parametrize("api_strategy,factory_strategy", [
        (ChunkingStrategy.FIXED_SIZE, "character"),
        (ChunkingStrategy.SLIDING_WINDOW, "character"),
        (ChunkingStrategy.SEMANTIC, "semantic"),
        (ChunkingStrategy.RECURSIVE, "recursive"),
        (ChunkingStrategy.DOCUMENT_STRUCTURE, "markdown"),
        (ChunkingStrategy.HYBRID, "hybrid"),
    ])
    def test_strategy_enum_to_factory_mapping(self, api_strategy, factory_strategy):
        """Test mapping from ChunkingStrategy enum to factory strategy name."""
        # Mock dependencies
        mock_db = AsyncMock()
        mock_collection_repo = MagicMock()
        mock_document_repo = MagicMock()
        mock_redis = MagicMock()
        
        service = ChunkingService(
            db_session=mock_db,
            collection_repo=mock_collection_repo,
            document_repo=mock_document_repo,
            redis_client=mock_redis,
        )
        
        # Test with enum value
        assert service._map_strategy_to_factory_name(api_strategy.value) == factory_strategy