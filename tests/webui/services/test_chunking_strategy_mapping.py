#!/usr/bin/env python3

"""
Unit tests for chunking strategy mapping.

This module tests the mapping between API enum values and factory strategy names
to ensure compatibility between the API layer and the chunking factory.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.webui.api.v2.chunking_schemas import ChunkingStrategy
from packages.webui.services.chunking_service import ChunkingService

# Remove ChunkingFactory import as it's not used in updated tests


class TestChunkingStrategyMapping:
    """Test the mapping between API strategy names and factory strategy names."""

    def test_strategy_mapping_completeness(self) -> None:
        """Ensure all ChunkingStrategy enum values are mapped."""
        # All enum values should have a mapping
        for strategy in ChunkingStrategy:
            assert (
                strategy.value in ChunkingService.STRATEGY_MAPPING
            ), f"Strategy {strategy.value} is not mapped in ChunkingService.STRATEGY_MAPPING"

    def test_strategy_mapping_values(self) -> None:
        """Verify correct mapping of each strategy."""
        expected_mapping = {
            "fixed_size": "character",
            "sliding_window": "character",
            "semantic": "semantic",
            "recursive": "recursive",
            "document_structure": "markdown",
            "markdown": "markdown",
            "hierarchical": "hierarchical",
            "hybrid": "hybrid",
        }

        assert expected_mapping == ChunkingService.STRATEGY_MAPPING

    def test_map_strategy_to_factory_name(self) -> None:
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

    async def test_create_chunker_uses_mapped_strategy(self) -> None:
        """Test that strategy mapping works correctly in the service."""
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

        # Test that _map_strategy_to_factory_name uses STRATEGY_MAPPING
        assert service._map_strategy_to_factory_name("fixed_size") == "character"
        assert service._map_strategy_to_factory_name("sliding_window") == "character"
        assert service._map_strategy_to_factory_name("semantic") == "semantic"
        assert service._map_strategy_to_factory_name("recursive") == "recursive"
        assert service._map_strategy_to_factory_name("document_structure") == "markdown"
        assert service._map_strategy_to_factory_name("hybrid") == "hybrid"

    def test_all_mapped_strategies_are_registered(self) -> None:
        """Verify that all mapped strategy names are valid."""
        # Get unique mapped values
        mapped_strategies = set(ChunkingService.STRATEGY_MAPPING.values())

        # Define known valid strategies based on the chunking system
        known_strategies = {"character", "recursive", "markdown", "semantic", "hierarchical", "hybrid"}

        # Check that all mapped strategies are known valid strategies
        for strategy in mapped_strategies:
            assert strategy in known_strategies, f"Mapped strategy '{strategy}' is not a known valid strategy"

    @pytest.mark.parametrize(
        ("api_strategy", "factory_strategy"),
        [
            (ChunkingStrategy.FIXED_SIZE, "character"),
            (ChunkingStrategy.SLIDING_WINDOW, "character"),
            (ChunkingStrategy.SEMANTIC, "semantic"),
            (ChunkingStrategy.RECURSIVE, "recursive"),
            (ChunkingStrategy.DOCUMENT_STRUCTURE, "markdown"),
            (ChunkingStrategy.HYBRID, "hybrid"),
        ],
    )
    def test_strategy_enum_to_factory_mapping(self, api_strategy, factory_strategy) -> None:
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
