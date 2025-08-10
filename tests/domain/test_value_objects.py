#!/usr/bin/env python3

"""Tests for domain value objects."""

from datetime import UTC, datetime

import pytest

from packages.shared.chunking.domain.exceptions import InvalidConfigurationError
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from packages.shared.chunking.domain.value_objects.operation_status import OperationStatus


class TestChunkConfig:
    """Test suite for ChunkConfig value object."""

    def test_valid_configuration(self) -> None:
        """Test creating a valid chunk configuration."""
        # Act
        config = ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=5)

        # Assert
        assert config.strategy_name == "character"
        assert config.min_tokens == 10
        assert config.max_tokens == 100
        assert config.overlap_tokens == 5

    def test_configuration_with_additional_params(self) -> None:
        """Test configuration with additional parameters."""
        # Act
        config = ChunkConfig(
            strategy_name="semantic",
            min_tokens=50,
            max_tokens=200,
            overlap_tokens=20,
            similarity_threshold=0.8,
            encoding="utf-8",  # Use an allowed parameter from the whitelist
            language="en",
        )  # Another allowed parameter

        # Assert
        assert config.strategy_name == "semantic"
        assert config.additional_params["similarity_threshold"] == 0.8
        assert config.additional_params["encoding"] == "utf-8"
        assert config.additional_params["language"] == "en"

    def test_unknown_parameter_rejected(self) -> None:
        """Test that unknown parameters are rejected for security."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(
                strategy_name="semantic", min_tokens=50, max_tokens=200, overlap_tokens=20, unknown_param="value"
            )  # This should be rejected

        assert "Unknown configuration parameter 'unknown_param'" in str(exc_info.value)
        assert "Allowed additional parameters" in str(exc_info.value)

    def test_invalid_min_tokens_negative(self) -> None:
        """Test that negative min_tokens raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(strategy_name="character", min_tokens=-1, max_tokens=100, overlap_tokens=5)

        assert "min_tokens must be positive" in str(exc_info.value)

    def test_invalid_min_tokens_zero(self) -> None:
        """Test that zero min_tokens raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(strategy_name="character", min_tokens=0, max_tokens=100, overlap_tokens=5)

        assert "min_tokens must be positive" in str(exc_info.value)

    def test_invalid_max_tokens_negative(self) -> None:
        """Test that negative max_tokens raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=-1, overlap_tokens=5)

        assert "max_tokens must be positive" in str(exc_info.value)

    def test_invalid_min_greater_than_max(self) -> None:
        """Test that min_tokens > max_tokens raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(strategy_name="character", min_tokens=100, max_tokens=50, overlap_tokens=5)

        assert "min_tokens cannot be greater than max_tokens" in str(exc_info.value)

    def test_invalid_overlap_negative(self) -> None:
        """Test that negative overlap_tokens raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=-1)

        assert "overlap_tokens cannot be negative" in str(exc_info.value)

    def test_invalid_overlap_greater_than_min(self) -> None:
        """Test that overlap_tokens >= min_tokens raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=10)

        assert "overlap_tokens must be less than min_tokens" in str(exc_info.value)

    def test_invalid_overlap_greater_than_min_excessive(self) -> None:
        """Test that overlap_tokens > min_tokens raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=15)

        assert "overlap_tokens must be less than min_tokens" in str(exc_info.value)

    def test_empty_strategy_name(self) -> None:
        """Test that empty strategy name raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(strategy_name="", min_tokens=10, max_tokens=100, overlap_tokens=5)

        assert "strategy_name cannot be empty" in str(exc_info.value)

    def test_estimate_chunks_basic(self) -> None:
        """Test basic chunk estimation."""
        # Arrange
        config = ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=5)

        # Act
        estimated = config.estimate_chunks(1000)  # 1000 tokens

        # Assert
        # With 1000 tokens and max_tokens=100, minimum would be 10 chunks
        # But with overlap, we need more chunks
        assert estimated >= 10
        assert estimated <= 100  # Reasonable upper bound

    def test_estimate_chunks_no_overlap(self) -> None:
        """Test chunk estimation without overlap."""
        # Arrange
        config = ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=0)

        # Act
        estimated = config.estimate_chunks(1000)

        # Assert
        # With no overlap, 1000 tokens / 100 max = 10 chunks minimum
        assert estimated >= 10

    def test_estimate_chunks_small_document(self) -> None:
        """Test chunk estimation for small document."""
        # Arrange
        config = ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=5)

        # Act
        estimated = config.estimate_chunks(50)  # Small document

        # Assert
        assert estimated >= 1
        assert estimated <= 5

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        # Arrange
        config = ChunkConfig(
            strategy_name="semantic", min_tokens=50, max_tokens=200, overlap_tokens=20, similarity_threshold=0.8
        )

        # Act
        config_dict = config.to_dict()

        # Assert
        assert config_dict["strategy_name"] == "semantic"
        assert config_dict["min_tokens"] == 50
        assert config_dict["max_tokens"] == 200
        assert config_dict["overlap_tokens"] == 20
        assert config_dict["similarity_threshold"] == 0.8

    def test_equality(self) -> None:
        """Test value object equality."""
        # Arrange
        config1 = ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=5)
        config2 = ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=5)
        config3 = ChunkConfig(
            strategy_name="character", min_tokens=10, max_tokens=200, overlap_tokens=5  # Different max_tokens
        )

        # Assert
        assert config1 == config2
        assert config1 != config3
        assert config1 != "not a config"


class TestOperationStatus:
    """Test suite for OperationStatus value object."""

    def test_all_statuses_defined(self) -> None:
        """Test that all expected statuses are defined."""
        # Assert
        assert OperationStatus.PENDING.value == "PENDING"
        assert OperationStatus.PROCESSING.value == "PROCESSING"
        assert OperationStatus.COMPLETED.value == "COMPLETED"
        assert OperationStatus.FAILED.value == "FAILED"
        assert OperationStatus.CANCELLED.value == "CANCELLED"

    def test_is_terminal_states(self) -> None:
        """Test identification of terminal states."""
        # Assert
        assert not OperationStatus.PENDING.is_terminal()
        assert not OperationStatus.PROCESSING.is_terminal()
        assert OperationStatus.COMPLETED.is_terminal()
        assert OperationStatus.FAILED.is_terminal()
        assert OperationStatus.CANCELLED.is_terminal()

    def test_valid_transitions_from_pending(self) -> None:
        """Test valid transitions from PENDING state."""
        # Assert
        assert OperationStatus.PENDING.can_transition_to(OperationStatus.PROCESSING)
        assert OperationStatus.PENDING.can_transition_to(OperationStatus.CANCELLED)
        assert not OperationStatus.PENDING.can_transition_to(OperationStatus.COMPLETED)
        assert not OperationStatus.PENDING.can_transition_to(OperationStatus.FAILED)

    def test_valid_transitions_from_processing(self) -> None:
        """Test valid transitions from PROCESSING state."""
        # Assert
        assert not OperationStatus.PROCESSING.can_transition_to(OperationStatus.PENDING)
        assert OperationStatus.PROCESSING.can_transition_to(OperationStatus.COMPLETED)
        assert OperationStatus.PROCESSING.can_transition_to(OperationStatus.FAILED)
        assert OperationStatus.PROCESSING.can_transition_to(OperationStatus.CANCELLED)

    def test_no_transitions_from_completed(self) -> None:
        """Test that no transitions are allowed from COMPLETED state."""
        # Assert
        for status in OperationStatus:
            if status != OperationStatus.COMPLETED:
                assert not OperationStatus.COMPLETED.can_transition_to(status)

    def test_no_transitions_from_failed(self) -> None:
        """Test that no transitions are allowed from FAILED state."""
        # Assert
        for status in OperationStatus:
            if status != OperationStatus.FAILED:
                assert not OperationStatus.FAILED.can_transition_to(status)

    def test_no_transitions_from_cancelled(self) -> None:
        """Test that no transitions are allowed from CANCELLED state."""
        # Assert
        for status in OperationStatus:
            if status != OperationStatus.CANCELLED:
                assert not OperationStatus.CANCELLED.can_transition_to(status)

    def test_self_transition_not_allowed(self) -> None:
        """Test that self-transitions are not allowed."""
        # Assert
        for status in OperationStatus:
            assert not status.can_transition_to(status)

    def test_string_representation(self) -> None:
        """Test string representation of status."""
        # Assert
        assert str(OperationStatus.PENDING) == "OperationStatus.PENDING"
        assert str(OperationStatus.PROCESSING) == "OperationStatus.PROCESSING"
        assert str(OperationStatus.COMPLETED) == "OperationStatus.COMPLETED"


class TestChunkMetadata:
    """Test suite for ChunkMetadata value object."""

    def test_basic_metadata_creation(self) -> None:
        """Test creating basic chunk metadata."""
        # Act
        metadata = ChunkMetadata(
            chunk_id="chunk-1",
            document_id="doc-123",
            chunk_index=0,
            start_offset=0,
            end_offset=50,
            token_count=50,
            strategy_name="character",
        )

        # Assert
        assert metadata.chunk_id == "chunk-1"
        assert metadata.document_id == "doc-123"
        assert metadata.chunk_index == 0
        assert metadata.start_offset == 0
        assert metadata.end_offset == 50
        assert metadata.token_count == 50
        assert metadata.strategy_name == "character"
        assert metadata.semantic_score is None
        assert metadata.hierarchy_level is None
        assert metadata.section_title is None

    def test_full_metadata_creation(self) -> None:
        """Test creating metadata with all fields."""

        # Act
        now = datetime.now(UTC)
        metadata = ChunkMetadata(
            chunk_id="chunk-full",
            document_id="doc-456",
            chunk_index=2,
            start_offset=100,
            end_offset=200,
            token_count=100,
            strategy_name="semantic",
            semantic_score=0.75,
            hierarchy_level=2,
            section_title="Introduction",
            created_at=now,
        )

        # Assert
        assert metadata.chunk_id == "chunk-full"
        assert metadata.document_id == "doc-456"
        assert metadata.chunk_index == 2
        assert metadata.start_offset == 100
        assert metadata.end_offset == 200
        assert metadata.token_count == 100
        assert metadata.strategy_name == "semantic"
        assert metadata.semantic_score == 0.75
        assert metadata.hierarchy_level == 2
        assert metadata.section_title == "Introduction"
        assert metadata.created_at == now

    def test_invalid_token_count_negative(self) -> None:
        """Test that negative token count raises error."""
        # Act & Assert
        with pytest.raises(ValueError, match="Token count must be positive"):
            ChunkMetadata(
                chunk_id="test",
                document_id="doc",
                chunk_index=0,
                start_offset=0,
                end_offset=10,
                token_count=-1,
                strategy_name="test",
            )

    def test_invalid_offsets(self) -> None:
        """Test that invalid offsets raise error."""
        # Act & Assert - end offset <= start offset
        with pytest.raises(ValueError, match="End offset.*must be greater than start offset"):
            ChunkMetadata(
                chunk_id="test",
                document_id="doc",
                chunk_index=0,
                start_offset=10,
                end_offset=5,  # Less than start
                token_count=10,
                strategy_name="test",
            )

    def test_invalid_semantic_score_range(self) -> None:
        """Test that semantic score outside 0-1 range raises error."""
        # Act & Assert - semantic score above 1
        with pytest.raises(ValueError, match="Semantic score must be between 0.0 and 1.0"):
            ChunkMetadata(
                chunk_id="test",
                document_id="doc",
                chunk_index=0,
                start_offset=0,
                end_offset=10,
                token_count=10,
                strategy_name="test",
                semantic_score=1.5,
            )

    def test_negative_start_offset(self) -> None:
        """Test that negative start offset raises error."""
        # Act & Assert
        with pytest.raises(ValueError, match="Start offset must be non-negative"):
            ChunkMetadata(
                chunk_id="test",
                document_id="doc",
                chunk_index=0,
                start_offset=-1,  # Negative
                end_offset=10,
                token_count=10,
                strategy_name="test",
            )

    def test_negative_chunk_index(self) -> None:
        """Test that negative chunk index raises error."""
        # Act & Assert
        with pytest.raises(ValueError, match="Chunk index must be non-negative"):
            ChunkMetadata(
                chunk_id="test",
                document_id="doc",
                chunk_index=-1,  # Negative
                start_offset=0,
                end_offset=10,
                token_count=10,
                strategy_name="test",
            )

    def test_character_count_property(self) -> None:
        """Test the character_count property."""
        # Arrange
        metadata = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            chunk_index=0,
            start_offset=10,
            end_offset=60,
            token_count=10,
            strategy_name="test",
        )

        # Assert
        assert metadata.character_count == 50

    def test_average_token_length_property(self) -> None:
        """Test the average_token_length property."""
        # Arrange
        metadata = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            chunk_index=0,
            start_offset=0,
            end_offset=100,
            token_count=20,
            strategy_name="test",
        )

        # Assert
        assert metadata.average_token_length == 5.0

    def test_overlaps_with_method(self) -> None:
        """Test the overlaps_with method."""
        # Arrange
        metadata1 = ChunkMetadata(
            chunk_id="chunk-1",
            document_id="doc-123",
            chunk_index=0,
            start_offset=0,
            end_offset=50,
            token_count=10,
            strategy_name="test",
        )
        metadata2 = ChunkMetadata(
            chunk_id="chunk-2",
            document_id="doc-123",
            chunk_index=1,
            start_offset=40,  # Overlaps with metadata1
            end_offset=90,
            token_count=10,
            strategy_name="test",
        )
        metadata3 = ChunkMetadata(
            chunk_id="chunk-3",
            document_id="doc-123",
            chunk_index=2,
            start_offset=60,  # No overlap with metadata1
            end_offset=100,
            token_count=8,
            strategy_name="test",
        )
        metadata4 = ChunkMetadata(
            chunk_id="chunk-4",
            document_id="doc-456",  # Different document
            chunk_index=0,
            start_offset=0,
            end_offset=50,
            token_count=10,
            strategy_name="test",
        )

        # Assert
        assert metadata1.overlaps_with(metadata2)
        assert not metadata1.overlaps_with(metadata3)
        assert not metadata1.overlaps_with(metadata4)  # Different document

    def test_overlap_size_method(self) -> None:
        """Test the overlap_size method."""
        # Arrange
        metadata1 = ChunkMetadata(
            chunk_id="chunk-1",
            document_id="doc-123",
            chunk_index=0,
            start_offset=0,
            end_offset=50,
            token_count=10,
            strategy_name="test",
        )
        metadata2 = ChunkMetadata(
            chunk_id="chunk-2",
            document_id="doc-123",
            chunk_index=1,
            start_offset=30,  # 20 character overlap
            end_offset=80,
            token_count=10,
            strategy_name="test",
        )
        metadata3 = ChunkMetadata(
            chunk_id="chunk-3",
            document_id="doc-123",
            chunk_index=2,
            start_offset=60,  # No overlap
            end_offset=100,
            token_count=8,
            strategy_name="test",
        )

        # Assert
        assert metadata1.overlap_size(metadata2) == 20
        assert metadata1.overlap_size(metadata3) == 0

    def test_immutability(self) -> None:
        """Test that metadata is immutable (frozen dataclass)."""
        # Arrange
        metadata = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            chunk_index=0,
            start_offset=0,
            end_offset=50,
            token_count=10,
            strategy_name="test",
        )

        # Act & Assert - attributes should not be settable
        with pytest.raises(AttributeError):
            metadata.token_count = 100

        with pytest.raises(AttributeError):
            metadata.chunk_id = "new-id"

    def test_equality(self) -> None:
        """Test value object equality."""
        # Arrange
        metadata1 = ChunkMetadata(
            chunk_id="chunk-1",
            document_id="doc-123",
            chunk_index=0,
            start_offset=0,
            end_offset=50,
            token_count=50,
            strategy_name="character",
            semantic_score=0.8,
        )
        metadata2 = ChunkMetadata(
            chunk_id="chunk-1",
            document_id="doc-123",
            chunk_index=0,
            start_offset=0,
            end_offset=50,
            token_count=50,
            strategy_name="character",
            semantic_score=0.8,
        )
        metadata3 = ChunkMetadata(
            chunk_id="chunk-2",  # Different ID
            document_id="doc-123",
            chunk_index=0,
            start_offset=0,
            end_offset=50,
            token_count=50,
            strategy_name="character",
            semantic_score=0.8,
        )

        # Assert
        assert metadata1 == metadata2
        assert metadata1 != metadata3
        assert metadata1 != "not metadata"

    def test_hierarchy_level_validation(self) -> None:
        """Test that negative hierarchy level raises error."""
        # Act & Assert
        with pytest.raises(ValueError, match="Hierarchy level must be non-negative"):
            ChunkMetadata(
                chunk_id="test",
                document_id="doc",
                chunk_index=0,
                start_offset=0,
                end_offset=10,
                token_count=10,
                strategy_name="test",
                hierarchy_level=-1,  # Negative hierarchy level
            )
