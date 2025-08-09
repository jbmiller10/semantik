#!/usr/bin/env python3
"""Tests for domain value objects."""

import pytest

from packages.shared.chunking.domain.exceptions import InvalidConfigurationError
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from packages.shared.chunking.domain.value_objects.operation_status import OperationStatus


class TestChunkConfig:
    """Test suite for ChunkConfig value object."""

    def test_valid_configuration(self):
        """Test creating a valid chunk configuration."""
        # Act
        config = ChunkConfig(
            strategy_name="character",
            min_tokens=10,
            max_tokens=100,
            overlap_tokens=5,
        )

        # Assert
        assert config.strategy_name == "character"
        assert config.min_tokens == 10
        assert config.max_tokens == 100
        assert config.overlap_tokens == 5

    def test_configuration_with_additional_params(self):
        """Test configuration with additional parameters."""
        # Act
        config = ChunkConfig(
            strategy_name="semantic",
            min_tokens=50,
            max_tokens=200,
            overlap_tokens=20,
            similarity_threshold=0.8,
            custom_param="value",
        )

        # Assert
        assert config.strategy_name == "semantic"
        assert config.additional_params["similarity_threshold"] == 0.8
        assert config.additional_params["custom_param"] == "value"

    def test_invalid_min_tokens_negative(self):
        """Test that negative min_tokens raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(
                strategy_name="character",
                min_tokens=-1,
                max_tokens=100,
                overlap_tokens=5,
            )
        
        assert "min_tokens must be positive" in str(exc_info.value)

    def test_invalid_min_tokens_zero(self):
        """Test that zero min_tokens raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(
                strategy_name="character",
                min_tokens=0,
                max_tokens=100,
                overlap_tokens=5,
            )
        
        assert "min_tokens must be positive" in str(exc_info.value)

    def test_invalid_max_tokens_negative(self):
        """Test that negative max_tokens raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(
                strategy_name="character",
                min_tokens=10,
                max_tokens=-1,
                overlap_tokens=5,
            )
        
        assert "max_tokens must be positive" in str(exc_info.value)

    def test_invalid_min_greater_than_max(self):
        """Test that min_tokens > max_tokens raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(
                strategy_name="character",
                min_tokens=100,
                max_tokens=50,
                overlap_tokens=5,
            )
        
        assert "min_tokens cannot be greater than max_tokens" in str(exc_info.value)

    def test_invalid_overlap_negative(self):
        """Test that negative overlap_tokens raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(
                strategy_name="character",
                min_tokens=10,
                max_tokens=100,
                overlap_tokens=-1,
            )
        
        assert "overlap_tokens cannot be negative" in str(exc_info.value)

    def test_invalid_overlap_greater_than_min(self):
        """Test that overlap_tokens >= min_tokens raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(
                strategy_name="character",
                min_tokens=10,
                max_tokens=100,
                overlap_tokens=10,
            )
        
        assert "overlap_tokens must be less than min_tokens" in str(exc_info.value)

    def test_invalid_overlap_greater_than_min_excessive(self):
        """Test that overlap_tokens > min_tokens raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(
                strategy_name="character",
                min_tokens=10,
                max_tokens=100,
                overlap_tokens=15,
            )
        
        assert "overlap_tokens must be less than min_tokens" in str(exc_info.value)

    def test_empty_strategy_name(self):
        """Test that empty strategy name raises error."""
        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ChunkConfig(
                strategy_name="",
                min_tokens=10,
                max_tokens=100,
                overlap_tokens=5,
            )
        
        assert "strategy_name cannot be empty" in str(exc_info.value)

    def test_estimate_chunks_basic(self):
        """Test basic chunk estimation."""
        # Arrange
        config = ChunkConfig(
            strategy_name="character",
            min_tokens=10,
            max_tokens=100,
            overlap_tokens=5,
        )

        # Act
        estimated = config.estimate_chunks(1000)  # 1000 tokens

        # Assert
        # With 1000 tokens and max_tokens=100, minimum would be 10 chunks
        # But with overlap, we need more chunks
        assert estimated >= 10
        assert estimated <= 100  # Reasonable upper bound

    def test_estimate_chunks_no_overlap(self):
        """Test chunk estimation without overlap."""
        # Arrange
        config = ChunkConfig(
            strategy_name="character",
            min_tokens=10,
            max_tokens=100,
            overlap_tokens=0,
        )

        # Act
        estimated = config.estimate_chunks(1000)

        # Assert
        # With no overlap, 1000 tokens / 100 max = 10 chunks minimum
        assert estimated >= 10

    def test_estimate_chunks_small_document(self):
        """Test chunk estimation for small document."""
        # Arrange
        config = ChunkConfig(
            strategy_name="character",
            min_tokens=10,
            max_tokens=100,
            overlap_tokens=5,
        )

        # Act
        estimated = config.estimate_chunks(50)  # Small document

        # Assert
        assert estimated >= 1
        assert estimated <= 5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        # Arrange
        config = ChunkConfig(
            strategy_name="semantic",
            min_tokens=50,
            max_tokens=200,
            overlap_tokens=20,
            similarity_threshold=0.8,
        )

        # Act
        config_dict = config.to_dict()

        # Assert
        assert config_dict["strategy_name"] == "semantic"
        assert config_dict["min_tokens"] == 50
        assert config_dict["max_tokens"] == 200
        assert config_dict["overlap_tokens"] == 20
        assert config_dict["similarity_threshold"] == 0.8

    def test_equality(self):
        """Test value object equality."""
        # Arrange
        config1 = ChunkConfig(
            strategy_name="character",
            min_tokens=10,
            max_tokens=100,
            overlap_tokens=5,
        )
        config2 = ChunkConfig(
            strategy_name="character",
            min_tokens=10,
            max_tokens=100,
            overlap_tokens=5,
        )
        config3 = ChunkConfig(
            strategy_name="character",
            min_tokens=10,
            max_tokens=200,  # Different max_tokens
            overlap_tokens=5,
        )

        # Assert
        assert config1 == config2
        assert config1 != config3
        assert config1 != "not a config"


class TestOperationStatus:
    """Test suite for OperationStatus value object."""

    def test_all_statuses_defined(self):
        """Test that all expected statuses are defined."""
        # Assert
        assert OperationStatus.PENDING.value == "PENDING"
        assert OperationStatus.PROCESSING.value == "PROCESSING"
        assert OperationStatus.COMPLETED.value == "COMPLETED"
        assert OperationStatus.FAILED.value == "FAILED"
        assert OperationStatus.CANCELLED.value == "CANCELLED"

    def test_is_terminal_states(self):
        """Test identification of terminal states."""
        # Assert
        assert not OperationStatus.PENDING.is_terminal()
        assert not OperationStatus.PROCESSING.is_terminal()
        assert OperationStatus.COMPLETED.is_terminal()
        assert OperationStatus.FAILED.is_terminal()
        assert OperationStatus.CANCELLED.is_terminal()

    def test_valid_transitions_from_pending(self):
        """Test valid transitions from PENDING state."""
        # Assert
        assert OperationStatus.PENDING.can_transition_to(OperationStatus.PROCESSING)
        assert OperationStatus.PENDING.can_transition_to(OperationStatus.CANCELLED)
        assert not OperationStatus.PENDING.can_transition_to(OperationStatus.COMPLETED)
        assert not OperationStatus.PENDING.can_transition_to(OperationStatus.FAILED)

    def test_valid_transitions_from_processing(self):
        """Test valid transitions from PROCESSING state."""
        # Assert
        assert not OperationStatus.PROCESSING.can_transition_to(OperationStatus.PENDING)
        assert OperationStatus.PROCESSING.can_transition_to(OperationStatus.COMPLETED)
        assert OperationStatus.PROCESSING.can_transition_to(OperationStatus.FAILED)
        assert OperationStatus.PROCESSING.can_transition_to(OperationStatus.CANCELLED)

    def test_no_transitions_from_completed(self):
        """Test that no transitions are allowed from COMPLETED state."""
        # Assert
        for status in OperationStatus:
            if status != OperationStatus.COMPLETED:
                assert not OperationStatus.COMPLETED.can_transition_to(status)

    def test_no_transitions_from_failed(self):
        """Test that no transitions are allowed from FAILED state."""
        # Assert
        for status in OperationStatus:
            if status != OperationStatus.FAILED:
                assert not OperationStatus.FAILED.can_transition_to(status)

    def test_no_transitions_from_cancelled(self):
        """Test that no transitions are allowed from CANCELLED state."""
        # Assert
        for status in OperationStatus:
            if status != OperationStatus.CANCELLED:
                assert not OperationStatus.CANCELLED.can_transition_to(status)

    def test_self_transition_not_allowed(self):
        """Test that self-transitions are not allowed."""
        # Assert
        for status in OperationStatus:
            assert not status.can_transition_to(status)

    def test_string_representation(self):
        """Test string representation of status."""
        # Assert
        assert str(OperationStatus.PENDING) == "OperationStatus.PENDING"
        assert str(OperationStatus.PROCESSING) == "OperationStatus.PROCESSING"
        assert str(OperationStatus.COMPLETED) == "OperationStatus.COMPLETED"


class TestChunkMetadata:
    """Test suite for ChunkMetadata value object."""

    def test_basic_metadata_creation(self):
        """Test creating basic chunk metadata."""
        # Act
        metadata = ChunkMetadata(
            token_count=50,
            semantic_density=0.8,
        )

        # Assert
        assert metadata.token_count == 50
        assert metadata.semantic_density == 0.8
        assert metadata.overlap_percentage == 0.0
        assert metadata.confidence_score == 1.0
        assert metadata.language is None
        assert metadata.custom_attributes == {}

    def test_full_metadata_creation(self):
        """Test creating metadata with all fields."""
        # Act
        metadata = ChunkMetadata(
            token_count=100,
            semantic_density=0.75,
            overlap_percentage=0.2,
            confidence_score=0.95,
            language="en",
            custom_attributes={"category": "technical", "importance": "high"},
        )

        # Assert
        assert metadata.token_count == 100
        assert metadata.semantic_density == 0.75
        assert metadata.overlap_percentage == 0.2
        assert metadata.confidence_score == 0.95
        assert metadata.language == "en"
        assert metadata.custom_attributes["category"] == "technical"
        assert metadata.custom_attributes["importance"] == "high"

    def test_invalid_token_count_negative(self):
        """Test that negative token count raises error."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            ChunkMetadata(token_count=-1)
        
        assert "token_count must be non-negative" in str(exc_info.value)

    def test_invalid_semantic_density_below_range(self):
        """Test that semantic density below 0 raises error."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            ChunkMetadata(token_count=10, semantic_density=-0.1)
        
        assert "semantic_density must be between 0 and 1" in str(exc_info.value)

    def test_invalid_semantic_density_above_range(self):
        """Test that semantic density above 1 raises error."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            ChunkMetadata(token_count=10, semantic_density=1.1)
        
        assert "semantic_density must be between 0 and 1" in str(exc_info.value)

    def test_invalid_overlap_percentage_below_range(self):
        """Test that overlap percentage below 0 raises error."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            ChunkMetadata(token_count=10, overlap_percentage=-0.1)
        
        assert "overlap_percentage must be between 0 and 1" in str(exc_info.value)

    def test_invalid_overlap_percentage_above_range(self):
        """Test that overlap percentage above 1 raises error."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            ChunkMetadata(token_count=10, overlap_percentage=1.5)
        
        assert "overlap_percentage must be between 0 and 1" in str(exc_info.value)

    def test_invalid_confidence_score_below_range(self):
        """Test that confidence score below 0 raises error."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            ChunkMetadata(token_count=10, confidence_score=-0.1)
        
        assert "confidence_score must be between 0 and 1" in str(exc_info.value)

    def test_invalid_confidence_score_above_range(self):
        """Test that confidence score above 1 raises error."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            ChunkMetadata(token_count=10, confidence_score=1.1)
        
        assert "confidence_score must be between 0 and 1" in str(exc_info.value)

    def test_boundary_values(self):
        """Test boundary values for all percentage fields."""
        # Test 0 values
        metadata_zero = ChunkMetadata(
            token_count=0,
            semantic_density=0.0,
            overlap_percentage=0.0,
            confidence_score=0.0,
        )
        assert metadata_zero.token_count == 0
        assert metadata_zero.semantic_density == 0.0
        
        # Test 1 values
        metadata_one = ChunkMetadata(
            token_count=100,
            semantic_density=1.0,
            overlap_percentage=1.0,
            confidence_score=1.0,
        )
        assert metadata_one.semantic_density == 1.0
        assert metadata_one.overlap_percentage == 1.0
        assert metadata_one.confidence_score == 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        # Arrange
        metadata = ChunkMetadata(
            token_count=75,
            semantic_density=0.6,
            overlap_percentage=0.15,
            confidence_score=0.9,
            language="en",
            custom_attributes={"source": "document"},
        )

        # Act
        metadata_dict = metadata.to_dict()

        # Assert
        assert metadata_dict["token_count"] == 75
        assert metadata_dict["semantic_density"] == 0.6
        assert metadata_dict["overlap_percentage"] == 0.15
        assert metadata_dict["confidence_score"] == 0.9
        assert metadata_dict["language"] == "en"
        assert metadata_dict["custom_attributes"]["source"] == "document"

    def test_to_dict_minimal(self):
        """Test conversion to dictionary with minimal fields."""
        # Arrange
        metadata = ChunkMetadata(token_count=50)

        # Act
        metadata_dict = metadata.to_dict()

        # Assert
        assert metadata_dict["token_count"] == 50
        assert metadata_dict["semantic_density"] == 1.0
        assert metadata_dict["overlap_percentage"] == 0.0
        assert metadata_dict["confidence_score"] == 1.0
        assert metadata_dict["language"] is None
        assert metadata_dict["custom_attributes"] == {}

    def test_equality(self):
        """Test value object equality."""
        # Arrange
        metadata1 = ChunkMetadata(
            token_count=50,
            semantic_density=0.8,
            language="en",
        )
        metadata2 = ChunkMetadata(
            token_count=50,
            semantic_density=0.8,
            language="en",
        )
        metadata3 = ChunkMetadata(
            token_count=50,
            semantic_density=0.7,  # Different density
            language="en",
        )

        # Assert
        assert metadata1 == metadata2
        assert metadata1 != metadata3
        assert metadata1 != "not metadata"

    def test_immutability(self):
        """Test that metadata is immutable."""
        # Arrange
        metadata = ChunkMetadata(
            token_count=50,
            custom_attributes={"key": "value"},
        )

        # Act & Assert - attributes should not be settable
        with pytest.raises(AttributeError):
            metadata.token_count = 100

        # The custom_attributes dict itself is mutable (Python limitation)
        # but we should document that it shouldn't be modified
        original_attrs = metadata.custom_attributes.copy()
        metadata.custom_attributes["new_key"] = "new_value"
        
        # This is a known limitation - we'd need to use frozen dataclasses
        # or return copies to truly prevent modification
        assert metadata.custom_attributes != original_attrs