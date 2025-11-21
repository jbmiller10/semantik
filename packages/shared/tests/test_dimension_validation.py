"""
Unit tests for dimension validation utilities.
"""

import pytest

from shared.database.exceptions import DimensionMismatchError
from shared.embedding.validation import (
    adjust_embeddings_dimension,
    validate_dimension_compatibility,
    validate_embedding_dimensions,
)


class TestDimensionValidation:
    """Test dimension validation functions."""

    def test_validate_dimension_compatibility_success(self) -> None:
        """Test that matching dimensions pass validation."""
        # Should not raise any exception
        validate_dimension_compatibility(
            expected_dimension=384,
            actual_dimension=384,
            collection_name="test_collection",
            model_name="all-MiniLM-L6-v2",
        )

    def test_validate_dimension_compatibility_failure(self) -> None:
        """Test that mismatched dimensions raise DimensionMismatchError."""
        with pytest.raises(DimensionMismatchError) as exc_info:
            validate_dimension_compatibility(
                expected_dimension=384,
                actual_dimension=1024,
                collection_name="test_collection",
                model_name="Qwen3-0.6B",
            )

        error = exc_info.value
        assert error.expected_dimension == 384
        assert error.actual_dimension == 1024
        assert error.collection_name == "test_collection"
        assert error.model_name == "Qwen3-0.6B"
        assert "expected 384, got 1024" in str(error)

    def test_validate_embedding_dimensions_success(self) -> None:
        """Test that embeddings with correct dimensions pass validation."""
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        # Should not raise any exception
        validate_embedding_dimensions(embeddings, expected_dimension=384)

    def test_validate_embedding_dimensions_failure(self) -> None:
        """Test that embeddings with wrong dimensions raise DimensionMismatchError."""
        embeddings = [[0.1] * 384, [0.2] * 1024, [0.3] * 384]  # Middle one is wrong

        with pytest.raises(DimensionMismatchError) as exc_info:
            validate_embedding_dimensions(embeddings, expected_dimension=384)

        error = exc_info.value
        assert error.expected_dimension == 384
        assert error.actual_dimension == 1024

    def test_validate_embedding_dimensions_empty_list(self) -> None:
        """Test that empty embedding list raises ValueError."""
        with pytest.raises(ValueError, match="No embeddings provided"):
            validate_embedding_dimensions([], expected_dimension=384)

    def test_adjust_embeddings_dimension_truncation(self) -> None:
        """Test truncating embeddings to smaller dimension."""
        original = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        adjusted = adjust_embeddings_dimension(original, target_dimension=2, normalize=False)

        assert len(adjusted) == 2
        assert len(adjusted[0]) == 2
        assert adjusted[0] == [1.0, 2.0]
        assert adjusted[1] == [5.0, 6.0]

    def test_adjust_embeddings_dimension_padding(self) -> None:
        """Test padding embeddings to larger dimension."""
        original = [[1.0, 2.0], [3.0, 4.0]]
        adjusted = adjust_embeddings_dimension(original, target_dimension=4, normalize=False)

        assert len(adjusted) == 2
        assert len(adjusted[0]) == 4
        assert adjusted[0] == [1.0, 2.0, 0.0, 0.0]
        assert adjusted[1] == [3.0, 4.0, 0.0, 0.0]

    def test_adjust_embeddings_dimension_normalization(self) -> None:
        """Test that normalization produces unit vectors."""
        original = [[3.0, 4.0]]  # Norm = 5
        adjusted = adjust_embeddings_dimension(original, target_dimension=2, normalize=True)

        # Check that it's normalized to unit length
        norm = sum(v**2 for v in adjusted[0]) ** 0.5
        assert abs(norm - 1.0) < 1e-6
        assert abs(adjusted[0][0] - 0.6) < 1e-6  # 3/5
        assert abs(adjusted[0][1] - 0.8) < 1e-6  # 4/5

    def test_adjust_embeddings_dimension_no_change(self) -> None:
        """Test that embeddings with correct dimension are unchanged."""
        original = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        adjusted = adjust_embeddings_dimension(original, target_dimension=3, normalize=False)

        assert adjusted == original

    def test_adjust_embeddings_dimension_empty_list(self) -> None:
        """Test that empty list returns empty list."""
        adjusted = adjust_embeddings_dimension([], target_dimension=10)
        assert adjusted == []
