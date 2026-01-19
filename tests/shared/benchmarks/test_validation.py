"""Tests for shared.benchmarks.validation module."""

import pytest

from shared.benchmarks.validation import (
    KValueConfig,
    parse_k_values,
    validate_top_k_values,
)


class TestParseKValues:
    """Tests for parse_k_values function."""

    def test_default_values(self) -> None:
        """Returns defaults when no values provided."""
        result = parse_k_values(raw_primary_k=None, raw_k_values=None)
        assert result == KValueConfig(primary_k=10, k_values_for_metrics=[10])

    def test_custom_primary_k(self) -> None:
        """Uses custom primary_k when valid."""
        result = parse_k_values(raw_primary_k=5, raw_k_values=None)
        assert result.primary_k == 5
        assert result.k_values_for_metrics == [5]

    def test_primary_k_from_string(self) -> None:
        """Parses primary_k from string."""
        result = parse_k_values(raw_primary_k="20", raw_k_values=None)
        assert result.primary_k == 20

    def test_invalid_primary_k_uses_default(self) -> None:
        """Falls back to default for invalid primary_k."""
        result = parse_k_values(raw_primary_k="invalid", raw_k_values=None)
        assert result.primary_k == 10

    def test_non_positive_primary_k_uses_default(self) -> None:
        """Falls back to default for non-positive primary_k."""
        result = parse_k_values(raw_primary_k=0, raw_k_values=None)
        assert result.primary_k == 10

        result = parse_k_values(raw_primary_k=-5, raw_k_values=None)
        assert result.primary_k == 10

    def test_custom_k_values(self) -> None:
        """Parses custom k_values_for_metrics."""
        result = parse_k_values(raw_primary_k=10, raw_k_values=[5, 10, 20])
        assert result.k_values_for_metrics == [5, 10, 20]

    def test_k_values_sorted(self) -> None:
        """Returns k_values sorted."""
        result = parse_k_values(raw_primary_k=10, raw_k_values=[20, 5, 15])
        assert result.k_values_for_metrics == [5, 10, 15, 20]

    def test_k_values_deduped(self) -> None:
        """Removes duplicate k_values."""
        result = parse_k_values(raw_primary_k=10, raw_k_values=[5, 10, 5, 10])
        assert result.k_values_for_metrics == [5, 10]

    def test_primary_k_included_in_k_values(self) -> None:
        """Ensures primary_k is included in k_values_for_metrics."""
        result = parse_k_values(raw_primary_k=10, raw_k_values=[5, 20])
        assert 10 in result.k_values_for_metrics
        assert result.k_values_for_metrics == [5, 10, 20]

    def test_invalid_k_values_ignored(self) -> None:
        """Invalid values in k_values list are ignored."""
        result = parse_k_values(raw_primary_k=10, raw_k_values=[5, "invalid", None, 20])
        assert result.k_values_for_metrics == [5, 10, 20]

    def test_non_positive_k_values_ignored(self) -> None:
        """Non-positive values in k_values list are ignored."""
        result = parse_k_values(raw_primary_k=10, raw_k_values=[5, 0, -3, 20])
        assert result.k_values_for_metrics == [5, 10, 20]

    def test_all_invalid_k_values_uses_primary(self) -> None:
        """Falls back to [primary_k] when all k_values are invalid."""
        result = parse_k_values(raw_primary_k=10, raw_k_values=["x", "y", None])
        assert result.k_values_for_metrics == [10]

    def test_empty_k_values_uses_primary(self) -> None:
        """Falls back to [primary_k] for empty k_values list."""
        result = parse_k_values(raw_primary_k=10, raw_k_values=[])
        assert result.k_values_for_metrics == [10]

    def test_k_values_from_strings(self) -> None:
        """Parses k_values from string representations."""
        result = parse_k_values(raw_primary_k=10, raw_k_values=["5", "15", "20"])
        assert result.k_values_for_metrics == [5, 10, 15, 20]

    def test_custom_default_primary_k(self) -> None:
        """Uses custom default_primary_k when provided."""
        result = parse_k_values(raw_primary_k=None, raw_k_values=None, default_primary_k=25)
        assert result.primary_k == 25
        assert result.k_values_for_metrics == [25]


class TestValidateTopKValues:
    """Tests for validate_top_k_values function."""

    def test_all_valid(self) -> None:
        """Returns all values as valid when they meet requirements."""
        valid, invalid, too_small = validate_top_k_values([10, 20, 30], min_required_k=10)
        assert valid == [10, 20, 30]
        assert invalid == []
        assert too_small == []

    def test_detects_invalid_values(self) -> None:
        """Detects values that cannot be parsed as positive integers."""
        valid, invalid, too_small = validate_top_k_values([10, "invalid", None, -5, 0, 20], min_required_k=10)
        assert valid == [10, 20]
        assert invalid == ["invalid", None, -5, 0]
        assert too_small == []

    def test_detects_too_small_values(self) -> None:
        """Detects values smaller than min_required_k."""
        valid, invalid, too_small = validate_top_k_values([5, 10, 15, 20], min_required_k=15)
        assert valid == [15, 20]
        assert invalid == []
        assert too_small == [5, 10]

    def test_combined_validation(self) -> None:
        """Handles combination of invalid and too-small values."""
        valid, invalid, too_small = validate_top_k_values([5, "bad", 10, None, 20], min_required_k=15)
        assert valid == [20]
        assert invalid == ["bad", None]
        assert too_small == [5, 10]

    def test_values_sorted(self) -> None:
        """Returns sorted lists."""
        valid, invalid, too_small = validate_top_k_values([30, 10, 20, 5], min_required_k=15)
        assert valid == [20, 30]
        assert too_small == [5, 10]

    def test_values_deduped(self) -> None:
        """Removes duplicates from results."""
        valid, invalid, too_small = validate_top_k_values([20, 20, 5, 5, "x", "x"], min_required_k=10)
        assert valid == [20]
        assert too_small == [5]

    def test_empty_input(self) -> None:
        """Handles empty input list."""
        valid, invalid, too_small = validate_top_k_values([], min_required_k=10)
        assert valid == []
        assert invalid == []
        assert too_small == []

    def test_string_values_parsed(self) -> None:
        """Parses string representations of integers."""
        valid, invalid, too_small = validate_top_k_values(["10", "20"], min_required_k=10)
        assert valid == [10, 20]
        assert invalid == []


class TestKValueConfig:
    """Tests for KValueConfig dataclass."""

    def test_is_frozen(self) -> None:
        """KValueConfig is immutable."""
        config = KValueConfig(primary_k=10, k_values_for_metrics=[5, 10, 20])
        with pytest.raises(AttributeError):
            config.primary_k = 5  # type: ignore[misc]

    def test_equality(self) -> None:
        """KValueConfig instances with same values are equal."""
        config1 = KValueConfig(primary_k=10, k_values_for_metrics=[5, 10])
        config2 = KValueConfig(primary_k=10, k_values_for_metrics=[5, 10])
        assert config1 == config2
