"""Validation utilities for benchmark k-value parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KValueConfig:
    """Validated k-value configuration for benchmark metrics.

    Attributes:
        primary_k: The primary k value for evaluation (defaults to 10).
        k_values_for_metrics: Sorted list of k values for metric computation.
            Always includes primary_k.
    """

    primary_k: int
    k_values_for_metrics: list[int]


def parse_k_values(
    raw_primary_k: Any,
    raw_k_values: list[Any] | None,
    default_primary_k: int = 10,
) -> KValueConfig:
    """Parse and validate k-value configuration.

    Converts raw primary_k and k_values_for_metrics into validated integers.
    Invalid values are ignored for k_values_for_metrics but cause fallback
    to defaults for primary_k.

    Args:
        raw_primary_k: Raw primary k value (may be int, str, or invalid).
        raw_k_values: Raw list of k values for metrics (may contain invalid values).
        default_primary_k: Default value for primary_k if invalid (default: 10).

    Returns:
        KValueConfig with validated primary_k and k_values_for_metrics.
        k_values_for_metrics is sorted and always includes primary_k.
    """
    parsed_primary = _parse_positive_int(raw_primary_k, default=None)
    primary_k = parsed_primary if parsed_primary is not None else default_primary_k

    if not raw_k_values:
        k_values: list[int] = [primary_k]
    else:
        k_values_set: set[int] = set()
        for value in raw_k_values:
            k_int = _parse_positive_int(value, default=None)
            if k_int is not None:
                k_values_set.add(k_int)
        k_values = sorted(k_values_set) if k_values_set else [primary_k]

    if primary_k not in k_values:
        k_values = sorted([*k_values, primary_k])

    return KValueConfig(primary_k=primary_k, k_values_for_metrics=k_values)


def validate_top_k_values(
    raw_top_k_values: list[Any],
    min_required_k: int,
) -> tuple[list[int], list[Any], list[int]]:
    """Validate top-k values for search configuration.

    Args:
        raw_top_k_values: Raw list of top-k values to validate.
        min_required_k: Minimum k value required (typically max of k_values_for_metrics).

    Returns:
        Tuple of (valid_values, invalid_values, too_small_values).
        - valid_values: Sorted list of valid positive integers >= min_required_k.
        - invalid_values: Values that couldn't be parsed as positive integers.
        - too_small_values: Valid integers that are < min_required_k.
    """
    valid: list[int] = []
    invalid: list[Any] = []
    too_small: list[int] = []

    for raw_value in raw_top_k_values:
        k_int = _parse_positive_int(raw_value, default=None)
        if k_int is None:
            invalid.append(raw_value)
        elif k_int < min_required_k:
            too_small.append(k_int)
        else:
            valid.append(k_int)

    return sorted(set(valid)), invalid, sorted(set(too_small))


def _parse_positive_int(value: Any, default: int | None) -> int | None:
    """Parse a value as a positive integer.

    Args:
        value: Value to parse.
        default: Default to return if parsing fails or result is non-positive.

    Returns:
        Parsed positive integer, or default if invalid.
    """
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default
