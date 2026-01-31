"""Predicate evaluation for pipeline edge routing.

This module implements the predicate expression language used in pipeline
edge `when` conditions. Predicates allow conditional routing of files
through different processing paths based on file attributes.

Predicate Expression Language:
    - Exact match: {"mime_type": "application/pdf"}
    - Glob pattern: {"mime_type": "application/*"}
    - Negation: {"mime_type": "!image/*"}
    - Numeric comparison: {"size_bytes": ">10000000"}
    - Array (OR): {"extension": [".md", ".txt"]}
    - Nested field: {"metadata.source.language": "zh"}
    - Legacy nested: {"source_metadata.language": "zh"} (auto-translated)
    - Catch-all: None or {}

Multiple fields in a predicate are AND'd together.
"""

from __future__ import annotations

import fnmatch
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from shared.pipeline.types import FileReference

# Pattern for numeric comparisons: operator followed by number
_NUMERIC_PATTERN = re.compile(r"^(>=|<=|>|<|==|!=)\s*(-?\d+(?:\.\d+)?)$")


def _translate_legacy_path(path: str) -> str:
    """Translate legacy source_metadata paths to new metadata.source paths.

    This provides backward compatibility for predicates using the old
    source_metadata.X path format, translating them to the new
    metadata.source.X format.

    Args:
        path: The field path to potentially translate

    Returns:
        The translated path (or original if no translation needed)

    Examples:
        >>> _translate_legacy_path("source_metadata.language")
        "metadata.source.language"
        >>> _translate_legacy_path("source_metadata")
        "metadata.source"
        >>> _translate_legacy_path("mime_type")
        "mime_type"
    """
    if path.startswith("source_metadata."):
        return path.replace("source_metadata.", "metadata.source.", 1)
    if path == "source_metadata":
        return "metadata.source"
    return path


def get_nested_value(obj: Any, path: str) -> Any:
    """Get a value from a nested object using dot notation.

    Args:
        obj: The object to extract from (dict or dataclass)
        path: Dot-separated path (e.g., "source_metadata.language")

    Returns:
        The value at the path, or None if not found
    """
    parts = path.split(".")
    current = obj

    for part in parts:
        if current is None:
            return None

        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None

    return current


def match_value(pattern: Any, value: Any) -> bool:
    """Match a single value against a pattern.

    Args:
        pattern: The pattern to match against
        value: The value to test

    Returns:
        True if the value matches the pattern
    """
    # Handle None pattern (always matches)
    if pattern is None:
        return True

    # Handle None value
    if value is None:
        # Only None pattern matches None value (handled above)
        # Non-None pattern does not match None value
        return False

    # Handle array pattern (OR logic)
    if isinstance(pattern, list):
        return any(match_value(p, value) for p in pattern)

    # Handle string patterns
    if isinstance(pattern, str):
        # Check for numeric comparison first (before negation, since != is numeric)
        numeric_match = _NUMERIC_PATTERN.match(pattern)
        if numeric_match:
            op = numeric_match.group(1)
            threshold = float(numeric_match.group(2))
            try:
                num_value = float(value)
            except (ValueError, TypeError):
                return False
            return _compare_numeric(num_value, op, threshold)

        # Check for negation (after numeric patterns to avoid conflict with !=)
        if pattern.startswith("!"):
            return not match_value(pattern[1:], value)

        # Handle string value
        if isinstance(value, str):
            # Check for glob pattern
            if any(c in pattern for c in "*?["):
                return fnmatch.fnmatch(value, pattern)
            # Exact match
            return value == pattern

        # Special-case: string boolean patterns against boolean values.
        # This enables predicates like {"metadata.detected.is_code": "!true"} when
        # UIs serialize negation by prefixing string values with '!'.
        if isinstance(value, bool):
            lowered = pattern.lower()
            if lowered in ("true", "1", "yes"):
                return value is True
            if lowered in ("false", "0", "no"):
                return value is False

        # Convert non-string value to string for comparison
        return str(value) == pattern

    # Handle boolean pattern
    if isinstance(pattern, bool):
        if isinstance(value, bool):
            return value == pattern
        # Convert string to bool for comparison
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes") if pattern else value.lower() in ("false", "0", "no")
        return bool(value) == pattern

    # Handle numeric pattern (exact match)
    if isinstance(pattern, int | float):
        try:
            return float(value) == float(pattern)
        except (ValueError, TypeError):
            return False

    # Unknown pattern type
    return False


def _compare_numeric(value: float, op: str, threshold: float) -> bool:
    """Compare a numeric value against a threshold using the given operator.

    Args:
        value: The value to compare
        op: Comparison operator (>, >=, <, <=, ==, !=)
        threshold: The threshold value

    Returns:
        True if the comparison succeeds
    """
    comparators: dict[str, Callable[[float, float], bool]] = {
        ">": lambda v, t: v > t,
        ">=": lambda v, t: v >= t,
        "<": lambda v, t: v < t,
        "<=": lambda v, t: v <= t,
        "==": lambda v, t: v == t,
        "!=": lambda v, t: v != t,
    }
    comparator = comparators.get(op)
    if comparator:
        return comparator(value, threshold)
    return False


def matches_predicate(file_ref: FileReference, predicate: dict[str, Any] | None) -> bool:
    """Check if a file reference matches a predicate.

    Args:
        file_ref: The file reference to test
        predicate: The predicate to match against, or None for catch-all

    Returns:
        True if the file reference matches the predicate

    Examples:
        # Catch-all (always matches)
        >>> matches_predicate(file_ref, None)
        True
        >>> matches_predicate(file_ref, {})
        True

        # Exact match
        >>> matches_predicate(file_ref, {"mime_type": "application/pdf"})
        True  # if file_ref.mime_type == "application/pdf"

        # Glob pattern
        >>> matches_predicate(file_ref, {"mime_type": "application/*"})
        True  # if file_ref.mime_type starts with "application/"

        # Negation
        >>> matches_predicate(file_ref, {"mime_type": "!image/*"})
        True  # if file_ref.mime_type does NOT start with "image/"

        # Numeric comparison
        >>> matches_predicate(file_ref, {"size_bytes": ">10000000"})
        True  # if file_ref.size_bytes > 10MB

        # Array (OR)
        >>> matches_predicate(file_ref, {"extension": [".md", ".txt"]})
        True  # if file_ref.extension is ".md" OR ".txt"

        # Nested field (new format)
        >>> matches_predicate(file_ref, {"metadata.source.language": "zh"})
        True  # if file_ref.metadata["source"]["language"] == "zh"

        # Nested field (legacy format - auto-translated)
        >>> matches_predicate(file_ref, {"source_metadata.language": "zh"})
        True  # Translated to metadata.source.language

        # Multiple fields (AND)
        >>> matches_predicate(file_ref, {"mime_type": "text/*", "size_bytes": "<100000"})
        True  # if both conditions are met
    """
    # Catch-all: None or empty dict always matches
    if predicate is None or not predicate:
        return True

    # All fields in predicate must match (AND logic)
    for field, pattern in predicate.items():
        # Translate legacy source_metadata.X paths to metadata.source.X
        translated_field = _translate_legacy_path(field)
        value = get_nested_value(file_ref, translated_field)
        if not match_value(pattern, value):
            return False

    return True


__all__ = [
    "matches_predicate",
    "match_value",
    "get_nested_value",
]
