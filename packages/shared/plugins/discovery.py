"""Plugin Discovery API for agent-driven pipeline building.

This module provides functions to discover plugins based on their AgentHints
metadata, enabling agents to intelligently select plugins for document
processing pipelines.

Example:
    >>> from shared.plugins.loader import load_plugins
    >>> from shared.plugins.discovery import (
    ...     list_plugins_for_agent,
    ...     find_plugins_for_input,
    ...     get_alternative_plugins,
    ... )
    >>>
    >>> load_plugins()  # Load all plugins first
    >>>
    >>> # Get all plugins with AgentHints
    >>> plugins = list_plugins_for_agent()
    >>>
    >>> # Find plugins that can handle PDFs
    >>> pdf_plugins = find_plugins_for_input("application/pdf")
    >>>
    >>> # Find alternatives to a specific plugin
    >>> alternatives = get_alternative_plugins("unstructured")
"""

from __future__ import annotations

import fnmatch
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manifest import PluginManifest

from .registry import plugin_registry

logger = logging.getLogger(__name__)


def list_plugins_for_agent(plugin_type: str | None = None) -> list[PluginManifest]:
    """Return manifests for all agent-visible plugins (those with agent_hints).

    Agent-visible plugins are those that have AgentHints metadata, which provides
    information about when and how to use the plugin.

    Args:
        plugin_type: Optional filter for plugin type (e.g., "parser", "chunking",
            "embedding", "extractor"). If None, returns all types.

    Returns:
        List of PluginManifest objects for plugins with agent_hints, sorted by
        plugin type and then by plugin ID.

    Example:
        >>> # Get all agent-visible plugins
        >>> all_plugins = list_plugins_for_agent()
        >>>
        >>> # Get only parser plugins
        >>> parsers = list_plugins_for_agent("parser")
    """
    records = plugin_registry.list_records(plugin_type=plugin_type)

    # Filter to only plugins with agent_hints
    manifests = [record.manifest for record in records if record.manifest.agent_hints is not None]

    # Sort by type, then by ID for consistent ordering
    manifests.sort(key=lambda m: (m.type, m.id))

    return manifests


def find_plugins_for_input(
    input_type: str,
    plugin_type: str | None = None,
) -> list[PluginManifest]:
    """Find plugins that accept a given input type (MIME type matching).

    Searches plugins with AgentHints that declare input_types. Supports
    both exact matches and wildcard patterns (e.g., "text/*", "*/*").

    Args:
        input_type: The MIME type to search for (e.g., "application/pdf",
            "text/plain", "image/png").
        plugin_type: Optional filter for plugin type.

    Returns:
        List of PluginManifest objects that can handle the input type,
        sorted by specificity (more specific matches first).

    Example:
        >>> # Find all plugins that can handle PDF files
        >>> pdf_plugins = find_plugins_for_input("application/pdf")
        >>>
        >>> # Find only parsers that can handle text
        >>> text_parsers = find_plugins_for_input("text/plain", plugin_type="parser")
    """
    records = plugin_registry.list_records(plugin_type=plugin_type)

    matching: list[tuple[int, PluginManifest]] = []

    for record in records:
        manifest = record.manifest
        hints = manifest.agent_hints

        if hints is None or hints.input_types is None:
            continue

        if _matches_input_type(input_type, hints.input_types):
            specificity = _input_type_specificity(input_type, manifest)
            matching.append((specificity, manifest))

    # Sort by specificity (higher is better), then by ID
    matching.sort(key=lambda x: (-x[0], x[1].id))

    return [manifest for _, manifest in matching]


def get_alternative_plugins(plugin_id: str) -> list[PluginManifest]:
    """Find plugins that are alternatives to the given plugin.

    Two plugins are considered alternatives if they have overlapping
    input_types in their AgentHints. This helps agents find fallback
    options when a preferred plugin is unavailable.

    Args:
        plugin_id: The ID of the plugin to find alternatives for.

    Returns:
        List of PluginManifest objects for plugins that can handle similar
        inputs, excluding the original plugin. Sorted by overlap score
        (more overlap first).

    Example:
        >>> # Find alternatives to the unstructured parser
        >>> alternatives = get_alternative_plugins("unstructured")
        >>> for alt in alternatives:
        ...     print(f"{alt.id}: {alt.agent_hints.purpose}")
    """
    # Find the original plugin
    original_record = plugin_registry.find_by_id(plugin_id)
    if original_record is None:
        logger.warning("Plugin '%s' not found in registry", plugin_id)
        return []

    original_manifest = original_record.manifest
    original_hints = original_manifest.agent_hints

    # If original has no hints or no input_types, can't find alternatives
    if original_hints is None or original_hints.input_types is None:
        return []

    # Get all records of the same type
    records = plugin_registry.list_records(plugin_type=original_record.plugin_type)

    alternatives: list[tuple[int, PluginManifest]] = []

    for record in records:
        # Skip the original plugin
        if record.plugin_id == plugin_id:
            continue

        manifest = record.manifest
        hints = manifest.agent_hints

        if hints is None or hints.input_types is None:
            continue

        # Check for overlapping input types
        if _has_overlapping_inputs(original_hints.input_types, hints.input_types):
            # Calculate overlap score (number of overlapping patterns)
            overlap = _calculate_overlap_score(original_hints.input_types, hints.input_types)
            alternatives.append((overlap, manifest))

    # Sort by overlap score (higher first), then by ID
    alternatives.sort(key=lambda x: (-x[0], x[1].id))

    return [manifest for _, manifest in alternatives]


def _matches_input_type(query: str, accepted: list[str]) -> bool:
    """Check if a query MIME type matches any accepted patterns.

    Supports exact matches and glob patterns:
    - "application/pdf" matches "application/pdf"
    - "text/plain" matches "text/*"
    - "image/png" matches "*/*"

    Args:
        query: The MIME type to check (e.g., "application/pdf").
        accepted: List of accepted MIME types/patterns from AgentHints.

    Returns:
        True if the query matches any accepted pattern.
    """
    query_lower = query.lower()

    for pattern in accepted:
        pattern_lower = pattern.lower()

        # Exact match
        if query_lower == pattern_lower:
            return True

        # Glob-style pattern matching (e.g., "text/*", "*/*")
        if fnmatch.fnmatch(query_lower, pattern_lower):
            return True

    return False


def _has_overlapping_inputs(a: list[str], b: list[str]) -> bool:
    """Check if two input_types lists have any overlap.

    Two lists overlap if any pattern in list A matches any pattern in list B,
    or vice versa. This includes wildcard pattern matching.

    Args:
        a: First list of MIME types/patterns.
        b: Second list of MIME types/patterns.

    Returns:
        True if any patterns overlap.
    """
    # Check if any pattern in A matches any pattern in B
    for pattern_a in a:
        for pattern_b in b:
            # Check both directions for wildcard matches
            if _patterns_overlap(pattern_a, pattern_b):
                return True

    return False


def _patterns_overlap(pattern_a: str, pattern_b: str) -> bool:
    """Check if two MIME type patterns overlap.

    Patterns overlap if:
    - They are equal
    - One matches the other via glob
    - They have the same major type and one has wildcard minor

    Args:
        pattern_a: First MIME type/pattern.
        pattern_b: Second MIME type/pattern.

    Returns:
        True if patterns overlap.
    """
    a_lower = pattern_a.lower()
    b_lower = pattern_b.lower()

    # Exact match
    if a_lower == b_lower:
        return True

    # Check glob matches in both directions
    if fnmatch.fnmatch(a_lower, b_lower) or fnmatch.fnmatch(b_lower, a_lower):
        return True

    # Check major type matching (e.g., "text/plain" overlaps with "text/*")
    a_parts = a_lower.split("/", 1)
    b_parts = b_lower.split("/", 1)

    if len(a_parts) == 2 and len(b_parts) == 2:
        a_major, a_minor = a_parts
        b_major, b_minor = b_parts

        # Same major type with wildcard minor
        if a_major == b_major and (a_minor == "*" or b_minor == "*"):
            return True

        # Wildcard major type
        if a_major == "*" or b_major == "*":
            return True

    return False


def _calculate_overlap_score(a: list[str], b: list[str]) -> int:
    """Calculate overlap score between two input_types lists.

    Higher score indicates more patterns that overlap.

    Args:
        a: First list of MIME types/patterns.
        b: Second list of MIME types/patterns.

    Returns:
        Number of overlapping pattern pairs.
    """
    score = 0
    for pattern_a in a:
        for pattern_b in b:
            if _patterns_overlap(pattern_a, pattern_b):
                score += 1
    return score


def _input_type_specificity(query: str, manifest: PluginManifest) -> int:
    """Calculate how specifically a plugin matches a query input type.

    Higher specificity indicates a more precise match:
    - Exact match: 100 points
    - Major type match (e.g., "text/*"): 50 points
    - Wildcard match (e.g., "*/*"): 10 points

    Args:
        query: The MIME type being queried.
        manifest: The plugin manifest to score.

    Returns:
        Specificity score (higher is better).
    """
    hints = manifest.agent_hints
    if hints is None or hints.input_types is None:
        return 0

    query_lower = query.lower()
    query_parts = query_lower.split("/", 1)
    query_major = query_parts[0] if query_parts else ""

    max_score = 0

    for pattern in hints.input_types:
        pattern_lower = pattern.lower()
        pattern_parts = pattern_lower.split("/", 1)

        # Exact match - highest specificity
        if query_lower == pattern_lower:
            max_score = max(max_score, 100)
            continue

        # Check for wildcard patterns
        if len(pattern_parts) == 2:
            pattern_major, pattern_minor = pattern_parts

            # Major type match with wildcard minor (e.g., "text/*")
            if pattern_major == query_major and pattern_minor == "*":
                max_score = max(max_score, 50)
                continue

            # Full wildcard (e.g., "*/*")
            if pattern_major == "*":
                max_score = max(max_score, 10)
                continue

    return max_score
