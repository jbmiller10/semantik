"""Version compatibility checking for plugins.

This module provides functions to check if a plugin is compatible with
the current Semantik version based on semver constraints.
"""

from __future__ import annotations

import logging

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)


def get_semantik_version() -> str:
    """Get the current Semantik version.

    Reads from the package metadata or falls back to hardcoded value.

    Returns:
        The current Semantik version string.
    """
    try:
        from importlib.metadata import version

        return version("semantik")
    except Exception:
        # Fallback if running in development without proper installation
        return "2.0.0"


def check_compatibility(
    plugin_constraint: str | None,
    semantik_version: str | None = None,
) -> tuple[bool, str | None]:
    """Check if a plugin is compatible with the current Semantik version.

    Args:
        plugin_constraint: Minimum Semantik version required (e.g., "2.0.0").
            If None or empty, the plugin is considered compatible.
        semantik_version: Semantik version to check against. If None,
            uses the current running version.

    Returns:
        Tuple of (is_compatible, error_message):
        - (True, None) if compatible
        - (False, "reason") if not compatible
    """
    if not plugin_constraint:
        return True, None

    if semantik_version is None:
        semantik_version = get_semantik_version()

    try:
        # Create specifier for minimum version
        spec = SpecifierSet(f">={plugin_constraint}")
        current = Version(semantik_version)

        if current in spec:
            return True, None
        return (
            False,
            f"Requires Semantik >= {plugin_constraint}, but running {semantik_version}",
        )

    except InvalidVersion as exc:
        logger.warning("Invalid version format: %s", exc)
        return False, f"Invalid version format: {exc}"
    except InvalidSpecifier as exc:
        logger.warning("Invalid version specifier: %s", exc)
        return False, f"Invalid constraint format: {exc}"


def is_compatible(
    plugin_constraint: str | None,
    semantik_version: str | None = None,
) -> bool:
    """Check if plugin is compatible (simple boolean version).

    This is a convenience wrapper around check_compatibility() that
    returns just the boolean result.

    Args:
        plugin_constraint: Minimum Semantik version required.
        semantik_version: Semantik version to check against.

    Returns:
        True if compatible, False otherwise.
    """
    compatible, _ = check_compatibility(plugin_constraint, semantik_version)
    return compatible
