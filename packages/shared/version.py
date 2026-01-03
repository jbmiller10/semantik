"""Centralized version management for Semantik.

This module provides a single source of truth for the Semantik version.
The version is read from:
1. Package metadata (when installed via pip/uv)
2. VERSION file at repo root (for development)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to VERSION file relative to this module
# In development: packages/shared/version.py -> ../../VERSION
# The VERSION file should be at the repository root
_VERSION_FILE_PATHS = [
    Path(__file__).parent.parent.parent / "VERSION",  # From packages/shared/
    Path("/app/VERSION"),  # Docker container path
]


def _read_version_file() -> str | None:
    """Read version from VERSION file."""
    for path in _VERSION_FILE_PATHS:
        if path.is_file():
            try:
                return path.read_text().strip()
            except OSError:
                continue
    return None


@lru_cache(maxsize=1)
def get_version() -> str:
    """Get the current Semantik version.

    Reads from package metadata first (for installed packages),
    then falls back to VERSION file (for development).

    Returns:
        The current Semantik version string.
    """
    # Try package metadata first (works when installed)
    try:
        from importlib.metadata import version

        return version("semantik")
    except Exception:
        pass

    # Fall back to VERSION file
    file_version = _read_version_file()
    if file_version:
        return file_version

    # Last resort fallback (should never happen in normal use)
    logger.warning("Could not determine Semantik version, using fallback")
    return "0.0.0"


# Convenience export
__version__ = get_version()
