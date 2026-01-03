"""Semantik plugin development CLI."""

from __future__ import annotations


def _get_version() -> str:
    """Get version from package metadata or VERSION file."""
    try:
        from importlib.metadata import version

        return version("semantik-cli")
    except Exception:
        pass

    # Fall back to VERSION file (for development)
    from pathlib import Path

    version_file = Path(__file__).parent.parent.parent.parent.parent / "VERSION"
    if version_file.is_file():
        return version_file.read_text().strip()

    return "0.0.0"


__version__ = _get_version()
__all__ = ["__version__"]
