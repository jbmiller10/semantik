"""Parser utility functions shared across parser plugins.

Contains common helpers for extension normalization, file type handling,
and metadata construction.
"""

from __future__ import annotations

import mimetypes
from typing import Any


def normalize_extension(ext: str | None) -> str:
    """Normalize a file extension to lowercase with leading dot.

    Args:
        ext: File extension with or without leading dot, or None.

    Returns:
        Normalized extension (e.g., ".pdf") or empty string if None/empty.
    """
    if not ext:
        return ""
    ext = ext.strip().lower()
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    return ext


def normalize_file_type(ext: str) -> str:
    """Convert extension to file type (extension without leading dot).

    Args:
        ext: File extension with or without leading dot.

    Returns:
        File type string (e.g., "pdf" from ".pdf").
    """
    return ext.lstrip(".").lower()


def build_parser_metadata(
    *,
    parser_name: str,
    filename: str | None = None,
    file_extension: str | None = None,
    mime_type: str | None = None,
    caller_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata dict with protected keys.

    Constructs a metadata dictionary for parser output, ensuring
    required keys are always present and correctly formatted.

    Args:
        parser_name: Identifier of the parser (e.g., "text", "unstructured").
        filename: Optional filename (defaults to "document").
        file_extension: Optional file extension.
        mime_type: Optional MIME type (will guess from filename/extension if not provided).
        caller_metadata: Optional metadata from caller to include.

    Returns:
        Metadata dict with keys: filename, file_extension, file_type, parser, mime_type.
    """
    ext_norm = normalize_extension(file_extension)

    # Start with caller metadata (or empty dict)
    result = dict(caller_metadata or {})

    # Determine filename: explicit > default
    resolved_filename = filename or "document"

    # Overwrite with protected/required keys
    result["filename"] = resolved_filename
    result["file_extension"] = ext_norm
    result["file_type"] = normalize_file_type(ext_norm)
    result["parser"] = parser_name

    # Determine MIME type: explicit > guess > default
    mime_norm = mime_type.strip().lower() if mime_type else None
    if mime_norm is None:
        guessed_mime: str | None = None
        if resolved_filename and resolved_filename != "document":
            guessed_mime, _ = mimetypes.guess_type(resolved_filename)
        if guessed_mime is None and ext_norm:
            guessed_mime, _ = mimetypes.guess_type(f"document{ext_norm}")
        result["mime_type"] = guessed_mime.lower() if guessed_mime else "application/octet-stream"
    else:
        result["mime_type"] = mime_norm

    return result


__all__ = [
    "normalize_extension",
    "normalize_file_type",
    "build_parser_metadata",
]
