"""Centralized normalization helpers for parser metadata and inputs.

This module provides consistent normalization across the parser system:
- Extension normalization (lowercase, leading dot)
- MIME type normalization
- Protected metadata key enforcement
- Safe metadata merging that prevents callers from overriding parser-set keys
"""

from __future__ import annotations

from typing import Any

# Keys that parsers must set and callers cannot override.
# These are critical for system integrity (e.g., knowing which parser produced output).
PROTECTED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "parser",
        "filename",
        "file_extension",
        "file_type",
        "mime_type",
    }
)


def normalize_extension(ext: str | None) -> str:
    """Normalize a file extension to lowercase with leading dot.

    Args:
        ext: File extension (with or without leading dot), or None.

    Returns:
        Normalized extension (e.g., ".txt", ".pdf") or empty string if None/empty.

    Examples:
        >>> normalize_extension(".TXT")
        '.txt'
        >>> normalize_extension("pdf")
        '.pdf'
        >>> normalize_extension(None)
        ''
        >>> normalize_extension("")
        ''
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
        ext: File extension (normalized or not).

    Returns:
        File type string without leading dot.

    Examples:
        >>> normalize_file_type(".txt")
        'txt'
        >>> normalize_file_type("pdf")
        'pdf'
        >>> normalize_file_type("")
        ''
    """
    return ext.lstrip(".").lower()


def normalize_mime_type(mime_type: str | None) -> str | None:
    """Normalize MIME type to lowercase.

    Args:
        mime_type: MIME type string or None.

    Returns:
        Lowercase MIME type or None if input is None/empty.

    Examples:
        >>> normalize_mime_type("Application/PDF")
        'application/pdf'
        >>> normalize_mime_type("  text/html  ")
        'text/html'
        >>> normalize_mime_type(None)
        None
        >>> normalize_mime_type("")
        None
    """
    if not mime_type:
        return None
    normalized = mime_type.strip().lower()
    return normalized if normalized else None


def build_parser_metadata(
    *,
    parser_name: str,
    filename: str | None = None,
    file_extension: str | None = None,
    mime_type: str | None = None,
    caller_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata dict with protected keys that cannot be overridden.

    This function merges caller-provided metadata first, then overwrites
    protected keys (parser, file_extension, file_type) to ensure they
    are always set correctly by the parser.

    Args:
        parser_name: Name of the parser (e.g., "text", "unstructured").
        filename: Document filename.
        file_extension: File extension (will be normalized).
        mime_type: MIME type (will be normalized).
        caller_metadata: Optional metadata from caller (cannot override protected keys).

    Returns:
        Merged metadata dict with protected keys guaranteed to be correct.

    Example:
        >>> # Even if caller tries to override 'parser', it won't work:
        >>> build_parser_metadata(
        ...     parser_name="text",
        ...     file_extension=".txt",
        ...     caller_metadata={"parser": "malicious", "custom": "value"}
        ... )
        {'custom': 'value', 'filename': 'document', 'file_extension': '.txt',
         'file_type': 'txt', 'mime_type': None, 'parser': 'text'}
    """
    import mimetypes

    ext_norm = normalize_extension(file_extension)

    # Start with caller metadata (or empty dict)
    result = dict(caller_metadata or {})

    # Determine filename: explicit > default (caller cannot override).
    resolved_filename = filename or "document"

    # Overwrite with protected/required keys - these cannot be overridden by caller
    result["filename"] = resolved_filename
    result["file_extension"] = ext_norm
    result["file_type"] = normalize_file_type(ext_norm)
    result["parser"] = parser_name

    # Determine MIME type: explicit > guess > default.
    # This guarantees a non-null, normalized value for downstream code.
    mime_norm = normalize_mime_type(mime_type)
    if mime_norm is None:
        guessed_mime: str | None = None
        if resolved_filename and resolved_filename != "document":
            guessed_mime, _ = mimetypes.guess_type(resolved_filename)
        if guessed_mime is None and ext_norm:
            guessed_mime, _ = mimetypes.guess_type(f"document{ext_norm}")
        result["mime_type"] = normalize_mime_type(guessed_mime) or "application/octet-stream"
    else:
        result["mime_type"] = mime_norm

    return result
