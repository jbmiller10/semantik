"""Parser exception hierarchy.

Provides distinct exception types so connectors can differentiate between
"this file format isn't supported" vs "something went wrong during extraction".
"""

from __future__ import annotations


class ParserError(Exception):
    """Base exception for all parser errors."""


class ParserConfigError(ParserError):
    """Raised when parser config validation fails.

    Example:
        raise ParserConfigError("Unknown config option 'stratgy'. Valid: strategy, include_page_breaks")
    """


class UnsupportedFormatError(ParserError):
    """Raised when a file format is not supported by the parser.

    This is a "expected" error - the file simply can't be handled.
    Connectors may want to try a different parser or skip the file.

    Example:
        raise UnsupportedFormatError("TextParser cannot handle .pdf files")
    """


class ExtractionFailedError(ParserError):
    """Raised when extraction fails for reasons other than format support.

    This indicates something went wrong during parsing - corrupt file,
    missing dependencies, resource exhaustion, etc.

    Example:
        raise ExtractionFailedError("PDF parsing failed: file appears corrupted")
    """

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause
