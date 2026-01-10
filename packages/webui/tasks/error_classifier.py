"""Error classification for document retry decisions.

This module provides utilities to classify errors into categories that determine
whether a failed document should be automatically retried.

Categories:
- TRANSIENT: Temporary errors that may succeed on retry (network, rate limits, etc.)
- PERMANENT: Errors that will not succeed on retry (corrupt files, auth failures, etc.)
- UNKNOWN: Errors that cannot be classified - treated as retryable with caution
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Optional httpx import - gracefully handle if not available
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]


class ErrorCategory(str, Enum):
    """Classification of errors for retry decisions."""

    TRANSIENT = "transient"  # Retry-safe: network, timeouts, rate limits
    PERMANENT = "permanent"  # Do not retry: corrupt files, auth failures
    UNKNOWN = "unknown"  # Unknown - retry with caution


# Patterns indicating transient (retryable) errors
TRANSIENT_PATTERNS = frozenset(
    [
        # Network issues
        "timeout",
        "timed out",
        "connection refused",
        "connection reset",
        "connection error",
        "network",
        "dns",
        "socket",
        # HTTP transient errors
        "503",
        "502",
        "504",
        "429",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "too many requests",
        "rate limit",
        # Resource issues
        "out of memory",
        "cuda out of memory",
        "resource exhausted",
        "memory",
        # Service-specific transient
        "vecpipe",
        "embedding request failed",
        "upsert request failed",
        "qdrant",
        "temporarily unavailable",
        "retry",
    ]
)

# Patterns indicating permanent (non-retryable) errors
PERMANENT_PATTERNS = frozenset(
    [
        # File issues
        "file not found",
        "filenotfound",
        "no such file",
        "permission denied",
        "access denied",
        "unsupported format",
        "unsupported file",
        "corrupt",
        "corrupted",
        "invalid file",
        "cannot read",
        "unreadable",
        # Content issues
        "empty content",
        "no text extracted",
        "encoding error",
        "decode error",
        "invalid encoding",
        "zero bytes",
        # Auth issues
        "401",
        "403",
        "404",
        "unauthorized",
        "forbidden",
        "not found",
        "authentication failed",
        "invalid credentials",
        # Validation issues
        "400",
        "422",
        "bad request",
        "validation error",
        "invalid input",
        "schema",
        # Permanent resource issues
        "dimension mismatch",
        "collection not found",
        "model not found",
    ]
)

# HTTP status codes that are definitely transient
TRANSIENT_HTTP_CODES = frozenset([429, 500, 502, 503, 504])

# HTTP status codes that are definitely permanent
PERMANENT_HTTP_CODES = frozenset([400, 401, 403, 404, 405, 422])


def classify_error(error: Exception | str) -> ErrorCategory:
    """Classify an error for retry eligibility.

    Args:
        error: The exception or error message string to classify

    Returns:
        ErrorCategory indicating whether the error is transient (retryable),
        permanent (not retryable), or unknown (retry with caution)

    Examples:
        >>> classify_error(TimeoutError("Connection timed out"))
        ErrorCategory.TRANSIENT
        >>> classify_error("File not found: document.pdf")
        ErrorCategory.PERMANENT
        >>> classify_error("Unknown processing error")
        ErrorCategory.UNKNOWN
    """
    # Handle httpx exceptions if available
    if HTTPX_AVAILABLE and httpx is not None:
        # Check HTTP status code errors first
        if isinstance(error, httpx.HTTPStatusError):
            code = error.response.status_code
            if code in TRANSIENT_HTTP_CODES:
                return ErrorCategory.TRANSIENT
            if code in PERMANENT_HTTP_CODES:
                return ErrorCategory.PERMANENT

        # Network/connection errors are always transient
        if isinstance(error, httpx.RequestError):
            return ErrorCategory.TRANSIENT

        # Timeout errors are transient
        if isinstance(error, httpx.TimeoutException):
            return ErrorCategory.TRANSIENT

    # Check for common transient exception types
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        # OSError can be many things, but connection-related ones are transient
        error_str = str(error).lower()
        if any(p in error_str for p in ["connection", "network", "socket", "timeout"]):
            return ErrorCategory.TRANSIENT

    # Pattern match on error string
    error_str = str(error).lower()

    # Check transient patterns first (more likely in network/service errors)
    for pattern in TRANSIENT_PATTERNS:
        if pattern in error_str:
            return ErrorCategory.TRANSIENT

    # Check permanent patterns
    for pattern in PERMANENT_PATTERNS:
        if pattern in error_str:
            return ErrorCategory.PERMANENT

    # Unknown - we'll allow retry but with caution
    return ErrorCategory.UNKNOWN


def is_retryable(error: Exception | str, max_retries: int = 3, current_retry: int = 0) -> bool:
    """Determine if an error should be retried.

    Args:
        error: The exception or error message to check
        max_retries: Maximum number of retry attempts allowed
        current_retry: Current retry attempt number (0-indexed)

    Returns:
        True if the error should be retried, False otherwise
    """
    if current_retry >= max_retries:
        return False

    category = classify_error(error)

    # Never retry permanent errors
    if category == ErrorCategory.PERMANENT:
        return False

    # Always retry transient errors (within limit)
    if category == ErrorCategory.TRANSIENT:
        return True

    # Unknown errors: retry up to half the max retries
    # This provides some resilience while limiting wasted retries
    return current_retry < (max_retries // 2 + 1)


def get_retry_delay(attempt: int, base_delay: float = 2.0, max_delay: float = 30.0) -> float:
    """Calculate exponential backoff delay for retry attempt.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Initial delay in seconds (default 2.0)
        max_delay: Maximum delay in seconds (default 30.0)

    Returns:
        Delay in seconds before the next retry attempt

    Examples:
        >>> get_retry_delay(0)  # First retry
        2.0
        >>> get_retry_delay(1)  # Second retry
        4.0
        >>> get_retry_delay(2)  # Third retry
        8.0
    """
    delay = base_delay * (2**attempt)
    return min(delay, max_delay)
