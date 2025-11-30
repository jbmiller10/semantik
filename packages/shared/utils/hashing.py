"""Content hashing utilities for document deduplication."""

import hashlib


def compute_content_hash(content: str | bytes) -> str:
    """Compute SHA-256 hash of content for deduplication.

    Args:
        content: Text content (str) or binary content (bytes)

    Returns:
        Lowercase hex string (64 characters) of SHA-256 hash

    Example:
        >>> compute_content_hash("Hello, world!")
        '315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3'
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    return hashlib.sha256(content).hexdigest()
