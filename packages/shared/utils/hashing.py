"""Content hashing utilities for document deduplication."""

import hashlib
from pathlib import Path

# Chunk size for streaming file reads (8KB)
HASH_CHUNK_SIZE = 8192


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


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file contents using streaming reads.

    Uses chunked reading (8KB) to efficiently handle large files without
    loading entire content into memory. Produces identical output to
    compute_content_hash() for the same content.

    Args:
        file_path: Path to the file to hash

    Returns:
        Lowercase hex string (64 characters) of SHA-256 hash

    Raises:
        OSError: If file cannot be read

    Example:
        >>> from pathlib import Path
        >>> compute_file_hash(Path("/path/to/file.txt"))
        '315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3'
    """
    sha256_hash = hashlib.sha256()

    try:
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        raise OSError(f"Failed to calculate hash for {file_path}: {e}") from e
