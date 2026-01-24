"""Content loader for the pipeline executor.

This module provides the PipelineLoader class that loads file content from
various sources and computes content hashes for change detection.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from shared.pipeline.types import FileReference, LoadResult

if TYPE_CHECKING:
    from shared.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class LoadError(Exception):
    """Error raised when loading file content fails.

    Attributes:
        file_uri: URI of the file that failed to load
        reason: Human-readable error description
    """

    def __init__(self, file_uri: str, reason: str) -> None:
        self.file_uri = file_uri
        self.reason = reason
        super().__init__(f"Failed to load {file_uri}: {reason}")


class PipelineLoader:
    """Loads file content and computes SHA-256 hashes.

    This class handles content loading for files enumerated by connectors.
    For file:// URIs, it reads directly from the filesystem. For other
    schemes, it delegates to the connector's load_content() method.

    Example:
        ```python
        loader = PipelineLoader(connector=my_connector)
        result = await loader.load(file_ref)
        print(f"Hash: {result.content_hash}")
        print(f"Size: {len(result.content)} bytes")
        ```
    """

    def __init__(self, connector: BaseConnector | None = None) -> None:
        """Initialize the loader.

        Args:
            connector: Optional connector for loading non-file:// URIs.
                      Required if loading content from remote sources.
        """
        self.connector = connector

    async def load(self, file_ref: FileReference) -> LoadResult:
        """Load content from a file reference.

        Args:
            file_ref: File reference to load content from

        Returns:
            LoadResult with content bytes and SHA-256 hash

        Raises:
            LoadError: If content cannot be loaded
        """
        try:
            content = await self._load_content(file_ref)
            content_hash = self._compute_hash(content)

            return LoadResult(
                file_ref=file_ref,
                content=content,
                content_hash=content_hash,
                retention="ephemeral",
                local_path=file_ref.source_metadata.get("local_path"),
            )

        except LoadError:
            raise
        except Exception as e:
            logger.error("Failed to load %s: %s", file_ref.uri, e, exc_info=True)
            raise LoadError(file_ref.uri, str(e)) from e

    async def _load_content(self, file_ref: FileReference) -> bytes:
        """Load raw content bytes from a file reference.

        Args:
            file_ref: File reference to load

        Returns:
            Raw content bytes

        Raises:
            LoadError: If content cannot be loaded
        """
        # Check for local_path in source_metadata (set by connectors)
        local_path = file_ref.source_metadata.get("local_path")
        if local_path:
            return self._load_from_path(local_path, file_ref.uri)

        # Parse URI scheme
        parsed = urlparse(file_ref.uri)
        scheme = parsed.scheme.lower()

        # Handle file:// URIs
        # Note: This method trusts that file_ref comes from a validated connector.
        # FileReference objects should not be constructed from untrusted input.
        # Connectors are responsible for path traversal protection (see LocalFileConnector).
        if scheme == "file":
            # Extract path from file:// URI
            path = parsed.path
            if not path:
                raise LoadError(file_ref.uri, "Invalid file:// URI: no path")
            return self._load_from_path(path, file_ref.uri)

        # For other schemes, delegate to connector
        if self.connector is None:
            raise LoadError(
                file_ref.uri,
                f"No connector configured to load {scheme}:// URIs",
            )

        content: bytes = await self.connector.load_content(file_ref)
        return content

    def _load_from_path(self, path: str, uri: str) -> bytes:
        """Load content from a local filesystem path.

        Args:
            path: Local filesystem path
            uri: Original URI for error messages

        Returns:
            File content bytes

        Raises:
            LoadError: If file cannot be read
        """
        file_path = Path(path)

        if not file_path.exists():
            raise LoadError(uri, f"File not found: {path}")

        if not file_path.is_file():
            raise LoadError(uri, f"Not a file: {path}")

        try:
            return file_path.read_bytes()
        except PermissionError as e:
            raise LoadError(uri, f"Permission denied: {path}") from e
        except OSError as e:
            raise LoadError(uri, f"Cannot read file: {e}") from e

    def _compute_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of content.

        Args:
            content: Content bytes to hash

        Returns:
            Lowercase hex-encoded SHA-256 hash (64 characters)
        """
        return hashlib.sha256(content).hexdigest()


__all__ = [
    "PipelineLoader",
    "LoadError",
]
