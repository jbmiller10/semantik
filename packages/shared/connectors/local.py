"""Local file system connector for directory document sources."""

import fnmatch
import logging
import mimetypes
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from shared.connectors.base import BaseConnector
from shared.dtos.ingestion import IngestedDocument
from shared.text_processing.extraction import extract_and_serialize
from shared.utils.hashing import compute_content_hash

logger = logging.getLogger(__name__)

# Supported file extensions (from DocumentScanningService)
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".text", ".pptx", ".eml", ".md", ".html"}

# Maximum file size (500 MB)
MAX_FILE_SIZE = 500 * 1024 * 1024


class LocalFileConnector(BaseConnector):
    """Connector for local filesystem directory sources.

    Config keys:
        path (required): Path to directory to scan
        recursive (optional): Whether to scan subdirectories (default: True)
        include_patterns (optional): List of glob patterns to include
        exclude_patterns (optional): List of glob patterns to exclude

    Example:
        ```python
        connector = LocalFileConnector({"path": "/data/docs", "recursive": True})
        if await connector.authenticate():
            async for doc in connector.load_documents():
                print(doc.unique_id, doc.content_hash)
        ```

    Security:
        This connector validates that all file paths stay within the configured
        base directory to prevent path traversal attacks via symlinks or relative paths.
    """

    def validate_config(self) -> None:
        """Validate required config keys."""
        if "path" not in self._config:
            raise ValueError("LocalFileConnector requires 'path' in config")

    async def authenticate(self) -> bool:
        """Verify the directory exists and is accessible.

        Returns:
            True if directory exists and is readable.

        Raises:
            ValueError: If path doesn't exist or is not a directory.
        """
        path = Path(self._config["path"])
        if not path.exists():
            raise ValueError(f"Directory does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        return True

    def _is_safe_path(self, file_path: Path, base_path: Path) -> bool:
        """Verify that a file path is safely within the base directory.

        Protects against path traversal attacks by:
        1. Resolving symlinks to get the canonical path
        2. Verifying the resolved path is within the base directory

        Args:
            file_path: The file path to validate
            base_path: The base directory that should contain the file

        Returns:
            True if the path is safe, False otherwise
        """
        try:
            # Resolve both paths to their canonical form (resolves symlinks)
            resolved_file = file_path.resolve()
            resolved_base = base_path.resolve()
            # Check if the resolved file path is within the base directory
            return resolved_file.is_relative_to(resolved_base)
        except (OSError, ValueError):
            # OSError: permission denied, broken symlink, etc.
            # ValueError: is_relative_to can raise on edge cases
            return False

    async def load_documents(
        self,
        source_id: int | None = None,  # noqa: ARG002
    ) -> AsyncIterator[IngestedDocument]:
        """Yield documents from the directory.

        Walks the directory, reads and parses each supported file,
        and yields IngestedDocument instances with full content.

        Yields:
            IngestedDocument for each supported file.
        """
        source_path = Path(self._config["path"])
        recursive = self._config.get("recursive", True)

        if recursive:
            # Use followlinks=False to avoid following symlinks during traversal
            for root, _, files in os.walk(source_path, followlinks=False):
                for filename in files:
                    file_path = Path(root) / filename
                    # Security: verify path stays within base directory
                    if not self._is_safe_path(file_path, source_path):
                        logger.warning(f"Skipping path outside base directory: {file_path}")
                        continue
                    if not self._should_include_file(file_path):
                        continue
                    doc = await self._process_file(file_path)
                    if doc is not None:
                        yield doc
        else:
            for filename in os.listdir(source_path):
                file_path = source_path / filename
                if file_path.is_file():
                    # Security: verify path stays within base directory
                    if not self._is_safe_path(file_path, source_path):
                        logger.warning(f"Skipping path outside base directory: {file_path}")
                        continue
                    if not self._should_include_file(file_path):
                        continue
                    doc = await self._process_file(file_path)
                    if doc is not None:
                        yield doc

    def _should_include_file(self, file_path: Path) -> bool:
        include_patterns = [p for p in (self._config.get("include_patterns") or []) if p]
        exclude_patterns = [p for p in (self._config.get("exclude_patterns") or []) if p]

        if not include_patterns and not exclude_patterns:
            return True

        source_path = self._config.get("path")
        if not source_path:
            return True

        rel_path = os.path.relpath(str(file_path), start=str(source_path)).replace(os.sep, "/")
        file_name = file_path.name

        if exclude_patterns and any(
            fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(file_name, pattern) for pattern in exclude_patterns
        ):
            return False

        if include_patterns:
            return any(
                fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(file_name, pattern)
                for pattern in include_patterns
            )

        return True

    async def _process_file(self, file_path: Path) -> IngestedDocument | None:
        """Process a single file and return IngestedDocument or None if skipped.

        Args:
            file_path: Path to the file

        Returns:
            IngestedDocument or None if file should be skipped
        """
        # Check extension
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return None

        # Check file size
        try:
            stat = file_path.stat()
            if stat.st_size > MAX_FILE_SIZE:
                logger.warning(f"Skipping file too large: {file_path} ({stat.st_size} bytes)")
                return None
            file_size = stat.st_size
        except Exception as e:
            logger.error(f"Cannot access file {file_path}: {e}")
            return None

        # Parse document content
        try:
            elements = extract_and_serialize(str(file_path))
            content = "\n\n".join(text for text, _ in elements)

            # Collect metadata from first element (has filename)
            base_metadata = elements[0][1] if elements else {}
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return None

        # Skip empty documents
        if not content.strip():
            logger.debug(f"Skipping empty document: {file_path}")
            return None

        # Compute hash of parsed text content
        content_hash = compute_content_hash(content)

        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))

        # Build metadata
        metadata: dict[str, Any] = {
            **base_metadata,
            "file_size": file_size,
            "mime_type": mime_type,
        }

        return IngestedDocument(
            content=content,
            unique_id=f"file://{file_path}",
            source_type="directory",
            metadata=metadata,
            content_hash=content_hash,
            file_path=str(file_path),
        )
