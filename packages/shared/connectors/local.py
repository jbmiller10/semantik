"""Local file system connector for directory document sources."""

import fnmatch
import logging
import mimetypes
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

from shared.connectors.base import BaseConnector
from shared.pipeline.types import FileReference

logger = logging.getLogger(__name__)


class LocalFileConnector(BaseConnector):
    """Connector for local filesystem directory sources.

    Enumerates files from a local directory, yielding FileReference objects
    that describe each file without loading or parsing content.

    Config keys:
        path (required): Path to directory to scan
        recursive (optional): Whether to scan subdirectories (default: True)
        include_patterns (optional): List of glob patterns to include
        exclude_patterns (optional): List of glob patterns to exclude

    Example:
        ```python
        connector = LocalFileConnector({"path": "/data/docs", "recursive": True})
        if await connector.authenticate():
            async for file_ref in connector.enumerate():
                print(file_ref.uri, file_ref.change_hint)

        # Check for files that couldn't be enumerated
        skipped = connector.get_skipped_files()
        for path, reason in skipped:
            print(f"Skipped {path}: {reason}")
        ```

    Security:
        This connector validates that all file paths stay within the configured
        base directory to prevent path traversal attacks via symlinks or relative paths.
    """

    PLUGIN_ID: ClassVar[str] = "directory"
    PLUGIN_TYPE: ClassVar[str] = "connector"
    METADATA: ClassVar[dict[str, Any]] = {
        "name": "Local Directory",
        "description": "Index files from a local directory on the server",
        "icon": "folder",
        "supports_sync": True,
    }

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the connector with tracking for skipped files."""
        super().__init__(config)
        self._skipped_files: list[tuple[str, str]] = []

    def get_skipped_files(self) -> list[tuple[str, str]]:
        """Get list of files that were skipped during enumeration.

        Returns:
            List of (path, reason) tuples for files that couldn't be stat'd.
        """
        return list(self._skipped_files)

    @classmethod
    def get_config_fields(cls) -> list[dict[str, Any]]:
        return [
            {
                "name": "path",
                "type": "text",
                "label": "Directory Path",
                "description": "Absolute path to the directory to index",
                "required": True,
                "placeholder": "/path/to/documents",
            },
            {
                "name": "recursive",
                "type": "boolean",
                "label": "Recursive",
                "description": "Include files from subdirectories",
                "default": True,
            },
            {
                "name": "include_patterns",
                "type": "glob_list",
                "label": "Include Patterns",
                "description": "Glob patterns to include (e.g., *.md, *.py)",
                "placeholder": "*.md, *.txt",
            },
            {
                "name": "exclude_patterns",
                "type": "glob_list",
                "label": "Exclude Patterns",
                "description": "Glob patterns to exclude",
                "placeholder": "*.log, __pycache__/**",
            },
        ]

    @classmethod
    def get_secret_fields(cls) -> list[dict[str, Any]]:
        return []

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

    def _should_include_file(self, file_path: Path) -> bool:
        """Check if a file should be included based on patterns.

        Args:
            file_path: The file path to check

        Returns:
            True if the file should be included, False otherwise
        """
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

    def _infer_content_type(self, file_path: Path) -> str:
        """Infer semantic content type from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Content type string (document, code, message, etc.)
        """
        ext = file_path.suffix.lower()

        # Code files
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".java",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".swift",
            ".kt",
            ".scala",
            ".sh",
            ".bash",
            ".zsh",
            ".sql",
            ".graphql",
        }
        if ext in code_extensions:
            return "code"

        # Configuration files
        config_extensions = {
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            ".xml",
            ".env",
        }
        if ext in config_extensions:
            return "config"

        # Default to document for everything else
        return "document"

    async def enumerate(
        self,
        source_id: int | None = None,  # noqa: ARG002
    ) -> AsyncIterator[FileReference]:
        """Yield file references from the directory.

        Walks the directory and yields FileReference objects for each file
        that passes the include/exclude filters. Does not load or parse content.

        Args:
            source_id: Optional source ID (unused for local files)

        Yields:
            FileReference for each matching file.
        """
        # Clear skipped files from any previous enumeration
        self._skipped_files = []

        source_path = Path(self._config["path"])
        recursive = self._config.get("recursive", True)

        iterator = source_path.rglob("*") if recursive else source_path.iterdir()

        for file_path in iterator:
            if not file_path.is_file():
                continue
            if not self._is_safe_path(file_path, source_path):
                logger.warning(f"Skipping path outside base directory: {file_path}")
                continue
            if not self._should_include_file(file_path):
                continue

            try:
                stat = file_path.stat()
                mime_type, _ = mimetypes.guess_type(str(file_path))
                rel_path = str(file_path.relative_to(source_path))

                yield FileReference(
                    uri=f"file://{file_path}",
                    source_type="directory",
                    content_type=self._infer_content_type(file_path),
                    filename=file_path.name,
                    extension=file_path.suffix.lower() or None,
                    mime_type=mime_type,
                    size_bytes=stat.st_size,
                    change_hint=f"mtime:{int(stat.st_mtime)},size:{stat.st_size}",
                    metadata={
                        "source": {
                            "local_path": str(file_path),
                            "relative_path": rel_path,
                        }
                    },
                )
            except OSError as e:
                logger.warning(f"Cannot stat {file_path}: {e}")
                self._skipped_files.append((str(file_path), str(e)))
                continue

    async def load_content(self, file_ref: FileReference) -> bytes:
        """Load raw content bytes from a local file.

        Args:
            file_ref: File reference with local_path in metadata.source

        Returns:
            Raw content bytes

        Raises:
            ValueError: If local_path is missing from metadata.source
            OSError: If file cannot be read
        """
        local_path = file_ref.metadata.get("source", {}).get("local_path")
        if not local_path:
            raise ValueError(f"Missing local_path in metadata.source for {file_ref.uri}")

        path = Path(local_path)

        # Validate path is within base directory
        source_path = Path(self._config["path"])
        if not self._is_safe_path(path, source_path):
            raise ValueError(f"Path traversal detected: {local_path}")

        return path.read_bytes()
