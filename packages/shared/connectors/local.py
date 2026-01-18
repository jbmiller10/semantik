"""Local file system connector for directory document sources."""

import asyncio
import fnmatch
import logging
import mimetypes
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar, Literal, TypedDict, cast

import billiard

from shared.connectors.base import BaseConnector
from shared.dtos.ingestion import IngestedDocument
from shared.text_processing.parsers import ExtractionFailedError, UnsupportedFormatError, parse_content
from shared.utils.hashing import compute_content_hash

logger = logging.getLogger(__name__)

# Supported file extensions (from DocumentScanningService)
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".text", ".pptx", ".eml", ".md", ".html"}

# Maximum file size (500 MB)
MAX_FILE_SIZE = 500 * 1024 * 1024

# Parallel processing settings
DEFAULT_PARALLEL_WORKERS = 4
MAX_PARALLEL_WORKERS = 0  # 0 = no limit, use all available CPUs
PARALLEL_BATCH_SIZE = 50


class _WorkerSuccessData(TypedDict):
    content: str
    unique_id: str
    source_type: str
    metadata: dict[str, Any]
    content_hash: str
    file_path: str


class _WorkerArgs(TypedDict, total=False):
    file_path: str
    base_path: str
    parser_overrides: dict[str, str]
    parser_configs: dict[str, dict[str, Any]]


class _WorkerSuccess(TypedDict):
    status: Literal["success"]
    data: _WorkerSuccessData


class _WorkerSkipped(TypedDict):
    status: Literal["skipped"]
    reason: str
    path: str


class _WorkerError(TypedDict):
    status: Literal["error"]
    reason: str
    path: str


_WorkerResult = _WorkerSuccess | _WorkerSkipped | _WorkerError


def _process_file_worker(worker_input: str | _WorkerArgs) -> _WorkerResult:
    """Module-level worker function for process pools (billiard.Pool).

    Process pools require picklable functions, so this must be
    at module level rather than an instance method.

    Args:
        worker_input: Either a string file path, or a dict containing the file path and
            additional context (base_path, parser configs, overrides).

    Returns:
        Dictionary with 'status' key indicating outcome:
        - status='success': Contains 'data' with document fields
        - status='skipped': Contains 'reason' (file_too_large, empty_content)
        - status='error': Contains 'reason' with error message
    """
    if isinstance(worker_input, str):
        file_path_str = worker_input
        base_path_str = ""
        parser_overrides: dict[str, str] | None = None
        parser_configs: dict[str, dict[str, Any]] | None = None
    else:
        file_path_str = worker_input.get("file_path", "")
        base_path_str = worker_input.get("base_path", "")
        parser_overrides = worker_input.get("parser_overrides")
        parser_configs = worker_input.get("parser_configs")

    file_path = Path(file_path_str)
    base_path = Path(base_path_str) if base_path_str else None

    # Check file size
    try:
        stat = file_path.stat()
        if stat.st_size > MAX_FILE_SIZE:
            logger.warning(f"Skipping file too large: {file_path} ({stat.st_size} bytes)")
            return {"status": "skipped", "reason": "file_too_large", "path": file_path_str}
        file_size = stat.st_size
    except Exception as e:
        logger.error(f"Cannot access file {file_path}: {e}")
        return {"status": "error", "reason": f"Cannot access file: {e}", "path": file_path_str}

    # Compute source path (relative to configured base) when available.
    if base_path is not None:
        try:
            source_path = str(file_path.relative_to(base_path))
        except Exception:
            source_path = file_path.name
    else:
        source_path = file_path.name

    # Parse document content
    try:
        result = parse_content(
            file_path.read_bytes(),
            filename=file_path.name,
            file_extension=file_path.suffix.lower(),
            metadata={
                "source_type": "directory",
                "source_path": source_path,
                "local_file_path": str(file_path),
            },
            parser_overrides=parser_overrides,
            parser_configs=parser_configs,
        )
        content = result.text
        base_metadata = result.metadata
    except UnsupportedFormatError:
        logger.debug(f"Skipping unsupported format: {file_path}")
        return {"status": "skipped", "reason": "unsupported_format", "path": file_path_str}
    except ExtractionFailedError as e:
        logger.error(f"Failed to parse {file_path}: {e}")
        return {"status": "error", "reason": f"Failed to parse: {e}", "path": file_path_str}
    except Exception as e:
        logger.error(f"Failed to parse {file_path}: {e}")
        return {"status": "error", "reason": f"Failed to parse: {e}", "path": file_path_str}

    # Skip empty documents
    if not content.strip():
        logger.debug(f"Skipping empty document: {file_path}")
        return {"status": "skipped", "reason": "empty_content", "path": file_path_str}

    # Compute hash of parsed text content
    content_hash = compute_content_hash(content)

    # Get MIME type
    mime_type, _ = mimetypes.guess_type(str(file_path))

    # Return as dict (IngestedDocument will be created in main process)
    worker_metadata: dict[str, Any] = {
        **base_metadata,
        "file_size": file_size,
    }
    if mime_type:
        worker_metadata["mime_type"] = mime_type

    return {
        "status": "success",
        "data": {
            "content": content,
            "unique_id": f"file://{file_path}",
            "source_type": "directory",
            "metadata": worker_metadata,
            "content_hash": content_hash,
            "file_path": str(file_path),
        },
    }


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

    PLUGIN_ID: ClassVar[str] = "directory"
    PLUGIN_TYPE: ClassVar[str] = "connector"
    METADATA: ClassVar[dict[str, Any]] = {
        "name": "Local Directory",
        "description": "Index files from a local directory on the server",
        "icon": "folder",
        "supports_sync": True,
    }

    PARSER_OVERRIDES: ClassVar[dict[str, str]] = {}

    def _build_parser_configs(self) -> dict[str, dict[str, Any]]:
        return {
            "unstructured": {"strategy": self._config.get("parsing_strategy", "auto")},
            "text": {"encoding": "utf-8", "errors": "replace"},
        }

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
            {
                "name": "parsing_strategy",
                "type": "select",
                "label": "Parsing Strategy",
                "description": "Unstructured parsing strategy (only applies to complex formats like PDF/DOCX)",
                "default": "auto",
                "options": [
                    {"value": "auto", "label": "Auto (recommended)"},
                    {"value": "fast", "label": "Fast (less accurate)"},
                    {"value": "hi_res", "label": "High Resolution (slower)"},
                    {"value": "ocr_only", "label": "OCR Only (scanned docs)"},
                ],
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

    async def load_documents(
        self,
        source_id: int | None = None,  # noqa: ARG002
    ) -> AsyncIterator[IngestedDocument]:
        """Yield documents from the directory.

        Walks the directory, reads and parses each supported file,
        and yields IngestedDocument instances with full content.

        Uses parallel processing for file parsing when enabled.

        Yields:
            IngestedDocument for each supported file.
        """
        source_path = Path(self._config["path"])
        recursive = self._config.get("recursive", True)
        parser_overrides = dict(self.PARSER_OVERRIDES) if self.PARSER_OVERRIDES else None
        parser_configs = self._build_parser_configs()

        # Collect all file paths first (fast)
        file_paths: list[Path] = []

        if recursive:
            for root, _, files in os.walk(source_path, followlinks=False):
                for filename in files:
                    file_path = Path(root) / filename
                    if not self._is_safe_path(file_path, source_path):
                        logger.warning(f"Skipping path outside base directory: {file_path}")
                        continue
                    if not self._should_include_file(file_path):
                        continue
                    if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        file_paths.append(file_path)
        else:
            for filename in os.listdir(source_path):
                file_path = source_path / filename
                if file_path.is_file():
                    if not self._is_safe_path(file_path, source_path):
                        logger.warning(f"Skipping path outside base directory: {file_path}")
                        continue
                    if not self._should_include_file(file_path):
                        continue
                    if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        file_paths.append(file_path)

        logger.info("Found %d files to process in %s", len(file_paths), source_path)

        if not file_paths:
            return

        # Check if parallel processing is enabled
        try:
            from shared.config import settings

            use_parallel = getattr(settings, "PARALLEL_INGESTION_ENABLED", True)
            num_workers = getattr(settings, "PARALLEL_INGESTION_WORKERS", 0)
            max_workers = getattr(settings, "PARALLEL_INGESTION_MAX_WORKERS", 0)

            # 0 means auto-detect based on CPU count
            if num_workers <= 0:
                num_workers = os.cpu_count() or DEFAULT_PARALLEL_WORKERS

            # Apply max cap if set (0 = no limit)
            if max_workers > 0:
                num_workers = min(num_workers, max_workers)

            # Don't use more workers than files
            num_workers = min(num_workers, len(file_paths))
        except ImportError:
            use_parallel = True
            num_workers = min(os.cpu_count() or DEFAULT_PARALLEL_WORKERS, len(file_paths))

        if use_parallel and len(file_paths) > 1:
            # Parallel processing
            logger.info("Processing %d files with %d parallel workers", len(file_paths), num_workers)
            async for doc in self._process_files_parallel(
                file_paths,
                num_workers,
                base_path=source_path,
                parser_overrides=parser_overrides,
                parser_configs=parser_configs,
            ):
                yield doc
        else:
            # Sequential processing (fallback)
            for file_path in file_paths:
                doc = await self._process_file(file_path)
                if doc is not None:
                    yield doc

    async def _process_files_parallel(
        self,
        file_paths: list[Path],
        num_workers: int,
        *,
        base_path: Path,
        parser_overrides: dict[str, str] | None,
        parser_configs: dict[str, dict[str, Any]] | None,
    ) -> AsyncIterator[IngestedDocument]:
        """Process files in parallel batches using multiple processes.

        Uses billiard.Pool which allows spawning child processes even from
        daemon processes (like Celery workers), bypassing Python's GIL for
        true CPU parallelism.

        Args:
            file_paths: List of file paths to process
            num_workers: Number of parallel workers

        Yields:
            IngestedDocument for each successfully processed file
        """
        loop = asyncio.get_running_loop()

        # Use billiard.Pool which works from daemon processes (unlike multiprocessing)
        # This is what Celery uses internally. Create pool ONCE and reuse for all batches.
        with billiard.Pool(processes=num_workers) as pool:
            # Process in batches to avoid memory issues
            for batch_start in range(0, len(file_paths), PARALLEL_BATCH_SIZE):
                batch_end = min(batch_start + PARALLEL_BATCH_SIZE, len(file_paths))
                batch = file_paths[batch_start:batch_end]
                batch_args: list[_WorkerArgs] = [
                    {
                        "file_path": str(fp),
                        "base_path": str(base_path),
                        "parser_overrides": parser_overrides or {},
                        "parser_configs": parser_configs or {},
                    }
                    for fp in batch
                ]

                # Run pool.map in a thread to avoid blocking the event loop
                try:
                    results: list[_WorkerResult] = await loop.run_in_executor(
                        None,  # Use default thread pool
                        pool.map,
                        _process_file_worker,
                        batch_args,
                    )
                except Exception as exc:
                    if isinstance(exc, MemoryError | SystemExit | KeyboardInterrupt):
                        raise
                    logger.error(
                        "Parallel file processing batch failed (%d-%d of %d): %s; falling back to sequential",
                        batch_start,
                        batch_end,
                        len(file_paths),
                        exc,
                        exc_info=True,
                    )
                    for file_path in batch:
                        doc = await self._process_file(file_path)
                        if doc is not None:
                            yield doc
                    continue

                for result in results:
                    status = result["status"]
                    if status == "success":
                        success_result = cast(_WorkerSuccess, result)
                        yield IngestedDocument(**success_result["data"])
                    elif status == "skipped":
                        logger.debug("Skipped %s: %s", result.get("path"), result.get("reason"))
                    else:
                        logger.error("Error processing %s: %s", result.get("path"), result.get("reason"))

                logger.debug(
                    "Processed batch %d-%d of %d files",
                    batch_start,
                    batch_end,
                    len(file_paths),
                )

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

        # Compute relative path from configured base directory
        base_path = Path(self._config["path"])
        try:
            rel_path = str(file_path.relative_to(base_path))
        except ValueError:
            rel_path = file_path.name

        # Parse document content
        try:
            result = parse_content(
                file_path.read_bytes(),
                filename=file_path.name,
                file_extension=file_path.suffix.lower(),
                metadata={
                    "source_type": "directory",
                    "source_path": rel_path,
                    "local_file_path": str(file_path),
                },
                parser_overrides=self.PARSER_OVERRIDES or None,
                parser_configs=self._build_parser_configs(),
            )
            content = result.text
            base_metadata = result.metadata
        except UnsupportedFormatError:
            logger.debug(f"Skipping unsupported format: {file_path}")
            return None
        except ExtractionFailedError as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return None
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
        metadata: dict[str, Any] = {**base_metadata, "file_size": file_size}
        if mime_type:
            metadata["mime_type"] = mime_type

        return IngestedDocument(
            content=content,
            unique_id=f"file://{file_path}",
            source_type="directory",
            metadata=metadata,
            content_hash=content_hash,
            file_path=str(file_path),
        )
