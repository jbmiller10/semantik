"""Directory scan service for previewing directory contents without creating collections."""

import hashlib
import logging
import mimetypes
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.webui.api.schemas import DirectoryScanFile, DirectoryScanProgress, DirectoryScanResponse
from packages.webui.websocket_manager import ws_manager

logger = logging.getLogger(__name__)

# Supported file extensions for document scanning
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".text", ".pptx", ".eml", ".md", ".html"}

# Maximum file size (500 MB)
MAX_FILE_SIZE = 500 * 1024 * 1024

# Chunk size for file reading (for hash calculation)
HASH_CHUNK_SIZE = 8192

# Progress update interval (every N files)
PROGRESS_UPDATE_INTERVAL = 50


class DirectoryScanService:
    """Service for scanning directories and previewing document contents."""

    def __init__(self) -> None:
        """Initialize the directory scan service."""

    async def scan_directory_preview(
        self,
        path: str,
        scan_id: str,
        user_id: int,
        recursive: bool = True,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> DirectoryScanResponse:
        """Scan a directory and return preview of documents without saving to database.

        Args:
            path: Path to directory to scan
            scan_id: UUID for tracking this scan session
            user_id: ID of user performing the scan
            recursive: Whether to scan subdirectories
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude

        Returns:
            DirectoryScanResponse with file list and statistics

        Raises:
            ValueError: If path is invalid
            PermissionError: If access denied
            FileNotFoundError: If path doesn't exist
        """
        scan_path = Path(path)

        # Validate path
        if not scan_path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not scan_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        # Check access permissions
        try:
            os.listdir(scan_path)
        except PermissionError as e:
            raise PermissionError(f"Access denied to directory: {path}") from e

        # Initialize scan state
        files: list[DirectoryScanFile] = []
        warnings: list[str] = []
        total_size = 0
        files_scanned = 0
        total_files = 0

        # WebSocket channel for progress updates
        channel_id = f"directory-scan:{scan_id}"

        # First, count total files for progress tracking
        logger.info(f"Starting file count for directory: {path}")
        try:
            await self._send_progress(
                channel_id=channel_id,
                scan_id=scan_id,
                msg_type="counting",
                data={"message": "Counting files...", "path": path},
            )

            total_files = await self._count_files(scan_path, recursive, include_patterns, exclude_patterns)

            logger.info(f"Found {total_files} supported files in {path}")

            await self._send_progress(
                channel_id=channel_id,
                scan_id=scan_id,
                msg_type="progress",
                data={
                    "total_files": total_files,
                    "files_scanned": 0,
                    "percentage": 0.0,
                    "message": f"Found {total_files} files to scan",
                },
            )
        except Exception as e:
            logger.warning(f"Error counting files: {e}")
            # Continue without total count

        # Scan files
        try:
            if recursive:
                async for file_info, warning in self._scan_recursive(scan_path, include_patterns, exclude_patterns):
                    if warning:
                        warnings.append(warning)
                        await self._send_progress(
                            channel_id=channel_id,
                            scan_id=scan_id,
                            msg_type="warning",
                            data={"message": warning},
                        )
                    elif file_info:
                        files.append(file_info)
                        total_size += file_info.file_size
                        files_scanned += 1

                        # Send progress update
                        if files_scanned % PROGRESS_UPDATE_INTERVAL == 0 or files_scanned == total_files:
                            percentage = (files_scanned / total_files * 100) if total_files > 0 else 0
                            await self._send_progress(
                                channel_id=channel_id,
                                scan_id=scan_id,
                                msg_type="progress",
                                data={
                                    "files_scanned": files_scanned,
                                    "total_files": total_files,
                                    "current_path": file_info.file_path,
                                    "percentage": round(percentage, 1),
                                },
                            )
            else:
                # Non-recursive scan
                for entry in os.listdir(scan_path):
                    entry_path = scan_path / entry
                    if entry_path.is_file():
                        file_info, warning = await self._scan_file(entry_path, include_patterns, exclude_patterns)
                        if warning:
                            warnings.append(warning)
                        elif file_info:
                            files.append(file_info)
                            total_size += file_info.file_size
                            files_scanned += 1

                            if files_scanned % PROGRESS_UPDATE_INTERVAL == 0:
                                percentage = (files_scanned / total_files * 100) if total_files > 0 else 0
                                await self._send_progress(
                                    channel_id=channel_id,
                                    scan_id=scan_id,
                                    msg_type="progress",
                                    data={
                                        "files_scanned": files_scanned,
                                        "total_files": total_files,
                                        "current_path": file_info.file_path,
                                        "percentage": round(percentage, 1),
                                    },
                                )

        except Exception as e:
            logger.error(f"Error during directory scan: {e}")
            await self._send_progress(
                channel_id=channel_id,
                scan_id=scan_id,
                msg_type="error",
                data={"message": f"Scan error: {str(e)}"},
            )
            raise

        # Create response
        response = DirectoryScanResponse(
            scan_id=scan_id,
            path=path,
            files=files,
            total_files=len(files),
            total_size=total_size,
            warnings=warnings,
        )

        # Send completion message
        await self._send_progress(
            channel_id=channel_id,
            scan_id=scan_id,
            msg_type="completed",
            data={
                "total_files": len(files),
                "total_size": total_size,
                "warnings": warnings,
                "message": f"Scan completed: {len(files)} files, {self._format_size(total_size)}",
            },
        )

        return response

    async def _count_files(
        self,
        path: Path,
        recursive: bool,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> int:
        """Count supported files in directory."""
        count = 0

        try:
            if recursive:
                for root, _, files in os.walk(path):
                    for filename in files:
                        file_path = Path(root) / filename
                        if self._should_include_file(file_path, include_patterns, exclude_patterns):
                            count += 1
            else:
                for entry in os.listdir(path):
                    entry_path = path / entry
                    if entry_path.is_file() and self._should_include_file(
                        entry_path, include_patterns, exclude_patterns
                    ):
                        count += 1
        except Exception as e:
            logger.warning(f"Error counting files in {path}: {e}")

        return count

    async def _scan_recursive(
        self,
        path: Path,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> AsyncIterator[tuple[DirectoryScanFile | None, str | None]]:
        """Recursively scan directory yielding file info or warnings."""
        try:
            for root, _, files in os.walk(path):
                for filename in files:
                    file_path = Path(root) / filename
                    file_info, warning = await self._scan_file(file_path, include_patterns, exclude_patterns)
                    yield file_info, warning
        except Exception as e:
            logger.error(f"Error scanning directory {path}: {e}")
            yield None, f"Error scanning directory {path}: {str(e)}"

    async def _scan_file(
        self,
        file_path: Path,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> tuple[DirectoryScanFile | None, str | None]:
        """Scan a single file and return its info or a warning."""
        try:
            # Check if file should be included
            if not self._should_include_file(file_path, include_patterns, exclude_patterns):
                return None, None

            # Get file stats
            stat = file_path.stat()

            # Check file size
            if stat.st_size > MAX_FILE_SIZE:
                return None, f"File too large (>{self._format_size(MAX_FILE_SIZE)}): {file_path}"

            # Calculate hash
            content_hash = await self._calculate_file_hash(file_path)

            # Get MIME type
            mime_type = self._get_mime_type(file_path)

            # Create file info
            file_info = DirectoryScanFile(
                file_path=str(file_path),
                file_name=file_path.name,
                file_size=stat.st_size,
                mime_type=mime_type,
                content_hash=content_hash,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            )

            return file_info, None

        except PermissionError:
            return None, f"Permission denied: {file_path}"
        except Exception as e:
            logger.warning(f"Error scanning file {file_path}: {e}")
            return None, f"Error scanning file {file_path}: {str(e)}"

    def _should_include_file(
        self,
        file_path: Path,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> bool:
        """Check if file should be included based on patterns and supported types."""
        # Check if extension is supported
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False

        # Check exclude patterns
        if exclude_patterns:
            for pattern in exclude_patterns:
                if file_path.match(pattern):
                    return False

        # Check include patterns
        if include_patterns:
            return any(file_path.match(pattern) for pattern in include_patterns)

        return True  # Include by default if no patterns specified

    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file contents."""
        sha256_hash = hashlib.sha256()

        try:
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            raise OSError(f"Failed to calculate hash for {file_path}: {e}") from e

    def _get_mime_type(self, file_path: Path) -> str | None:
        """Get MIME type for a file."""
        mime_type, _ = mimetypes.guess_type(str(file_path))

        if not mime_type:
            mime_map = {
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".doc": "application/msword",
                ".txt": "text/plain",
                ".text": "text/plain",
                ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ".eml": "message/rfc822",
                ".md": "text/markdown",
                ".html": "text/html",
            }
            mime_type = mime_map.get(file_path.suffix.lower())

        return mime_type

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size = size / 1024.0
        return f"{size:.1f} PB"

    async def _send_progress(
        self,
        channel_id: str,
        scan_id: str,
        msg_type: str,
        data: dict[str, Any],
    ) -> None:
        """Send progress update via WebSocket."""
        try:
            progress_msg = DirectoryScanProgress(
                type=msg_type,
                scan_id=scan_id,
                data=data,
            )
            await ws_manager._broadcast(channel_id, progress_msg.model_dump())
        except Exception as e:
            logger.warning(f"Failed to send WebSocket progress update: {e}")
