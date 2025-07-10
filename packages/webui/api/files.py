"""
File and directory scanning routes for the Web UI
"""

import asyncio
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from webui.auth import get_current_user
from webui.schemas import FileInfo

from .jobs import SUPPORTED_EXTENSIONS, manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["files"])


# Request models
class ScanDirectoryRequest(BaseModel):
    path: str
    recursive: bool = True


def compute_file_content_hash(file_path: Path, chunk_size: int = 8192) -> str | None:
    """Compute SHA256 hash of file content

    Args:
        file_path: Path to the file to hash
        chunk_size: Size of chunks to read at a time

    Returns:
        SHA256 hash as hex string, or None if computation fails
    """
    sha256_hash = hashlib.sha256()
    try:
        # Check if it's a symbolic link
        if file_path.is_symlink():
            # For symlinks, hash the link target path instead of content
            target = str(file_path.resolve())
            sha256_hash.update(target.encode("utf-8"))
            return f"symlink:{sha256_hash.hexdigest()}"

        with file_path.open("rb") as f:
            while chunk := f.read(chunk_size):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except PermissionError:
        logger.warning(f"Permission denied reading {file_path}")
        return None
    except OSError as e:
        logger.warning(f"IO error reading {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error computing hash for {file_path}: {e}")
        return None


async def compute_file_content_hash_async(file_path: Path, chunk_size: int = 65536) -> str | None:
    """Compute SHA256 hash of file content asynchronously

    For large files, this prevents blocking the event loop.

    Args:
        file_path: Path to the file to hash
        chunk_size: Size of chunks to read at a time (larger for async)

    Returns:
        SHA256 hash as hex string, or None if computation fails
    """
    sha256_hash = hashlib.sha256()
    try:
        # Check if it's a symbolic link
        if file_path.is_symlink():
            # For symlinks, hash the link target path instead of content
            target = str(file_path.resolve())
            sha256_hash.update(target.encode("utf-8"))
            return f"symlink:{sha256_hash.hexdigest()}"

        # Get file size to decide whether to use async
        file_size = file_path.stat().st_size

        # For files smaller than 10MB, use sync version in thread pool
        if file_size < 10 * 1024 * 1024:
            return await asyncio.get_event_loop().run_in_executor(
                None, compute_file_content_hash, file_path, chunk_size
            )

        # For larger files, read asynchronously
        # Note: aiofiles is not in dependencies, so we'll use thread pool
        def hash_large_file() -> str:
            with Path(file_path).open("rb") as f:
                while chunk := f.read(chunk_size):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()

        return await asyncio.get_event_loop().run_in_executor(None, hash_large_file)

    except PermissionError:
        logger.warning(f"Permission denied reading {file_path}")
        return None
    except OSError as e:
        logger.warning(f"IO error reading {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error computing hash for {file_path}: {e}")
        return None


def scan_directory(path: str, recursive: bool = True) -> dict[str, Any]:
    """Scan directory for supported files

    Args:
        path: Directory path to scan
        recursive: Whether to scan subdirectories

    Returns:
        Dictionary with files list and any warnings

    Raises:
        ValueError: If path doesn't exist or is not a directory
    """
    files = []
    warnings = []
    path_obj = Path(path)

    if not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")

    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # Determine search pattern
    pattern = "**/*" if recursive else "*"

    file_count = 0
    total_size = 0
    warning_file_limit = 10000
    warning_size_limit = 50 * 1024 * 1024 * 1024  # 50GB

    for file_path in path_obj.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                stat = file_path.stat()
                total_size += stat.st_size

                content_hash = compute_file_content_hash(file_path)
                files.append(
                    FileInfo(
                        path=str(file_path),
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                        extension=file_path.suffix.lower(),
                        content_hash=content_hash,
                    )
                )
                file_count += 1
            except OSError as e:
                logger.warning(f"Failed to stat file {file_path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing file {file_path}: {e}")

    # Generate warnings if thresholds exceeded
    if file_count > warning_file_limit:
        warnings.append(
            {
                "type": "high_file_count",
                "message": f"Found {file_count:,} files, which exceeds the recommended limit of {warning_file_limit:,} files. Processing this many files may take a long time.",
                "severity": "warning",
            }
        )

    if total_size > warning_size_limit:
        warnings.append(
            {
                "type": "high_total_size",
                "message": f"Total file size is {total_size / (1024**3):.2f}GB, which exceeds the recommended limit of {warning_size_limit / (1024**3):.0f}GB. Processing this much data may take a long time and consume significant resources.",
                "severity": "warning",
            }
        )

    return {"files": files, "warnings": warnings, "total_files": file_count, "total_size": total_size}


async def scan_directory_async(path: str, recursive: bool = True, scan_id: str | None = None) -> dict[str, Any]:
    """Scan directory for supported files with progress updates

    Args:
        path: Directory path to scan
        recursive: Whether to scan subdirectories
        scan_id: Optional scan ID for WebSocket updates

    Returns:
        Dictionary with files list and any warnings

    Raises:
        ValueError: If path doesn't exist or is not a directory
    """
    files = []
    warnings = []
    path_obj = Path(path)

    if not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")

    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # First, count total files to scan
    total_files = 0

    # Determine search pattern
    pattern = "**/*" if recursive else "*"

    # Count phase
    for _ in path_obj.glob(pattern):
        total_files += 1
        if total_files % 100 == 0 and scan_id:
            await manager.send_update(f"scan_{scan_id}", {"type": "counting", "count": total_files})

    # Scan phase with warning thresholds
    file_count = 0
    total_size = 0
    warning_file_limit = 10000
    warning_size_limit = 50 * 1024 * 1024 * 1024  # 50GB

    for scanned_files, file_path in enumerate(path_obj.glob(pattern), 1):
        # Send progress update every 10 files or at specific percentages
        if scan_id and (scanned_files % 10 == 0 or scanned_files == total_files):
            await manager.send_update(
                f"scan_{scan_id}",
                {"type": "progress", "scanned": scanned_files, "total": total_files, "current_path": str(file_path)},
            )

        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                stat = file_path.stat()
                total_size += stat.st_size

                content_hash = await compute_file_content_hash_async(file_path)
                files.append(
                    FileInfo(
                        path=str(file_path),
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                        extension=file_path.suffix.lower(),
                        content_hash=content_hash,
                    )
                )
                file_count += 1
            except OSError as e:
                logger.warning(f"Failed to stat file {file_path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing file {file_path}: {e}")

    # Generate warnings if thresholds exceeded
    if file_count > warning_file_limit:
        warning = {
            "type": "high_file_count",
            "message": f"Found {file_count:,} files, which exceeds the recommended limit of {warning_file_limit:,} files. Processing this many files may take a long time.",
            "severity": "warning",
        }
        warnings.append(warning)
        if scan_id:
            await manager.send_update(f"scan_{scan_id}", {"type": "warning", "warning": warning})

    if total_size > warning_size_limit:
        warning = {
            "type": "high_total_size",
            "message": f"Total file size is {total_size / (1024**3):.2f}GB, which exceeds the recommended limit of {warning_size_limit / (1024**3):.0f}GB. Processing this much data may take a long time and consume significant resources.",
            "severity": "warning",
        }
        warnings.append(warning)
        if scan_id:
            await manager.send_update(f"scan_{scan_id}", {"type": "warning", "warning": warning})

    return {"files": files, "warnings": warnings, "total_files": file_count, "total_size": total_size}


@router.post("/scan-directory")
async def scan_directory_endpoint(
    request: ScanDirectoryRequest, current_user: dict[str, Any] = Depends(get_current_user)  # noqa: ARG001
) -> dict[str, Any]:
    """Scan a directory for supported files"""
    try:
        result = scan_directory(request.path, request.recursive)
        return {
            "files": result["files"],
            "count": result["total_files"],
            "total_size": result["total_size"],
            "warnings": result["warnings"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# WebSocket handler for scan progress - export this separately so it can be mounted at the app level
async def scan_websocket(websocket: WebSocket, scan_id: str) -> None:
    """WebSocket for real-time scan progress"""
    await manager.connect(websocket, f"scan_{scan_id}")
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("action") == "scan":
                path = data.get("path")
                recursive = data.get("recursive", True)

                try:
                    # Send initial status
                    await manager.send_update(f"scan_{scan_id}", {"type": "started", "path": path})

                    # Perform scan with progress updates
                    result = await scan_directory_async(path, recursive, scan_id)

                    # Send completion
                    await manager.send_update(
                        f"scan_{scan_id}",
                        {
                            "type": "completed",
                            "files": [f.dict() for f in result["files"]],
                            "count": result["total_files"],
                            "total_size": result["total_size"],
                            "warnings": result["warnings"],
                        },
                    )
                except Exception as e:
                    await manager.send_update(f"scan_{scan_id}", {"type": "error", "error": str(e)})
            elif data.get("action") == "cancel":
                # Handle cancellation
                await manager.send_update(f"scan_{scan_id}", {"type": "cancelled"})
                break
    except WebSocketDisconnect:
        manager.disconnect(websocket, f"scan_{scan_id}")
