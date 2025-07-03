"""
File and directory scanning routes for the Web UI
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..auth import get_current_user
from ..schemas import FileInfo
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

        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except PermissionError:
        logger.warning(f"Permission denied reading {file_path}")
        return None
    except IOError as e:
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
        def hash_large_file():
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()

        return await asyncio.get_event_loop().run_in_executor(None, hash_large_file)

    except PermissionError:
        logger.warning(f"Permission denied reading {file_path}")
        return None
    except IOError as e:
        logger.warning(f"IO error reading {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error computing hash for {file_path}: {e}")
        return None


def scan_directory(path: str, recursive: bool = True, max_files: int = 10000) -> list[FileInfo]:
    """Scan directory for supported files

    Args:
        path: Directory path to scan
        recursive: Whether to scan subdirectories
        max_files: Maximum number of files to process (default 10000)

    Raises:
        ValueError: If path doesn't exist or too many files
    """
    files = []
    path_obj = Path(path)

    if not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")

    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # Determine search pattern
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    file_count = 0
    total_size = 0
    max_total_size = 50 * 1024 * 1024 * 1024  # 50GB limit

    for file_path in path_obj.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            # Check file count limit
            if file_count >= max_files:
                raise ValueError(f"Too many files to process. Found {file_count} files, limit is {max_files}")

            try:
                stat = file_path.stat()

                # Check total size limit
                total_size += stat.st_size
                if total_size > max_total_size:
                    raise ValueError(
                        f"Total file size exceeds limit. Total: {total_size / (1024**3):.2f}GB, limit: {max_total_size / (1024**3):.2f}GB"
                    )

                content_hash = compute_file_content_hash(file_path)
                files.append(
                    FileInfo(
                        path=str(file_path),
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        extension=file_path.suffix.lower(),
                        content_hash=content_hash,
                    )
                )
                file_count += 1
            except OSError as e:
                logger.warning(f"Failed to stat file {file_path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing file {file_path}: {e}")

    return files


async def scan_directory_async(
    path: str, recursive: bool = True, scan_id: str = None, max_files: int = 10000
) -> list[FileInfo]:
    """Scan directory for supported files with progress updates

    Args:
        path: Directory path to scan
        recursive: Whether to scan subdirectories
        scan_id: Optional scan ID for WebSocket updates
        max_files: Maximum number of files to process (default 10000)

    Raises:
        ValueError: If path doesn't exist or too many files
    """
    files = []
    path_obj = Path(path)

    if not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")

    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # First, count total files to scan
    total_files = 0
    scanned_files = 0

    # Determine search pattern
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    # Count phase
    for _ in path_obj.glob(pattern):
        total_files += 1
        if total_files % 100 == 0 and scan_id:
            await manager.send_update(f"scan_{scan_id}", {"type": "counting", "count": total_files})

    # Scan phase with resource limits
    file_count = 0
    total_size = 0
    max_total_size = 50 * 1024 * 1024 * 1024  # 50GB limit

    for file_path in path_obj.glob(pattern):
        scanned_files += 1

        # Send progress update every 10 files or at specific percentages
        if scan_id and (scanned_files % 10 == 0 or scanned_files == total_files):
            await manager.send_update(
                f"scan_{scan_id}",
                {"type": "progress", "scanned": scanned_files, "total": total_files, "current_path": str(file_path)},
            )

        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            # Check file count limit
            if file_count >= max_files:
                error_msg = f"Too many files to process. Found {file_count} files, limit is {max_files}"
                if scan_id:
                    await manager.send_update(f"scan_{scan_id}", {"type": "error", "error": error_msg})
                raise ValueError(error_msg)

            try:
                stat = file_path.stat()

                # Check total size limit
                total_size += stat.st_size
                if total_size > max_total_size:
                    error_msg = f"Total file size exceeds limit. Total: {total_size / (1024**3):.2f}GB, limit: {max_total_size / (1024**3):.2f}GB"
                    if scan_id:
                        await manager.send_update(f"scan_{scan_id}", {"type": "error", "error": error_msg})
                    raise ValueError(error_msg)

                content_hash = await compute_file_content_hash_async(file_path)
                files.append(
                    FileInfo(
                        path=str(file_path),
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        extension=file_path.suffix.lower(),
                        content_hash=content_hash,
                    )
                )
                file_count += 1
            except OSError as e:
                logger.warning(f"Failed to stat file {file_path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing file {file_path}: {e}")

    return files


@router.post("/scan-directory")
async def scan_directory_endpoint(
    request: ScanDirectoryRequest, current_user: dict[str, Any] = Depends(get_current_user)
):
    """Scan a directory for supported files"""
    try:
        files = scan_directory(request.path, request.recursive)
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# WebSocket handler for scan progress - export this separately so it can be mounted at the app level
async def scan_websocket(websocket: WebSocket, scan_id: str):
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
                    files = await scan_directory_async(path, recursive, scan_id)

                    # Send completion
                    await manager.send_update(
                        f"scan_{scan_id}",
                        {"type": "completed", "files": [f.dict() for f in files], "count": len(files)},
                    )
                except Exception as e:
                    await manager.send_update(f"scan_{scan_id}", {"type": "error", "error": str(e)})
            elif data.get("action") == "cancel":
                # Handle cancellation
                await manager.send_update(f"scan_{scan_id}", {"type": "cancelled"})
                break
    except WebSocketDisconnect:
        manager.disconnect(websocket, f"scan_{scan_id}")
