"""
File and directory scanning routes for the Web UI
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel

from webui.auth import get_current_user
from webui.api.jobs import manager, SUPPORTED_EXTENSIONS
from webui.schemas import FileInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["files"])


# Request models
class ScanDirectoryRequest(BaseModel):
    path: str
    recursive: bool = True


def scan_directory(path: str, recursive: bool = True) -> List[FileInfo]:
    """Scan directory for supported files"""
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

    for file_path in path_obj.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                stat = file_path.stat()
                files.append(
                    FileInfo(
                        path=str(file_path),
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        extension=file_path.suffix.lower(),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to stat file {file_path}: {e}")

    return files


async def scan_directory_async(path: str, recursive: bool = True, scan_id: str = None) -> List[FileInfo]:
    """Scan directory for supported files with progress updates"""
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

    # Scan phase
    for file_path in path_obj.glob(pattern):
        scanned_files += 1

        # Send progress update every 10 files or at specific percentages
        if scan_id and (scanned_files % 10 == 0 or scanned_files == total_files):
            await manager.send_update(
                f"scan_{scan_id}",
                {"type": "progress", "scanned": scanned_files, "total": total_files, "current_path": str(file_path)},
            )

        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                stat = file_path.stat()
                files.append(
                    FileInfo(
                        path=str(file_path),
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        extension=file_path.suffix.lower(),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to stat file {file_path}: {e}")

    return files


@router.post("/scan-directory")
async def scan_directory_endpoint(
    request: ScanDirectoryRequest, current_user: Dict[str, Any] = Depends(get_current_user)
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
