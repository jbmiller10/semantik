"""
Filesystem API for directory browsing
"""

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from shared.config import settings
from webui.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fs", tags=["filesystem"])


class FileItem(BaseModel):
    name: str
    type: str  # 'dir' or 'file'
    path: str
    size: int | None = None


class FileListResponse(BaseModel):
    items: list[FileItem]
    current_path: str
    parent_path: str | None


@router.get("/list", response_model=FileListResponse)
async def list_directory(
    path: str | None = Query(None, description="Path to list (relative to root)"),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> FileListResponse:
    """
    List contents of a directory.
    Restricted to the configured DOCUMENT_PATH.
    """
    root_path = Path(settings.DOCUMENT_PATH).resolve()
    
    if not path or path == "/":
        target_path = root_path
        relative_path = ""
    else:
        # Sanitize path to prevent traversal
        # Remove leading slashes to treat as relative
        clean_path = path.lstrip("/")
        target_path = (root_path / clean_path).resolve()
        
        # Security check: ensure target is within root
        if not str(target_path).startswith(str(root_path)):
            raise HTTPException(status_code=403, detail="Access denied: Path outside allowed directory")
            
        relative_path = str(target_path.relative_to(root_path))

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Path not found")
        
    if not target_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    items = []
    try:
        # List directory contents
        with os.scandir(target_path) as it:
            for entry in it:
                # Skip hidden files
                if entry.name.startswith("."):
                    continue
                    
                item_type = "dir" if entry.is_dir() else "file"
                
                # Calculate absolute path for the item, but return relative to root for UI
                # or return full path if that's what the ingestion API expects.
                # The ingestion API expects absolute paths on the server.
                full_path = str(Path(entry.path).absolute())
                
                size = None
                if item_type == "file":
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        pass
                        
                items.append(FileItem(
                    name=entry.name,
                    type=item_type,
                    path=full_path,
                    size=size
                ))
                
        # Sort: directories first, then files, alphabetically
        items.sort(key=lambda x: (0 if x.type == "dir" else 1, x.name.lower()))
        
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except OSError as e:
        logger.error(f"Error listing directory {target_path}: {e}")
        raise HTTPException(status_code=500, detail="Error listing directory")

    # Determine parent path
    parent_path = None
    if target_path != root_path:
        parent = target_path.parent
        # Ensure parent is still within root
        if str(parent).startswith(str(root_path)):
            parent_path = str(parent.relative_to(root_path))
            if parent_path == ".":
                parent_path = ""

    return FileListResponse(
        items=items,
        current_path=str(target_path),
        parent_path=parent_path
    )
