"""
Document serving routes for the Web UI
Provides secure access to original documents for viewing
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import mimetypes

from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import FileResponse, StreamingResponse
from webui import database
from webui.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Supported file extensions for document viewer
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.text', '.pptx', '.eml', '.md', '.html'}

def validate_file_access(job_id: str, doc_id: str, current_user: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that the user has access to the requested document"""
    # Get job to verify it exists and user has access
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get file record from database
    files = database.get_job_files(job_id)
    file_record = None
    
    for file in files:
        if file.get('doc_id') == doc_id:
            file_record = file
            break
    
    if not file_record:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Verify file path is within job directory (prevent path traversal)
    file_path = Path(file_record['path'])
    job_dir = Path(job['directory_path'])
    
    try:
        # Resolve paths to absolute and check containment
        file_path_resolved = file_path.resolve()
        job_dir_resolved = job_dir.resolve()
        
        if not str(file_path_resolved).startswith(str(job_dir_resolved)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception as e:
        logger.error(f"Path validation error: {e}")
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Verify file exists
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    return file_record

@router.get("/{job_id}/{doc_id}")
async def get_document(
    job_id: str,
    doc_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    range: Optional[str] = Header(None)
):
    """
    Retrieve a document by job_id and doc_id
    Supports HTTP Range requests for streaming large files
    """
    # Validate access and get file info
    file_record = validate_file_access(job_id, doc_id, current_user)
    file_path = Path(file_record['path'])
    
    # Check if file extension is supported
    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415, 
            detail=f"Unsupported file type: {extension}"
        )
    
    # Get file size for range requests
    file_size = file_path.stat().st_size
    
    # Determine content type
    content_type = mimetypes.guess_type(str(file_path))[0]
    if not content_type:
        # Default content types for specific extensions
        content_type_map = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.txt': 'text/plain',
            '.text': 'text/plain',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.eml': 'message/rfc822',
            '.md': 'text/markdown',
            '.html': 'text/html'
        }
        content_type = content_type_map.get(extension, 'application/octet-stream')
    
    # Handle range requests for large files
    if range:
        try:
            # Parse range header (e.g., "bytes=0-1023")
            range_start = 0
            range_end = file_size - 1
            
            if range.startswith('bytes='):
                range_spec = range[6:]
                if '-' in range_spec:
                    start_str, end_str = range_spec.split('-', 1)
                    if start_str:
                        range_start = int(start_str)
                    if end_str:
                        range_end = int(end_str)
            
            # Validate range
            if range_start >= file_size:
                raise HTTPException(
                    status_code=416,
                    detail="Requested range not satisfiable"
                )
            
            range_end = min(range_end, file_size - 1)
            content_length = range_end - range_start + 1
            
            # Return partial content
            def file_generator():
                with open(file_path, 'rb') as f:
                    f.seek(range_start)
                    remaining = content_length
                    chunk_size = 8192  # 8KB chunks
                    
                    while remaining > 0:
                        to_read = min(chunk_size, remaining)
                        chunk = f.read(to_read)
                        if not chunk:
                            break
                        yield chunk
                        remaining -= len(chunk)
            
            return StreamingResponse(
                file_generator(),
                status_code=206,  # Partial Content
                media_type=content_type,
                headers={
                    'Content-Length': str(content_length),
                    'Accept-Ranges': 'bytes',
                    'Content-Range': f'bytes {range_start}-{range_end}/{file_size}',
                    'Content-Disposition': f'inline; filename="{file_path.name}"'
                }
            )
            
        except (ValueError, IndexError):
            # Invalid range header, return full file
            pass
    
    # Return full file
    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        headers={
            'Accept-Ranges': 'bytes',
            'Content-Length': str(file_size),
            'Content-Disposition': f'inline; filename="{file_path.name}"'
        }
    )

@router.get("/{job_id}/{doc_id}/info")
async def get_document_info(
    job_id: str,
    doc_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get document metadata without downloading the file"""
    # Validate access and get file info
    file_record = validate_file_access(job_id, doc_id, current_user)
    file_path = Path(file_record['path'])
    
    return {
        "doc_id": doc_id,
        "filename": file_path.name,
        "path": file_record['path'],
        "size": file_record['size'],
        "extension": file_record['extension'],
        "modified": file_record['modified'],
        "supported": file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    }