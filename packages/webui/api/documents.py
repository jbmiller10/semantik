"""
Document serving routes for the Web UI

Provides secure access to original documents with support for:
- Direct file serving with range requests
- PPTX to Markdown conversion with image extraction
- Multi-format document preview (PDF, DOCX, TXT, MD, HTML, PPTX, EML)
"""

import contextlib
import logging
import mimetypes
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from webui import database
from webui.auth import get_current_user
from webui.rate_limiter import limiter

logger = logging.getLogger(__name__)

# PPTX conversion configuration
PPTX2MD_AVAILABLE = False
PPTX2MD_COMMAND = None
with contextlib.suppress(ImportError):
    pass

# Try different methods to find pptx2md
methods_to_try = [
    # Method 1: Try with poetry run (if we're in a poetry project)
    (["poetry", "run", "python", "-m", "pptx2md", "--help"], "poetry run"),
    # Method 2: Try with current Python executable
    ([sys.executable, "-m", "pptx2md", "--help"], "direct python"),
    # Method 3: Try as a direct command
    (["pptx2md", "--help"], "direct command"),
]

for cmd, method in methods_to_try:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            PPTX2MD_AVAILABLE = True
            # Store the successful method for later use
            if method == "poetry run":
                PPTX2MD_COMMAND = ["poetry", "run", "python", "-m", "pptx2md"]
            elif method == "direct python":
                PPTX2MD_COMMAND = [sys.executable, "-m", "pptx2md"]
            else:
                PPTX2MD_COMMAND = ["pptx2md"]

            logger.info(f"pptx2md is available using: {' '.join(PPTX2MD_COMMAND)}")
            break

    except Exception:
        pass

if not PPTX2MD_AVAILABLE:
    logger.warning("pptx2md not available. PPTX preview will be disabled.")

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Supported file extensions for document viewer
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".text", ".pptx", ".eml", ".md", ".html"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
CHUNK_SIZE = 8192  # 8KB chunks for streaming

# Temporary image storage configuration
TEMP_IMAGE_DIR = Path(tempfile.gettempdir()) / "webui_temp_images"
TEMP_IMAGE_DIR.mkdir(exist_ok=True)
TEMP_IMAGE_TTL = 3600  # 1 hour in seconds
IMAGE_SESSIONS = {}  # Maps session_id to (user_id, created_time, image_dir)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"}

# Cleanup thread
cleanup_lock = threading.Lock()


def cleanup_old_sessions():
    """Remove temporary image directories older than TTL"""
    with cleanup_lock:
        current_time = time.time()
        expired_sessions = []

        for session_id, (_user_id, created_time, image_dir) in IMAGE_SESSIONS.items():
            if current_time - created_time > TEMP_IMAGE_TTL:
                expired_sessions.append(session_id)
                try:
                    if image_dir.exists():
                        shutil.rmtree(image_dir)
                except Exception as e:
                    logger.error(f"Error cleaning up session {session_id}: {e}")

        # Remove expired sessions from the dictionary
        for session_id in expired_sessions:
            del IMAGE_SESSIONS[session_id]


def start_cleanup_thread():
    """Start background thread for cleanup"""

    def cleanup_worker():
        while True:
            time.sleep(300)  # Run cleanup every 5 minutes
            cleanup_old_sessions()

    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    logger.info("Started temporary image cleanup thread")


# Start cleanup thread when module loads
start_cleanup_thread()


def validate_file_access(job_id: str, doc_id: str, current_user: dict[str, Any]) -> dict[str, Any]:
    """Validate that the user has access to the requested document"""
    # Get job to verify it exists and user has access
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if the current user owns the job
    if job.get("user_id") != current_user.get("id"):
        raise HTTPException(status_code=403, detail="Access denied")

    # Get file record from database
    files = database.get_job_files(job_id)
    file_record = None

    for file in files:
        if file.get("doc_id") == doc_id:
            file_record = file
            break

    if not file_record:
        raise HTTPException(status_code=404, detail="Document not found")

    # Verify file path is within job directory (prevent path traversal)
    file_path = Path(file_record["path"])
    job_dir = Path(job["directory_path"])

    try:
        # Resolve paths to absolute and check containment
        file_path_resolved = file_path.resolve()
        job_dir_resolved = job_dir.resolve()

        if not str(file_path_resolved).startswith(str(job_dir_resolved)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception as e:
        logger.error(f"Path validation error: {e}")
        raise HTTPException(status_code=403, detail="Access denied") from e

    # Verify file exists
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    # Check file size
    if file_path.stat().st_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File is too large to preview")

    return file_record


@router.get("/{job_id}/{doc_id}")
@limiter.limit("10/minute")
async def get_document(
    request: Request,
    job_id: str,
    doc_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    range: str | None = Header(None),
):
    """
    Retrieve a document by job_id and doc_id.

    This endpoint serves the raw file content for a given document, identified by its `job_id` and `doc_id`.
    It performs security checks to ensure the user has access to the file and that the file is within the job's directory.
    It supports HTTP Range requests for efficient streaming of large files, which is essential for PDF viewing and other large documents.

    - **job_id**: The ID of the ingestion job.
    - **doc_id**: The unique ID of the document within the job.
    - **current_user**: The authenticated user, injected by Depends.
    - **range**: The HTTP Range header, used for partial content requests.

    Returns a `FileResponse` or `StreamingResponse` for the document.
    """
    # Validate access and get file info
    file_record = validate_file_access(job_id, doc_id, current_user)
    file_path = Path(file_record["path"])

    # Check if file extension is supported
    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {extension}")

    # Special handling for PPTX files - convert to Markdown
    if extension == ".pptx" and PPTX2MD_AVAILABLE:
        try:

            # Create a temporary output directory for images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert PPTX to Markdown using command line
                output_path = Path(temp_dir) / "output.md"

                # Create a session ID for this conversion
                session_id = str(uuid.uuid4())
                session_image_dir = TEMP_IMAGE_DIR / session_id
                session_image_dir.mkdir(parents=True, exist_ok=True)

                # Store session info
                with cleanup_lock:
                    user_id = current_user.get("id", "anonymous")
                    IMAGE_SESSIONS[session_id] = (user_id, time.time(), session_image_dir)

                # Run the conversion command using the detected method
                cmd = PPTX2MD_COMMAND + [
                    str(file_path),
                    "-o",
                    str(output_path),
                    "--disable_wmf",  # Avoid WMF issues on Linux
                    "--disable_escaping",  # Don't escape markdown characters
                    "--image_dir",
                    str(session_image_dir),  # Extract images to session directory
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0 and output_path.exists():
                    # Read the converted markdown
                    with output_path.open(encoding="utf-8") as f:
                        markdown_content = f.read()

                    # Update image paths in markdown to use our temp image endpoint
                    # pptx2md appears to generate paths like: webui_temp_images/{session_id}/filename
                    # We need to replace these with our API endpoint

                    # Simple string replacement for the common case
                    markdown_content = markdown_content.replace(
                        f"webui_temp_images/{session_id}/", f"/api/documents/temp-images/{session_id}/"
                    )

                    # Also handle any other image path patterns
                    def replace_image_path(match):
                        alt_text = match.group(1)
                        image_path = match.group(2)

                        # If already an API path, keep it
                        if image_path.startswith("/api/documents/temp-images/"):
                            return match.group(0)

                        # If it contains our session ID, update the path
                        if session_id in image_path:
                            # Extract filename from end of path
                            image_filename = Path(image_path).name
                            new_url = f"/api/documents/temp-images/{session_id}/{image_filename}"
                            return f"![{alt_text}]({new_url})"

                        # For other paths, just use filename
                        image_filename = Path(image_path).name
                        new_url = f"/api/documents/temp-images/{session_id}/{image_filename}"
                        return f"![{alt_text}]({new_url})"

                    # Apply regex replacement for any remaining cases
                    markdown_content = re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", replace_image_path, markdown_content)
                else:
                    # Conversion failed, provide fallback
                    logger.error(f"pptx2md conversion failed for {file_path.name}")
                    markdown_content = (
                        f"# {file_path.name}\n\nError converting presentation. Please download the file to view it."
                    )

                # Return the markdown content with appropriate headers
                return Response(
                    content=markdown_content,
                    media_type="text/markdown",
                    headers={
                        "Content-Disposition": f'inline; filename="{file_path.stem}.md"',
                        "X-Original-Filename": file_path.name,
                        "X-Converted-From": "pptx",
                        "X-Image-Session-Id": session_id,  # Send session ID to client
                    },
                )

        except Exception as e:
            logger.error(f"Error converting PPTX to Markdown: {e}")
            # Fall back to serving the original file

    # Get file size for range requests
    file_size = file_path.stat().st_size

    # Determine content type
    content_type = mimetypes.guess_type(str(file_path))[0]
    if not content_type:
        # Default content types for specific extensions
        content_type_map = {
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
        content_type = content_type_map.get(extension, "application/octet-stream")

    # Handle range requests for large files
    if range:
        try:
            # Parse range header (e.g., "bytes=0-1023")
            range_start = 0
            range_end = file_size - 1

            if range.startswith("bytes="):
                range_spec = range[6:]
                if "-" in range_spec:
                    start_str, end_str = range_spec.split("-", 1)
                    if start_str:
                        range_start = int(start_str)
                    if end_str:
                        range_end = int(end_str)

            # Validate range
            if range_start >= file_size:
                raise HTTPException(status_code=416, detail="Requested range not satisfiable")

            range_end = min(range_end, file_size - 1)
            content_length = range_end - range_start + 1

            # Return partial content
            def file_generator():
                with Path(file_path).open("rb") as f:
                    f.seek(range_start)
                    remaining = content_length

                    while remaining > 0:
                        to_read = min(CHUNK_SIZE, remaining)
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
                    "Content-Length": str(content_length),
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes {range_start}-{range_end}/{file_size}",
                    "Content-Disposition": f'inline; filename="{file_path.name}"',
                },
            )

        except (ValueError, IndexError):
            # Invalid range header, return full file
            pass

    # Return full file
    # Generate ETag based on file metadata
    file_stat = file_path.stat()
    etag = f'"{file_stat.st_mtime}-{file_stat.st_size}"'

    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Disposition": f'inline; filename="{file_path.name}"',
            "Cache-Control": "private, max-age=3600",  # Cache for 1 hour
            "ETag": etag,
            "Last-Modified": datetime.fromtimestamp(file_stat.st_mtime, UTC).strftime("%a, %d %b %Y %H:%M:%S GMT"),
        },
    )


@router.get("/{job_id}/{doc_id}/info")
@limiter.limit("30/minute")
async def get_document_info(
    request: Request, job_id: str, doc_id: str, current_user: dict[str, Any] = Depends(get_current_user)
):
    """
    Get document metadata without downloading the file.

    This endpoint provides metadata about a document, such as its filename, size, and whether it's supported by the viewer.
    It's a lightweight alternative to downloading the entire file.

    - **job_id**: The ID of the ingestion job.
    - **doc_id**: The unique ID of the document within the job.
    - **current_user**: The authenticated user, injected by Depends.

    Returns a JSON object with document metadata.
    """
    # Validate access and get file info
    file_record = validate_file_access(job_id, doc_id, current_user)
    file_path = Path(file_record["path"])

    return {
        "doc_id": doc_id,
        "filename": file_path.name,
        "path": file_record["path"],
        "size": file_record["size"],
        "extension": file_record["extension"],
        "modified": file_record["modified"],
        "supported": file_path.suffix.lower() in SUPPORTED_EXTENSIONS,
    }


@router.get("/temp-images/{session_id}/{filename}")
@limiter.limit("30/minute")
async def get_temp_image(
    request: Request,
    session_id: str,
    filename: str,
    token: str | None = None,
    current_user: dict[str, Any] | None = None,
):
    """
    Serve temporary images extracted from documents (e.g., PPTX slides).

    Images are stored in session-specific directories and automatically cleaned up after TTL.
    The session ID acts as a temporary access token for security.
    """

    # Validate session exists
    with cleanup_lock:
        session_info = IMAGE_SESSIONS.get(session_id)

    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    user_id, created_time, image_dir = session_info

    # Check if session has expired
    if time.time() - created_time > TEMP_IMAGE_TTL:
        raise HTTPException(status_code=410, detail="Session has expired")

    # Validate filename (prevent path traversal)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Check file extension
    file_path = image_dir / filename
    if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported image type")

    # Check if file exists
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    # Verify file is within session directory (extra security)
    try:
        file_path_resolved = file_path.resolve()
        image_dir_resolved = image_dir.resolve()

        if not str(file_path_resolved).startswith(str(image_dir_resolved)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception as e:
        logger.error(f"Path validation error: {e}")
        raise HTTPException(status_code=403, detail="Access denied") from e

    # Determine content type
    content_type = mimetypes.guess_type(str(file_path))[0]
    if not content_type:
        # Default content types for image extensions
        content_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".svg": "image/svg+xml",
            ".webp": "image/webp",
        }
        content_type = content_type_map.get(file_path.suffix.lower(), "application/octet-stream")

    # Return the image file
    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        headers={
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            "Content-Disposition": f'inline; filename="{filename}"',
        },
    )
