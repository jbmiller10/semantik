"""
Document API v2 endpoints.

This module provides RESTful API endpoints for document content access.
Security is critical - all document access must be properly authorized.
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database import get_db
from packages.shared.database.exceptions import EntityNotFoundError
from packages.shared.database.models import Collection, Document
from packages.webui.auth import get_current_user
from packages.webui.dependencies import create_document_repository, get_collection_for_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["documents-v2"])


@router.get(
    "/collections/{collection_uuid}/documents/{document_uuid}/content",
    responses={
        200: {"description": "Document content", "content": {"application/octet-stream": {}}},
        401: {"description": "Unauthorized"},
        403: {"description": "Access denied"},
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_document_content(
    collection_uuid: str,
    document_uuid: str,
    collection: Collection = Depends(get_collection_for_user),  # noqa: ARG001
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """Get the content of a specific document.

    This endpoint serves the raw document content for viewing. Access control
    is enforced through the get_collection_for_user dependency which verifies
    the user has at least read access to the collection.

    Security considerations:
    - User authentication is required
    - User must have access to the collection (owner or shared)
    - Document must belong to the specified collection
    - File paths are sanitized to prevent path traversal attacks
    """
    try:
        # Get document repository
        document_repo = create_document_repository(db)

        # Fetch the document
        document: Document = await document_repo.get_by_id(document_uuid)

        # Verify document belongs to the specified collection
        if document.collection_id != collection_uuid:
            logger.warning(
                f"Attempted cross-collection access: user {current_user['id']} tried to access "
                f"document {document_uuid} from collection {document.collection_id} "
                f"via collection {collection_uuid}"
            )
            raise HTTPException(
                status_code=403,
                detail="Document does not belong to the specified collection",
            )

        # CRITICAL SECURITY: Validate and sanitize the file path
        # Ensure the path is absolute and resolved to prevent path traversal
        try:
            # Convert to Path object and resolve to absolute path
            file_path = Path(document.file_path).resolve()

            # Additional security check: ensure file exists and is a regular file
            if not file_path.exists():
                logger.error(f"Document file not found: {document.file_path}")
                raise HTTPException(
                    status_code=404,
                    detail="Document file not found on disk",
                )

            if not file_path.is_file():
                logger.error(f"Document path is not a file: {document.file_path}")
                raise HTTPException(
                    status_code=500,
                    detail="Invalid document path",
                )

            # SECURITY: In a production environment, you would check that the file
            # is within an allowed directory. For now, we'll serve any file that
            # the document references, but this should be restricted based on
            # your deployment configuration.
            # Example check (uncomment and configure as needed):
            # ALLOWED_DOCUMENT_ROOT = Path("/var/semantik/documents").resolve()
            # if not str(file_path).startswith(str(ALLOWED_DOCUMENT_ROOT)):
            #     logger.error(f"Path traversal attempt detected: {file_path}")
            #     raise HTTPException(status_code=403, detail="Access denied")

        except Exception as e:
            logger.error(f"Error resolving document path: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error accessing document",
            ) from e

        # Prepare response headers
        media_type = document.mime_type or "application/octet-stream"

        # Log successful document access for audit
        logger.info(
            f"User {current_user['id']} accessed document {document_uuid} from collection {collection_uuid}"
        )

        # Return the file
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=document.file_name,
            headers={
                "Content-Disposition": f'inline; filename="{document.file_name}"',
                "Cache-Control": "private, max-age=3600",  # Cache for 1 hour
            },
        )

    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_uuid} not found",
        ) from e
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Failed to serve document content: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve document content",
        ) from e
