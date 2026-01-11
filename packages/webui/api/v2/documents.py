"""
Document API v2 endpoints.

This module provides RESTful API endpoints for document content access.
Security is critical - all document access must be properly authorized.

Content is served from two sources:
1. Database artifacts (for non-file sources like Git, IMAP)
2. Filesystem files (for local directory sources)

The endpoint checks for artifacts first, then falls back to file serving.
"""

import logging
import urllib.parse
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import settings
from shared.database import get_db
from shared.database.exceptions import EntityNotFoundError
from shared.database.models import Collection
from shared.database.repositories.document_artifact_repository import DocumentArtifactRepository
from webui.api.schemas import DocumentResponse, ErrorResponse
from webui.auth import get_current_user
from webui.dependencies import create_document_repository, get_collection_for_user

logger = logging.getLogger(__name__)


def _get_status_value(status: Any) -> str:
    """Get status value as string, handling both enum and string types.

    SQLAlchemy may return the status as either an enum object (with .value)
    or as a raw string depending on session state and query patterns.
    """
    return status.value if hasattr(status, "value") else str(status)


def sanitize_filename_for_header(filename: str) -> str:
    """Sanitize filename for use in Content-Disposition header.

    Removes control characters and encodes special chars per RFC 5987.

    Args:
        filename: The original filename

    Returns:
        RFC 5987 encoded filename safe for Content-Disposition header
    """
    # Remove control characters (CR, LF, etc.) that could break header parsing
    safe = filename.replace('"', "'").replace("\r", "").replace("\n", "").replace("\x00", "")
    # URL-encode the filename per RFC 5987
    return urllib.parse.quote(safe, safe="")


router = APIRouter(prefix="/api/v2", tags=["documents-v2"])


@router.get(
    "/collections/{collection_uuid}/documents/{document_uuid}",
    response_model=DocumentResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_document(
    collection_uuid: str,
    document_uuid: str,
    collection: Collection = Depends(get_collection_for_user),  # noqa: ARG001
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Get document metadata by ID.

    This endpoint returns document metadata (not content). Access control is enforced
    via the get_collection_for_user dependency and a cross-collection check.
    """
    try:
        document_repo = create_document_repository(db)
        document = await document_repo.get_by_id(document_uuid)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_uuid} not found")

        if document.collection_id != collection_uuid:
            logger.warning(
                "Attempted cross-collection metadata access: user %s tried to access document %s from collection %s via %s",
                current_user.get("id"),
                document_uuid,
                document.collection_id,
                collection_uuid,
            )
            raise HTTPException(status_code=403, detail="Document does not belong to the specified collection")

        return DocumentResponse(
            id=document.id,
            collection_id=document.collection_id,
            file_name=document.file_name,
            file_path=document.file_path,
            file_size=document.file_size,
            mime_type=document.mime_type,
            content_hash=document.content_hash,
            status=_get_status_value(document.status),
            error_message=document.error_message,
            chunk_count=document.chunk_count,
            retry_count=document.retry_count or 0,
            last_retry_at=document.last_retry_at,
            error_category=document.error_category,
            metadata=document.meta,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document metadata: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get document metadata") from e


@router.get(
    "/collections/{collection_uuid}/documents/{document_uuid}/content",
    response_model=None,
    responses={
        200: {"description": "Document content", "content": {"application/octet-stream": {}}},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_document_content(
    collection_uuid: str,
    document_uuid: str,
    collection: Collection = Depends(get_collection_for_user),  # noqa: ARG001
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FileResponse | Response:
    """Get the content of a specific document.

    This endpoint serves the raw document content for viewing. Access control
    is enforced through the get_collection_for_user dependency which verifies
    the user has at least read access to the collection.

    Content is served from two sources (checked in order):
    1. Database artifacts (for non-file sources like Git, IMAP)
    2. Filesystem files (for local directory sources)

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
        document = await document_repo.get_by_id(document_uuid)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_uuid} not found")

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

        # Check for database artifact first (for non-file sources like Git, IMAP)
        artifact_repo = DocumentArtifactRepository(db, max_artifact_bytes=settings.MAX_ARTIFACT_BYTES)
        artifact_content = await artifact_repo.get_content(document_uuid)

        if artifact_content is not None:
            content, mime_type, charset = artifact_content

            # Log successful artifact access
            logger.info(
                f"User {current_user['id']} accessed artifact for document {document_uuid} "
                f"from collection {collection_uuid}"
            )

            # Build media type with charset if applicable
            media_type_header = f"{mime_type}; charset={charset}" if charset else mime_type

            # Return artifact content as Response
            # Use RFC 5987 encoding for filename to prevent header injection
            safe_filename = sanitize_filename_for_header(str(document.file_name))
            if isinstance(content, str):
                return Response(
                    content=content.encode(charset or "utf-8"),
                    media_type=media_type_header,
                    headers={
                        "Content-Disposition": f"inline; filename*=UTF-8''{safe_filename}",
                        "Cache-Control": "private, max-age=3600",
                    },
                )
            return Response(
                content=content,
                media_type=media_type_header,
                headers={
                    "Content-Disposition": f"inline; filename*=UTF-8''{safe_filename}",
                    "Cache-Control": "private, max-age=3600",
                },
            )

        # Fallback to file-based serving for local directory sources
        # CRITICAL SECURITY: Validate and sanitize the file path
        # Ensure the path is absolute and resolved to prevent path traversal
        try:
            # Convert to Path object and resolve to absolute path
            file_path = Path(document.file_path).resolve()
        except Exception as e:
            logger.error("Error resolving document path: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Error accessing document",
            ) from e

        # Ensure the resolved path stays within the configured document root when one is set
        allowed_roots = settings.document_allowed_roots
        if settings.should_enforce_document_roots and allowed_roots:
            for root in allowed_roots:
                try:
                    file_path.relative_to(root)
                    break
                except ValueError:
                    continue
            else:
                logger.warning(
                    "Path traversal attempt blocked: user %s attempted to access %s outside allowed roots %s",
                    current_user.get("id", "unknown"),
                    file_path,
                    ", ".join(str(root) for root in allowed_roots),
                )
                raise HTTPException(status_code=403, detail="Access to the requested document is forbidden")

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
                status_code=400,
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

        # Prepare response headers
        media_type = document.mime_type or "application/octet-stream"

        # Log successful document access for audit
        logger.info(f"User {current_user['id']} accessed document {document_uuid} from collection {collection_uuid}")

        # Return the file with sanitized filename header
        safe_filename = sanitize_filename_for_header(str(document.file_name))
        return FileResponse(
            path=str(file_path),
            media_type=str(media_type),
            filename=str(document.file_name),
            headers={
                "Content-Disposition": f"inline; filename*=UTF-8''{safe_filename}",
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
        logger.error("Failed to serve document content: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve document content",
        ) from e


# ========== Retry Endpoints ==========


@router.post(
    "/collections/{collection_uuid}/documents/{document_uuid}/retry",
    response_model=DocumentResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Document cannot be retried"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def retry_document(
    collection_uuid: str,
    document_uuid: str,
    collection: Collection = Depends(get_collection_for_user),  # noqa: ARG001
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Retry a single failed document.

    Resets the document status to PENDING so it can be reprocessed.
    Only documents in FAILED status can be retried.
    """
    try:
        document_repo = create_document_repository(db)
        document = await document_repo.get_by_id(document_uuid)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_uuid} not found")

        if document.collection_id != collection_uuid:
            logger.warning(
                "Attempted cross-collection retry: user %s tried to retry document %s from collection %s via %s",
                current_user.get("id"),
                document_uuid,
                document.collection_id,
                collection_uuid,
            )
            raise HTTPException(status_code=403, detail="Document does not belong to the specified collection")

        # Reset the document for retry
        try:
            document = await document_repo.reset_for_retry(document_uuid)
            await db.commit()
        except Exception as e:
            await db.rollback()
            if "not in FAILED status" in str(e):
                raise HTTPException(status_code=400, detail="Only failed documents can be retried") from e
            raise

        logger.info(
            "User %s retried document %s in collection %s (retry_count=%d)",
            current_user.get("id"),
            document_uuid,
            collection_uuid,
            document.retry_count,
        )

        return DocumentResponse(
            id=document.id,
            collection_id=document.collection_id,
            file_name=document.file_name,
            file_path=document.file_path,
            file_size=document.file_size,
            mime_type=document.mime_type,
            content_hash=document.content_hash,
            status=_get_status_value(document.status),
            error_message=document.error_message,
            chunk_count=document.chunk_count,
            retry_count=document.retry_count or 0,
            last_retry_at=document.last_retry_at,
            error_category=document.error_category,
            metadata=document.meta,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retry document: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retry document") from e


@router.post(
    "/collections/{collection_uuid}/documents/retry-failed",
    response_model=None,
    responses={
        200: {"description": "Bulk retry operation dispatched"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Collection not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def retry_failed_documents(
    collection_uuid: str,
    collection: Collection = Depends(get_collection_for_user),  # noqa: ARG001
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Retry all failed and stuck pending documents in a collection.

    Resets all retryable failed documents (transient or unknown errors) to PENDING,
    then creates a RETRY_DOCUMENTS operation to process all PENDING documents.
    This also handles documents that got stuck in PENDING status due to interrupted
    operations or worker crashes.
    """
    import uuid as uuid_module

    from shared.database.models import OperationType
    from shared.database.repositories.operation_repository import OperationRepository
    from webui.celery_app import celery_app

    try:
        document_repo = create_document_repository(db)
        operation_repo = OperationRepository(db)

        # Count stuck pending documents (more than 5 min old, never processed)
        pending_count = await document_repo.count_stuck_pending_documents(collection_uuid)

        # Reset all retryable failed documents to PENDING
        reset_count = await document_repo.bulk_reset_failed_for_retry(collection_uuid)

        if reset_count == 0 and pending_count == 0:
            # No documents to retry
            return {
                "reset_count": 0,
                "pending_count": 0,
                "operation_id": None,
                "message": "No retryable documents found",
            }

        # Create RETRY_DOCUMENTS operation (processes all PENDING documents)
        user_id = current_user.get("id") or current_user.get("sub")
        operation = await operation_repo.create(
            collection_id=collection_uuid,
            user_id=user_id,
            operation_type=OperationType.RETRY_DOCUMENTS,
            config={
                "reset_count": reset_count,
                "pending_count": pending_count,
                "triggered_by": "manual",
            },
        )

        # Commit BEFORE dispatching Celery task (critical to avoid race condition)
        await db.commit()

        # Dispatch Celery task
        celery_app.send_task(
            "webui.tasks.process_collection_operation",
            args=[operation.uuid],
            task_id=str(uuid_module.uuid4()),
        )

        total_count = reset_count + pending_count
        logger.info(
            "User %s triggered retry of %d documents (%d failed reset, %d stuck pending) "
            "in collection %s (operation=%s)",
            user_id,
            total_count,
            reset_count,
            pending_count,
            collection_uuid,
            operation.uuid,
        )

        return {
            "reset_count": reset_count,
            "pending_count": pending_count,
            "operation_id": operation.uuid,
            "message": f"Dispatched retry operation for {total_count} document(s)",
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("Failed to bulk retry documents: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retry failed documents") from e


@router.get(
    "/collections/{collection_uuid}/documents/failed/count",
    response_model=None,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Collection not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_failed_document_count(
    collection_uuid: str,
    retryable_only: bool = False,
    collection: Collection = Depends(get_collection_for_user),  # noqa: ARG001
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    db: AsyncSession = Depends(get_db),
) -> dict[str, int]:
    """Get count of failed documents by error category.

    Returns counts broken down by error category (transient, permanent, unknown)
    and a total count. Optionally filter to only retryable documents.
    """
    from webui.api.schemas import FailedDocumentCountResponse

    try:
        document_repo = create_document_repository(db)
        counts = await document_repo.get_failed_document_count(
            collection_uuid,
            retryable_only=retryable_only,
        )

        return FailedDocumentCountResponse(**counts).model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get failed document count: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get failed document count") from e
