"""Internal API endpoints for system services"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from shared.config import settings
from shared.database.base import JobRepository
from shared.database.factory import create_job_repository
from shared.database.models import CollectionStatus
from shared.database.repositories.collection_repository import CollectionRepository
from sqlalchemy.ext.asyncio import AsyncSession
from webui.database import get_async_session

router = APIRouter(prefix="/api/internal", tags=["internal"])


def verify_internal_api_key(x_internal_api_key: Annotated[str | None, Header()] = None) -> None:
    """Verify the internal API key."""
    if x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing internal API key")


@router.get("/jobs/all-ids", dependencies=[Depends(verify_internal_api_key)])
async def get_all_job_ids(job_repo: JobRepository = Depends(create_job_repository)) -> list[str]:
    """
    Get all job IDs from the database.
    This endpoint is intended for internal services like maintenance/cleanup.
    """
    jobs = await job_repo.list_jobs()
    return [job["id"] for job in jobs]


class CompleteReindexRequest(BaseModel):
    """Request model for completing a reindex operation."""

    collection_id: str
    operation_id: str
    staging_collection_name: str
    new_config: dict[str, Any] | None = None
    vector_count: int


class CompleteReindexResponse(BaseModel):
    """Response model for completing a reindex operation."""

    success: bool
    old_collection_names: list[str]
    message: str


@router.post(
    "/complete-reindex", response_model=CompleteReindexResponse, dependencies=[Depends(verify_internal_api_key)]
)
async def complete_reindex(
    request: CompleteReindexRequest, session: AsyncSession = Depends(get_async_session)
) -> CompleteReindexResponse:
    """
    Atomically complete a reindex operation by switching from staging to active.

    This endpoint performs a single atomic transaction to:
    1. Copy staging collection names to active collection list
    2. Clear the staging field
    3. Update collection status to ready

    Args:
        request: The reindex completion request details
        session: Database session

    Returns:
        Response containing the old collection names for cleanup

    Raises:
        HTTPException: If the operation fails
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Begin atomic transaction
        async with session.begin():
            # Initialize repositories
            collection_repo = CollectionRepository(session)

            # Get the collection
            collection = await collection_repo.get_by_uuid(request.collection_id)
            if not collection:
                raise HTTPException(status_code=404, detail=f"Collection {request.collection_id} not found")

            # Save old collection names for cleanup
            old_collection_names = collection.qdrant_collections or []

            # Prepare updates
            updates = {
                "qdrant_collections": [request.staging_collection_name],
                "qdrant_staging": None,
                "status": CollectionStatus.READY,
                "status_message": None,
                "vector_count": request.vector_count,
            }

            # Add new config if provided
            if request.new_config:
                updates["config"] = request.new_config
                # Update individual config fields
                updates["embedding_model"] = request.new_config.get("embedding_model", collection.embedding_model)
                updates["chunk_size"] = request.new_config.get("chunk_size", collection.chunk_size)
                updates["chunk_overlap"] = request.new_config.get("chunk_overlap", collection.chunk_overlap)

            # Perform atomic update
            await collection_repo.update(request.collection_id, updates)

            logger.info(
                f"Completed atomic switch for collection {request.collection_id}: "
                f"{old_collection_names} -> [{request.staging_collection_name}]"
            )

        # Transaction committed successfully
        return CompleteReindexResponse(
            success=True, old_collection_names=old_collection_names, message="Reindex completed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to complete reindex for collection {request.collection_id}: {e}")
        # Rollback happened automatically
        raise HTTPException(status_code=500, detail=f"Failed to complete reindex: {str(e)}") from e
