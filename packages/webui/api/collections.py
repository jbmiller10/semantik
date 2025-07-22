"""
Collections management API endpoints
"""

import logging
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from qdrant_client.models import CollectionInfo
from shared.database.base import CollectionRepository
from shared.database.exceptions import AccessDeniedError, EntityAlreadyExistsError, EntityNotFoundError
from webui.auth import get_current_user
from webui.dependencies import get_collection_repository
from webui.utils.qdrant_manager import qdrant_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collections", tags=["collections"])


# Pydantic Models
class CollectionSummary(BaseModel):
    name: str
    total_files: int
    total_vectors: int
    model_name: str
    created_at: str
    updated_at: str
    operation_count: int


class CollectionStats(BaseModel):
    total_files: int
    total_vectors: int
    total_size: int
    operation_count: int


class CollectionConfig(BaseModel):
    model_name: str
    chunk_size: int
    chunk_overlap: int
    quantization: str
    vector_dim: int | None
    instruction: str | None


class OperationInfo(BaseModel):
    id: str
    status: str
    created_at: str
    updated_at: str
    directory_path: str
    total_files: int
    processed_files: int
    failed_files: int
    mode: str


class CollectionDetails(BaseModel):
    name: str
    stats: CollectionStats
    configuration: CollectionConfig
    source_directories: list[str]
    operations: list[OperationInfo]


class CollectionRenameRequest(BaseModel):
    new_name: str = Field(..., min_length=1, max_length=255)

    @field_validator("new_name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        # Remove leading/trailing whitespace
        v = v.strip()
        if not v:
            raise ValueError("Collection name cannot be empty")
        # Check for invalid characters
        invalid_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
        for char in invalid_chars:
            if char in v:
                raise ValueError(f"Collection name cannot contain '{char}'")
        return v


class FileInfo(BaseModel):
    id: int
    operation_id: str
    path: str
    size: int
    modified: str
    extension: str
    status: str
    chunks_created: int
    vectors_created: int
    collection_name: str


class PaginatedFileList(BaseModel):
    files: list[FileInfo]
    total: int
    page: int
    pages: int


@router.get("", response_model=list[CollectionSummary])
async def list_collections(
    current_user: dict[str, Any] = Depends(get_current_user),
    collection_repo: CollectionRepository = Depends(get_collection_repository),
) -> list[CollectionSummary]:
    """List all unique collections with summary stats"""
    try:
        # Get collections from database
        collections = await collection_repo.list_collections(user_id=str(current_user["id"]))

        # Get Qdrant client
        qdrant_manager.get_client()

        # Enhance with actual vector counts from Qdrant
        result = []
        for collection in collections:
            # The collection dict already has aggregated data from list_collections
            # For now, we'll use the database-calculated vector count
            # TODO: Could enhance this by querying Qdrant for each job's actual count

            result.append(
                CollectionSummary(
                    name=collection["name"],
                    total_files=collection["total_files"] or 0,
                    total_vectors=collection["total_vectors"] or 0,
                    model_name=collection["model_name"] or "Unknown",
                    created_at=collection["created_at"],
                    updated_at=collection["updated_at"],
                    operation_count=collection["operation_count"],
                )
            )

        return result

    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{collection_name}", response_model=CollectionDetails)
async def get_collection_details(
    collection_name: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    collection_repo: CollectionRepository = Depends(get_collection_repository),
) -> CollectionDetails:
    """Get detailed information for a single collection"""
    try:
        # Get collection details from database
        details = await collection_repo.get_collection_details(collection_name, user_id=str(current_user["id"]))

        if not details:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found or access denied",
            )

        # Get Qdrant client for actual vector counts
        qdrant = qdrant_manager.get_client()
        actual_vectors = 0

        for operation in details["operations"]:
            if operation["status"] == "completed":
                try:
                    qdrant_collection = f"operation_{operation['id']}"
                    info = qdrant.get_collection(qdrant_collection)
                    if isinstance(info, CollectionInfo) and info.points_count is not None:
                        actual_vectors += info.points_count
                except Exception:
                    # Collection might not exist
                    pass

        # Use actual count if available
        if actual_vectors > 0:
            details["stats"]["total_vectors"] = actual_vectors

        return CollectionDetails(
            name=details["name"],
            stats=CollectionStats(**details["stats"]),
            configuration=CollectionConfig(**details["configuration"]),
            source_directories=details["source_directories"],
            operations=[OperationInfo(**operation) for operation in details["operations"]],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collection details: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/{collection_name}")
async def rename_collection(
    collection_name: str,
    request: CollectionRenameRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    collection_repo: CollectionRepository = Depends(get_collection_repository),
) -> dict[str, str]:
    """Rename the display name of a collection"""
    try:
        # Attempt to rename
        success = await collection_repo.rename_collection(
            old_name=collection_name,
            new_name=request.new_name,
            user_id=str(current_user["id"]),
        )

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to rename collection. Either you don't have access, or the new name already exists.",
            )

        logger.info(f"User {current_user['username']} renamed collection '{collection_name}' to '{request.new_name}'")

        return {"message": "Collection renamed successfully", "new_name": request.new_name}

    except EntityAlreadyExistsError:
        raise HTTPException(
            status_code=409,
            detail=f"A collection with the name '{request.new_name}' already exists.",
        ) from None
    except AccessDeniedError:
        raise HTTPException(
            status_code=403,
            detail=f"You don't have permission to rename collection '{collection_name}'.",
        ) from None
    except EntityNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found.",
        ) from None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming collection: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/{collection_name}")
async def delete_collection(
    collection_name: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    collection_repo: CollectionRepository = Depends(get_collection_repository),
) -> dict[str, Any]:
    """Delete a collection and all associated data"""
    try:
        # Get deletion info from database
        deletion_info = await collection_repo.delete_collection(
            collection_name=collection_name, user_id=str(current_user["id"])
        )

        if not deletion_info["operation_ids"]:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found or access denied",
            )

        # Delete from Qdrant
        qdrant = qdrant_manager.get_client()
        deleted_qdrant = []
        failed_qdrant = []

        for qdrant_collection in deletion_info["qdrant_collections"]:
            try:
                qdrant.delete_collection(qdrant_collection)
                deleted_qdrant.append(qdrant_collection)
            except Exception as e:
                logger.error(f"Failed to delete Qdrant collection {qdrant_collection}: {e}")
                failed_qdrant.append(qdrant_collection)

        # Clean up job artifacts from filesystem
        deleted_artifacts = []
        failed_artifacts = []

        for operation_id in deletion_info["operation_ids"]:
            # Try to delete operation directory
            operation_dir = Path("/app/operations") / operation_id
            if operation_dir.exists():
                try:
                    shutil.rmtree(operation_dir)
                    deleted_artifacts.append(str(operation_dir))
                except Exception as e:
                    logger.error(f"Failed to delete operation directory {operation_dir}: {e}")
                    failed_artifacts.append(str(operation_dir))

            # Try to delete output files
            output_dir = Path("/app/output") / operation_id
            if output_dir.exists():
                try:
                    shutil.rmtree(output_dir)
                    deleted_artifacts.append(str(output_dir))
                except Exception as e:
                    logger.error(f"Failed to delete output directory {output_dir}: {e}")
                    failed_artifacts.append(str(output_dir))

        logger.info(
            f"User {current_user['username']} deleted collection '{collection_name}' "
            f"({len(deletion_info['operation_ids'])} operations, {len(deleted_qdrant)} Qdrant collections)"
        )

        return {
            "message": "Collection deleted successfully",
            "deleted": {
                "operations": len(deletion_info["operation_ids"]),
                "qdrant_collections": len(deleted_qdrant),
                "artifacts": len(deleted_artifacts),
            },
            "errors": {
                "qdrant_failures": failed_qdrant,
                "artifact_failures": failed_artifacts,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{collection_name}/files", response_model=PaginatedFileList)
async def get_collection_files(
    collection_name: str,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    current_user: dict[str, Any] = Depends(get_current_user),
    collection_repo: CollectionRepository = Depends(get_collection_repository),
) -> PaginatedFileList:
    """Get paginated list of files in a collection"""
    try:
        # Get files from database
        result = await collection_repo.get_collection_files(
            collection_name=collection_name,
            user_id=str(current_user["id"]),
            page=page,
            limit=limit,
        )

        return PaginatedFileList(
            files=[FileInfo(**file) for file in result["files"]],
            total=result["total"],
            page=result["page"],
            pages=result["pages"],
        )

    except Exception as e:
        logger.error(f"Error getting collection files: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
