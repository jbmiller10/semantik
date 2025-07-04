"""
Collections management API endpoints
"""

import logging
import os
import shutil
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from qdrant_client import QdrantClient
from qdrant_client.models import CollectionInfo

from .. import database
from ..auth import get_current_user
from ..utils.qdrant_manager import qdrant_manager

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
    job_count: int


class CollectionStats(BaseModel):
    total_files: int
    total_vectors: int
    total_size: int
    job_count: int


class CollectionConfig(BaseModel):
    model_name: str
    chunk_size: int
    chunk_overlap: int
    quantization: str
    vector_dim: int | None
    instruction: str | None


class JobInfo(BaseModel):
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
    jobs: list[JobInfo]


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
    job_id: str
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
async def list_collections(current_user: dict[str, Any] = Depends(get_current_user)):
    """List all unique collections with summary stats"""
    try:
        # Get collections from database
        collections = database.list_collections(user_id=current_user["id"])

        # Get Qdrant client
        qdrant = qdrant_manager.get_client()

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
                    job_count=collection["job_count"],
                )
            )

        return result

    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{collection_name}", response_model=CollectionDetails)
async def get_collection_details(collection_name: str, current_user: dict[str, Any] = Depends(get_current_user)):
    """Get detailed information for a single collection"""
    try:
        # Get collection details from database
        details = database.get_collection_details(collection_name, user_id=current_user["id"])

        if not details:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found or access denied",
            )

        # Get Qdrant client for actual vector counts
        qdrant = qdrant_manager.get_client()
        actual_vectors = 0

        for job in details["jobs"]:
            if job["status"] == "completed":
                try:
                    qdrant_collection = f"job_{job['id']}"
                    info = qdrant.get_collection(qdrant_collection)
                    if isinstance(info, CollectionInfo):
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
            jobs=[JobInfo(**job) for job in details["jobs"]],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collection details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{collection_name}")
async def rename_collection(
    collection_name: str,
    request: CollectionRenameRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Rename the display name of a collection"""
    try:
        # Attempt to rename
        success = database.rename_collection(
            old_name=collection_name,
            new_name=request.new_name,
            user_id=current_user["id"],
        )

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to rename collection. Either you don't have access, or the new name already exists.",
            )

        logger.info(f"User {current_user['username']} renamed collection '{collection_name}' to '{request.new_name}'")

        return {"message": "Collection renamed successfully", "new_name": request.new_name}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{collection_name}")
async def delete_collection(collection_name: str, current_user: dict[str, Any] = Depends(get_current_user)):
    """Delete a collection and all associated data"""
    try:
        # Get deletion info from database
        deletion_info = database.delete_collection(collection_name=collection_name, user_id=current_user["id"])

        if not deletion_info["job_ids"]:
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

        for job_id in deletion_info["job_ids"]:
            # Try to delete job directory
            job_dir = os.path.join("/app/jobs", job_id)
            if os.path.exists(job_dir):
                try:
                    shutil.rmtree(job_dir)
                    deleted_artifacts.append(job_dir)
                except Exception as e:
                    logger.error(f"Failed to delete job directory {job_dir}: {e}")
                    failed_artifacts.append(job_dir)

            # Try to delete output files
            output_dir = os.path.join("/app/output", job_id)
            if os.path.exists(output_dir):
                try:
                    shutil.rmtree(output_dir)
                    deleted_artifacts.append(output_dir)
                except Exception as e:
                    logger.error(f"Failed to delete output directory {output_dir}: {e}")
                    failed_artifacts.append(output_dir)

        logger.info(
            f"User {current_user['username']} deleted collection '{collection_name}' "
            f"({len(deletion_info['job_ids'])} jobs, {len(deleted_qdrant)} Qdrant collections)"
        )

        return {
            "message": "Collection deleted successfully",
            "deleted": {
                "jobs": len(deletion_info["job_ids"]),
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{collection_name}/files", response_model=PaginatedFileList)
async def get_collection_files(
    collection_name: str,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get paginated list of files in a collection"""
    try:
        # Get files from database
        result = database.get_collection_files(
            collection_name=collection_name,
            user_id=current_user["id"],
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
        raise HTTPException(status_code=500, detail=str(e))

