"""
Job management routes and WebSocket handlers for the Web UI
"""

import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

from fastapi import APIRouter, Body, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance
from shared.contracts.jobs import AddToCollectionRequest
from shared.contracts.jobs import CreateJobRequest as SharedCreateJobRequest

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from shared.config import settings
from shared.database.base import CollectionRepository, FileRepository, JobRepository
from shared.database.factory import create_collection_repository, create_file_repository, create_job_repository
from shared.embedding import POPULAR_MODELS, embedding_service
from webui.auth import get_current_user, get_current_user_websocket
from webui.tasks import process_embedding_job_task
from webui.utils.qdrant_manager import qdrant_manager
from webui.websocket_manager import ws_manager

logger = logging.getLogger(__name__)

# Constants
JOBS_DIR = str(settings.jobs_dir)
OUTPUT_DIR = str(settings.output_dir)
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt", ".text", ".pptx", ".eml", ".md", ".html"]

# Create necessary directories
Path(JOBS_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# Request/Response models are imported from shared.contracts.jobs
# Create a proper type alias for mypy
CreateJobRequest: TypeAlias = SharedCreateJobRequest


# Legacy JobStatus for WebSocket updates (different from JobResponse)
class JobStatus(BaseModel):
    id: str
    name: str
    status: str  # created, scanning, processing, completed, failed
    created_at: str
    updated_at: str
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    current_file: str | None = None
    error: str | None = None
    model_name: str
    directory_path: str
    quantization: str | None = None
    batch_size: int | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None


# WebSocket manager is imported at the top with other webui imports

# Task tracking moved to database in future refactoring
# TODO: Add celery_task_id field to jobs table for task management


# API Routes
@router.get("/new-id")
async def get_new_job_id(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, str]:  # noqa: ARG001
    """Generate a new job ID for WebSocket connection"""
    return {"job_id": str(uuid.uuid4())}


@router.post("", response_model=JobStatus, deprecated=True)
async def create_job(
    request: CreateJobRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    job_repo: JobRepository = Depends(create_job_repository),
    file_repo: FileRepository = Depends(create_file_repository),
    collection_repo: CollectionRepository = Depends(create_collection_repository),  # noqa: ARG001
) -> JobStatus:
    """Create a new embedding job

    **DEPRECATED**: This endpoint is deprecated and will be removed in v2.0.
    Use collection-based operations instead:
    - For new collections: Collections are created automatically when processing files
    - For adding to existing collections: Use POST /api/jobs/add-to-collection

    Migration guide:
    1. Instead of creating individual jobs, work with collections as logical units
    2. Collections group related documents and can span multiple indexing operations
    3. Use /api/collections endpoints for management and /api/search for querying
    """
    logger.warning(
        "POST /api/jobs is deprecated. Use collection-based operations instead. "
        "This endpoint will be removed in v2.0."
    )
    # Accept job_id from request if provided, otherwise generate new one
    job_id = request.job_id if request.job_id else str(uuid.uuid4())

    try:
        # Import here to avoid circular import
        from .files import scan_directory_async

        # Scan directory first - use async version to avoid blocking UI
        scan_result = await scan_directory_async(request.directory_path, recursive=True, scan_id=job_id)
        files = scan_result["files"]

        if not files:
            raise HTTPException(status_code=400, detail="No supported files found in directory")

        # Create job record
        now = datetime.now(UTC).isoformat()
        job_data = {
            "id": job_id,
            "name": request.name,
            "description": request.description,
            "status": "created",
            "created_at": now,
            "updated_at": now,
            "directory_path": request.directory_path,
            "model_name": request.model_name,
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "batch_size": request.batch_size,
            "vector_dim": request.vector_dim,
            "quantization": request.quantization,
            "instruction": request.instruction,
            "user_id": current_user["id"],
        }
        await job_repo.create_job(job_data)

        # Create file records
        file_records = [
            {
                "path": f.path,
                "size": f.size,
                "modified": f.modified,
                "extension": f.extension,
                "hash": getattr(f, "hash", None),
                "content_hash": getattr(f, "content_hash", None),
            }
            for f in files
        ]
        await file_repo.add_files_to_job(job_id, file_records)

        # Get Qdrant client from connection manager
        qdrant = qdrant_manager.get_client()

        # Determine vector size
        vector_size = request.vector_dim
        if not vector_size:
            # Try to get from POPULAR_MODELS first
            if request.model_name in POPULAR_MODELS:
                dim_value = POPULAR_MODELS[request.model_name].get("dim") or POPULAR_MODELS[request.model_name].get(
                    "dimension"
                )
                if isinstance(dim_value, int):
                    vector_size = dim_value

            # If still not found, get from actual model
            if not vector_size:
                try:
                    model_info = embedding_service.get_model_info(request.model_name, request.quantization)
                    if not model_info.get("error"):
                        vector_size = model_info["embedding_dim"]
                    else:
                        logger.warning(f"Could not get model info: {model_info.get('error')}")
                        vector_size = 1024  # Default fallback
                except Exception as e:
                    logger.warning(f"Error getting model info: {e}")
                    vector_size = 1024  # Default fallback

        # Ensure we have a valid vector size
        if not vector_size:
            vector_size = 1024  # Final fallback

        # Update job with actual vector dimension
        await job_repo.update_job(job_id, {"vector_dim": vector_size})

        # Create collection using connection manager with retry logic
        collection_name = f"job_{job_id}"
        try:
            qdrant_manager.create_collection(
                collection_name=collection_name, vector_size=vector_size, distance=Distance.COSINE
            )

            # Verify collection was created
            qdrant_manager.verify_collection(collection_name)
            logger.info(f"Successfully created collection {collection_name}")

            # Store metadata about this collection
            from shared.database.collection_metadata import store_collection_metadata

            store_collection_metadata(
                qdrant=qdrant,
                collection_name=collection_name,
                model_name=request.model_name,
                quantization=request.quantization,
                vector_dim=vector_size,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                instruction=request.instruction or "",
            )

        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            # Clean up job and return error
            await job_repo.delete_job(job_id)
            raise HTTPException(status_code=500, detail=f"Failed to create Qdrant collection: {str(e)}") from e

        # Start processing with Celery
        try:
            celery_task = process_embedding_job_task.delay(job_id)
            logger.info(f"Started Celery task {celery_task.id} for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to start Celery task for job {job_id}: {e}")
            await job_repo.update_job(job_id, {"status": "failed", "error": f"Failed to start task: {str(e)}"})
            raise HTTPException(status_code=500, detail=f"Failed to start processing task: {str(e)}") from e

        return JobStatus(
            id=job_id,
            name=request.name,
            status="created",
            created_at=now,
            updated_at=now,
            total_files=len(files),
            processed_files=0,
            failed_files=0,
            model_name=request.model_name,
            directory_path=request.directory_path,
            quantization=request.quantization,
            batch_size=request.batch_size,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/add-to-collection", response_model=JobStatus)
async def add_to_collection(
    request: AddToCollectionRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    job_repo: JobRepository = Depends(create_job_repository),
    file_repo: FileRepository = Depends(create_file_repository),
    collection_repo: CollectionRepository = Depends(create_collection_repository),
) -> JobStatus:
    """Add new documents to an existing collection"""
    # Accept job_id from request if provided, otherwise generate new one
    job_id = request.job_id if request.job_id else str(uuid.uuid4())

    try:
        # Import here to avoid circular import
        from .files import scan_directory_async

        # Get parent collection metadata
        collection_metadata = await collection_repo.get_collection_metadata(request.collection_name)
        if not collection_metadata:
            raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found")

        # Scan directory for new files
        scan_result = await scan_directory_async(request.directory_path, recursive=True, scan_id=job_id)
        files = scan_result["files"]

        if not files:
            raise HTTPException(status_code=400, detail="No supported files found in directory")

        # Check for duplicates
        content_hashes = [f.content_hash for f in files if f.content_hash]
        existing_hashes = await file_repo.get_duplicate_files_in_collection(request.collection_name, content_hashes)

        # Filter out duplicates
        new_files = [f for f in files if f.content_hash not in existing_hashes]

        if not new_files:
            raise HTTPException(status_code=400, detail=f"All {len(files)} files already exist in the collection")

        # Get the actual Qdrant collection to verify vector dimensions
        qdrant = qdrant_manager.get_client()
        parent_collection_name = f"job_{collection_metadata['id']}"

        try:
            collection_info = qdrant.get_collection(parent_collection_name)
            # Get the actual vector dimension from Qdrant
            vectors_config = collection_info.config.params.vectors
            if vectors_config is None:
                raise ValueError(f"No vector configuration found for collection {parent_collection_name}")
            if hasattr(vectors_config, "size"):
                actual_vector_dim = vectors_config.size
            else:
                # It's a dict, get the first vector config
                actual_vector_dim = next(iter(vectors_config.values())).size

            # Verify that the stored metadata matches the actual collection
            if actual_vector_dim != collection_metadata["vector_dim"]:
                logger.warning(
                    f"Vector dimension mismatch: Qdrant has {actual_vector_dim}, "
                    f"metadata has {collection_metadata['vector_dim']}. Using Qdrant value."
                )
                # Use the actual dimension from Qdrant
                collection_metadata["vector_dim"] = actual_vector_dim

            # Verify the model can generate embeddings of the required dimension
            model_name = collection_metadata["model_name"]
            expected_dim = actual_vector_dim

            # Get the model's natural dimension
            model_natural_dim = None
            if model_name in POPULAR_MODELS:
                model_natural_dim = POPULAR_MODELS[model_name].get("dim") or POPULAR_MODELS[model_name].get("dimension")

            if not model_natural_dim:
                try:
                    model_info = embedding_service.get_model_info(model_name, collection_metadata["quantization"])
                    if not model_info.get("error"):
                        model_natural_dim = model_info["embedding_dim"]
                except Exception as e:
                    logger.warning(f"Could not get model dimension info: {e}")

            # Check if dimension adjustment would be needed
            if model_natural_dim and model_natural_dim != expected_dim:
                logger.info(
                    f"Model {model_name} naturally produces {model_natural_dim}-dimensional embeddings, "
                    f"but collection expects {expected_dim}. Embeddings will be adjusted."
                )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to verify collection dimensions: {str(e)}") from e

        # Create job record with inherited settings
        now = datetime.now(UTC).isoformat()
        job_data = {
            "id": job_id,
            "name": request.collection_name,  # Use same collection name
            "description": request.description
            or f"Adding {len(new_files)} new files (skipped {len(files) - len(new_files)} duplicates)",
            "status": "created",
            "created_at": now,
            "updated_at": now,
            "directory_path": request.directory_path,
            "model_name": collection_metadata["model_name"],
            "chunk_size": collection_metadata["chunk_size"],
            "chunk_overlap": collection_metadata["chunk_overlap"],
            "batch_size": collection_metadata["batch_size"],
            "vector_dim": collection_metadata["vector_dim"],
            "quantization": collection_metadata["quantization"],
            "instruction": collection_metadata["instruction"],
            "user_id": current_user["id"],
            "parent_job_id": collection_metadata["id"],
            "mode": "append",
        }
        await job_repo.create_job(job_data)

        # Create file records for new files only
        file_records = [
            {
                "path": f.path,
                "size": f.size,
                "modified": f.modified,
                "extension": f.extension,
                "hash": getattr(f, "hash", None),
                "content_hash": getattr(f, "content_hash", None),
            }
            for f in new_files
        ]
        await file_repo.add_files_to_job(job_id, file_records)

        # Update total files count
        await job_repo.update_job(job_id, {"total_files": len(new_files)})

        # Start processing with Celery
        try:
            celery_task = process_embedding_job_task.delay(job_id)
            logger.info(f"Started Celery task {celery_task.id} for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to start Celery task for job {job_id}: {e}")
            await job_repo.update_job(job_id, {"status": "failed", "error": f"Failed to start task: {str(e)}"})
            raise HTTPException(status_code=500, detail=f"Failed to start processing task: {str(e)}") from e

        # Return job status
        return JobStatus(
            id=job_id,
            name=request.collection_name,
            status="created",
            created_at=now,
            updated_at=now,
            total_files=len(new_files),
            processed_files=0,
            failed_files=0,
            model_name=collection_metadata["model_name"],
            directory_path=request.directory_path,
            quantization=collection_metadata["quantization"],
            batch_size=collection_metadata["batch_size"],
            chunk_size=collection_metadata["chunk_size"],
            chunk_overlap=collection_metadata["chunk_overlap"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("", response_model=list[JobStatus], deprecated=True)
async def list_jobs(
    current_user: dict[str, Any] = Depends(get_current_user),
    job_repo: JobRepository = Depends(create_job_repository),
) -> list[JobStatus]:
    """List all jobs for the current user

    **DEPRECATED**: This endpoint is deprecated and will be removed in v2.0.
    Use GET /api/collections instead for a collection-centric view.

    Migration guide:
    1. GET /api/collections provides aggregated view of all collections
    2. Each collection shows total files, vectors, and associated jobs
    3. Use GET /api/collections/{name} for detailed job information per collection
    """
    logger.warning(
        "GET /api/jobs is deprecated. Use GET /api/collections instead. This endpoint will be removed in v2.0."
    )
    jobs = await job_repo.list_jobs(user_id=str(current_user["id"]))

    result = []
    for job in jobs:
        result.append(
            JobStatus(
                id=job["id"],
                name=job["name"],
                status=job["status"],
                created_at=job["created_at"],
                updated_at=job["updated_at"],
                total_files=job.get("total_files", 0),
                processed_files=job.get("processed_files", 0),
                failed_files=job.get("failed_files", 0),
                current_file=job.get("current_file"),
                error=job.get("error"),
                model_name=job["model_name"],
                directory_path=job["directory_path"],
                quantization=job.get("quantization"),
                batch_size=job.get("batch_size"),
                chunk_size=job.get("chunk_size"),
                chunk_overlap=job.get("chunk_overlap"),
            )
        )

    return result


@router.get("/collection-metadata/{collection_name}")
async def get_collection_metadata(
    collection_name: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    collection_repo: CollectionRepository = Depends(create_collection_repository),
) -> dict[str, Any]:
    """Get metadata for a specific collection"""
    metadata = await collection_repo.get_collection_metadata(collection_name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

    # Only return necessary fields
    return {
        "id": metadata["id"],
        "name": metadata["name"],
        "model_name": metadata["model_name"],
        "chunk_size": metadata["chunk_size"],
        "chunk_overlap": metadata["chunk_overlap"],
        "batch_size": metadata["batch_size"],
        "quantization": metadata["quantization"],
        "instruction": metadata.get("instruction"),
        "vector_dim": metadata["vector_dim"],
    }


@router.post("/check-duplicates")
async def check_duplicates(
    collection_name: str = Body(...),
    content_hashes: list[str] = Body(...),
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    file_repo: FileRepository = Depends(create_file_repository),
) -> dict[str, list[str]]:
    """Check which content hashes already exist in a collection"""
    existing_hashes = await file_repo.get_duplicate_files_in_collection(collection_name, content_hashes)
    return {"existing_hashes": list(existing_hashes)}


@router.get("/collections-status")
async def check_collections_status(
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    job_repo: JobRepository = Depends(create_job_repository),
) -> dict[str, dict[str, Any]]:
    """Check which job collections exist in Qdrant"""
    try:
        qdrant = qdrant_manager.get_client()

        # Get all collections
        collections = qdrant.get_collections().collections
        collection_names = {c.name for c in collections}

        # Get all jobs
        jobs = await job_repo.list_jobs()

        # Check each job's collection
        results = {}
        for job in jobs:
            collection_name = f"job_{job['id']}"
            exists = collection_name in collection_names

            point_count: int | None = 0
            if exists:
                try:
                    collection_info = qdrant.get_collection(collection_name)
                    point_count = collection_info.points_count
                except Exception as e:
                    logger.warning(f"Could not get point count for collection {collection_name}: {e}")

            results[job["id"]] = {"exists": exists, "point_count": point_count, "status": job["status"]}

        return results

    except Exception as e:
        logger.error(f"Failed to check collections status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check collections: {str(e)}") from e


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    job_repo: JobRepository = Depends(create_job_repository),
) -> dict[str, str]:
    """Cancel a running job"""
    # Check current job status
    job = await job_repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] not in ["created", "scanning", "processing"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job in status: {job['status']}")

    # Update job status to cancelled
    await job_repo.update_job(job_id, {"status": "cancelled"})

    # TODO: Implement task cancellation when celery_task_id is added to database
    # The Celery task ID needs to be stored persistently to support cancellation
    # after server restarts or when tasks are distributed across workers.
    #
    # Future implementation:
    # task_id = await job_repo.get_celery_task_id(job_id)
    # if task_id:
    #     from webui.celery_app import celery_app
    #     try:
    #         celery_app.control.revoke(task_id, terminate=True, signal="SIGKILL")
    #         logger.info(f"Successfully revoked Celery task {task_id} for job {job_id}")
    #     except Exception as e:
    #         logger.warning(f"Failed to revoke Celery task {task_id} for job {job_id}: {e}")

    logger.warning(f"Job cancellation requested for {job_id} but task revocation not yet implemented")

    return {"message": "Job marked as cancelled (task revocation pending implementation)"}


@router.delete("/{job_id}", deprecated=True)
async def delete_job(
    job_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    job_repo: JobRepository = Depends(create_job_repository),
) -> dict[str, str]:
    """Delete a job and its associated collection

    **DEPRECATED**: This endpoint is deprecated and will be removed in v2.0.
    Use DELETE /api/collections/{collection_name} instead.

    Migration guide:
    1. DELETE /api/collections/{name} removes entire collection and all associated jobs
    2. This provides cleaner semantics for managing document sets
    3. Deletion is atomic at the collection level
    """
    logger.warning(
        f"DELETE /api/jobs/{job_id} is deprecated. Use DELETE /api/collections/{{collection_name}} instead. "
        "This endpoint will be removed in v2.0."
    )
    # Check if job exists
    job = await job_repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete from Qdrant
    collection_name = f"job_{job_id}"
    try:
        async_client = AsyncQdrantClient(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        await async_client.delete_collection(collection_name)
    except Exception as e:
        logger.warning(f"Failed to delete Qdrant collection: {e}")

    # Delete from database
    await job_repo.delete_job(job_id)

    # Clean up Redis stream
    try:
        await ws_manager.cleanup_job_stream(job_id)
        logger.info(f"Cleaned up Redis stream for deleted job {job_id}")
    except Exception as e:
        logger.warning(f"Failed to clean up Redis stream: {e}")

    return {"message": "Job deleted successfully"}


@router.get("/{job_id}", response_model=JobStatus, deprecated=True)
async def get_job(
    job_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    job_repo: JobRepository = Depends(create_job_repository),
) -> JobStatus:
    """Get job details

    **DEPRECATED**: This endpoint is deprecated and will be removed in v2.0.
    Use GET /api/collections/{collection_name} instead.

    Migration guide:
    1. Jobs are now grouped under collections
    2. GET /api/collections/{name} shows all jobs for a collection
    3. Collection details include configuration, stats, and job history
    """
    logger.warning(
        f"GET /api/jobs/{job_id} is deprecated. Use GET /api/collections/{{collection_name}} instead. "
        "This endpoint will be removed in v2.0."
    )
    job = await job_repo.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(
        id=job["id"],
        name=job["name"],
        status=job["status"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        total_files=job["total_files"],
        processed_files=job["processed_files"],
        failed_files=job["failed_files"],
        current_file=job["current_file"],
        error=job["error"],
        model_name=job["model_name"],
        directory_path=job["directory_path"],
        quantization=job.get("quantization"),
        batch_size=job.get("batch_size"),
        chunk_size=job.get("chunk_size"),
        chunk_overlap=job.get("chunk_overlap"),
    )


@router.get("/{job_id}/collection-exists")
async def check_collection_exists(
    job_id: str, current_user: dict[str, Any] = Depends(get_current_user)  # noqa: ARG001
) -> dict[str, Any]:
    """Check if a job's collection exists in Qdrant"""
    try:
        qdrant = qdrant_manager.get_client()
        collection_name = f"job_{job_id}"

        # Get all collections
        collections = qdrant.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)

        # If collection exists, get point count
        point_count: int | None = 0
        if collection_exists:
            try:
                collection_info = qdrant.get_collection(collection_name)
                point_count = collection_info.points_count
            except Exception as e:
                logger.warning(f"Could not get point count for collection {collection_name}: {e}")

        return {"exists": collection_exists, "collection_name": collection_name, "point_count": point_count}

    except Exception as e:
        logger.error(f"Failed to check collection existence: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check collection: {str(e)}") from e


# WebSocket handler - export this separately so it can be mounted at the app level
async def websocket_endpoint(websocket: WebSocket, job_id: str) -> None:
    """WebSocket for real-time job updates with Redis pub/sub.

    Authentication is handled via JWT token passed as query parameter.
    The token should be passed as ?token=<jwt_token> in the WebSocket URL.
    """
    job_repo = create_job_repository()

    # Extract token from query parameters
    token = websocket.query_params.get("token")

    try:
        # Authenticate the user
        user = await get_current_user_websocket(token)
        user_id = str(user["id"])

        # Verify the user has access to this job
        job = await job_repo.get_job(job_id)
        if not job:
            await websocket.close(code=1008, reason="Job not found")
            return

        # Check if user owns this job (unless auth is disabled)
        if not settings.DISABLE_AUTH and job.get("user_id") != user["id"]:
            await websocket.close(code=1008, reason="Access denied")
            return

    except ValueError as e:
        # Authentication failed
        await websocket.close(code=1008, reason=str(e))
        return
    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        await websocket.close(code=1011, reason="Internal server error")
        return

    # Authentication successful, connect the WebSocket
    await ws_manager.connect(websocket, job_id, user_id)

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, job_id, user_id)


# Operation-based WebSocket handler for collection operations
async def operation_websocket_endpoint(websocket: WebSocket, operation_id: str) -> None:
    """WebSocket for real-time operation updates with Redis pub/sub.

    Authentication is handled via JWT token passed as query parameter.
    The token should be passed as ?token=<jwt_token> in the WebSocket URL.
    """
    from shared.database.factory import create_operation_repository

    operation_repo = create_operation_repository()

    # Extract token from query parameters
    token = websocket.query_params.get("token")

    try:
        # Authenticate the user
        user = await get_current_user_websocket(token)
        user_id = user["id"]

        # Verify the user has access to this operation
        await operation_repo.get_by_uuid_with_permission_check(operation_id, user_id)

    except ValueError as e:
        # Authentication failed
        await websocket.close(code=1008, reason=str(e))
        return
    except Exception as e:
        logger.error(f"Operation WebSocket authentication error: {e}")
        await websocket.close(code=1011, reason="Internal server error")
        return

    # Authentication successful, connect the WebSocket
    await ws_manager.connect_operation(websocket, operation_id, str(user_id))

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect_operation(websocket, operation_id, str(user_id))


# TODO: Remove this function when Redis pub/sub is implemented
# This polling approach is being replaced with a more efficient pub/sub pattern
# async def poll_celery_task_state(job_id: str) -> None:
#     """Poll Celery task state and send WebSocket updates with adaptive polling interval."""
#     from webui.celery_app import celery_app
#
#     # Need celery_task_id from database instead of active_job_tasks
#     # task_id = await job_repo.get_celery_task_id(job_id)
#     # if not task_id:
#         # return

#     # Adaptive polling configuration
#     poll_interval = 1.0  # Start with 1 second
#     max_poll_interval = 10.0  # Max 10 seconds for long-running tasks
#     poll_increase_rate = 1.2  # Increase by 20% each iteration
#     task_start_time = asyncio.get_event_loop().time()
#     consecutive_errors = 0
#     max_consecutive_errors = 5
#
#     while True:
#         try:
#             # Get task result with connection error handling
#             try:
#                 result = celery_app.AsyncResult(task_id)
#                 task_state = result.state
#                 task_info = result.info
#                 consecutive_errors = 0  # Reset error counter on success
#             except Exception as e:
#                 logger.warning(f"Failed to get Celery task state for job {job_id}: {e}")
#                 consecutive_errors += 1
#
#                 if consecutive_errors >= max_consecutive_errors:
#                     logger.error(f"Too many consecutive errors polling task {task_id}, stopping")
#                     await manager.send_update(job_id, {"type": "error", "message": "Lost connection to task queue"})
#                     break
#
#                 await asyncio.sleep(5)  # Wait before retry
#                 continue
#
#             if task_state == "PENDING":
#                 # Task hasn't started yet
#                 pass
#             elif task_state == "PROCESSING":
#                 # Task is running, send progress update
#                 if task_info:
#                     await manager.send_update(
#                         job_id,
#                         {
#                             "type": task_info.get("status", "processing"),
#                             "total_files": task_info.get("total_files", 0),
#                             "processed_files": task_info.get("processed_files", 0),
#                             "current_file": task_info.get("current_file"),
#                         },
#                     )
#             elif task_state == "SUCCESS":
#                 # Task completed successfully
#                 await manager.send_update(job_id, {"type": "job_completed", "message": "Job completed successfully"})
#                 # Clean up - no longer needed without active_job_tasks
#                 break
#             elif task_state == "FAILURE":
#                 # Task failed
#                 await manager.send_update(
#                     job_id, {"type": "error", "message": str(task_info) if task_info else "Job failed"}
#                 )
#                 # Clean up - no longer needed without active_job_tasks
#                 break
#
#             # Adaptive polling: increase interval for long-running tasks
#             await asyncio.sleep(poll_interval)
#
#             # Calculate elapsed time
#             elapsed_time = asyncio.get_event_loop().time() - task_start_time
#
#             # Increase polling interval for long-running tasks
#             if elapsed_time > 30:  # After 30 seconds, start increasing interval
#                 poll_interval = min(poll_interval * poll_increase_rate, max_poll_interval)
#
#         except Exception as e:
#             logger.error(f"Unexpected error in poll_celery_task_state: {e}")
#             await asyncio.sleep(5)  # Longer delay on error
