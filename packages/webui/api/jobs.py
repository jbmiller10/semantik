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

import os

from shared.config import settings
from shared.database.base import CollectionRepository, FileRepository, JobRepository
from shared.database.factory import create_collection_repository, create_file_repository, create_job_repository
from shared.embedding import POPULAR_MODELS, embedding_service
from webui.auth import get_current_user, get_current_user_websocket
from webui.tasks import process_embedding_job_task
from webui.utils.qdrant_manager import qdrant_manager
from webui.websocket_manager import get_websocket_manager

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


# Get Redis URL from environment
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Get WebSocket manager instance
manager = get_websocket_manager(redis_url)

# Task tracking moved to database in future refactoring
# TODO: Add celery_task_id field to jobs table for task management


# API Routes
@router.get("/new-id")
async def get_new_job_id(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, str]:  # noqa: ARG001
    """Generate a new job ID for WebSocket connection"""
    return {"job_id": str(uuid.uuid4())}


@router.post("", response_model=JobStatus)
async def create_job(
    request: CreateJobRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    job_repo: JobRepository = Depends(create_job_repository),
    file_repo: FileRepository = Depends(create_file_repository),
    collection_repo: CollectionRepository = Depends(create_collection_repository),  # noqa: ARG001
) -> JobStatus:
    """Create a new embedding job"""
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


@router.get("", response_model=list[JobStatus])
async def list_jobs(
    current_user: dict[str, Any] = Depends(get_current_user),
    job_repo: JobRepository = Depends(create_job_repository),
) -> list[JobStatus]:
    """List all jobs for the current user"""
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


@router.delete("/{job_id}")
async def delete_job(
    job_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    job_repo: JobRepository = Depends(create_job_repository),
) -> dict[str, str]:
    """Delete a job and its associated collection"""
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

    return {"message": "Job deleted successfully"}


@router.get("/{job_id}", response_model=JobStatus)
async def get_job(
    job_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    job_repo: JobRepository = Depends(create_job_repository),
) -> JobStatus:
    """Get job details"""
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
async def websocket_endpoint(websocket: WebSocket, job_id: str, token: str | None = None) -> None:
    """WebSocket for real-time job updates with Redis stream support.

    This endpoint:
    1. Authenticates the WebSocket client
    2. Connects the WebSocket client
    3. Sends the current job state from the database
    4. Subscribes to Redis streams for live updates

    Args:
        websocket: The WebSocket connection
        job_id: The job ID to subscribe to
        token: JWT authentication token (passed as query parameter)
    """
    job_repo = create_job_repository()

    try:
        # Authenticate the user if authentication is enabled
        user = None
        if not settings.DISABLE_AUTH:
            if not token:
                await websocket.close(code=1008, reason="Missing authentication token")
                return

            try:
                user = await get_current_user_websocket(token)
            except ValueError as e:
                await websocket.close(code=1008, reason=str(e))
                return

        # Get current job state from database
        job = await job_repo.get_job(job_id)
        if not job:
            await websocket.close(code=1008, reason="Job not found")
            return

        # Check if user has access to this job (if auth is enabled)
        if user and job.get("user_id") != user["id"]:
            await websocket.close(code=1008, reason="Access denied")
            return

        # Connect the WebSocket
        await manager.connect(websocket, job_id)

        if job:
            # Send initial state to client
            await manager.send_initial_state(
                websocket,
                job_id,
                {
                    "id": job["id"],
                    "name": job["name"],
                    "status": job["status"],
                    "total_files": job.get("total_files", 0),
                    "processed_files": job.get("processed_files", 0),
                    "failed_files": job.get("failed_files", 0),
                    "current_file": job.get("current_file"),
                    "error": job.get("error"),
                    "created_at": job["created_at"],
                    "updated_at": job["updated_at"],
                },
            )

        # Keep connection alive and handle incoming messages
        while True:
            # Wait for messages from client (like ping/pong)
            await websocket.receive_text()
            # Could handle client messages here if needed

    except WebSocketDisconnect:
        await manager.disconnect(websocket, job_id)
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        await manager.disconnect(websocket, job_id)


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
