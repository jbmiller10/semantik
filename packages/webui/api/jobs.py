"""
Job management routes and WebSocket handlers for the Web UI
"""

import asyncio
import gc
import hashlib
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, validator
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import contextlib

from shared.config import settings
from shared.text_processing.chunking import TokenChunker
from shared.text_processing.extraction import extract_text

from webui import database
from webui.auth import get_current_user
from webui.embedding_service import POPULAR_MODELS, embedding_service
from webui.utils.qdrant_manager import qdrant_manager

logger = logging.getLogger(__name__)

# Constants
JOBS_DIR = str(settings.jobs_dir)
OUTPUT_DIR = str(settings.output_dir)
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt", ".text", ".pptx", ".eml", ".md", ".html"]

# Create necessary directories
Path(JOBS_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# Request/Response models
class CreateJobRequest(BaseModel):
    name: str
    description: str = ""
    directory_path: str
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    chunk_size: int = 600
    chunk_overlap: int = 200
    batch_size: int = 96
    vector_dim: int | None = None
    quantization: str = "float32"
    instruction: str | None = None
    job_id: str | None = None  # Allow pre-generated job_id for WebSocket connection

    @validator("chunk_size")
    def validate_chunk_size(cls, v: int) -> int:  # noqa: N805
        if v <= 0:
            raise ValueError("chunk_size must be positive")
        if v < 100:
            raise ValueError("chunk_size must be at least 100 tokens")
        if v > 50000:
            raise ValueError("chunk_size must not exceed 50000 tokens")
        return v

    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v: int, values: dict[str, Any]) -> int:  # noqa: N805
        if v < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError(f'chunk_overlap ({v}) must be less than chunk_size ({values["chunk_size"]})')
        return v


class AddToCollectionRequest(BaseModel):
    collection_name: str
    directory_path: str
    description: str = ""
    job_id: str | None = None  # Allow pre-generated job_id for WebSocket connection


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


# WebSocket manager for real-time updates
class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str) -> None:
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def send_update(self, job_id: str, message: dict[str, Any]) -> None:
        if job_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)

            # Clean up disconnected clients
            for conn in disconnected:
                self.disconnect(conn, job_id)


manager = ConnectionManager()

# Global task tracking for cancellation support
active_job_tasks: dict[str, asyncio.Task] = {}

# Background task executor - increased workers for parallel processing
executor = ThreadPoolExecutor(max_workers=8)


def extract_text_thread_safe(filepath: str) -> str:
    """Thread-safe version of extract_text that uses the unified extraction"""
    return extract_text(filepath)


def extract_and_serialize_thread_safe(filepath: str) -> list[tuple[str, dict[str, Any]]]:
    """Thread-safe version of extract_and_serialize that preserves metadata"""
    from shared.text_processing.extraction import extract_and_serialize

    return extract_and_serialize(filepath)


# Import will be done inside functions to avoid circular import


async def update_metrics_continuously() -> None:
    """Background task to update resource metrics during job processing"""
    # Since webui/api/metrics.py already has a background thread updating metrics,
    # we only need to ensure metrics are being updated, not force updates
    while True:
        try:
            from shared.metrics.prometheus import metrics_collector

            # Only update if it's been more than 0.5 seconds since last update
            current_time = time.time()
            if current_time - metrics_collector.last_update > 0.5:
                metrics_collector.update_resource_metrics()
        except Exception as e:
            logger.warning(f"Error updating metrics in job: {e}")
        await asyncio.sleep(1)  # Check every 1 second


async def process_embedding_job(job_id: str) -> None:
    """Process an embedding job asynchronously"""
    metrics_task = None  # Initialize to avoid undefined reference

    # Import metrics if available
    try:
        from shared.metrics.prometheus import (
            embedding_batch_duration,
            metrics_collector,
            record_chunks_created,
            record_embeddings_generated,
            record_file_failed,
            record_file_processed,
        )

        metrics_tracking = True
        # Start background metrics updater
        metrics_task = asyncio.create_task(update_metrics_continuously())
    except ImportError:
        metrics_tracking = False
        logger.warning("Metrics tracking not available for embedding job")

    try:
        # Update job status and set start time
        database.update_job(job_id, {"status": "processing", "start_time": datetime.now(UTC).isoformat()})

        # Get job details
        job = database.get_job(job_id)
        if not job:
            raise Exception(f"Job {job_id} not found")

        # Determine collection name based on mode
        if job.get("mode") == "append" and job.get("parent_job_id"):
            # For append mode, use the parent job's collection
            collection_name = f"job_{job['parent_job_id']}"
        else:
            # For create mode, use this job's ID
            collection_name = f"job_{job_id}"

        # Get pending files
        files = database.get_job_files(job_id, status="pending")

        # Send initial update with total files
        await manager.send_update(job_id, {"type": "job_started", "total_files": len(files)})

        # Get Qdrant client with retry logic
        try:
            qdrant = qdrant_manager.get_client()
        except Exception as e:
            error_msg = f"Failed to connect to Qdrant after retries: {e}"
            logger.error(error_msg)
            database.update_job(job_id, {"status": "failed", "error": error_msg})
            await manager.send_update(job_id, {"type": "error", "message": error_msg})
            return

        for file_idx, file_row in enumerate(files):
            try:
                # For append mode, check if file already exists in collection
                if job.get("mode") == "append" and file_row.get("content_hash"):
                    # Check if this content hash already exists in the collection
                    existing_hashes = database.get_duplicate_files_in_collection(
                        job["name"], [file_row["content_hash"]]
                    )
                    if file_row["content_hash"] in existing_hashes:
                        logger.info(
                            f"Skipping duplicate file: {file_row['path']} (content_hash: {file_row['content_hash']})"
                        )
                        # Mark as completed since it already exists
                        database.update_file_status(
                            job_id, file_row["path"], "completed", chunks_created=0, vectors_created=0
                        )
                        # Update processed files count
                        current_job = database.get_job(job_id)
                        if current_job:
                            database.update_job(job_id, {"processed_files": current_job.get("processed_files", 0) + 1})
                        continue

                # Update current file
                database.update_job(job_id, {"current_file": file_row["path"]})

                # Send progress update
                await manager.send_update(
                    job_id,
                    {
                        "type": "file_processing",
                        "current_file": file_row["path"],
                        "processed_files": file_idx,
                        "total_files": len(files),
                        "status": "Processing",
                    },
                )

                # Yield control to event loop to keep UI responsive
                await asyncio.sleep(0)

                # Extract text and create chunks
                logger.info(f"Processing file: {file_row['path']}")

                # Add memory tracking
                import psutil

                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(f"Memory before extraction: {initial_memory:.2f} MB")

                logger.info(f"Starting text extraction for: {file_row['path']}")
                # Run text extraction in thread pool to avoid blocking
                try:
                    loop = asyncio.get_event_loop()
                    # Use asyncio timeout instead of signal-based timeout
                    file_path = file_row["path"]
                    text_blocks = await asyncio.wait_for(
                        loop.run_in_executor(executor, extract_and_serialize_thread_safe, file_path),
                        timeout=300,  # 5 minute timeout
                    )
                except TimeoutError:
                    raise TimeoutError(f"Text extraction timed out after 300 seconds for {file_row['path']}") from None

                memory_after_extract = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(
                    f"Memory after extraction: {memory_after_extract:.2f} MB (delta: {memory_after_extract - initial_memory:.2f} MB)"
                )
                logger.info(f"Extracted {len(text_blocks)} text blocks")

                doc_id = hashlib.md5(file_row["path"].encode()).hexdigest()[:16]

                logger.info(
                    f"Starting chunking for: {file_row['path']} with chunk_size={job['chunk_size']}, overlap={job['chunk_overlap']}"
                )

                # Create a chunker with job-specific settings
                chunker = TokenChunker(chunk_size=job["chunk_size"] or 600, chunk_overlap=job["chunk_overlap"] or 200)

                # Process each text block with its metadata
                all_chunks = []
                for text, metadata in text_blocks:
                    if not text.strip():
                        continue

                    # Run chunking in thread pool to avoid blocking
                    chunks = await loop.run_in_executor(
                        executor, chunker.chunk_text, text, doc_id, metadata  # Pass metadata to preserve page numbers
                    )
                    all_chunks.extend(chunks)

                memory_after_chunk = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(
                    f"Memory after chunking: {memory_after_chunk:.2f} MB (delta: {memory_after_chunk - memory_after_extract:.2f} MB)"
                )
                logger.info(f"Created {len(all_chunks)} chunks")

                # Record chunks created
                if metrics_tracking:
                    record_chunks_created(len(all_chunks))

                # Update chunks variable to use all_chunks
                chunks = all_chunks

                # Generate embeddings
                texts = [chunk["text"] for chunk in chunks]

                # Use unified embedding service - run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()

                # Time the embedding generation
                import time

                embed_start_time = time.time()

                embeddings_array = await loop.run_in_executor(
                    executor,
                    embedding_service.generate_embeddings,
                    texts,
                    job["model_name"],
                    job["quantization"] or "float32",
                    job["batch_size"],
                    False,  # show_progress
                    job["instruction"],
                )

                # Record embedding time
                if metrics_tracking:
                    embed_duration = time.time() - embed_start_time
                    logger.info(f"Embedding generation took {embed_duration:.3f} seconds for {len(texts)} texts")
                    embedding_batch_duration.observe(embed_duration)

                # Free texts list after embedding generation
                del texts

                if embeddings_array is None:
                    raise Exception("Failed to generate embeddings")

                # Record embeddings generated
                if metrics_tracking:
                    record_embeddings_generated(len(embeddings_array))

                embeddings = embeddings_array.tolist()

                # Free the numpy array after conversion
                del embeddings_array

                # Handle dimension override if specified
                target_dim = job["vector_dim"]
                if target_dim and len(embeddings) > 0:
                    model_dim = len(embeddings[0])
                    if target_dim != model_dim:
                        logger.info(f"Adjusting embeddings from {model_dim} to {target_dim} dimensions")
                        adjusted_embeddings = []
                        for emb in embeddings:
                            if target_dim < model_dim:
                                # Truncate
                                adjusted = emb[:target_dim]
                            else:
                                # Pad with zeros
                                adjusted = emb + [0.0] * (target_dim - model_dim)

                            # Renormalize
                            norm = sum(x**2 for x in adjusted) ** 0.5
                            if norm > 0:
                                adjusted = [x / norm for x in adjusted]
                            adjusted_embeddings.append(adjusted)
                        embeddings = adjusted_embeddings

                # Prepare points for Qdrant in batches to avoid memory spikes
                upload_batch_size = 100
                total_points = len(chunks)
                successfully_uploaded = 0

                for batch_start in range(0, total_points, upload_batch_size):
                    batch_end = min(batch_start + upload_batch_size, total_points)
                    points = []

                    for i in range(batch_start, batch_end):
                        chunk = chunks[i]
                        embedding = embeddings[i]
                        point = {
                            "id": str(uuid.uuid4()),
                            "vector": embedding,
                            "payload": {
                                "job_id": job_id,
                                "doc_id": doc_id,
                                "chunk_id": chunk["chunk_id"],
                                "path": file_row["path"],
                                "content": chunk["text"],  # Store full text for hybrid search
                                "metadata": chunk.get("metadata", {}),  # Store metadata including page_number
                            },
                        }
                        points.append(point)

                    # Upload batch to Qdrant
                    if points:
                        try:
                            point_structs = [
                                PointStruct(id=point["id"], vector=point["vector"], payload=point["payload"])
                                for point in points
                            ]
                            qdrant.upsert(collection_name=collection_name, points=point_structs, wait=True)
                            successfully_uploaded += len(point_structs)
                            logger.info(f"Successfully uploaded {len(point_structs)} points to Qdrant")
                        except Exception as e:
                            error_msg = f"Failed to upload vectors to Qdrant: {e}"
                            logger.error(error_msg)
                            database.update_file_status(job_id, file_row["path"], "failed", error=str(e))
                            raise  # Re-raise to be caught by outer exception handler

                    # Free the points batch
                    del points

                # Update database after all batches uploaded
                database.update_file_status(
                    job_id, file_row["path"], "completed", vectors_created=total_points, chunks_created=len(chunks)
                )
                # Get current job to update processed files count
                current_job = database.get_job(job_id)
                if current_job:
                    database.update_job(job_id, {"processed_files": current_job.get("processed_files", 0) + 1})

                # Record file processed
                if metrics_tracking:
                    record_file_processed("embedding")

                # Send file completed update
                await manager.send_update(
                    job_id,
                    {
                        "type": "file_completed",
                        "processed_files": current_job.get("processed_files", 0) + 1 if current_job else 1,
                        "total_files": len(files),
                    },
                )

                # Free chunks and embeddings after upload
                del chunks
                del embeddings

                # Force garbage collection after each file
                gc.collect()

                # Update resource metrics periodically (force update)
                if metrics_tracking:
                    # Force update by resetting the last update time
                    metrics_collector.last_update = 0
                    metrics_collector.update_resource_metrics()

                # Yield control to event loop after processing each file
                await asyncio.sleep(0)

            except Exception as e:
                logger.error(f"Failed to process file {file_row['path']}: {e}")
                # Record file failed
                if metrics_tracking:
                    record_file_failed("embedding", type(e).__name__)
                # File status already updated in the database.update_file_status call above

        # Check total vectors created from database
        total_vectors_created = database.get_job_total_vectors(job_id)

        if total_vectors_created == 0:
            # No vectors were created - this is a failure condition
            error_msg = "No vectors were created. All files either failed to process or contained no extractable text."
            logger.error(f"Job {job_id} failed: {error_msg}")
            database.update_job(job_id, {"status": "failed", "error": error_msg})
            await manager.send_update(job_id, {"type": "error", "message": error_msg})
            return

        # Verify collection has points before marking as completed
        try:
            # Use the correct collection name (considering append mode)
            collection_info = qdrant.get_collection(collection_name)

            if collection_info.points_count == 0:
                # This shouldn't happen if total_vectors_created > 0, but check anyway
                error_msg = f"Qdrant collection has 0 points but {total_vectors_created} vectors were expected"
                raise Exception(error_msg)

            # Allow up to 10% discrepancy between database count and Qdrant count
            # This accounts for potential race conditions, network issues during upload,
            # or partial batch failures that were retried. The exact count isn't critical
            # as long as most vectors were successfully uploaded.
            if collection_info.points_count is not None and collection_info.points_count < total_vectors_created * 0.9:
                logger.warning(
                    f"Vector count mismatch: {collection_info.points_count} in Qdrant vs {total_vectors_created} expected"
                )

            logger.info(f"Collection {collection_name} has {collection_info.points_count} points")
        except Exception as e:
            error_msg = f"Failed to verify Qdrant collection: {e}"
            logger.error(error_msg)
            database.update_job(job_id, {"status": "failed", "error": error_msg})
            await manager.send_update(job_id, {"type": "error", "message": error_msg})
            return

        # Mark job as completed only if vectors were successfully created
        database.update_job(job_id, {"status": "completed", "current_file": None})

        await manager.send_update(job_id, {"type": "job_completed", "message": "Job completed successfully"})

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        database.update_job(job_id, {"status": "failed", "error": str(e)})

        await manager.send_update(job_id, {"type": "error", "message": str(e)})

    finally:
        # Cancel metrics updater task if it exists
        if metrics_task:
            metrics_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await metrics_task


# API Routes
@router.get("/new-id")
async def get_new_job_id(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, str]:  # noqa: ARG001
    """Generate a new job ID for WebSocket connection"""
    return {"job_id": str(uuid.uuid4())}


@router.post("", response_model=JobStatus)
async def create_job(request: CreateJobRequest, current_user: dict[str, Any] = Depends(get_current_user)) -> JobStatus:
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
        database.create_job(job_data)

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
        database.add_files_to_job(job_id, file_records)

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
        database.update_job(job_id, {"vector_dim": vector_size})

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
            from .collection_metadata import store_collection_metadata

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
            database.delete_job(job_id)
            raise HTTPException(status_code=500, detail=f"Failed to create Qdrant collection: {str(e)}") from e

        # Start processing in background with cancellation support
        task = asyncio.create_task(process_embedding_job(job_id))
        active_job_tasks[job_id] = task

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
    request: AddToCollectionRequest, current_user: dict[str, Any] = Depends(get_current_user)
) -> JobStatus:
    """Add new documents to an existing collection"""
    # Accept job_id from request if provided, otherwise generate new one
    job_id = request.job_id if request.job_id else str(uuid.uuid4())

    try:
        # Import here to avoid circular import
        from .files import scan_directory_async

        # Get parent collection metadata
        collection_metadata = database.get_collection_metadata(request.collection_name)
        if not collection_metadata:
            raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found")

        # Scan directory for new files
        scan_result = await scan_directory_async(request.directory_path, recursive=True, scan_id=job_id)
        files = scan_result["files"]

        if not files:
            raise HTTPException(status_code=400, detail="No supported files found in directory")

        # Check for duplicates
        content_hashes = [f.content_hash for f in files if f.content_hash]
        existing_hashes = database.get_duplicate_files_in_collection(request.collection_name, content_hashes)

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
        database.create_job(job_data)

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
        database.add_files_to_job(job_id, file_records)

        # Update total files count
        database.update_job(job_id, {"total_files": len(new_files)})

        # Start processing in background with cancellation support
        task = asyncio.create_task(process_embedding_job(job_id))
        active_job_tasks[job_id] = task

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
async def list_jobs(current_user: dict[str, Any] = Depends(get_current_user)) -> list[JobStatus]:
    """List all jobs for the current user"""
    jobs = database.list_jobs(user_id=current_user["id"])

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
    collection_name: str, current_user: dict[str, Any] = Depends(get_current_user)  # noqa: ARG001
) -> dict[str, Any]:
    """Get metadata for a specific collection"""
    metadata = database.get_collection_metadata(collection_name)
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
) -> dict[str, list[str]]:
    """Check which content hashes already exist in a collection"""
    existing_hashes = database.get_duplicate_files_in_collection(collection_name, content_hashes)
    return {"existing_hashes": list(existing_hashes)}


@router.get("/collections-status")
async def check_collections_status(
    current_user: dict[str, Any] = Depends(get_current_user)  # noqa: ARG001
) -> dict[str, dict[str, Any]]:
    """Check which job collections exist in Qdrant"""
    try:
        qdrant = qdrant_manager.get_client()

        # Get all collections
        collections = qdrant.get_collections().collections
        collection_names = {c.name for c in collections}

        # Get all jobs
        jobs = database.list_jobs()

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
    job_id: str, current_user: dict[str, Any] = Depends(get_current_user)  # noqa: ARG001
) -> dict[str, str]:
    """Cancel a running job"""
    # Check current job status
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] not in ["created", "scanning", "processing"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job in status: {job['status']}")

    # Update job status to cancelled
    database.update_job(job_id, {"status": "cancelled"})

    # Cancel the task if it's running
    if job_id in active_job_tasks:
        active_job_tasks[job_id].cancel()

    return {"message": "Job cancellation requested"}


@router.delete("/{job_id}")
async def delete_job(
    job_id: str, current_user: dict[str, Any] = Depends(get_current_user)  # noqa: ARG001
) -> dict[str, str]:
    """Delete a job and its associated collection"""
    # Check if job exists
    job = database.get_job(job_id)
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
    database.delete_job(job_id)

    return {"message": "Job deleted successfully"}


@router.get("/{job_id}", response_model=JobStatus)
async def get_job(job_id: str, current_user: dict[str, Any] = Depends(get_current_user)) -> JobStatus:  # noqa: ARG001
    """Get job details"""
    job = database.get_job(job_id)

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
    """WebSocket for real-time job updates"""
    await manager.connect(websocket, job_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
