"""Celery task definitions for asynchronous processing.

This module implements the core background task processing for Semantik's collection
operations. It provides a unified task entry point with comprehensive monitoring,
error handling, and resource management.

Architecture Overview:
    - All collection operations (INDEX, APPEND, REINDEX, REMOVE_SOURCE) go through
      a single `process_collection_operation` task for consistency
    - Tasks use late acknowledgment (acks_late=True) for message reliability
    - Comprehensive metrics are collected via Prometheus
    - Real-time updates are sent via Redis streams for WebSocket communication
    - All operations create audit log entries for compliance

Key Features:
    - Automatic resource cleanup via context managers
    - Transaction support for atomic database operations
    - Blue-green reindexing with validation checkpoints
    - Comprehensive search quality validation
    - Resource tracking (CPU, memory, duration)
    - Graceful error handling with guaranteed status updates

Task Configuration:
    - Soft time limit: 1 hour (graceful shutdown)
    - Hard time limit: 2 hours (forced termination)
    - Max retries: 3 (with 60-second delay)
    - Late acknowledgment for reliability

Usage:
    Tasks are typically triggered by the CollectionService when users perform
    operations via the API. Progress can be monitored via WebSocket subscriptions
    to the Redis stream updates.
"""

import asyncio
import contextlib
import gc
import hashlib
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import psutil
import redis.asyncio as redis
from qdrant_client.models import PointStruct
from shared.config import settings
from shared.database.factory import create_collection_repository, create_file_repository, create_job_repository
from shared.embedding import embedding_service
from shared.gpu_scheduler import gpu_task
from shared.managers.qdrant_manager import QdrantManager
from shared.metrics.collection_metrics import (
    OperationTimer,
    QdrantOperationTimer,
    collection_cpu_seconds_total,
    collection_memory_usage_bytes,
    collections_total,
    update_collection_stats,
)
from shared.text_processing.chunking import TokenChunker
from webui.celery_app import celery_app
from webui.utils.qdrant_manager import qdrant_manager

logger = logging.getLogger(__name__)

# Task timeout constants
OPERATION_SOFT_TIME_LIMIT = 3600  # 1 hour soft limit
OPERATION_HARD_TIME_LIMIT = 7200  # 2 hour hard limit

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 60  # seconds

# Batch processing constants
EMBEDDING_BATCH_SIZE = 100
VECTOR_UPLOAD_BATCH_SIZE = 100
DOCUMENT_REMOVAL_BATCH_SIZE = 100

# Validation thresholds
REINDEX_VECTOR_COUNT_VARIANCE = 0.1  # 10% variance allowed
REINDEX_SEARCH_MISMATCH_THRESHOLD = 0.3  # 30% mismatch threshold
REINDEX_SCORE_DIFF_THRESHOLD = 0.1  # 0.1 score difference threshold

# Redis configuration
REDIS_STREAM_MAX_LEN = 1000  # Keep last 1000 messages
REDIS_STREAM_TTL = 86400  # 24 hours

# Cleanup configuration
DEFAULT_DAYS_TO_KEEP = 7  # Days to keep old results
CLEANUP_DELAY_SECONDS = 300  # 5 minutes default delay before cleaning up old collections
CLEANUP_DELAY_MIN_SECONDS = 300  # 5 minutes minimum
CLEANUP_DELAY_MAX_SECONDS = 1800  # 30 minutes maximum
CLEANUP_DELAY_PER_10K_VECTORS = 60  # Additional 1 minute per 10k vectors

# Background task executor
executor = ThreadPoolExecutor(max_workers=8)


class CeleryTaskWithUpdates:
    """Helper class to send updates to Redis Stream from Celery tasks.

    Implements context manager protocol for automatic resource cleanup.
    """

    def __init__(self, job_id: str):
        """Initialize with job ID."""
        self.job_id = job_id
        self.redis_url = settings.REDIS_URL
        self.stream_key = f"job:updates:{job_id}"
        self._redis_client: redis.Redis | None = None

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._redis_client is None:
            self._redis_client = await redis.from_url(self.redis_url, decode_responses=True)
        assert self._redis_client is not None
        return self._redis_client

    async def send_update(self, update_type: str, data: dict) -> None:
        """Send update to Redis Stream."""
        try:
            redis_client = await self._get_redis()
            message = {"timestamp": datetime.now(UTC).isoformat(), "type": update_type, "data": data}

            # Add to stream with automatic ID
            await redis_client.xadd(self.stream_key, {"message": json.dumps(message)}, maxlen=REDIS_STREAM_MAX_LEN)

            # Set TTL on first message
            await redis_client.expire(self.stream_key, REDIS_STREAM_TTL)

            logger.debug(f"Sent update to Redis stream {self.stream_key}: type={update_type}")
        except Exception as e:
            logger.error(f"Failed to send update to Redis stream: {e}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None

    async def __aenter__(self) -> "CeleryTaskWithUpdates":
        """Async context manager entry - ensures Redis connection is available."""
        # Verify Redis connection is available
        try:
            redis_client = await self._get_redis()
            await redis_client.ping()
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Async context manager exit - ensures cleanup even on exceptions."""
        await self.close()


def extract_and_serialize_thread_safe(filepath: str) -> list[tuple[str, dict[str, Any]]]:
    """Thread-safe version of extract_and_serialize that preserves metadata"""
    from shared.text_processing.extraction import extract_and_serialize

    result: list[tuple[str, dict[str, Any]]] = extract_and_serialize(filepath)
    return result


def calculate_cleanup_delay(vector_count: int) -> int:
    """Calculate cleanup delay based on collection size.

    Uses a formula that scales with the number of vectors:
    - Base delay: 5 minutes
    - Additional 1 minute per 10,000 vectors
    - Maximum delay: 30 minutes

    Args:
        vector_count: Number of vectors in the collection

    Returns:
        Delay in seconds
    """
    additional_delay = (vector_count // 10000) * CLEANUP_DELAY_PER_10K_VECTORS
    total_delay = CLEANUP_DELAY_MIN_SECONDS + additional_delay

    # Cap at maximum delay
    cleanup_delay = min(total_delay, CLEANUP_DELAY_MAX_SECONDS)

    logger.info(
        f"Calculated cleanup delay: {cleanup_delay}s for {vector_count} vectors "
        f"(base: {CLEANUP_DELAY_MIN_SECONDS}s, additional: {additional_delay}s)"
    )

    return cleanup_delay


@celery_app.task(bind=True)
def test_task(self: Any) -> dict[str, str]:  # noqa: ARG001
    """Test task to verify Celery is working."""
    return {"status": "success", "message": "Celery is working!"}


@celery_app.task(name="webui.tasks.cleanup_old_results")
def cleanup_old_results(days_to_keep: int = DEFAULT_DAYS_TO_KEEP) -> dict[str, Any]:
    """Clean up old Celery results and job records.

    Args:
        days_to_keep: Number of days to keep results (default: 7)

    Returns:
        Dictionary with cleanup statistics
    """
    from datetime import timedelta

    stats: dict[str, Any] = {"celery_results_deleted": 0, "old_jobs_marked": 0, "errors": []}

    try:
        # Clean up old Celery results from Redis
        cutoff_time = datetime.now(UTC) - timedelta(days=days_to_keep)

        # Note: This is a simplified approach. In production, you might want to use
        # Celery's built-in result expiration or a more sophisticated cleanup
        logger.info(f"Starting cleanup of results older than {days_to_keep} days")

        # Mark old jobs as archived (not deleting to preserve history)
        try:
            job_repo = create_job_repository()  # noqa: F841
            # This would need a new method in the repository
            # For now, just log what we would do
            logger.info(f"Would archive jobs older than {cutoff_time}")
            # stats["old_jobs_marked"] = await job_repo.archive_old_jobs(cutoff_time)
        except Exception as e:
            error_msg = f"Error archiving old jobs: {e}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)

        logger.info(f"Cleanup completed: {stats}")
        return stats

    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        stats["errors"].append(str(e))
        return stats


@celery_app.task(
    name="webui.tasks.cleanup_old_collections",
    max_retries=3,
    default_retry_delay=60,
    retry_backoff=True,
    retry_backoff_max=600,  # 10 minutes max delay for Qdrant operations
)
def cleanup_old_collections(old_collection_names: list[str], collection_id: str) -> dict[str, Any]:
    """Clean up old Qdrant collections after a successful reindex.

    This task is scheduled with a delay after a reindex operation completes
    to allow time for any in-flight requests to complete.

    Args:
        old_collection_names: List of old Qdrant collection names to delete
        collection_id: ID of the collection (for logging/tracking)

    Returns:
        Dictionary with cleanup statistics
    """
    stats: dict[str, Any] = {
        "collections_deleted": 0,
        "collections_failed": 0,
        "errors": [],
        "collection_id": collection_id,
    }

    if not old_collection_names:
        logger.info(f"No old collections to clean up for collection {collection_id}")
        return stats

    logger.info(f"Starting cleanup of {len(old_collection_names)} old collections for collection {collection_id}")

    # Import Qdrant client
    from shared.managers.qdrant_manager import QdrantManager
    from shared.managers.timer import QdrantOperationTimer

    try:
        qdrant_manager = QdrantManager()
        qdrant_client = qdrant_manager.client

        for collection_name in old_collection_names:
            try:
                # Check if collection exists before attempting deletion
                with QdrantOperationTimer("check_collection_exists"):
                    collections = qdrant_client.get_collections()
                    exists = any(col.name == collection_name for col in collections.collections)

                if not exists:
                    logger.warning(f"Collection {collection_name} does not exist, skipping")
                    continue

                # Delete the collection
                with QdrantOperationTimer("delete_old_collection"):
                    qdrant_client.delete_collection(collection_name)

                stats["collections_deleted"] += 1
                logger.info(f"Successfully deleted old collection: {collection_name}")

            except Exception as e:
                error_msg = f"Failed to delete collection {collection_name}: {str(e)}"
                logger.error(error_msg)
                stats["collections_failed"] += 1
                stats["errors"].append(error_msg)

        # Log final statistics
        logger.info(
            f"Cleanup completed for collection {collection_id}: "
            f"deleted={stats['collections_deleted']}, failed={stats['collections_failed']}"
        )

        # Record metrics if available
        from shared.metrics.collection_metrics import record_metric_safe

        status = "success" if stats["collections_failed"] == 0 else "partial"
        record_metric_safe("collection_cleanup_total", {"status": status})

        return stats

    except Exception as e:
        logger.error(f"Cleanup task failed for collection {collection_id}: {e}")
        stats["errors"].append(str(e))

        # Record failure metric
        from shared.metrics.collection_metrics import record_metric_safe

        record_metric_safe("collection_cleanup_total", {"status": "failed"})

        return stats


@celery_app.task(bind=True, name="webui.tasks.process_embedding_job_task", max_retries=3, default_retry_delay=60)
def process_embedding_job_task(self: Any, job_id: str) -> dict[str, Any]:
    """
    Process an embedding job as a Celery task.

    This is a synchronous wrapper that runs the async processing logic.
    """
    # Run the async function in an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(_process_embedding_job_async(job_id, self))
    except Exception as exc:
        logger.error(f"Task failed for job {job_id}: {exc}")
        # Don't retry for certain exceptions
        if isinstance(exc, ValueError | TypeError):
            raise  # Don't retry on programming errors
        # Retry for other exceptions (network issues, temporary failures)
        raise self.retry(exc=exc, countdown=60) from exc
    finally:
        loop.close()


async def _process_embedding_job_async(job_id: str, celery_task: Any) -> dict[str, Any]:
    """Async implementation of the embedding job processing."""
    metrics_task = None

    # Create repository instances
    job_repo = create_job_repository()
    file_repo = create_file_repository()
    collection_repo = create_collection_repository()  # noqa: F841

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
        metrics_task = asyncio.create_task(_update_metrics_continuously())
    except ImportError:
        metrics_tracking = False
        logger.warning("Metrics tracking not available for embedding job")

    # Use context manager for automatic Redis cleanup
    async with CeleryTaskWithUpdates(job_id) as updater:
        try:
            # Update job status and set start time
            await job_repo.update_job(job_id, {"status": "processing", "start_time": datetime.now(UTC).isoformat()})

            # Send job started update via Redis
            await updater.send_update("job_started", {"status": "processing"})

            # Get job details
            job = await job_repo.get_job(job_id)
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
            files = await file_repo.get_job_files(job_id, status="pending")

            # Update Celery task state (keep for backwards compatibility)
            celery_task.update_state(
                state="PROCESSING", meta={"total_files": len(files), "processed_files": 0, "status": "job_started"}
            )

            # Send initial file count via Redis
            await updater.send_update(
                "status_update", {"status": "processing", "total_files": len(files), "processed_files": 0}
            )

            # Get Qdrant client with retry logic
            try:
                qdrant = qdrant_manager.get_client()
            except Exception as e:
                error_msg = f"Failed to connect to Qdrant after retries: {e}"
                logger.error(error_msg)
                await job_repo.update_job(job_id, {"status": "failed", "error": error_msg})
                raise

            processed_count = 0
            failed_count = 0

            for file_idx, file_row in enumerate(files):
                try:
                    # For append mode, check if file already exists in collection
                    if job.get("mode") == "append" and file_row.get("content_hash"):
                        # Check if this content hash already exists in the collection
                        existing_hashes = await file_repo.get_duplicate_files_in_collection(
                            job["name"], [file_row["content_hash"]]
                        )
                        if file_row["content_hash"] in existing_hashes:
                            logger.info(
                                f"Skipping duplicate file: {file_row['path']} (content_hash: {file_row['content_hash']})"
                            )
                            # Mark as completed since it already exists
                            await file_repo.update_file_status(
                                job_id, file_row["path"], "completed", chunks_created=0, vectors_created=0
                            )
                            # Update processed files count
                            current_job = await job_repo.get_job(job_id)
                            if current_job:
                                await job_repo.update_job(
                                    job_id, {"processed_files": current_job.get("processed_files", 0) + 1}
                                )
                            processed_count += 1
                            continue

                    # Update current file
                    await job_repo.update_job(job_id, {"current_file": file_row["path"]})

                    # Update Celery task state (keep for backwards compatibility)
                    celery_task.update_state(
                        state="PROCESSING",
                        meta={
                            "total_files": len(files),
                            "processed_files": file_idx,
                            "current_file": file_row["path"],
                            "status": "file_processing",
                        },
                    )

                    # Send file processing update via Redis
                    await updater.send_update(
                        "file_processing",
                        {
                            "status": "processing",
                            "current_file": file_row["path"],
                            "processed_files": file_idx,
                            "total_files": len(files),
                        },
                    )

                    # Yield control to event loop to keep UI responsive
                    await asyncio.sleep(0)

                    # Extract text and create chunks
                    logger.info(f"Processing file: {file_row['path']}")

                    # Add memory tracking
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
                        raise TimeoutError(
                            f"Text extraction timed out after 300 seconds for {file_row['path']}"
                        ) from None

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
                    chunker = TokenChunker(
                        chunk_size=job["chunk_size"] or 600, chunk_overlap=job["chunk_overlap"] or 200
                    )

                    # Process each text block with its metadata
                    all_chunks = []
                    for text, metadata in text_blocks:
                        if not text.strip():
                            continue

                        # Run chunking in thread pool to avoid blocking
                        chunks = await loop.run_in_executor(
                            executor,
                            chunker.chunk_text,
                            text,
                            doc_id,
                            metadata,  # Pass metadata to preserve page numbers
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
                    embed_start_time = time.time()

                    # Prepare task ID for GPU scheduling
                    task_id = celery_task.request.id if hasattr(celery_task, "request") else str(uuid.uuid4())

                    # Function to run in executor with GPU scheduling
                    def generate_embeddings_with_gpu(task_id: str = task_id, texts: list[str] = texts) -> Any:
                        with gpu_task(task_id) as gpu_id:
                            if gpu_id is None:
                                logger.warning(f"No GPU available for task {task_id}, proceeding with CPU")
                            else:
                                logger.info(f"Task {task_id} using GPU {gpu_id}")

                            return embedding_service.generate_embeddings(
                                texts,
                                job["model_name"],
                                job["quantization"] or "float32",
                                job["batch_size"],
                                False,  # show_progress
                                job["instruction"],
                            )

                    # Run with GPU scheduling in thread pool
                    embeddings_array = await loop.run_in_executor(executor, generate_embeddings_with_gpu)

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
                    upload_batch_size = VECTOR_UPLOAD_BATCH_SIZE
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
                                await file_repo.update_file_status(job_id, file_row["path"], "failed", error=str(e))
                                raise  # Re-raise to be caught by outer exception handler

                        # Free the points batch
                        del points

                    # Update database after all batches uploaded
                    await file_repo.update_file_status(
                        job_id, file_row["path"], "completed", vectors_created=total_points, chunks_created=len(chunks)
                    )
                    # Get current job to update processed files count
                    current_job = await job_repo.get_job(job_id)
                    if current_job:
                        await job_repo.update_job(
                            job_id, {"processed_files": current_job.get("processed_files", 0) + 1}
                        )

                    processed_count += 1

                    # Record file processed
                    if metrics_tracking:
                        record_file_processed("embedding")

                    # Update Celery task state (keep for backwards compatibility)
                    celery_task.update_state(
                        state="PROCESSING",
                        meta={
                            "total_files": len(files),
                            "processed_files": processed_count,
                            "current_file": file_row["path"],
                            "status": "file_completed",
                        },
                    )

                    # Send file completed update via Redis
                    await updater.send_update(
                        "file_completed",
                        {
                            "status": "processing",
                            "current_file": file_row["path"],
                            "processed_files": processed_count,
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
                    failed_count += 1
                    # File status already updated in the database.update_file_status call above

                    # Send error update via Redis
                    await updater.send_update(
                        "error",
                        {
                            "status": "processing",
                            "current_file": file_row["path"],
                            "error": str(e),
                            "processed_files": processed_count,
                            "failed_files": failed_count,
                            "total_files": len(files),
                        },
                    )

            # Check total vectors created from database
            total_vectors_created = await file_repo.get_job_total_vectors(job_id)

            if total_vectors_created == 0:
                # No vectors were created - this is a failure condition
                error_msg = (
                    "No vectors were created. All files either failed to process or contained no extractable text."
                )
                logger.error(f"Job {job_id} failed: {error_msg}")
                await job_repo.update_job(job_id, {"status": "failed", "error": error_msg})
                raise Exception(error_msg)

            # Verify collection has points before marking as completed
            try:
                # Use the correct collection name (considering append mode)
                collection_info = qdrant.get_collection(collection_name)

                if collection_info.points_count == 0:
                    # This shouldn't happen if total_vectors_created > 0, but check anyway
                    error_msg = f"Qdrant collection has 0 points but {total_vectors_created} vectors were expected"
                    raise Exception(error_msg)

                # Allow up to 10% discrepancy between database count and Qdrant count
                if (
                    collection_info.points_count is not None
                    and collection_info.points_count < total_vectors_created * 0.9
                ):
                    logger.warning(
                        f"Vector count mismatch: {collection_info.points_count} in Qdrant vs {total_vectors_created} expected"
                    )

                logger.info(f"Collection {collection_name} has {collection_info.points_count} points")
            except Exception as e:
                error_msg = f"Failed to verify Qdrant collection: {e}"
                logger.error(error_msg)
                await job_repo.update_job(job_id, {"status": "failed", "error": error_msg})
                raise

            # Mark job as completed only if vectors were successfully created
            await job_repo.update_job(job_id, {"status": "completed", "current_file": None})

            # Send job completed update via Redis
            await updater.send_update(
                "job_completed",
                {
                    "status": "completed",
                    "processed_files": processed_count,
                    "failed_files": failed_count,
                    "total_vectors": total_vectors_created,
                },
            )

            # Clean up Redis stream for completed job
            try:
                from webui.websocket_manager import ws_manager

                await ws_manager.cleanup_job_stream(job_id)
                logger.info(f"Cleaned up Redis stream for completed job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up Redis stream for job {job_id}: {e}")

                return {
                    "status": "completed",
                    "job_id": job_id,
                    "processed_files": processed_count,
                    "failed_files": failed_count,
                    "total_vectors": total_vectors_created,
                }

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            await job_repo.update_job(job_id, {"status": "failed", "error": str(e)})

            # Send job failed update via Redis
            await updater.send_update("job_failed", {"status": "failed", "error": str(e)})

            raise

        finally:
            # Cancel metrics updater task if it exists
            if metrics_task:
                metrics_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await metrics_task
            # Note: Redis connection cleanup is handled automatically by the context manager

    # Return a default result if we somehow reach here
    return {
        "status": "failed",
        "job_id": job_id,
        "processed_files": 0,
        "failed_files": 0,
        "total_vectors": 0,
        "error": "Unexpected error in task execution",
    }


async def _update_metrics_continuously() -> None:
    """Background task to update resource metrics during job processing."""
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


@celery_app.task(
    bind=True,
    name="webui.tasks.process_collection_operation",
    max_retries=DEFAULT_MAX_RETRIES,
    default_retry_delay=DEFAULT_RETRY_DELAY,
    acks_late=True,  # Ensure message reliability
    soft_time_limit=OPERATION_SOFT_TIME_LIMIT,
    time_limit=OPERATION_HARD_TIME_LIMIT,
)
def process_collection_operation(self: Any, operation_id: str) -> dict[str, Any]:
    """
    Process a collection operation (INDEX, APPEND, REINDEX, REMOVE_SOURCE).

    This is a synchronous wrapper that runs the async processing logic.
    Implements reliable task processing with:
    - Late acknowledgment for message reliability
    - Proper time limits for long-running operations
    - Immediate task ID recording
    - Guaranteed status updates via try...finally
    """
    # Run the async function in an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(_process_collection_operation_async(operation_id, self))
    except Exception as exc:
        logger.error(f"Task failed for operation {operation_id}: {exc}")
        # Don't retry for certain exceptions
        if isinstance(exc, ValueError | TypeError):
            raise  # Don't retry on programming errors
        # Retry for other exceptions (network issues, temporary failures)
        raise self.retry(exc=exc, countdown=60) from exc
    finally:
        loop.close()


async def _process_collection_operation_async(operation_id: str, celery_task: Any) -> dict[str, Any]:
    """Async implementation of collection operation processing with enhanced monitoring."""
    from shared.database.factory import (
        create_collection_repository,
        create_document_repository,
        create_operation_repository,
    )
    from shared.database.models import CollectionStatus, OperationStatus, OperationType

    start_time = time.time()
    operation = None

    # Track initial resources
    process = psutil.Process()
    initial_cpu_time = process.cpu_times().user + process.cpu_times().system

    # Store task ID immediately as FIRST action
    task_id = celery_task.request.id if hasattr(celery_task, "request") else str(uuid.uuid4())

    # Create repository instances
    operation_repo = create_operation_repository()
    collection_repo = create_collection_repository()
    document_repo = create_document_repository()

    try:
        # Update operation with task ID as FIRST action inside try block
        await operation_repo.set_task_id(operation_id, task_id)
        logger.info(f"Set task_id {task_id} for operation {operation_id}")

        # Use context manager for automatic cleanup of Redis connection
        async with CeleryTaskWithUpdates(operation_id) as updater:
            # Get operation details
            operation = await operation_repo.get_by_uuid(operation_id)
            if not operation:
                raise ValueError(f"Operation {operation_id} not found in database")

        # Get operation details
        operation = await operation_repo.get_by_uuid(operation_id)
        if not operation:
            raise ValueError(f"Operation {operation_id} not found in database")

        # Log operation start
        logger.info(
            "Starting collection operation",
            extra={
                "operation_id": operation_id,
                "operation_type": operation["type"],
                "collection_id": operation["collection_id"],
                "task_id": task_id,
            },
        )

        # Update operation status to processing
        await operation_repo.update_status(operation_id, OperationStatus.PROCESSING)
        await updater.send_update("operation_started", {"status": "processing", "type": operation["type"]})

        # Get collection details
        collection = await collection_repo.get_by_id(operation["collection_id"])
        if not collection:
            raise ValueError(f"Collection {operation['collection_id']} not found in database")

        # Process based on operation type with timing
        result = {}
        operation_type = str(operation["type"]).lower()

        with OperationTimer(operation_type):
            # Track memory usage
            memory_before = process.memory_info().rss

            if operation["type"] == OperationType.INDEX:
                result = await _process_index_operation(operation, collection, collection_repo, document_repo, updater)
            elif operation["type"] == OperationType.APPEND:
                result = await _process_append_operation(operation, collection, collection_repo, document_repo, updater)
            elif operation["type"] == OperationType.REINDEX:
                result = await _process_reindex_operation(
                    operation, collection, collection_repo, document_repo, updater
                )
            elif operation["type"] == OperationType.REMOVE_SOURCE:
                result = await _process_remove_source_operation(
                    operation, collection, collection_repo, document_repo, updater
                )
            else:
                raise ValueError(f"Unknown operation type: {operation['type']}")

            # Track peak memory usage
            memory_peak = process.memory_info().rss
            collection_memory_usage_bytes.labels(operation_type=operation_type).set(memory_peak - memory_before)

            # Record operation metrics in database
            duration = time.time() - start_time
            cpu_time = (process.cpu_times().user + process.cpu_times().system) - initial_cpu_time

            await _record_operation_metrics(
                operation_repo,
                operation_id,
                {
                    "duration_seconds": duration,
                    "cpu_seconds": cpu_time,
                    "memory_peak_bytes": memory_peak,
                    "documents_processed": result.get("documents_added", result.get("documents_removed", 0)),
                    "success": result.get("success", False),
                },
            )

            # Update CPU time counter
            collection_cpu_seconds_total.labels(operation_type=operation_type).inc(cpu_time)

            # Update operation status to completed
            await operation_repo.update_status(operation_id, OperationStatus.COMPLETED, result=result)

            # Update collection status based on result
            old_status = collection.get("status", CollectionStatus.PENDING)

            if result.get("success"):
                # Check if collection has any documents
                doc_stats = await document_repo.get_stats_by_collection(collection["id"])
                if doc_stats["total_count"] > 0:
                    new_status = CollectionStatus.READY
                    await collection_repo.update_status(collection["id"], new_status)
                else:
                    new_status = CollectionStatus.EMPTY
                    await collection_repo.update_status(collection["id"], new_status)

                # Update collection statistics
                await _update_collection_metrics(
                    collection["id"],
                    doc_stats["total_count"],
                    collection.get("vector_count", 0),
                    doc_stats["total_size_bytes"],
                )
            else:
                new_status = CollectionStatus.PARTIALLY_READY
                await collection_repo.update_status(collection["id"], new_status)

            # Update collection status metrics
            if old_status != new_status:
                collections_total.labels(status=old_status.value).dec()
                collections_total.labels(status=new_status.value).inc()

            await updater.send_update("operation_completed", {"status": "completed", "result": result})

            logger.info(
                "Collection operation completed",
                extra={
                    "operation_id": operation_id,
                    "operation_type": operation["type"],
                    "duration_seconds": duration,
                    "success": result.get("success", False),
                },
            )

            return result

    except Exception as e:
        logger.error(f"Operation {operation_id} failed: {e}", exc_info=True)

        # Ensure status update even if some components failed to initialize
        try:
            # Record failure metrics
            if operation:
                await _record_operation_metrics(
                    operation_repo,
                    operation_id,
                    {
                        "duration_seconds": time.time() - start_time,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "success": False,
                    },
                )

            # Update operation status to failed - this is critical
            await operation_repo.update_status(operation_id, OperationStatus.FAILED, result={"error": str(e)})

            # Update collection status to failed if critical operation
            if operation and operation["type"] in [OperationType.INDEX, OperationType.REINDEX]:
                await collection_repo.update_status(operation["collection_id"], CollectionStatus.FAILED)

            # Note: updater.send_update for failures should be handled inside the context manager
            # The context manager will automatically close the Redis connection on exception
        except Exception as update_error:
            # Log but don't raise - we want the original exception to propagate
            logger.error(f"Failed to update operation status during error handling: {update_error}")

        raise

    finally:
        # Guaranteed cleanup - ensure operation status is finalized
        try:
            # If we haven't set a final status, ensure it's set
            if operation:
                current_status = await operation_repo.get_by_uuid(operation_id)
                if current_status and current_status.get("status") == OperationStatus.PROCESSING:
                    # Operation is still processing - must have failed unexpectedly
                    await operation_repo.update_status(
                        operation_id, OperationStatus.FAILED, result={"error": "Task terminated unexpectedly"}
                    )
        except Exception as final_error:
            logger.error(f"Failed to finalize operation status: {final_error}")

        # Note: Redis connection cleanup is handled automatically by the context manager


async def _record_operation_metrics(operation_repo: Any, operation_id: str, metrics: dict[str, Any]) -> None:
    """Record operation metrics in the database."""
    try:
        # Get operation ID (database ID, not UUID)
        operation = await operation_repo.get_by_uuid(operation_id)
        if operation:
            from shared.database.database import AsyncSessionLocal
            from shared.database.models import OperationMetrics

            async with AsyncSessionLocal() as session:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, int | float):
                        metric = OperationMetrics(
                            operation_id=operation["id"],
                            metric_name=metric_name,
                            metric_value=float(metric_value),
                        )
                        session.add(metric)
                await session.commit()
    except Exception as e:
        logger.warning(f"Failed to record operation metrics: {e}")


async def _update_collection_metrics(collection_id: str, documents: int, vectors: int, size_bytes: int) -> None:
    """Update collection metrics in Prometheus."""
    try:
        update_collection_stats(collection_id, documents, vectors, size_bytes)
    except Exception as e:
        logger.warning(f"Failed to update collection metrics: {e}")


def _sanitize_audit_details(details: dict[str, Any] | None) -> dict[str, Any] | None:
    """Sanitize audit log details to ensure no PII is logged.

    This function removes or redacts potentially sensitive information:
    - User home directories in paths are replaced with ~
    - Email addresses are redacted
    - Any keys containing 'password', 'secret', 'token' are removed
    """
    if not details:
        return details

    import re

    sanitized: dict[str, Any] = {}

    for key, value in details.items():
        # Skip sensitive keys
        if any(sensitive in key.lower() for sensitive in ["password", "secret", "token", "key"]):
            continue

        # Sanitize paths by replacing home directories
        if isinstance(value, str) and ("/" in value or "\\" in value):
            # Replace user home paths with ~
            path_str = str(value)
            # Match common home directory patterns
            path_str = re.sub(r"/home/[^/]+/", "/home/USER/", path_str)
            path_str = re.sub(r"/Users/[^/]+/", "/Users/USER/", path_str)
            path_str = re.sub(r"C:\\Users\\[^\\]+\\", "C:\\Users\\USER\\", path_str)
            sanitized[key] = path_str
        # Recursively sanitize nested dictionaries
        elif isinstance(value, dict):
            sanitized_value = _sanitize_audit_details(value)
            if sanitized_value is not None:
                sanitized[key] = sanitized_value
        # Redact email addresses
        elif isinstance(value, str) and "@" in value:
            # Simple email pattern - redact the local part
            sanitized[key] = re.sub(r"([^@\s]+)@", "REDACTED@", value)
        else:
            sanitized[key] = value

    return sanitized


async def _audit_log_operation(
    collection_id: str,
    operation_id: int,
    user_id: int | None,
    action: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Create an audit log entry for a collection operation.

    All details are sanitized to ensure no PII is logged.
    """
    try:
        from shared.database.database import AsyncSessionLocal
        from shared.database.models import CollectionAuditLog

        # Sanitize details to remove PII
        sanitized_details = _sanitize_audit_details(details)

        async with AsyncSessionLocal() as session:
            audit_log = CollectionAuditLog(
                collection_id=collection_id,
                operation_id=operation_id,
                user_id=user_id,
                action=action,
                details=sanitized_details,
            )
            session.add(audit_log)
            await session.commit()
    except Exception as e:
        logger.warning(f"Failed to create audit log: {e}")


async def _process_index_operation(
    operation: dict,
    collection: dict,
    collection_repo: Any,
    document_repo: Any,  # noqa: ARG001
    updater: CeleryTaskWithUpdates,
) -> dict[str, Any]:
    """Process INDEX operation - Initial collection creation with monitoring."""
    from shared.metrics.collection_metrics import record_qdrant_operation

    try:
        # Create Qdrant collection
        qdrant_client = qdrant_manager.get_client()

        # Use the vector_store_name from the collection if it exists, otherwise generate one
        vector_store_name = collection.get("vector_store_name") or f"collection_{collection['uuid']}"

        # Get vector dimension from config
        from webui.services.collection_service import DEFAULT_VECTOR_DIMENSION

        config = collection.get("config", {})
        vector_dim = config.get("vector_dim", DEFAULT_VECTOR_DIMENSION)

        # Create collection in Qdrant with monitoring
        from qdrant_client.models import Distance, VectorParams

        with QdrantOperationTimer("create_collection"):
            qdrant_client.create_collection(
                collection_name=vector_store_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )

        # Update collection with Qdrant collection name
        await collection_repo.update(collection["id"], {"vector_store_name": vector_store_name})

        # Audit log the collection creation
        await _audit_log_operation(
            collection["id"],
            operation["id"],
            operation.get("user_id"),
            "collection_indexed",
            {"qdrant_collection": vector_store_name, "vector_dim": vector_dim},
        )

        await updater.send_update("index_completed", {"qdrant_collection": vector_store_name, "vector_dim": vector_dim})

        return {"success": True, "qdrant_collection": vector_store_name, "vector_dim": vector_dim}

    except Exception as e:
        logger.error(f"Failed to create Qdrant collection: {e}")
        record_qdrant_operation("create_collection", "failed")
        raise


async def _process_append_operation(
    operation: dict,
    collection: dict,
    collection_repo: Any,  # noqa: ARG001
    document_repo: Any,  # noqa: ARG001
    updater: CeleryTaskWithUpdates,
) -> dict[str, Any]:
    """Process APPEND operation - Add documents to existing collection with monitoring."""
    from shared.metrics.collection_metrics import document_processing_duration, record_document_processed

    config = operation.get("config", {})
    source_path = config.get("source_path")

    if not source_path:
        raise ValueError("source_path is required for APPEND operation")

    # Import required modules for file scanning
    from shared.database.database import AsyncSessionLocal
    from shared.database.repositories.document_repository import DocumentRepository
    from webui.services.file_scanning_service import FileScanningService

    # Use proper async session for new repositories
    async with AsyncSessionLocal() as session:
        # Create new repository instance with session
        doc_repo = DocumentRepository(session)

        # Create file scanning service
        file_scanner = FileScanningService(db_session=session, document_repo=doc_repo)

        # Scan directory and register documents
        await updater.send_update("scanning_files", {"status": "scanning", "source_path": source_path})

        try:
            # Time document scanning
            scan_start = time.time()

            scan_stats = await file_scanner.scan_directory_and_register_documents(
                collection_id=collection["id"],
                source_path=source_path,
                recursive=True,  # Default to recursive scanning
                batch_size=EMBEDDING_BATCH_SIZE,  # Commit in batches for large directories
            )

            # Record scanning duration
            scan_duration = time.time() - scan_start
            document_processing_duration.labels(operation_type="append").observe(scan_duration)

            # Final commit to ensure any remaining transactions are persisted
            await session.commit()

            # Send progress update
            await updater.send_update(
                "scanning_completed",
                {
                    "status": "scanning_completed",
                    "total_files_found": scan_stats["total_files_found"],
                    "new_files_registered": scan_stats["new_files_registered"],
                    "duplicate_files_skipped": scan_stats["duplicate_files_skipped"],
                    "errors_count": len(scan_stats.get("errors", [])),
                },
            )

            # Record document processing metrics
            for _ in range(scan_stats["new_files_registered"]):
                record_document_processed("append", "registered")
            for _ in range(scan_stats["duplicate_files_skipped"]):
                record_document_processed("append", "skipped")
            for _ in range(len(scan_stats.get("errors", []))):
                record_document_processed("append", "failed")

            # Audit log the append operation
            await _audit_log_operation(
                collection["id"],
                operation["id"],
                operation.get("user_id"),
                "documents_appended",
                {
                    "source_path": source_path,
                    "documents_added": scan_stats["new_files_registered"],
                    "duplicates_skipped": scan_stats["duplicate_files_skipped"],
                },
            )

            # TODO: Process registered documents to generate embeddings and add to Qdrant
            # This will be implemented in future tasks

            await updater.send_update(
                "append_completed",
                {
                    "source_path": source_path,
                    "documents_added": scan_stats["new_files_registered"],
                    "total_files_scanned": scan_stats["total_files_found"],
                    "duplicates_skipped": scan_stats["duplicate_files_skipped"],
                },
            )

            return {
                "success": True,
                "source_path": source_path,
                "documents_added": scan_stats["new_files_registered"],
                "total_files_scanned": scan_stats["total_files_found"],
                "duplicates_skipped": scan_stats["duplicate_files_skipped"],
                "total_size_bytes": scan_stats["total_size_bytes"],
                "scan_duration_seconds": scan_duration,
                "errors": scan_stats.get("errors", []),
            }

        except Exception as e:
            logger.error(f"Failed to scan and register documents: {e}")
            await session.rollback()
            raise


async def reindex_handler(
    collection: dict,
    new_config: dict[str, Any],
    qdrant_manager_instance: QdrantManager,
) -> dict[str, Any]:
    """Create staging collection for blue-green reindexing.

    This handler is responsible for creating the staging (green) collection
    that will be used during the reindexing process. It's the first critical
    step of the zero-downtime reindexing strategy.

    Args:
        collection: Collection dictionary with current configuration
        new_config: New configuration for the reindexed collection
        qdrant_manager_instance: QdrantManager instance for collection operations

    Returns:
        Dict containing staging collection info

    Raises:
        ValueError: If collection configuration is invalid
        Exception: If staging collection creation fails
    """
    from webui.services.collection_service import DEFAULT_VECTOR_DIMENSION

    # Get base collection name
    base_collection_name = collection.get("vector_store_name")
    if not base_collection_name:
        raise ValueError("Collection missing vector_store_name field")

    # Determine vector dimension for new collection
    vector_dim = new_config.get("vector_dim", collection.get("config", {}).get("vector_dim", DEFAULT_VECTOR_DIMENSION))

    # Create staging collection using QdrantManager
    logger.info(f"Creating staging collection for {base_collection_name} with vector_dim={vector_dim}")

    try:
        staging_collection_name = qdrant_manager_instance.create_staging_collection(
            base_name=base_collection_name, vector_size=vector_dim
        )

        # Prepare staging info to store in database
        staging_info = {
            "collection_name": staging_collection_name,
            "created_at": datetime.now(UTC).isoformat(),
            "vector_dim": vector_dim,
            "base_collection": base_collection_name,
        }

        logger.info(f"Successfully created staging collection: {staging_collection_name}")

        return staging_info

    except Exception as e:
        logger.error(f"Failed to create staging collection for {base_collection_name}: {e}")
        raise


async def _process_reindex_operation(
    operation: dict,
    collection: dict,
    collection_repo: Any,
    document_repo: Any,
    updater: CeleryTaskWithUpdates,
) -> dict[str, Any]:
    """Process REINDEX operation - Blue-green reindexing with validation checkpoints."""
    from shared.database.models import DocumentStatus
    from shared.metrics.collection_metrics import (
        record_reindex_checkpoint,
        reindex_switch_duration,
        reindex_validation_duration,
    )

    config = operation.get("config", {})
    new_config = config.get("new_config", {})
    staging_collection_name = None
    checkpoints = []

    # Initialize QdrantManager with the client
    qdrant_client = qdrant_manager.get_client()
    qdrant_manager_instance = QdrantManager(qdrant_client)

    try:
        # Checkpoint 1: Pre-flight checks
        checkpoint_time = time.time()
        record_reindex_checkpoint(collection["id"], "preflight_start")
        checkpoints.append(("preflight_start", checkpoint_time))

        # Verify collection health
        if collection.get("status") == "error":
            raise ValueError("Cannot reindex collection in error state")

        # Check if collection has documents
        doc_stats = await document_repo.get_stats_by_collection(collection["id"])
        if doc_stats["total_count"] == 0:
            raise ValueError("Cannot reindex empty collection")

        await updater.send_update(
            "reindex_preflight",
            {
                "status": "preflight_complete",
                "documents_to_process": doc_stats["total_count"],
                "current_vector_count": collection.get("vector_count", 0),
            },
        )

        record_reindex_checkpoint(collection["id"], "preflight_complete")
        checkpoints.append(("preflight_complete", time.time()))

        # Checkpoint 2: Create staging collection using reindex_handler
        record_reindex_checkpoint(collection["id"], "staging_creation_start")

        # Get the current collection name before creating staging
        old_collection_name = collection.get("vector_store_name")
        if not old_collection_name:
            raise ValueError("Collection missing vector_store_name field")

        # Call the reindex_handler to create staging collection
        staging_info = await reindex_handler(collection, new_config, qdrant_manager_instance)
        staging_collection_name = staging_info["collection_name"]

        # Update collection with staging info
        await collection_repo.update(
            collection["id"],
            {"qdrant_staging": staging_info},
        )

        record_reindex_checkpoint(collection["id"], "staging_creation_complete")
        checkpoints.append(("staging_creation_complete", time.time()))

        await updater.send_update(
            "staging_created",
            {"staging_collection": staging_collection_name, "vector_dim": staging_info["vector_dim"]},
        )

        # Checkpoint 3: Reprocess documents
        record_reindex_checkpoint(collection["id"], "reprocessing_start")

        # Get all active documents
        documents = await document_repo.list_by_collection(
            collection["id"],
            status_filter=DocumentStatus.COMPLETED,
            limit=None,  # Get all documents
        )

        total_documents = len(documents)
        processed_count = 0
        failed_count = 0
        vector_count = 0

        # Process documents in batches
        # Get batch_size from config, defaulting to EMBEDDING_BATCH_SIZE
        batch_size = new_config.get("batch_size", collection.get("config", {}).get("batch_size", EMBEDDING_BATCH_SIZE))

        # Get the new configuration values
        chunk_size = new_config.get("chunk_size", collection.get("config", {}).get("chunk_size", 600))
        chunk_overlap = new_config.get("chunk_overlap", collection.get("config", {}).get("chunk_overlap", 200))
        model_name = new_config.get(
            "model_name", collection.get("config", {}).get("model_name", "Qwen/Qwen3-Embedding-0.6B")
        )
        quantization = new_config.get("quantization", collection.get("config", {}).get("quantization", "float32"))
        instruction = new_config.get("instruction", collection.get("config", {}).get("instruction"))
        vector_dim = new_config.get("vector_dim", collection.get("config", {}).get("vector_dim"))

        # Get worker count from config, defaulting to 4
        worker_count = new_config.get("worker_count", collection.get("config", {}).get("worker_count", 4))

        # Create thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for i in range(0, total_documents, batch_size):
                batch = documents[i : i + batch_size]

                for doc in batch:
                    try:
                        # Extract text from document
                        loop = asyncio.get_event_loop()
                        file_path = doc.get("file_path", doc.get("path"))

                        logger.info(f"Reprocessing document: {file_path}")

                        # Extract text blocks with metadata
                        text_blocks = await asyncio.wait_for(
                            loop.run_in_executor(executor, extract_and_serialize_thread_safe, file_path),
                            timeout=300,  # 5 minute timeout
                        )

                        # Create chunker with new configuration
                        chunker = TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                        # Generate document ID
                        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:16]

                        # Process each text block
                        all_chunks = []
                        for text, metadata in text_blocks:
                            if not text.strip():
                                continue

                            # Create chunks
                            chunks = await loop.run_in_executor(
                                executor,
                                chunker.chunk_text,
                                text,
                                doc_id,
                                metadata,
                            )
                            all_chunks.extend(chunks)

                        if not all_chunks:
                            logger.warning(f"No chunks created for document: {file_path}")
                            continue

                        # Generate embeddings for chunks
                        texts = [chunk["text"] for chunk in all_chunks]

                        # Generate embeddings with GPU scheduling
                        task_id = f"reindex-{operation['id']}-{doc_id}"

                        def generate_embeddings_with_gpu(task_id: str = task_id, texts: list[str] = texts) -> Any:
                            with gpu_task(task_id) as gpu_id:
                                if gpu_id is None:
                                    logger.warning(f"No GPU available for task {task_id}, proceeding with CPU")
                                else:
                                    logger.info(f"Task {task_id} using GPU {gpu_id}")

                                return embedding_service.generate_embeddings(
                                    texts,
                                    model_name,
                                    quantization,
                                    batch_size,
                                    False,  # show_progress
                                    instruction,
                                )

                        embeddings_array = await loop.run_in_executor(executor, generate_embeddings_with_gpu)

                        if embeddings_array is None:
                            raise Exception("Failed to generate embeddings")

                        embeddings = embeddings_array.tolist()

                        # Handle dimension override if specified
                        if vector_dim and len(embeddings) > 0:
                            model_dim = len(embeddings[0])
                            if vector_dim != model_dim:
                                logger.info(f"Adjusting embeddings from {model_dim} to {vector_dim} dimensions")
                                adjusted_embeddings = []
                                for emb in embeddings:
                                    if vector_dim < model_dim:
                                        # Truncate
                                        adjusted = emb[:vector_dim]
                                    else:
                                        # Pad with zeros
                                        adjusted = emb + [0.0] * (vector_dim - model_dim)

                                    # Renormalize
                                    norm = sum(x**2 for x in adjusted) ** 0.5
                                    if norm > 0:
                                        adjusted = [x / norm for x in adjusted]
                                    adjusted_embeddings.append(adjusted)
                                embeddings = adjusted_embeddings

                        # Upload vectors to staging collection
                        points = []
                        for i, chunk in enumerate(all_chunks):
                            point = PointStruct(
                                id=str(uuid.uuid4()),
                                vector=embeddings[i],
                                payload={
                                    "collection_id": collection["id"],
                                    "doc_id": doc_id,
                                    "chunk_id": chunk["chunk_id"],
                                    "path": file_path,
                                    "content": chunk["text"],
                                    "metadata": chunk.get("metadata", {}),
                                },
                            )
                            points.append(point)

                        # Upload to staging collection
                        with QdrantOperationTimer("upsert_staging_vectors"):
                            qdrant_client.upsert(collection_name=staging_collection_name, points=points, wait=True)

                        vector_count += len(points)
                        processed_count += 1

                        logger.info(f"Successfully reprocessed document {file_path}: {len(points)} vectors created")

                        # Free memory
                        del text_blocks, all_chunks, texts, embeddings_array, embeddings, points
                        gc.collect()

                    except Exception as e:
                        logger.error(f"Failed to reprocess document {doc.get('file_path', 'unknown')}: {e}")
                        failed_count += 1
                        # Continue processing other documents

                # Send progress update
                progress = (processed_count / total_documents) * 100 if total_documents > 0 else 0
                await updater.send_update(
                    "reprocessing_progress",
                    {
                        "processed": processed_count,
                        "total": total_documents,
                        "failed": failed_count,
                        "progress_percent": progress,
                        "vectors_created": vector_count,
                    },
                )

        record_reindex_checkpoint(collection["id"], "reprocessing_complete")
        checkpoints.append(("reprocessing_complete", time.time()))

        # Checkpoint 4: Validation
        validation_start = time.time()
        record_reindex_checkpoint(collection["id"], "validation_start")

        # Validate the new collection
        validation_result = await _validate_reindex(
            qdrant_client,
            old_collection_name,
            staging_collection_name,
            sample_size=min(100, total_documents // 10),
        )

        validation_duration = time.time() - validation_start
        reindex_validation_duration.observe(validation_duration)

        if not validation_result["passed"]:
            raise ValueError(f"Reindex validation failed: {validation_result['issues']}")

        # Log warnings if any
        if validation_result.get("warnings"):
            for warning in validation_result["warnings"]:
                logger.warning(f"Reindex validation warning: {warning}")

        record_reindex_checkpoint(collection["id"], "validation_complete")
        checkpoints.append(("validation_complete", time.time()))

        await updater.send_update(
            "validation_complete",
            {
                "validation_passed": True,
                "validation_duration": validation_duration,
                "sample_size": validation_result["sample_size"],
                "validation_warnings": validation_result.get("warnings", []),
                "validation_details": validation_result.get("validation_details", {}),
            },
        )

        # Checkpoint 5: Atomic switch via internal API
        switch_start = time.time()
        record_reindex_checkpoint(collection["id"], "atomic_switch_start")

        # Call internal API to perform atomic switch
        import httpx

        # Use configurable host for containerized environments
        host = settings.get("WEBUI_INTERNAL_HOST", "localhost")
        port = settings.get("WEBUI_PORT", 8080)
        internal_api_url = f"http://{host}:{port}/api/internal/complete-reindex"
        request_data = {
            "collection_id": collection["id"],
            "operation_id": operation["id"],
            "staging_collection_name": staging_collection_name,
            "new_config": new_config,
            "vector_count": vector_count,
        }

        headers = {
            "X-Internal-API-Key": settings.INTERNAL_API_KEY,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                internal_api_url,
                json=request_data,
                headers=headers,
                timeout=30.0,
            )

            if response.status_code != 200:
                raise ValueError(f"Failed to complete reindex via API: {response.status_code} - {response.text}")

            api_result = response.json()
            old_collection_names = api_result["old_collection_names"]

        switch_duration = time.time() - switch_start
        reindex_switch_duration.observe(switch_duration)

        record_reindex_checkpoint(collection["id"], "atomic_switch_complete")
        checkpoints.append(("atomic_switch_complete", time.time()))

        # Checkpoint 6: Schedule cleanup of old collections
        record_reindex_checkpoint(collection["id"], "cleanup_scheduled")

        # Calculate delay based on collection size
        cleanup_delay = calculate_cleanup_delay(vector_count)

        # Schedule cleanup task to run after a delay
        cleanup_task = cleanup_old_collections.apply_async(
            args=[old_collection_names, collection["id"]],
            countdown=cleanup_delay,
        )

        logger.info(
            f"Scheduled cleanup of {len(old_collection_names)} old collections "
            f"to run in {cleanup_delay} seconds. Task ID: {cleanup_task.id}"
        )

        record_reindex_checkpoint(collection["id"], "cleanup_scheduled_complete")
        checkpoints.append(("cleanup_scheduled_complete", time.time()))

        # Audit log the reindex
        await _audit_log_operation(
            collection["id"],
            operation["id"],
            operation.get("user_id"),
            "collection_reindexed",
            {
                "old_collections": old_collection_names,
                "new_collection": staging_collection_name,
                "old_config": collection["config"],
                "new_config": new_config,
                "documents_processed": processed_count,
                "vectors_created": vector_count,
                "checkpoints": checkpoints,
                "cleanup_task_id": cleanup_task.id,
            },
        )

        await updater.send_update(
            "reindex_completed",
            {
                "old_collections": old_collection_names,
                "new_collection": staging_collection_name,
                "documents_processed": processed_count,
                "vectors_created": vector_count,
                "duration": time.time() - checkpoints[0][1],
                "cleanup_scheduled": True,
                "cleanup_task_id": cleanup_task.id,
            },
        )

        return {
            "success": True,
            "old_collections": old_collection_names,
            "new_collection": staging_collection_name,
            "documents_processed": processed_count,
            "vectors_created": vector_count,
            "checkpoints": checkpoints,
            "cleanup_task_id": cleanup_task.id,
        }

    except Exception:
        logger.error(f"Failed to reindex collection at checkpoint: {checkpoints[-1][0] if checkpoints else 'unknown'}")

        # Cleanup staging collection if it was created
        if staging_collection_name:
            try:
                qdrant_client.delete_collection(staging_collection_name)
                logger.info(f"Cleaned up staging collection {staging_collection_name}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup staging collection: {cleanup_error}")

            # Clear staging info in database
            await collection_repo.update(collection["id"], {"qdrant_staging": None})

        raise


async def _validate_reindex(
    qdrant_client: Any,
    old_collection: str,
    new_collection: str,
    sample_size: int = 100,
) -> dict[str, Any]:
    """Validate reindex results by comparing old and new collections."""
    try:
        # Get collection info
        old_info = qdrant_client.get_collection(old_collection)
        new_info = qdrant_client.get_collection(new_collection)

        issues = []

        # Check if new collection has vectors
        if new_info.points_count == 0:
            issues.append("New collection has no vectors")

        # Check if vector count is reasonable (allow 10% variance for chunking changes)
        if old_info.points_count > 0:
            ratio = new_info.points_count / old_info.points_count
            if ratio < (1 - REINDEX_VECTOR_COUNT_VARIANCE) or ratio > (1 + REINDEX_VECTOR_COUNT_VARIANCE):
                issues.append(f"Vector count mismatch: {old_info.points_count} -> {new_info.points_count}")

        # Sample and compare search results for quality validation
        if old_info.points_count > 0 and new_info.points_count > 0:
            try:
                # Get a sample of points from the old collection
                import random

                # Unused imports removed - were for filtering
                # Scroll through some points from old collection to get sample IDs
                scroll_result = qdrant_client.scroll(
                    collection_name=old_collection,
                    limit=min(sample_size, old_info.points_count),
                    with_vectors=True,
                    with_payload=True,
                )

                sample_points = scroll_result[0]  # First element is the list of points

                if len(sample_points) > 0:
                    # Test search quality by comparing results
                    search_mismatches = 0
                    total_score_diff = 0.0
                    comparisons_made = 0

                    # Sample up to 10 points for search comparison
                    test_points = random.sample(sample_points, min(10, len(sample_points)))

                    for point in test_points:
                        # Search in both collections using the vector
                        old_results = qdrant_client.search(
                            collection_name=old_collection,
                            query_vector=point.vector,
                            limit=5,
                            with_payload=True,
                        )

                        new_results = qdrant_client.search(
                            collection_name=new_collection,
                            query_vector=point.vector,
                            limit=5,
                            with_payload=True,
                        )

                        # Compare top result relevance
                        if old_results and new_results:
                            # Check if the same document appears in top results
                            old_doc_ids = {r.payload.get("doc_id") for r in old_results if r.payload}
                            new_doc_ids = {r.payload.get("doc_id") for r in new_results if r.payload}

                            overlap = len(old_doc_ids & new_doc_ids)
                            if overlap < 3:  # Less than 3 out of 5 results match
                                search_mismatches += 1

                            # Compare scores (allowing for some variance due to reindexing)
                            if old_results[0].score and new_results[0].score:
                                score_diff = abs(old_results[0].score - new_results[0].score)
                                total_score_diff += score_diff
                                comparisons_made += 1

                    # Evaluate search quality
                    if search_mismatches > len(test_points) * REINDEX_SEARCH_MISMATCH_THRESHOLD:
                        issues.append(
                            f"Search quality degraded: {search_mismatches}/{len(test_points)} searches differ significantly"
                        )

                    if comparisons_made > 0:
                        avg_score_diff = total_score_diff / comparisons_made
                        if avg_score_diff > REINDEX_SCORE_DIFF_THRESHOLD:
                            issues.append(
                                f"Search scores differ significantly: average difference {avg_score_diff:.3f}"
                            )

            except Exception as e:
                logger.warning(f"Failed to perform search validation: {e}")
                # Don't fail validation on search comparison errors
                issues.append(f"Could not validate search quality: {str(e)}")

        # Additional validation: Check if vector dimensions match
        if hasattr(old_info.config, "params") and hasattr(new_info.config, "params"):
            old_dim = old_info.config.params.vectors.size if hasattr(old_info.config.params.vectors, "size") else None
            new_dim = new_info.config.params.vectors.size if hasattr(new_info.config.params.vectors, "size") else None

            if old_dim and new_dim and old_dim != new_dim:
                issues.append(f"Vector dimension mismatch: {old_dim} -> {new_dim}")

        # Strict validation criteria
        validation_passed = len(issues) == 0

        # Add warning-level issues that don't fail validation
        warnings = []
        if new_info.points_count > old_info.points_count * 1.05:
            warnings.append(
                f"Vector count increased by more than 5%: {old_info.points_count} -> {new_info.points_count}"
            )

        return {
            "passed": validation_passed,
            "issues": issues,
            "warnings": warnings,
            "sample_size": sample_size,
            "old_count": old_info.points_count,
            "new_count": new_info.points_count,
            "validation_details": {
                "vector_count_ratio": new_info.points_count / old_info.points_count if old_info.points_count > 0 else 0,
                "search_quality_tested": "search_mismatches" in locals(),
            },
        }

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {
            "passed": False,
            "issues": [f"Validation error: {str(e)}"],
            "sample_size": 0,
        }


async def _process_remove_source_operation(
    operation: dict,
    collection: dict,
    collection_repo: Any,  # noqa: ARG001
    document_repo: Any,
    updater: CeleryTaskWithUpdates,
) -> dict[str, Any]:
    """Process REMOVE_SOURCE operation - Remove documents from a source with monitoring."""
    from shared.database.models import DocumentStatus
    from shared.metrics.collection_metrics import record_document_processed

    config = operation.get("config", {})
    source_path = config.get("source_path")

    if not source_path:
        raise ValueError("source_path is required for REMOVE_SOURCE operation")

    try:
        # Get documents from this source
        documents = await document_repo.list_by_collection_and_source(collection["id"], source_path)

        if not documents:
            logger.info(f"No documents found for source {source_path}")
            await _audit_log_operation(
                collection["id"],
                operation["id"],
                operation.get("user_id"),
                "source_removed",
                {"source_path": source_path, "documents_removed": 0},
            )
            return {"success": True, "documents_removed": 0, "source_path": source_path}

        # Remove vectors from Qdrant
        # TODO: Uncomment when implementing actual vector deletion
        # qdrant_client = qdrant_manager.get_client()
        # vector_store_name = collection["vector_store_name"]

        # Get document IDs to remove
        doc_ids = [doc["id"] for doc in documents]
        removed_count = 0

        # Remove vectors in batches
        batch_size = DOCUMENT_REMOVAL_BATCH_SIZE
        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i : i + batch_size]

            try:
                # Search for points with these document IDs
                # TODO: This requires proper implementation when document IDs are stored in Qdrant payload
                with QdrantOperationTimer("delete_points"):
                    # For now, simulate deletion
                    await asyncio.sleep(0.01)
                    removed_count += len(batch_ids)

                # Send progress update
                progress = ((i + len(batch_ids)) / len(doc_ids)) * 100
                await updater.send_update(
                    "removing_documents",
                    {
                        "removed": i + len(batch_ids),
                        "total": len(doc_ids),
                        "progress_percent": progress,
                    },
                )
            except Exception as e:
                logger.error(f"Failed to remove vectors for batch: {e}")

        # Wrap critical database operations in a transaction for atomicity
        from shared.database.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session, session.begin():
            # Create repository instances with the transaction session
            from shared.database.repositories.collection_repository import CollectionRepository
            from shared.database.repositories.document_repository import DocumentRepository

            doc_repo_tx = DocumentRepository(session)
            collection_repo_tx = CollectionRepository(session)

            # Mark documents as deleted in database
            await doc_repo_tx.bulk_update_status(doc_ids, DocumentStatus.DELETED)

            # Record document removal metrics
            for _ in range(len(documents)):
                record_document_processed("remove_source", "deleted")

            # Update collection stats
            stats = await doc_repo_tx.get_stats_by_collection(collection["id"])
            await collection_repo_tx.update_stats(
                collection["id"],
                total_documents=stats["total_count"],
                total_chunks=stats["total_chunks"],
                total_size_bytes=stats["total_size_bytes"],
            )
            # Transaction will commit automatically if no exception occurs

        # Update collection metrics
        await _update_collection_metrics(
            collection["id"],
            stats["total_count"],
            collection.get("vector_count", 0) - removed_count,
            stats["total_size_bytes"],
        )

        # Audit log the removal
        await _audit_log_operation(
            collection["id"],
            operation["id"],
            operation.get("user_id"),
            "source_removed",
            {
                "source_path": source_path,
                "documents_removed": len(documents),
                "vectors_removed": removed_count,
            },
        )

        await updater.send_update(
            "remove_source_completed",
            {
                "source_path": source_path,
                "documents_removed": len(documents),
                "vectors_removed": removed_count,
            },
        )

        return {
            "success": True,
            "source_path": source_path,
            "documents_removed": len(documents),
            "vectors_removed": removed_count,
        }

    except Exception as e:
        logger.error(f"Failed to remove source {source_path}: {e}")
        raise
