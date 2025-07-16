"""Celery task definitions for asynchronous processing."""

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

# Background task executor
executor = ThreadPoolExecutor(max_workers=8)


class CeleryTaskWithUpdates:
    """Helper class to send updates to Redis Stream from Celery tasks."""

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
            await redis_client.xadd(
                self.stream_key, {"message": json.dumps(message)}, maxlen=1000  # Keep last 1000 messages
            )

            # Set TTL on first message (24 hours)
            await redis_client.expire(self.stream_key, 86400)

            logger.debug(f"Sent update to Redis stream {self.stream_key}: type={update_type}")
        except Exception as e:
            logger.error(f"Failed to send update to Redis stream: {e}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()


def extract_and_serialize_thread_safe(filepath: str) -> list[tuple[str, dict[str, Any]]]:
    """Thread-safe version of extract_and_serialize that preserves metadata"""
    from shared.text_processing.extraction import extract_and_serialize

    result: list[tuple[str, dict[str, Any]]] = extract_and_serialize(filepath)
    return result


@celery_app.task(bind=True)
def test_task(self: Any) -> dict[str, str]:  # noqa: ARG001
    """Test task to verify Celery is working."""
    return {"status": "success", "message": "Celery is working!"}


@celery_app.task(name="webui.tasks.cleanup_old_results")
def cleanup_old_results(days_to_keep: int = 7) -> dict[str, Any]:
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
    updater = None

    # Create repository instances
    job_repo = create_job_repository()
    file_repo = create_file_repository()
    collection_repo = create_collection_repository()  # noqa: F841

    # Create updater for sending Redis updates
    updater = CeleryTaskWithUpdates(job_id)

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
                    await job_repo.update_job(job_id, {"processed_files": current_job.get("processed_files", 0) + 1})

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
            error_msg = "No vectors were created. All files either failed to process or contained no extractable text."
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
            if collection_info.points_count is not None and collection_info.points_count < total_vectors_created * 0.9:
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

        # Close Redis connection
        if updater:
            await updater.close()


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
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,  # Ensure message reliability
    soft_time_limit=3600,  # 1 hour soft limit
    time_limit=7200,  # 2 hour hard limit
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

    updater = None
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

        # Create updater for sending Redis updates after task ID is set
        updater = CeleryTaskWithUpdates(operation_id)

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

            # Send update if updater exists
            if updater:
                await updater.send_update("operation_failed", {"status": "failed", "error": str(e)})
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

        # Close updater if it exists
        if updater:
            try:
                await updater.close()
            except Exception as close_error:
                logger.error(f"Failed to close updater: {close_error}")


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


async def _audit_log_operation(
    collection_id: str,
    operation_id: int,
    user_id: int | None,
    action: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Create an audit log entry for a collection operation."""
    try:
        from shared.database.database import AsyncSessionLocal
        from shared.database.models import CollectionAuditLog

        async with AsyncSessionLocal() as session:
            audit_log = CollectionAuditLog(
                collection_id=collection_id,
                operation_id=operation_id,
                user_id=user_id,
                action=action,
                details=details,
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
        qdrant_collection_name = f"collection_{collection['uuid']}"

        # Get vector dimension from config
        from webui.services.collection_service import DEFAULT_VECTOR_DIMENSION

        config = collection.get("config", {})
        vector_dim = config.get("vector_dim", DEFAULT_VECTOR_DIMENSION)

        # Create collection in Qdrant with monitoring
        from qdrant_client.models import Distance, VectorParams

        with QdrantOperationTimer("create_collection"):
            qdrant_client.create_collection(
                collection_name=qdrant_collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )

        # Update collection with Qdrant collection name
        await collection_repo.update(collection["id"], {"qdrant_collection_name": qdrant_collection_name})

        # Audit log the collection creation
        await _audit_log_operation(
            collection["id"],
            operation["id"],
            operation.get("user_id"),
            "collection_indexed",
            {"qdrant_collection": qdrant_collection_name, "vector_dim": vector_dim},
        )

        await updater.send_update(
            "index_completed", {"qdrant_collection": qdrant_collection_name, "vector_dim": vector_dim}
        )

        return {"success": True, "qdrant_collection": qdrant_collection_name, "vector_dim": vector_dim}

    except Exception as e:
        logger.error(f"Failed to create Qdrant collection: {e}")
        record_qdrant_operation("create_collection", "failed")
        raise


async def _process_append_operation(
    operation: dict,
    collection: dict,
    collection_repo: Any,
    document_repo: Any,  # noqa: ARG001
    updater: CeleryTaskWithUpdates,
) -> dict[str, Any]:
    """Process APPEND operation - Add documents to existing collection with monitoring."""
    from shared.metrics.collection_metrics import (
        document_processing_duration,
        record_document_processed,
    )

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
                batch_size=100,  # Commit every 100 files for large directories
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

        # Checkpoint 2: Create staging collection
        qdrant_client = qdrant_manager.get_client()
        old_collection_name = collection["qdrant_collection_name"]
        staging_collection_name = f"{old_collection_name}_staging_{int(time.time())}"

        record_reindex_checkpoint(collection["id"], "staging_creation_start")

        from qdrant_client.models import Distance, VectorParams
        from webui.services.collection_service import DEFAULT_VECTOR_DIMENSION

        vector_dim = new_config.get("vector_dim", collection["config"].get("vector_dim", DEFAULT_VECTOR_DIMENSION))

        with QdrantOperationTimer("create_staging_collection"):
            qdrant_client.create_collection(
                collection_name=staging_collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )

        # Update collection with staging info
        await collection_repo.update(
            collection["id"],
            {
                "qdrant_staging": {
                    "collection_name": staging_collection_name,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            },
        )

        record_reindex_checkpoint(collection["id"], "staging_creation_complete")
        checkpoints.append(("staging_creation_complete", time.time()))

        await updater.send_update(
            "staging_created",
            {"staging_collection": staging_collection_name, "vector_dim": vector_dim},
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
        batch_size = 100
        for i in range(0, total_documents, batch_size):
            batch = documents[i : i + batch_size]

            # TODO: Actually reprocess documents with new configuration
            # For now, simulate processing
            await asyncio.sleep(0.1)  # Simulate processing time

            processed_count += len(batch)
            vector_count += len(batch) * 10  # Simulate vector creation

            # Send progress update
            progress = (processed_count / total_documents) * 100
            await updater.send_update(
                "reprocessing_progress",
                {
                    "processed": processed_count,
                    "total": total_documents,
                    "failed": failed_count,
                    "progress_percent": progress,
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

        record_reindex_checkpoint(collection["id"], "validation_complete")
        checkpoints.append(("validation_complete", time.time()))

        await updater.send_update(
            "validation_complete",
            {
                "validation_passed": True,
                "validation_duration": validation_duration,
                "sample_size": validation_result["sample_size"],
            },
        )

        # Checkpoint 5: Atomic switch
        switch_start = time.time()
        record_reindex_checkpoint(collection["id"], "atomic_switch_start")

        # Update collection to point to new Qdrant collection
        await collection_repo.update(
            collection["id"],
            {
                "config": new_config,
                "qdrant_collection_name": staging_collection_name,
                "qdrant_staging": None,  # Clear staging info
                "vector_count": vector_count,
            },
        )

        switch_duration = time.time() - switch_start
        reindex_switch_duration.observe(switch_duration)

        record_reindex_checkpoint(collection["id"], "atomic_switch_complete")
        checkpoints.append(("atomic_switch_complete", time.time()))

        # Checkpoint 6: Cleanup old collection
        record_reindex_checkpoint(collection["id"], "cleanup_start")

        try:
            with QdrantOperationTimer("delete_old_collection"):
                qdrant_client.delete_collection(old_collection_name)
        except Exception as e:
            logger.warning(f"Failed to delete old collection {old_collection_name}: {e}")

        record_reindex_checkpoint(collection["id"], "cleanup_complete")
        checkpoints.append(("cleanup_complete", time.time()))

        # Audit log the reindex
        await _audit_log_operation(
            collection["id"],
            operation["id"],
            operation.get("user_id"),
            "collection_reindexed",
            {
                "old_collection": old_collection_name,
                "new_collection": staging_collection_name,
                "old_config": collection["config"],
                "new_config": new_config,
                "documents_processed": processed_count,
                "vectors_created": vector_count,
                "checkpoints": checkpoints,
            },
        )

        await updater.send_update(
            "reindex_completed",
            {
                "old_collection": old_collection_name,
                "new_collection": staging_collection_name,
                "documents_processed": processed_count,
                "vectors_created": vector_count,
                "duration": time.time() - checkpoints[0][1],
            },
        )

        return {
            "success": True,
            "old_collection": old_collection_name,
            "new_collection": staging_collection_name,
            "documents_processed": processed_count,
            "vectors_created": vector_count,
            "checkpoints": checkpoints,
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
            if ratio < 0.9 or ratio > 1.1:
                issues.append(f"Vector count mismatch: {old_info.points_count} -> {new_info.points_count}")

        # TODO: Sample and compare search results
        # This would involve:
        # 1. Getting random vectors from old collection
        # 2. Searching in both collections
        # 3. Comparing result quality

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "sample_size": sample_size,
            "old_count": old_info.points_count,
            "new_count": new_info.points_count,
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
    collection_repo: Any,
    document_repo: Any,
    updater: CeleryTaskWithUpdates,
) -> dict[str, Any]:
    """Process REMOVE_SOURCE operation - Remove documents from a source with monitoring."""
    from shared.database.models import DocumentStatus
    from shared.metrics.collection_metrics import (
        record_document_processed,
    )

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
        # qdrant_collection_name = collection["qdrant_collection_name"]

        # Get document IDs to remove
        doc_ids = [doc["id"] for doc in documents]
        removed_count = 0

        # Remove vectors in batches
        batch_size = 100
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

        # Mark documents as deleted in database
        await document_repo.bulk_update_status(doc_ids, DocumentStatus.DELETED)

        # Record document removal metrics
        for _ in range(len(documents)):
            record_document_processed("remove_source", "deleted")

        # Update collection stats
        stats = await document_repo.get_stats_by_collection(collection["id"])
        await collection_repo.update_stats(
            collection["id"],
            total_documents=stats["total_count"],
            total_chunks=stats["total_chunks"],
            total_size_bytes=stats["total_size_bytes"],
        )

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
