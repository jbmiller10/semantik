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
        self._redis_client = None
        
    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis client."""
        if not self._redis_client:
            self._redis_client = await redis.from_url(
                self.redis_url,
                decode_responses=True
            )
        return self._redis_client
        
    async def send_update(self, update_type: str, data: dict) -> None:
        """Send update to Redis Stream."""
        try:
            redis_client = await self._get_redis()
            message = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": update_type,
                "data": data
            }
            
            # Add to stream with automatic ID
            await redis_client.xadd(
                self.stream_key,
                {"message": json.dumps(message)},
                maxlen=1000  # Keep last 1000 messages
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
        await updater.send_update("status_update", {
            "status": "processing",
            "total_files": len(files),
            "processed_files": 0
        })

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
                await updater.send_update("file_processing", {
                    "status": "processing",
                    "current_file": file_row["path"],
                    "processed_files": file_idx,
                    "total_files": len(files)
                })

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
                await updater.send_update("file_completed", {
                    "status": "processing",
                    "current_file": file_row["path"],
                    "processed_files": processed_count,
                    "total_files": len(files)
                })

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
                await updater.send_update("error", {
                    "status": "processing",
                    "current_file": file_row["path"],
                    "error": str(e),
                    "processed_files": processed_count,
                    "failed_files": failed_count,
                    "total_files": len(files)
                })

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
        await updater.send_update("job_completed", {
            "status": "completed",
            "processed_files": processed_count,
            "failed_files": failed_count,
            "total_vectors": total_vectors_created
        })
        
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
        await updater.send_update("job_failed", {
            "status": "failed",
            "error": str(e)
        })
        
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
