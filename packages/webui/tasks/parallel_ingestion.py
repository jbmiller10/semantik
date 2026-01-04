"""Parallel ingestion pipeline with sequential embedding.

This module provides a producer-consumer pattern where:
- Multiple workers extract and chunk documents in parallel (CPU-bound)
- A single worker processes embeddings sequentially (GPU-bound)

This avoids VRAM issues while maximizing CPU utilization for extraction/chunking.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import psutil
from qdrant_client.models import PointStruct

from shared.database.models import DocumentStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from shared.database.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_EXTRACTION_WORKERS = 4  # Parallel extraction/chunking workers
MIN_EXTRACTION_WORKERS = 1
MAX_EXTRACTION_WORKERS = 16  # Hard cap to prevent resource exhaustion
EMBEDDING_BATCH_SIZE = 32  # Chunks to batch before embedding
VECTOR_UPLOAD_BATCH_SIZE = 100
QUEUE_MAX_SIZE = 100  # Max pending chunks in queue
QUEUE_PUT_POLL_INTERVAL_SECONDS = 0.5  # How often producers check for downstream shutdown when backpressured

# Resource thresholds for dynamic scaling
MIN_MEMORY_PER_WORKER_MB = 512  # Minimum memory per worker
MIN_FREE_MEMORY_PERCENT = 20  # Keep at least 20% memory free
CPU_LOAD_THRESHOLD = 0.7  # Don't scale up if CPU load > 70%


def _vecpipe_url(path: str) -> str:
    """Build URL for vecpipe service.

    Uses Docker service name 'vecpipe' for container-to-container communication.
    This runs inside the worker container, so we use internal Docker networking.
    """
    return f"http://vecpipe:8000/{path.lstrip('/')}"


def calculate_optimal_workers(
    max_workers: int | None = None,
    min_workers: int = MIN_EXTRACTION_WORKERS,
) -> int:
    """Calculate optimal number of extraction workers based on available resources.

    Considers:
    - CPU cores available
    - Available memory
    - Current system load
    - Container memory limits (if running in Docker)

    Args:
        max_workers: Maximum workers cap (defaults to MAX_EXTRACTION_WORKERS)
        min_workers: Minimum workers to use

    Returns:
        Optimal number of workers
    """
    if max_workers is None:
        max_workers = MAX_EXTRACTION_WORKERS

    try:
        # Get CPU count
        cpu_count = os.cpu_count() or 2

        # Leave some cores for embedding worker and system
        cpu_based_workers = max(1, cpu_count - 2)

        # Get memory info
        mem = psutil.virtual_memory()
        available_memory_mb = mem.available / (1024 * 1024)
        total_memory_mb = mem.total / (1024 * 1024)

        # Check for container memory limits (cgroup v1 and v2)
        container_limit_mb = _get_container_memory_limit_mb()
        if container_limit_mb:
            # Use container limit instead of system total
            total_memory_mb = min(total_memory_mb, container_limit_mb)
            # Estimate available as limit minus current usage
            mem_usage_mb = mem.used / (1024 * 1024)
            available_memory_mb = min(available_memory_mb, container_limit_mb - mem_usage_mb)

        # Calculate memory-based workers
        # Reserve memory for system and embedding
        reserved_memory_mb = total_memory_mb * (MIN_FREE_MEMORY_PERCENT / 100)
        usable_memory_mb = max(0, available_memory_mb - reserved_memory_mb)
        memory_based_workers = max(1, int(usable_memory_mb / MIN_MEMORY_PER_WORKER_MB))

        # Check current CPU load
        try:
            load_avg = os.getloadavg()[0]  # 1-minute load average
            load_ratio = load_avg / cpu_count if cpu_count > 0 else 1.0

            if load_ratio > CPU_LOAD_THRESHOLD:
                # System is busy, reduce workers
                load_factor = max(0.5, 1.0 - (load_ratio - CPU_LOAD_THRESHOLD))
                cpu_based_workers = max(1, int(cpu_based_workers * load_factor))
                logger.info(
                    "High system load (%.1f), reducing workers by factor %.2f",
                    load_ratio,
                    load_factor,
                )
        except (OSError, AttributeError):
            # getloadavg not available on Windows
            pass

        # Take minimum of CPU and memory constraints
        optimal = min(cpu_based_workers, memory_based_workers)

        # Apply bounds
        optimal = max(min_workers, min(optimal, max_workers))

        logger.info(
            "Dynamic worker calculation: cpu_cores=%d, available_mem=%.0fMB, "
            "cpu_based=%d, mem_based=%d, optimal=%d",
            cpu_count,
            available_memory_mb,
            cpu_based_workers,
            memory_based_workers,
            optimal,
        )

        return optimal

    except (OSError, RuntimeError, ValueError, psutil.Error) as exc:
        logger.warning("Failed to calculate optimal workers: %s, using default", exc)
        return max(min_workers, min(DEFAULT_EXTRACTION_WORKERS, max_workers))


def _get_container_memory_limit_mb() -> float | None:
    """Get container memory limit if running in Docker/cgroup.

    Returns:
        Memory limit in MB, or None if not in container or unlimited
    """
    # Try cgroup v2 first
    cgroup_v2_path = "/sys/fs/cgroup/memory.max"
    # Then cgroup v1
    cgroup_v1_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

    for path in [cgroup_v2_path, cgroup_v1_path]:
        try:
            with Path(path).open() as f:
                value = f.read().strip()
                if value == "max":
                    # Unlimited
                    return None
                limit_bytes = int(value)
                # Sanity check - if limit is huge, probably not meaningful
                if limit_bytes > 1e15:  # > 1 PB
                    return None
                return limit_bytes / (1024 * 1024)
        except (FileNotFoundError, ValueError, PermissionError):
            continue

    return None


@dataclass(frozen=True, slots=True)
class ChunkBatch:
    """A batch of chunks ready for embedding."""

    doc_id: str
    doc_identifier: str
    chunks: list[dict[str, Any]]
    texts: list[str]

    def __post_init__(self) -> None:
        if len(self.chunks) != len(self.texts):
            raise ValueError(f"ChunkBatch invariant violated: {len(self.chunks)} chunks != {len(self.texts)} texts")


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Result of embedding a batch."""

    doc_id: str
    doc_identifier: str
    chunks: list[dict[str, Any]]
    embeddings: list[list[float]]
    success: bool
    error: str | None = None

    def __post_init__(self) -> None:
        if self.success:
            if self.error is not None:
                raise ValueError("EmbeddingResult invariant violated: success=True but error is set")
            if len(self.embeddings) != len(self.chunks):
                raise ValueError(
                    f"EmbeddingResult invariant violated: {len(self.embeddings)} embeddings != {len(self.chunks)} chunks"
                )
        else:
            if not self.error:
                raise ValueError("EmbeddingResult invariant violated: success=False but error is empty")


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Result of extracting and chunking a document.

    Distinguishes between success, intentional skip, and error conditions.
    """

    success: bool
    batch: ChunkBatch | None = None
    error: str | None = None
    skip_reason: str | None = None  # "no_text", "empty_content", "no_chunks"

    def __post_init__(self) -> None:
        if self.success:
            if self.error is not None:
                raise ValueError("ExtractionResult invariant violated: success=True but error is set")
            if (self.batch is None) == (self.skip_reason is None):
                raise ValueError("ExtractionResult invariant violated: success=True must have exactly one of batch/skip_reason")
        else:
            if not self.error:
                raise ValueError("ExtractionResult invariant violated: success=False but error is empty")
            if self.batch is not None or self.skip_reason is not None:
                raise ValueError("ExtractionResult invariant violated: success=False cannot have batch or skip_reason")


async def _queue_put_with_shutdown(
    queue: asyncio.Queue,
    item: Any,
    *,
    downstream_stopped: asyncio.Event | None = None,
) -> None:
    """Put into a queue without hanging forever if downstream has stopped.

    This preserves backpressure (will wait for space) but periodically checks a
    downstream shutdown signal to avoid producer deadlocks when a consumer dies.
    """
    while True:
        if downstream_stopped and downstream_stopped.is_set():
            raise RuntimeError("Downstream consumer stopped")
        try:
            queue.put_nowait(item)
            return
        except asyncio.QueueFull:
            # Wait a bit and re-check downstream signal.
            if downstream_stopped:
                try:
                    await asyncio.wait_for(downstream_stopped.wait(), timeout=QUEUE_PUT_POLL_INTERVAL_SECONDS)
                except TimeoutError:
                    continue
            else:
                await asyncio.sleep(QUEUE_PUT_POLL_INTERVAL_SECONDS)


async def _best_effort(label: str, awaitable: Any) -> None:
    """Run best-effort async cleanup without swallowing fatal exceptions."""
    try:
        await awaitable
    except (SystemExit, KeyboardInterrupt, MemoryError):
        raise
    except Exception as exc:
        logger.debug("Best-effort %s failed: %s", label, exc, exc_info=True)


async def _update_document_status(
    document_repo: DocumentRepository,
    db_lock: asyncio.Lock,
    document_id: str,
    status: DocumentStatus,
    *,
    error_message: str | None = None,
    chunk_count: int | None = None,
) -> None:
    """Update status and commit immediately to make progress durable/visible."""
    async with db_lock:
        try:
            await document_repo.update_status(
                document_id,
                status,
                error_message=error_message,
                chunk_count=chunk_count,
            )
            await document_repo.session.commit()
        except Exception:
            # Keep the session usable for subsequent updates.
            await _best_effort(
                f"document status rollback for {document_id}",
                document_repo.session.rollback(),
            )
            raise


async def _incr_stat(stats: dict[str, int], stats_lock: asyncio.Lock, key: str, delta: int = 1) -> None:
    async with stats_lock:
        stats[key] += delta


async def extract_and_chunk_document(
    doc: Any,
    extract_fn: Callable,
    chunking_service: Any,
    collection: dict[str, Any],
    executor_pool: Any,
    new_doc_contents: dict[str, str],
) -> ExtractionResult:
    """Extract text and chunk a single document.

    This is CPU-bound work that can run in parallel.

    Returns:
        ExtractionResult with:
        - success=True, batch=ChunkBatch: Document successfully processed
        - success=True, skip_reason=str: Document intentionally skipped (empty, no text)
        - success=False, error=str: Document processing failed
    """
    doc_identifier = doc.file_path or doc.uri or f"doc:{doc.id}"
    logger.info("Extracting and chunking: %s", doc_identifier)

    try:
        # Check if we have pre-parsed content from connector
        combined_metadata: dict[str, Any] = {}
        if doc.id in new_doc_contents:
            combined_text = new_doc_contents[doc.id]
        else:
            # Extract from file
            loop = asyncio.get_running_loop()
            try:
                text_blocks = await asyncio.wait_for(
                    loop.run_in_executor(executor_pool, extract_fn, doc.file_path),
                    timeout=300,
                )
            except Exception as exc:
                logger.error("Extraction failed for %s: %s", doc_identifier, exc)
                return ExtractionResult(success=False, error=f"Extraction failed: {exc}")

            if not text_blocks:
                logger.warning("No text extracted from %s", doc_identifier)
                return ExtractionResult(success=True, skip_reason="no_text_extracted")

            combined_text = ""
            for text, metadata in text_blocks:
                if text.strip():
                    combined_text += text + "\n\n"
                    if metadata:
                        combined_metadata.update(metadata)

        if not combined_text.strip():
            logger.warning("No content for document %s", doc_identifier)
            return ExtractionResult(success=True, skip_reason="empty_content")

        # Chunk the text
        strategy = collection.get("chunking_strategy") or "recursive"
        chunking_config = collection.get("chunking_config") or {}

        chunks = await chunking_service.execute_ingestion_chunking(
            content=combined_text,
            strategy=strategy,
            config=chunking_config,
            metadata={**combined_metadata, "document_id": doc.id} if combined_metadata else {"document_id": doc.id},
        )

        if not chunks:
            logger.warning("No chunks created for %s", doc_identifier)
            return ExtractionResult(success=True, skip_reason="no_chunks_created")

        texts: list[str] = []
        for chunk in chunks:
            text = chunk.get("text") or chunk.get("content")
            if not isinstance(text, str):
                raise ValueError(f"Chunk missing text/content for {doc_identifier}")
            texts.append(text)

        logger.info("Created %s chunks for %s", len(chunks), doc_identifier)

        batch = ChunkBatch(
            doc_id=doc.id,
            doc_identifier=doc_identifier,
            chunks=chunks,
            texts=texts,
        )
        return ExtractionResult(success=True, batch=batch)

    except Exception as exc:
        logger.error("Failed to process %s: %s", doc_identifier, exc, exc_info=True)
        return ExtractionResult(success=False, error=f"Processing failed: {exc}")


async def extraction_worker(
    worker_id: int,
    doc_queue: asyncio.Queue,
    chunk_queue: asyncio.Queue,
    extract_fn: Callable,
    chunking_service: Any,
    collection: dict[str, Any],
    executor_pool: Any,
    new_doc_contents: dict[str, str],
    document_repo: DocumentRepository,
    stats: dict[str, int],
    stats_lock: asyncio.Lock,
    db_lock: asyncio.Lock,
    embedding_stopped: asyncio.Event,
) -> None:
    """Worker that pulls documents from queue, extracts and chunks them.

    Multiple instances of this worker run in parallel.
    """
    logger.info("Extraction worker %d started", worker_id)

    try:
        while True:
            doc = None
            try:
                doc = await doc_queue.get()

                if doc is None:  # Poison pill
                    doc_queue.task_done()
                    break

                result = await extract_and_chunk_document(
                    doc=doc,
                    extract_fn=extract_fn,
                    chunking_service=chunking_service,
                    collection=collection,
                    executor_pool=executor_pool,
                    new_doc_contents=new_doc_contents,
                )

                if not result.success:
                    # Processing failed - mark as FAILED with error message
                    await _update_document_status(
                        document_repo,
                        db_lock,
                        doc.id,
                        DocumentStatus.FAILED,
                        error_message=result.error[:500] if result.error else "Unknown error",
                    )
                    await _incr_stat(stats, stats_lock, "failed")
                elif result.skip_reason:
                    # Intentionally skipped - mark as COMPLETED with 0 chunks
                    await _update_document_status(
                        document_repo,
                        db_lock,
                        doc.id,
                        DocumentStatus.COMPLETED,
                        chunk_count=0,
                    )
                    await _incr_stat(stats, stats_lock, "skipped")
                else:
                    # Success - queue chunks for embedding (may backpressure)
                    await _queue_put_with_shutdown(chunk_queue, result.batch, downstream_stopped=embedding_stopped)

                doc_queue.task_done()

            except Exception as exc:
                logger.error("Extraction worker %d error: %s", worker_id, exc, exc_info=True)
                # CRITICAL: Update document status on exception to prevent stuck PROCESSING state
                if doc is not None:
                    try:
                        await _update_document_status(
                            document_repo,
                            db_lock,
                            doc.id,
                            DocumentStatus.FAILED,
                            error_message=f"Extraction worker error: {str(exc)[:500]}",
                        )
                        await _incr_stat(stats, stats_lock, "failed")
                    except Exception as status_exc:
                        logger.error(
                            "Failed to mark document %s as failed: %s",
                            doc.id if doc else "unknown",
                            status_exc,
                        )
                doc_queue.task_done()
    finally:
        # Always signal producer completion to the embedding worker (best-effort).
        await _best_effort(
            f"extraction worker {worker_id} poison pill",
            _queue_put_with_shutdown(chunk_queue, None, downstream_stopped=embedding_stopped),
        )

    logger.info("Extraction worker %d finished", worker_id)


async def embedding_worker(
    chunk_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    embedding_model: str,
    quantization: str,
    instruction: str | None,
    batch_size: int,
    num_producers: int,
    embedding_stopped: asyncio.Event,
) -> None:
    """Single worker that processes embedding requests sequentially.

    This ensures only one embedding request hits the GPU at a time.
    """
    logger.info("Embedding worker started")

    producers_done = 0
    pending_batches: list[ChunkBatch] = []
    pending_texts: list[str] = []

    async def flush_batch() -> None:
        """Send accumulated texts for embedding."""
        nonlocal pending_batches, pending_texts

        if not pending_texts:
            return

        logger.info("Embedding batch of %d texts from %d documents", len(pending_texts), len(pending_batches))

        try:
            vecpipe_url = _vecpipe_url("/embed")
            embed_request = {
                "texts": pending_texts,
                "model_name": embedding_model,
                "quantization": quantization,
                "instruction": instruction,
                "batch_size": batch_size,
                "mode": "document",
            }

            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(vecpipe_url, json=embed_request)
                response.raise_for_status()

                embed_response = response.json()
                all_embeddings = embed_response.get("embeddings")
                if not isinstance(all_embeddings, list):
                    raise ValueError(f"Invalid embedding response: embeddings={type(all_embeddings)}")
                if len(all_embeddings) != len(pending_texts):
                    raise ValueError(
                        f"Embedding response size mismatch: got {len(all_embeddings)} vectors, expected {len(pending_texts)}"
                    )

            # Distribute embeddings back to batches
            offset = 0
            for batch in pending_batches:
                batch_embeddings = all_embeddings[offset : offset + len(batch.texts)]
                offset += len(batch.texts)

                await result_queue.put(
                    EmbeddingResult(
                        doc_id=batch.doc_id,
                        doc_identifier=batch.doc_identifier,
                        chunks=batch.chunks,
                        embeddings=batch_embeddings,
                        success=True,
                    )
                )

        except httpx.RequestError as exc:
            error_message = f"Embedding request failed: {exc}"
            logger.error(error_message, exc_info=True)
            # Mark all batches as failed
            for batch in pending_batches:
                await result_queue.put(
                    EmbeddingResult(
                        doc_id=batch.doc_id,
                        doc_identifier=batch.doc_identifier,
                        chunks=batch.chunks,
                        embeddings=[],
                        success=False,
                        error=error_message,
                    )
                )
        except httpx.HTTPStatusError as exc:
            response = exc.response
            error_message = f"Embedding failed: {response.status_code} - {response.text}"
            logger.error(error_message, exc_info=True)
            # Mark all batches as failed
            for batch in pending_batches:
                await result_queue.put(
                    EmbeddingResult(
                        doc_id=batch.doc_id,
                        doc_identifier=batch.doc_identifier,
                        chunks=batch.chunks,
                        embeddings=[],
                        success=False,
                        error=error_message,
                    )
                )
        except Exception as exc:
            error_message = str(exc)
            logger.error("Embedding batch failed: %s", error_message, exc_info=True)
            # Mark all batches as failed
            for batch in pending_batches:
                await result_queue.put(
                    EmbeddingResult(
                        doc_id=batch.doc_id,
                        doc_identifier=batch.doc_identifier,
                        chunks=batch.chunks,
                        embeddings=[],
                        success=False,
                        error=error_message,
                    )
                )

        pending_batches = []
        pending_texts = []

    try:
        while True:
            try:
                # Use timeout to periodically flush partial batches
                try:
                    batch = await asyncio.wait_for(chunk_queue.get(), timeout=2.0)
                except TimeoutError:
                    # Flush any pending batches
                    if pending_texts:
                        await flush_batch()
                    continue

                if batch is None:  # Poison pill from a producer
                    producers_done += 1
                    chunk_queue.task_done()

                    if producers_done >= num_producers:
                        # All producers done, flush remaining and exit
                        if pending_texts:
                            await flush_batch()
                        break
                    continue

                # Accumulate batch
                pending_batches.append(batch)
                pending_texts.extend(batch.texts)

                chunk_queue.task_done()

                # Flush if we have enough texts
                if len(pending_texts) >= EMBEDDING_BATCH_SIZE:
                    await flush_batch()

            except Exception as exc:
                logger.error("Embedding worker error: %s", exc, exc_info=True)
    finally:
        embedding_stopped.set()
        # Ensure any pending documents don't remain stuck.
        if pending_batches:
            for batch in pending_batches:
                await _best_effort(
                    f"embedding failure enqueue for {batch.doc_identifier}",
                    asyncio.shield(
                        result_queue.put(
                            EmbeddingResult(
                                doc_id=batch.doc_id,
                                doc_identifier=batch.doc_identifier,
                                chunks=batch.chunks,
                                embeddings=[],
                                success=False,
                                error="Embedding worker terminated before flush",
                            )
                        )
                    ),
                )
        # Signal completion to result processor, even on final-flush failure.
        await _best_effort(
            "embedding completion signal",
            asyncio.shield(result_queue.put(None)),
        )
        logger.info("Embedding worker finished")


async def result_processor(
    result_queue: asyncio.Queue,
    qdrant_collection_name: str,
    collection_id: str,
    document_repo: DocumentRepository,
    stats: dict[str, int],
    stats_lock: asyncio.Lock,
    db_lock: asyncio.Lock,
    updater: Any,
) -> None:
    """Process embedding results and upsert to Qdrant."""
    logger.info("Result processor started")

    while True:
        result = await result_queue.get()

        if result is None:  # Done signal
            result_queue.task_done()
            break

        try:
            if not result.success:
                await _update_document_status(
                    document_repo,
                    db_lock,
                    result.doc_id,
                    DocumentStatus.FAILED,
                    error_message=result.error,
                )
                await _incr_stat(stats, stats_lock, "failed")
                continue

            # Build points
            points = []
            for i, chunk in enumerate(result.chunks):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=result.embeddings[i],
                    payload={
                        "collection_id": collection_id,
                        "doc_id": result.doc_id,
                        "chunk_id": chunk["chunk_id"],
                        "path": result.doc_identifier,
                        "content": chunk.get("text") or chunk.get("content") or "",
                        "metadata": chunk.get("metadata", {}),
                    },
                )
                points.append(point)

            # Upsert to Qdrant in batches
            for batch_start in range(0, len(points), VECTOR_UPLOAD_BATCH_SIZE):
                batch_end = min(batch_start + VECTOR_UPLOAD_BATCH_SIZE, len(points))
                batch_points = points[batch_start:batch_end]

                points_data = [{"id": point.id, "vector": point.vector, "payload": point.payload} for point in batch_points]

                upsert_request = {
                    "collection_name": qdrant_collection_name,
                    "points": points_data,
                    "wait": True,
                }

                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.post(_vecpipe_url("/upsert"), json=upsert_request)
                        response.raise_for_status()
                except httpx.RequestError as exc:
                    raise RuntimeError(f"Upsert request failed: {exc}") from exc
                except httpx.HTTPStatusError as exc:
                    response = exc.response
                    raise RuntimeError(f"Upsert failed: {response.status_code} - {response.text}") from exc

            # Update document status
            await _update_document_status(
                document_repo,
                db_lock,
                result.doc_id,
                DocumentStatus.COMPLETED,
                chunk_count=len(result.chunks),
            )

            await _incr_stat(stats, stats_lock, "processed")
            await _incr_stat(stats, stats_lock, "vectors", delta=len(points))

            logger.info(
                "Completed %s: %d chunks, %d vectors",
                result.doc_identifier,
                len(result.chunks),
                len(points),
            )

            # Send progress update
            if updater and stats["processed"] % 10 == 0:
                total = stats["processed"] + stats["failed"] + stats["skipped"]
                await updater.send_update(
                    "processing_progress",
                    {
                        "processed": stats["processed"],
                        "failed": stats["failed"],
                        "skipped": stats["skipped"],
                        "vectors_created": stats["vectors"],
                        "total_processed": total,
                    },
                )

        except Exception as exc:
            logger.error("Result processing failed for %s: %s", result.doc_identifier, exc, exc_info=True)
            await _best_effort(
                f"result status update for {result.doc_identifier}",
                _update_document_status(
                    document_repo,
                    db_lock,
                    result.doc_id,
                    DocumentStatus.FAILED,
                    error_message=str(exc)[:500],
                ),
            )
            await _incr_stat(stats, stats_lock, "failed")
        finally:
            result_queue.task_done()

    logger.info("Result processor finished")


async def process_documents_parallel(
    documents: list[Any],
    extract_fn: Callable,
    chunking_service: Any,
    collection: dict[str, Any],
    executor_pool: Any,
    new_doc_contents: dict[str, str],
    document_repo: DocumentRepository,
    qdrant_collection_name: str,
    embedding_model: str,
    quantization: str,
    instruction: str | None,
    batch_size: int,
    updater: Any = None,
    num_extraction_workers: int | None = None,
    max_extraction_workers: int | None = None,
) -> dict[str, int]:
    """Process documents with parallel extraction/chunking and sequential embedding.

    Args:
        documents: List of document objects to process
        extract_fn: Function to extract text from documents
        chunking_service: Service for chunking text
        collection: Collection configuration dict
        executor_pool: Thread pool for CPU-bound extraction
        new_doc_contents: Pre-parsed content map
        document_repo: Document repository for status updates
        qdrant_collection_name: Qdrant collection name
        embedding_model: Embedding model name
        quantization: Quantization setting
        instruction: Optional embedding instruction
        batch_size: Embedding batch size
        updater: Optional progress updater
        num_extraction_workers: Number of parallel extraction workers.
            If None or 0, dynamically calculated based on available resources.
        max_extraction_workers: Maximum workers cap for dynamic calculation.

    Returns:
        Dict with processing statistics
    """
    # Calculate workers dynamically if not specified
    if not num_extraction_workers:
        num_extraction_workers = calculate_optimal_workers(max_workers=max_extraction_workers)
    elif max_extraction_workers:
        # Apply cap even if explicit workers specified
        num_extraction_workers = min(num_extraction_workers, max_extraction_workers)

    logger.info(
        "Starting parallel ingestion: %d documents, %d extraction workers",
        len(documents),
        num_extraction_workers,
    )

    # Create queues
    doc_queue: asyncio.Queue = asyncio.Queue()
    chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    result_queue: asyncio.Queue = asyncio.Queue()

    embedding_stopped = asyncio.Event()
    stats_lock = asyncio.Lock()
    db_lock = asyncio.Lock()

    # Shared statistics
    stats = {"processed": 0, "failed": 0, "skipped": 0, "vectors": 0}

    # Start workers
    extraction_tasks = [
        asyncio.create_task(
            extraction_worker(
                worker_id=i,
                doc_queue=doc_queue,
                chunk_queue=chunk_queue,
                extract_fn=extract_fn,
                chunking_service=chunking_service,
                collection=collection,
                executor_pool=executor_pool,
                new_doc_contents=new_doc_contents,
                document_repo=document_repo,
                stats=stats,
                stats_lock=stats_lock,
                db_lock=db_lock,
                embedding_stopped=embedding_stopped,
            )
        )
        for i in range(num_extraction_workers)
    ]

    embedding_task = asyncio.create_task(
        embedding_worker(
            chunk_queue=chunk_queue,
            result_queue=result_queue,
            embedding_model=embedding_model,
            quantization=quantization,
            instruction=instruction,
            batch_size=batch_size,
            num_producers=num_extraction_workers,
            embedding_stopped=embedding_stopped,
        )
    )

    result_task = asyncio.create_task(
        result_processor(
            result_queue=result_queue,
            qdrant_collection_name=qdrant_collection_name,
            collection_id=collection["id"],
            document_repo=document_repo,
            stats=stats,
            stats_lock=stats_lock,
            db_lock=db_lock,
            updater=updater,
        )
    )

    # Queue all documents (workers are already waiting)
    for doc in documents:
        await doc_queue.put(doc)

    # Add poison pills for extraction workers after all docs are queued
    for _ in range(num_extraction_workers):
        await doc_queue.put(None)

    # Wait for all workers to complete
    await asyncio.gather(*extraction_tasks)
    await embedding_task
    await result_task

    logger.info(
        "Parallel ingestion complete: %d processed, %d failed, %d skipped, %d vectors",
        stats["processed"],
        stats["failed"],
        stats["skipped"],
        stats["vectors"],
    )

    return stats
