#!/usr/bin/env python3
"""
Celery tasks for chunking operations with comprehensive error handling.

This module provides robust task execution for document chunking with:
- Automatic retry with exponential backoff
- Resource limit enforcement and monitoring
- Idempotency through operation fingerprinting
- Graceful shutdown on soft time limits
- Dead letter queue for unrecoverable failures
- Real-time progress tracking via Redis streams
- Circuit breaker pattern for external services
"""

import asyncio
import gc
import json
import logging
import signal
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any, cast

import psutil
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from prometheus_client import Counter, Gauge, Histogram
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database import pg_connection_manager
from packages.shared.database.database import AsyncSessionLocal
from packages.shared.database.models import CollectionStatus, DocumentStatus, OperationStatus, OperationType
from packages.shared.database.repositories.chunk_repository import ChunkRepository
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.webui.api.chunking_exceptions import (
    ChunkingDependencyError,
    ChunkingMemoryError,
    ChunkingPartialFailureError,
    ChunkingStrategyError,
    ChunkingTimeoutError,
)
from packages.webui.celery_app import celery_app
from packages.webui.middleware.correlation import get_or_generate_correlation_id
from packages.webui.services.chunking.container import (
    build_chunking_operation_manager,
    resolve_celery_chunking_service,
)
from packages.webui.services.chunking.operation_manager import ChunkingOperationManager
from packages.webui.services.chunking_error_handler import ChunkingErrorHandler
from packages.webui.services.factory import get_redis_manager
from packages.webui.services.progress_manager import ProgressPayload, ProgressSendResult, ProgressUpdateManager
from packages.webui.services.type_guards import ensure_sync_redis
from packages.webui.tasks import executor as chunk_executor
from packages.webui.tasks import extract_and_serialize_thread_safe
from packages.webui.utils.error_classifier import get_default_chunking_error_classifier

logger = logging.getLogger(__name__)


def get_redis_client() -> Redis:
    """Get sync Redis client instance for Celery tasks."""
    redis_manager = get_redis_manager()
    client = redis_manager.sync_client
    return ensure_sync_redis(client)


_progress_update_manager: ProgressUpdateManager | None = None


def get_progress_update_manager() -> ProgressUpdateManager:
    """Return a cached progress update manager for chunking tasks."""

    global _progress_update_manager
    if _progress_update_manager is None:
        manager = ProgressUpdateManager(
            sync_redis=get_redis_client(),
            default_stream_template="stream:chunking:{operation_id}",
            default_ttl=None,
            default_maxlen=1000,
            logger_=logger.getChild("progress"),
        )
        _progress_update_manager = manager
    return _progress_update_manager


# Metrics
chunking_tasks_started = Counter(
    "chunking_tasks_started_total",
    "Total number of chunking tasks started",
    ["operation_type"],
)

chunking_tasks_completed = Counter(
    "chunking_tasks_completed_total",
    "Total number of chunking tasks completed",
    ["operation_type", "status"],
)

chunking_tasks_failed = Counter(
    "chunking_tasks_failed_total",
    "Total number of chunking tasks failed",
    ["operation_type", "error_type"],
)

chunking_task_duration = Histogram(
    "chunking_task_duration_seconds",
    "Duration of chunking tasks in seconds",
    ["operation_type"],
)

chunking_operation_memory_usage = Gauge(
    "chunking_operation_memory_usage_bytes",
    "Current memory usage by specific chunking operation",
    ["operation_id"],
)

chunking_active_operations = Gauge(
    "chunking_active_operations",
    "Number of currently active chunking operations",
)

# Task configuration
CHUNKING_SOFT_TIME_LIMIT = 3600  # 1 hour
CHUNKING_HARD_TIME_LIMIT = 7200  # 2 hours
CHUNKING_MAX_RETRIES = 3
CHUNKING_RETRY_BACKOFF = True
CHUNKING_RETRY_BACKOFF_MAX = 600  # 10 minutes max backoff
CHUNKING_MEMORY_LIMIT_GB = 4
CHUNKING_CPU_TIME_LIMIT = 1800  # 30 minutes of CPU time


class ChunkingTask(Task):
    """Base task class for chunking operations with enhanced error handling.

    This task class provides:
    - Automatic retry for specific exceptions
    - Resource tracking and limits
    - Dead letter queue for failed tasks
    - Correlation ID propagation
    - Comprehensive error reporting
    """

    # Task configuration
    autoretry_for = (
        ChunkingMemoryError,
        ChunkingTimeoutError,
        ChunkingStrategyError,
        ChunkingDependencyError,
        ConnectionError,
        TimeoutError,
    )
    retry_backoff = CHUNKING_RETRY_BACKOFF
    retry_backoff_max = CHUNKING_RETRY_BACKOFF_MAX
    max_retries = CHUNKING_MAX_RETRIES
    acks_late = True
    track_started = True
    reject_on_worker_lost = True

    def __init__(self) -> None:
        """Initialize the chunking task."""
        super().__init__()
        self._shutdown_handler_registered = False
        self._graceful_shutdown = False
        self._redis_client: Redis | None = None
        self._error_handler: ChunkingErrorHandler | None = None
        self._error_classifier = get_default_chunking_error_classifier()
        self._operation_manager: ChunkingOperationManager | None = None

    def before_start(self, task_id: str, args: tuple, kwargs: dict) -> None:
        """Set up task before execution starts.

        Args:
            task_id: Celery task ID
            args: Task positional arguments
            kwargs: Task keyword arguments
        """
        # Register shutdown handler for graceful termination
        if not self._shutdown_handler_registered:
            signal.signal(signal.SIGTERM, self._handle_shutdown)
            self._shutdown_handler_registered = True

        # Initialize Redis client and error handler
        try:
            # Get sync Redis client for Celery
            self._redis_client = get_redis_client()
            # Initialize error handler with async Redis client (None for sync context)
            # ChunkingErrorHandler can work without Redis for basic error handling
            self._error_handler = ChunkingErrorHandler(redis_client=None)
        except Exception as e:
            logger.error(f"Failed to initialize task resources: {e}")

        # Ensure operation manager is ready with current dependencies
        self._ensure_operation_manager()

        # Extract operation ID from args
        operation_id = args[0] if args else kwargs.get("operation_id", "unknown")

        # Track task start
        chunking_tasks_started.labels(operation_type="chunking").inc()
        chunking_active_operations.inc()

        # Log task start with correlation ID
        correlation_id = kwargs.get("correlation_id") or get_or_generate_correlation_id()
        logger.info(
            f"Starting chunking task {task_id}",
            extra={
                "task_id": task_id,
                "operation_id": operation_id,
                "correlation_id": correlation_id,
            },
        )

    def _ensure_operation_manager(self) -> ChunkingOperationManager:
        """Initialise or return the operation manager bound to this task."""

        if self._operation_manager is None:
            self._operation_manager = build_chunking_operation_manager(
                redis_client=self._redis_client,
                error_handler=self._error_handler or ChunkingErrorHandler(redis_client=None),
                error_classifier=self._error_classifier,
                logger_=logger.getChild("operation_manager"),
                expected_circuit_breaker_exceptions=(ChunkingDependencyError,),
                memory_usage_gauge=chunking_operation_memory_usage,
            )
        return self._operation_manager

    def on_success(self, retval: Any, task_id: str, args: tuple, kwargs: dict) -> None:
        """Handle successful task completion.

        Args:
            retval: Task return value
            task_id: Celery task ID
            args: Task positional arguments
            kwargs: Task keyword arguments
        """
        operation_id = args[0] if args else kwargs.get("operation_id", "unknown")

        # Track success metrics
        chunking_tasks_completed.labels(
            operation_type="chunking",
            status="success",
        ).inc()
        chunking_active_operations.dec()

        manager = self._ensure_operation_manager()
        manager.handle_success(task_id=task_id, operation_id=operation_id, result=retval)

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,  # noqa: ARG002
    ) -> None:
        """Handle task failure with dead letter queue.

        Args:
            exc: Exception that caused the failure
            task_id: Celery task ID
            args: Task positional arguments
            kwargs: Task keyword arguments
            einfo: Exception info
        """
        operation_id = args[0] if args else kwargs.get("operation_id", "unknown")
        correlation_id = kwargs.get("correlation_id") or get_or_generate_correlation_id()
        manager = self._ensure_operation_manager()
        error_type = manager.handle_failure(
            exc=exc,
            task_id=task_id,
            operation_id=operation_id,
            correlation_id=correlation_id,
            retry_count=self.request.retries,
            max_retries=self.max_retries,
            args=args,
            kwargs=kwargs,
        )

        chunking_tasks_failed.labels(
            operation_type="chunking",
            error_type=error_type,
        ).inc()
        chunking_active_operations.dec()

    def on_retry(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,  # noqa: ARG002
    ) -> None:
        """Handle task retry with state preservation.

        Args:
            exc: Exception that caused the retry
            task_id: Celery task ID
            args: Task positional arguments
            kwargs: Task keyword arguments
            einfo: Exception info
        """
        operation_id = args[0] if args else kwargs.get("operation_id", "unknown")
        correlation_id = kwargs.get("correlation_id") or get_or_generate_correlation_id()
        manager = self._ensure_operation_manager()
        manager.handle_retry(
            exc=exc,
            task_id=task_id,
            operation_id=operation_id,
            correlation_id=correlation_id,
            retry_count=self.request.retries,
        )

    def _handle_shutdown(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        """Handle graceful shutdown on SIGTERM.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info("Received shutdown signal, initiating graceful shutdown")
        self._graceful_shutdown = True


@celery_app.task(
    base=ChunkingTask,
    bind=True,
    name="webui.tasks.chunking.process_chunking_operation",
    soft_time_limit=CHUNKING_SOFT_TIME_LIMIT,
    time_limit=CHUNKING_HARD_TIME_LIMIT,
)
def process_chunking_operation(
    self: ChunkingTask,
    operation_id: str,
    correlation_id: str,
) -> dict[str, Any]:
    """Celery entrypoint that delegates to the manager-driven executor.

    The heavy lifting happens inside :func:`_execute_chunking_task`, which
    ensures that :class:`ChunkingOperationManager` owns retries, circuit
    breaker state, DLQ hand-offs, and resource monitoring for the task.
    """

    return _execute_chunking_task(self, operation_id, correlation_id)


def _execute_chunking_task(
    task: ChunkingTask,
    operation_id: str,
    correlation_id: str,
) -> dict[str, Any]:
    """Execute the chunking task synchronously for Celery and tests."""

    operation_manager = task._ensure_operation_manager()
    if not operation_manager.allow_execution():
        raise ChunkingDependencyError(
            detail="Circuit breaker is open - external service unavailable",
            correlation_id=correlation_id,
            dependency="chunking_service",
            operation_id=operation_id,
        )

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():  # pragma: no cover - defensive
                raise RuntimeError("event loop closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            _process_chunking_operation_async(
                operation_id=operation_id,
                correlation_id=correlation_id,
                celery_task=task,
            )
        )
    except SoftTimeLimitExceeded:
        logger.warning(
            "Soft time limit exceeded for operation %s",
            operation_id,
            extra={"operation_id": operation_id, "correlation_id": correlation_id},
        )
        _handle_soft_timeout_sync(operation_id, correlation_id, task)
        raise ChunkingTimeoutError(
            detail="Operation exceeded soft time limit",
            correlation_id=correlation_id,
            operation_id=operation_id,
            elapsed_time=CHUNKING_SOFT_TIME_LIMIT,
            timeout_limit=CHUNKING_SOFT_TIME_LIMIT,
        ) from None


def _collection_to_payload(collection: Any) -> dict[str, Any]:
    """Return a mapping with chunking configuration for the collection."""

    def _get(field: str, default: Any = None) -> Any:
        if isinstance(collection, dict):
            return collection.get(field, default)
        return getattr(collection, field, default)

    return {
        "id": _get("id"),
        "name": _get("name"),
        "chunking_strategy": _get("chunking_strategy"),
        "chunking_config": _get("chunking_config", {}) or {},
        "chunk_size": _get("chunk_size", 1000),
        "chunk_overlap": _get("chunk_overlap", 200),
        "embedding_model": _get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
        "quantization": _get("quantization", "float16"),
        "vector_store_name": _get("vector_store_name") or _get("vector_collection_id"),
    }


def _extract_document_ids(operation: Any) -> list[str]:
    """Extract document identifiers from operation config/meta if present."""

    containers: list[dict[str, Any]] = []
    for candidate in (getattr(operation, "config", None), getattr(operation, "meta", None)):
        if isinstance(candidate, dict):
            containers.append(candidate)

    candidates: list[str] = []
    keys = ("document_ids", "documents", "document_uuids", "pending_document_ids")
    for container in containers:
        for key in keys:
            value = container.get(key)
            if not value:
                continue
            if isinstance(value, str):
                candidates.append(value)
            elif isinstance(value, Iterable):
                for item in value:
                    document_id: str | None = None
                    if isinstance(item, dict):
                        document_id = item.get("id") or item.get("document_id")
                    elif item:
                        document_id = str(item)
                    if document_id:
                        candidates.append(document_id)

    seen: set[str] = set()
    unique_ids: list[str] = []
    for document_id in candidates:
        if document_id not in seen:
            seen.add(document_id)
            unique_ids.append(document_id)
    return unique_ids


def _normalize_document_status(value: Any) -> DocumentStatus | None:
    """Return DocumentStatus enum when possible."""

    if isinstance(value, DocumentStatus):
        return value
    if isinstance(value, str):
        try:
            return DocumentStatus(value.lower())
        except ValueError:
            return None
    return None


def _should_process_document(document: Any) -> bool:
    """Return True when the document should be chunked."""

    status = _normalize_document_status(getattr(document, "status", None))
    if status in {DocumentStatus.DELETED}:
        return False
    chunk_count = getattr(document, "chunk_count", 0) or 0
    if status == DocumentStatus.COMPLETED and chunk_count > 0:
        return False
    return True


def _combine_text_blocks(blocks: Sequence[tuple[str, dict[str, Any]]]) -> tuple[str, dict[str, Any]]:
    """Combine extracted text blocks into a single payload."""

    combined_text_parts: list[str] = []
    combined_metadata: dict[str, Any] = {}
    for text, metadata in blocks:
        if isinstance(text, str) and text.strip():
            combined_text_parts.append(text)
        if isinstance(metadata, dict):
            combined_metadata.update(metadata)
    return "\n\n".join(combined_text_parts).strip(), combined_metadata


def _build_chunk_rows(
    collection_id: str,
    document_id: str,
    chunks: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Transform chunking results into database rows."""

    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        content = chunk.get("text") or chunk.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        metadata = chunk.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {"value": metadata}
        rows.append(
            {
                "collection_id": collection_id,
                "document_id": document_id,
                "chunk_index": chunk.get("chunk_index", index),
                "content": content,
                "start_offset": chunk.get("start_offset"),
                "end_offset": chunk.get("end_offset"),
                "token_count": chunk.get("token_count"),
                "metadata": metadata,
            }
        )
    return rows


async def _resolve_documents_for_operation(
    operation: Any,
    document_repo: DocumentRepository,
) -> list[Any]:
    """Return documents relevant to the chunking operation."""

    document_ids = _extract_document_ids(operation)
    documents: list[Any] = []
    if document_ids:
        for document_id in document_ids:
            doc = await document_repo.get_by_id(document_id)
            if doc and _should_process_document(doc):
                documents.append(doc)
    else:
        docs, _ = await document_repo.list_by_collection(
            operation.collection_id,
            status=None,
            limit=10_000,
        )
        for doc in docs:
            if _should_process_document(doc):
                documents.append(doc)
    return documents


async def _process_document_chunking(
    *,
    chunking_service: Any,
    chunk_repo: ChunkRepository,
    document_repo: DocumentRepository,
    collection_payload: dict[str, Any],
    collection_id: str,
    document: Any,
    correlation_id: str,
) -> tuple[int, dict[str, Any] | None]:
    """Chunk a single document and persist the resulting rows."""

    loop = asyncio.get_running_loop()
    file_path = getattr(document, "file_path", "")
    text_blocks = await loop.run_in_executor(chunk_executor, extract_and_serialize_thread_safe, file_path)
    if not text_blocks:
        raise ValueError(f"No text content extracted for document {getattr(document, 'id', file_path)}")

    combined_text, metadata = _combine_text_blocks(text_blocks)
    if not combined_text:
        raise ValueError(f"Extracted content empty for document {getattr(document, 'id', file_path)}")

    file_type = file_path.rsplit(".", 1)[-1] if "." in file_path else None
    chunking_result = await chunking_service.execute_ingestion_chunking(
        text=combined_text,
        document_id=document.id,
        collection=collection_payload,
        metadata=metadata,
        file_type=file_type,
    )

    chunks = chunking_result.get("chunks") or []
    if not chunks:
        raise ValueError(f"No chunks produced for document {getattr(document, 'id', file_path)}")

    chunk_rows = _build_chunk_rows(collection_id, document.id, chunks)
    if not chunk_rows:
        raise ValueError(f"No chunk rows generated for document {getattr(document, 'id', file_path)}")

    await chunk_repo.create_chunks_bulk(chunk_rows)
    await document_repo.update_status(
        document.id,
        DocumentStatus.COMPLETED,
        chunk_count=len(chunk_rows),
    )

    logger.info(
        "Chunked document %s with %s chunks",
        getattr(document, "id", "unknown"),
        len(chunk_rows),
        extra={"correlation_id": correlation_id},
    )

    return len(chunk_rows), chunking_result.get("stats")


async def _process_chunking_operation_async(
    *,
    operation_id: str,
    correlation_id: str,
    celery_task: ChunkingTask,
) -> dict[str, Any]:
    """Async implementation for chunking operations executed by Celery."""

    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    cpu_times = process.cpu_times()
    initial_cpu_time = cpu_times.user + cpu_times.system

    chunking_operation_memory_usage.labels(operation_id=operation_id).set(initial_memory)

    redis_client = get_redis_client()
    manager = celery_task._ensure_operation_manager()

    documents_processed = 0
    total_documents = 0
    total_chunks = 0
    failed_documents: list[str] = []
    partial_failure = False

    try:
        if not pg_connection_manager._sessionmaker:
            await pg_connection_manager.initialize()

        session_factory = cast(Callable[[], AsyncSession], AsyncSessionLocal)
        async with session_factory() as db:
            operation_repo = OperationRepository(db)
            collection_repo = CollectionRepository(db)
            document_repo = DocumentRepository(db)
            chunk_repo = ChunkRepository(db)
            error_handler = celery_task._error_handler or ChunkingErrorHandler(redis_client=None)

            try:
                operation = await operation_repo.get_by_uuid(operation_id)
                if not operation:
                    raise ValueError(f"Operation {operation_id} not found")

                collection = await collection_repo.get_by_uuid(operation.collection_id)
                if not collection:
                    raise ValueError(f"Collection {operation.collection_id} not found for operation {operation_id}")

                task_id = getattr(getattr(celery_task, "request", None), "id", None) or str(uuid.uuid4())
                await operation_repo.set_task_id(operation_id, task_id)
                await operation_repo.update_status(
                    operation_id,
                    OperationStatus.PROCESSING,
                    started_at=datetime.now(UTC),
                )

                documents = await _resolve_documents_for_operation(operation, document_repo)
                total_documents = len(documents)

                collection_payload = _collection_to_payload(collection)
                collection_identifier = collection_payload.get("id") or getattr(collection, "id", None)
                if not collection_identifier:
                    raise ValueError("Collection identifier is required to persist chunks")
                collection_identifier = str(collection_identifier)

                await manager.check_resource_limits(
                    operation_id=operation_id,
                    correlation_id=correlation_id,
                )

                await _send_progress_update(
                    redis_client,
                    operation_id,
                    correlation_id,
                    0,
                    f"Preparing {total_documents} documents for chunking",
                )

                if total_documents == 0:
                    await operation_repo.update_status(
                        operation_id,
                        OperationStatus.COMPLETED,
                        completed_at=datetime.now(UTC),
                    )
                    await collection_repo.update_status(
                        collection_identifier,
                        CollectionStatus.READY,
                    )
                    await db.commit()
                    duration = time.time() - start_time
                    chunking_task_duration.labels(operation_type="chunking").observe(duration)
                    return {
                        "operation_id": operation_id,
                        "status": "success",
                        "chunks_created": 0,
                        "documents_processed": 0,
                        "documents_failed": 0,
                        "duration_seconds": duration,
                    }

                chunking_service = await resolve_celery_chunking_service(
                    db,
                    collection_repo=collection_repo,
                    document_repo=document_repo,
                )

                batch_size = max(1, await manager.calculate_batch_size())

                for batch_start in range(0, total_documents, batch_size):
                    batch = documents[batch_start : batch_start + batch_size]

                    await manager.monitor_resources(
                        process=process,
                        operation_id=operation_id,
                        initial_memory=initial_memory,
                        initial_cpu_time=initial_cpu_time,
                        correlation_id=correlation_id,
                    )

                    for document in batch:
                        if celery_task._graceful_shutdown:
                            partial_failure = True
                            break

                        try:
                            chunks_created, stats = await _process_document_chunking(
                                chunking_service=chunking_service,
                                chunk_repo=chunk_repo,
                                document_repo=document_repo,
                                collection_payload=collection_payload,
                                collection_id=collection_identifier,
                                document=document,
                                correlation_id=correlation_id,
                            )
                        except ChunkingPartialFailureError as exc:
                            partial_failure = True
                            failed_documents.extend(exc.failed_documents)
                            documents_processed += exc.total_documents - len(exc.failed_documents)
                            total_chunks += exc.successful_chunks
                        except Exception as exc:  # noqa: BLE001
                            partial_failure = True
                            document_id = str(getattr(document, "id", "unknown"))
                            failed_documents.append(document_id)
                            await document_repo.update_status(
                                document_id,
                                DocumentStatus.FAILED,
                                error_message=str(exc),
                            )
                            logger.exception(
                                "Failed to chunk document %s in operation %s",
                                document_id,
                                operation_id,
                                extra={"correlation_id": correlation_id},
                                exc_info=exc,
                            )
                        else:
                            documents_processed += 1
                            total_chunks += chunks_created
                            if stats:
                                logger.debug(
                                    "Chunked document %s with stats %s",
                                    getattr(document, "id", "unknown"),
                                    stats,
                                    extra={"correlation_id": correlation_id},
                                )

                        progress = int((documents_processed / total_documents) * 100)
                        await _send_progress_update(
                            redis_client,
                            operation_id,
                            correlation_id,
                            progress,
                            f"Processed {documents_processed}/{total_documents} documents",
                        )
                        await db.flush()
                        gc.collect()

                    await db.commit()

                    if celery_task._graceful_shutdown:
                        break

                operation.meta = {
                    "chunks_created": total_chunks,
                    "documents_processed": documents_processed,
                    "documents_failed": len(failed_documents),
                    "failed_documents": failed_documents,
                    "partial_failure": partial_failure or bool(failed_documents) or celery_task._graceful_shutdown,
                    "completed_at": datetime.now(UTC).isoformat(),
                }
                await operation_repo.update_status(
                    operation_id,
                    OperationStatus.COMPLETED,
                    completed_at=datetime.now(UTC),
                )
                await collection_repo.update_status(
                    collection_identifier,
                    CollectionStatus.READY,
                )
                await db.commit()
            except Exception as exc:  # pragma: no cover - exercised in integration
                await db.rollback()
                logger.exception(
                    "Chunking operation %s failed",
                    operation_id,
                    extra={
                        "operation_id": operation_id,
                        "correlation_id": correlation_id,
                        "documents_processed": documents_processed,
                    },
                )

                failure_details = {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "failed_at": datetime.now(UTC).isoformat(),
                    "documents_processed": documents_processed,
                    "documents_failed": len(failed_documents),
                    "failed_documents": failed_documents,
                    "chunks_created": total_chunks,
                }

                if redis_client:
                    try:
                        redis_client.hset(
                            f"operation:{operation_id}",
                            mapping={
                                key: (
                                    value
                                    if isinstance(value, str | bytes | int | float)
                                    else json.dumps(value, default=str)
                                )
                                for key, value in {
                                    **failure_details,
                                    "status": "failed",
                                }.items()
                            },
                        )
                    except Exception as redis_error:  # pragma: no cover - defensive
                        logger.warning(
                            "Failed to record Redis failure state for %s: %s",
                            operation_id,
                            redis_error,
                            extra={"correlation_id": correlation_id},
                        )

                try:
                    failed_operation = await operation_repo.update_status(
                        operation_id,
                        OperationStatus.FAILED,
                        error_message=str(exc),
                        completed_at=datetime.now(UTC),
                    )
                    # Merge with existing metadata to preserve prior context.
                    existing_meta = dict(getattr(failed_operation, "meta", {}) or {})
                    existing_meta.update(failure_details)
                    existing_meta["partial_failure"] = True
                    failed_operation.meta = existing_meta
                    await db.flush()
                    await db.commit()
                except Exception as status_error:  # pragma: no cover - defensive
                    await db.rollback()
                    logger.exception(
                        "Failed to persist failure status for %s: %s",
                        operation_id,
                        status_error,
                        extra={"correlation_id": correlation_id},
                    )

                try:
                    progress = int((documents_processed / total_documents) * 100) if total_documents else 0
                    await _send_progress_update(
                        redis_client,
                        operation_id,
                        correlation_id,
                        progress,
                        "Chunking operation failed",
                    )
                except Exception as progress_error:  # pragma: no cover - defensive
                    logger.debug(
                        "Unable to send failure progress update for %s: %s",
                        operation_id,
                        progress_error,
                    )

                try:
                    cleanup_strategy = "save_partial" if total_chunks > 0 else "rollback"
                    await error_handler.cleanup_failed_operation(
                        operation_id=operation_id,
                        partial_results=None,
                        cleanup_strategy=cleanup_strategy,
                    )
                except Exception as cleanup_error:  # pragma: no cover - defensive
                    logger.warning(
                        "Cleanup failed for operation %s: %s",
                        operation_id,
                        cleanup_error,
                        exc_info=True,
                    )

                raise
    finally:
        chunking_operation_memory_usage.labels(operation_id=operation_id).set(0)

    await _send_progress_update(
        redis_client,
        operation_id,
        correlation_id,
        100,
        f"Chunking operation processed {documents_processed}/{total_documents} documents",
    )

    duration = time.time() - start_time
    chunking_task_duration.labels(operation_type="chunking").observe(duration)

    status = "success"
    if partial_failure or failed_documents or celery_task._graceful_shutdown:
        status = "partial_success"

    return {
        "operation_id": operation_id,
        "status": status,
        "chunks_created": total_chunks,
        "documents_processed": documents_processed,
        "documents_failed": len(failed_documents),
        "duration_seconds": duration,
    }


def _handle_soft_timeout_sync(
    operation_id: str,
    correlation_id: str,
    celery_task: ChunkingTask,
) -> None:
    """Handle soft timeout by saving partial results (sync version).

    Args:
        operation_id: Operation identifier
        correlation_id: Correlation ID
        celery_task: Celery task instance
    """
    try:
        redis_client = get_redis_client()

        # Save operation state for potential resume
        redis_client.hset(
            f"operation:{operation_id}:state",
            mapping={
                "soft_timeout": "true",
                "task_id": celery_task.request.id,
                "timestamp": datetime.now(UTC).isoformat(),
                "correlation_id": correlation_id,
            },
        )

        logger.info(
            f"Saved state for operation {operation_id} after soft timeout",
            extra={
                "operation_id": operation_id,
                "correlation_id": correlation_id,
            },
        )

    except Exception as e:
        logger.error(f"Failed to handle soft timeout: {e}", exc_info=e)


def _send_progress_update_sync(
    redis_client: Redis | None,
    operation_id: str,
    correlation_id: str,
    progress: int,
    message: str,
) -> None:
    """Send real-time progress update via Redis (sync version).

    Args:
        redis_client: Redis client
        operation_id: Operation identifier
        correlation_id: Correlation ID
        progress: Progress percentage (0-100)
        message: Progress message
    """
    if not redis_client:
        return

    manager = get_progress_update_manager()
    payload = ProgressPayload(
        operation_id=operation_id,
        correlation_id=correlation_id,
        progress=progress,
        message=message,
        extra={"timestamp": str(time.time())},
    )

    hash_mapping = {
        "progress": progress,
        "message": message,
        "updated_at": datetime.now(UTC).isoformat(),
        "correlation_id": correlation_id,
    }

    manager.send_sync_update(
        payload,
        stream_template="stream:chunking:{operation_id}",
        maxlen=1000,
        ttl=None,
        hash_key_template="operation:{operation_id}:progress",
        hash_mapping=hash_mapping,
        use_throttle=False,
        redis_client=redis_client,
    )


async def _handle_soft_timeout(
    operation_id: str,
    correlation_id: str,
    celery_task: ChunkingTask,
) -> None:
    """Handle soft timeout by saving partial results.

    Args:
        operation_id: Operation identifier
        correlation_id: Correlation ID
        celery_task: Celery task instance
    """
    try:
        # ChunkingErrorHandler expects async Redis, but we have sync Redis
        # For now, pass None as the handler can work without Redis
        error_handler = ChunkingErrorHandler(None)

        # Save operation state for potential resume
        await error_handler._save_operation_state(
            operation_id=operation_id,
            correlation_id=correlation_id,
            context={
                "soft_timeout": True,
                "task_id": celery_task.request.id,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            error_type=error_handler.classify_error(TimeoutError()),
        )

        logger.info(
            f"Saved state for operation {operation_id} after soft timeout",
            extra={
                "operation_id": operation_id,
                "correlation_id": correlation_id,
            },
        )

    except Exception as e:
        logger.error(f"Failed to handle soft timeout: {e}", exc_info=e)


async def _get_documents_for_operation(
    operation: Any,
    collection: Any,  # noqa: ARG001
    operation_repo: OperationRepository,  # noqa: ARG001
    db: AsyncSession,  # noqa: ARG001
) -> list[dict[str, Any]]:
    """Get documents to process based on operation type.

    Args:
        operation: Operation object
        collection: Collection object
        operation_repo: Operation repository
        db: Database session

    Returns:
        List of documents to process
    """
    # This is a simplified implementation
    # In production, this would fetch actual documents from the database
    # based on the operation type and metadata

    if operation.type == OperationType.INDEX:
        # For INDEX operations, process all documents in collection
        return operation.metadata.get("documents", [])  # type: ignore[no-any-return]
    if operation.type == OperationType.REINDEX:
        # For REINDEX, process all existing documents
        return operation.metadata.get("documents", [])  # type: ignore[no-any-return]
    if operation.type == OperationType.APPEND:
        # For APPEND, process only new documents
        return operation.metadata.get("new_documents", [])  # type: ignore[no-any-return]
    return []


async def _send_progress_update(
    redis_client: Redis | None,
    operation_id: str,
    correlation_id: str,
    progress: int,
    message: str,
) -> None:
    """Send real-time progress update via Redis streams.

    Args:
        redis_client: Redis client
        operation_id: Operation identifier
        correlation_id: Correlation ID
        progress: Progress percentage (0-100)
        message: Progress message
    """
    if not redis_client:
        return

    payload = ProgressPayload(
        operation_id=operation_id,
        correlation_id=correlation_id,
        progress=progress,
        message=message,
    )

    manager = get_progress_update_manager()
    loop = asyncio.get_running_loop()

    def _send() -> ProgressSendResult:
        return manager.send_sync_update(
            payload,
            stream_template="chunking:progress:{operation_id}",
            ttl=3600,
            maxlen=0,
            use_throttle=False,
            redis_client=redis_client,
        )

    result = await loop.run_in_executor(None, _send)
    if result is ProgressSendResult.FAILED:
        logger.error("Failed to send progress update for operation %s", operation_id)


# Retry helper for specific operations
@celery_app.task(
    bind=True,
    name="webui.tasks.chunking.retry_failed_documents",
    max_retries=1,
)
def retry_failed_documents(
    self: Task,  # noqa: ARG001
    operation_id: str,
    failed_documents: list[str],
    correlation_id: str,
) -> dict[str, Any]:
    """Retry processing for specific failed documents.

    Args:
        self: Task instance
        operation_id: Original operation ID
        failed_documents: List of document IDs to retry
        correlation_id: Correlation ID

    Returns:
        Retry operation results
    """
    # Create a new operation for the retry
    retry_operation_id = f"{operation_id}_retry_{uuid.uuid4().hex[:8]}"

    logger.info(
        f"Retrying {len(failed_documents)} failed documents",
        extra={
            "original_operation_id": operation_id,
            "retry_operation_id": retry_operation_id,
            "correlation_id": correlation_id,
            "document_count": len(failed_documents),
        },
    )

    # Process with the main task
    result = process_chunking_operation.apply_async(
        args=[retry_operation_id, correlation_id],
        task_id=retry_operation_id,
    )
    return result.get()  # type: ignore[no-any-return]


# Monitoring task for dead letter queue
@celery_app.task(
    name="webui.tasks.chunking.monitor_dead_letter_queue",
)
def monitor_dead_letter_queue() -> dict[str, Any]:
    """Monitor dead letter queue and alert on failures.

    Returns:
        Monitoring results
    """
    redis_client = get_redis_client()
    dlq_key = "chunking:dlq:tasks"

    try:
        dlq_size = redis_client.llen(dlq_key)

        if dlq_size > 0:
            # Get sample of failed tasks
            failed_tasks = [json.loads(task) for task in redis_client.lrange(dlq_key, 0, 9)]

            logger.warning(
                f"Dead letter queue contains {dlq_size} failed tasks",
                extra={
                    "dlq_size": dlq_size,
                    "sample_tasks": failed_tasks,
                },
            )

            return {
                "dlq_size": dlq_size,
                "sample_tasks": failed_tasks,
                "alert": dlq_size > 10,
            }

        return {
            "dlq_size": 0,
            "alert": False,
        }

    except Exception as e:
        logger.error(f"Failed to monitor DLQ: {e}", exc_info=e)
        return {
            "error": str(e),
            "alert": True,
        }
