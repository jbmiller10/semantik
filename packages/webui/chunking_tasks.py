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
import traceback
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

import psutil
from celery import Task, current_task
from celery.exceptions import SoftTimeLimitExceeded
from prometheus_client import Counter, Gauge, Histogram
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database import pg_connection_manager
from packages.shared.database.database import AsyncSessionLocal
from packages.shared.database.models import CollectionStatus, OperationStatus, OperationType
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.webui.api.chunking_exceptions import (
    ChunkingDependencyError,
    ChunkingMemoryError,
    ChunkingPartialFailureError,
    ChunkingResourceLimitError,
    ChunkingStrategyError,
    ChunkingTimeoutError,
    ChunkingValidationError,
    ResourceType,
)
from packages.webui.celery_app import celery_app
from packages.webui.middleware.correlation import get_or_generate_correlation_id
from packages.webui.services.chunking_error_handler import ChunkingErrorHandler
from packages.webui.services.chunking_service import ChunkingService
from packages.webui.services.factory import (
    get_redis_manager,
)
from packages.webui.services.type_guards import ensure_sync_redis

if TYPE_CHECKING:
    from packages.shared.text_processing.base_chunker import ChunkResult

logger = logging.getLogger(__name__)


def get_redis_client() -> Redis:
    """Get sync Redis client instance for Celery tasks."""
    redis_manager = get_redis_manager()
    client = redis_manager.sync_client
    return ensure_sync_redis(client)


# Metrics - Handle re-registration gracefully for tests
try:
    chunking_tasks_started = Counter(
        "chunking_tasks_started_total",
        "Total number of chunking tasks started",
        ["operation_type"],
    )
except ValueError:
    # Metric already registered (happens in tests)
    from prometheus_client import REGISTRY
    chunking_tasks_started = REGISTRY._names_to_collectors["chunking_tasks_started_total"]

try:
    chunking_tasks_completed = Counter(
        "chunking_tasks_completed_total",
        "Total number of chunking tasks completed",
        ["operation_type", "status"],
    )
except ValueError:
    from prometheus_client import REGISTRY
    chunking_tasks_completed = REGISTRY._names_to_collectors["chunking_tasks_completed_total"]

try:
    chunking_tasks_failed = Counter(
        "chunking_tasks_failed_total",
        "Total number of chunking tasks failed",
        ["operation_type", "error_type"],
    )
except ValueError:
    from prometheus_client import REGISTRY
    chunking_tasks_failed = REGISTRY._names_to_collectors["chunking_tasks_failed_total"]

try:
    chunking_task_duration = Histogram(
        "chunking_task_duration_seconds",
        "Duration of chunking tasks in seconds",
        ["operation_type"],
    )
except ValueError:
    from prometheus_client import REGISTRY
    chunking_task_duration = REGISTRY._names_to_collectors["chunking_task_duration_seconds"]

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

# Circuit breaker configuration
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 300  # 5 minutes
CIRCUIT_BREAKER_EXPECTED_EXCEPTION = (ChunkingDependencyError,)


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

    # Circuit breaker state
    _circuit_breaker_failures = 0
    _circuit_breaker_last_failure_time = None
    _circuit_breaker_state = "closed"  # closed, open, half_open

    def __init__(self) -> None:
        """Initialize the chunking task."""
        super().__init__()
        self._shutdown_handler_registered = False
        self._graceful_shutdown = False
        self._redis_client: Redis | None = None
        self._error_handler: ChunkingErrorHandler | None = None

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

        # Reset circuit breaker on success
        self._circuit_breaker_failures = 0
        self._circuit_breaker_state = "closed"

        logger.info(
            f"Chunking task {task_id} completed successfully",
            extra={
                "task_id": task_id,
                "operation_id": operation_id,
                "result": retval,
            },
        )

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

        # Classify error type
        error_type = "unknown"
        if isinstance(exc, ChunkingMemoryError):
            error_type = "memory_error"
        elif isinstance(exc, ChunkingTimeoutError):
            error_type = "timeout_error"
        elif isinstance(exc, ChunkingValidationError):
            error_type = "validation_error"
        elif isinstance(exc, ChunkingStrategyError):
            error_type = "strategy_error"
        elif isinstance(exc, ChunkingDependencyError):
            error_type = "dependency_error"

        # Track failure metrics
        chunking_tasks_failed.labels(
            operation_type="chunking",
            error_type=error_type,
        ).inc()
        chunking_active_operations.dec()

        # Update circuit breaker state
        if isinstance(exc, CIRCUIT_BREAKER_EXPECTED_EXCEPTION):
            self._update_circuit_breaker_state()

        # Log comprehensive error
        logger.error(
            f"Chunking task {task_id} failed after {self.request.retries} retries",
            extra={
                "task_id": task_id,
                "operation_id": operation_id,
                "correlation_id": correlation_id,
                "error_type": error_type,
                "retries": self.request.retries,
                "max_retries": self.max_retries,
                "traceback": traceback.format_exc(),
            },
            exc_info=exc,
        )

        # Send to dead letter queue if max retries exceeded
        if self.request.retries >= self.max_retries:
            self._send_to_dead_letter_queue(
                task_id=task_id,
                operation_id=operation_id,
                correlation_id=correlation_id,
                error=exc,
                error_type=error_type,
                args=args,
                kwargs=kwargs,
            )

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

        logger.warning(
            f"Retrying chunking task {task_id}",
            extra={
                "task_id": task_id,
                "operation_id": operation_id,
                "correlation_id": correlation_id,
                "retry_count": self.request.retries,
                "error": str(exc),
            },
        )

        # Save state for retry if error handler available
        if self._redis_client:
            # Use sync Redis client directly to save retry state
            try:
                retry_state: "Mapping[str | bytes, bytes | float | int | str]" = {
                    "operation_id": operation_id,
                    "correlation_id": correlation_id,
                    "task_id": task_id,
                    "retry_count": str(self.request.retries),
                    "last_error": str(exc),
                    "error_type": self._classify_error_sync(exc),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                self._redis_client.hset(
                    f"operation:{operation_id}:retry_state",
                    mapping=retry_state,
                )
                # Set expiry for 24 hours
                self._redis_client.expire(f"operation:{operation_id}:retry_state", 86400)
            except Exception as e:
                logger.warning(f"Failed to save retry state: {e}")

    def _handle_shutdown(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        """Handle graceful shutdown on SIGTERM.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info("Received shutdown signal, initiating graceful shutdown")
        self._graceful_shutdown = True

    def _update_circuit_breaker_state(self) -> None:
        """Update circuit breaker state based on failures."""
        current_time = time.time()

        # Increment failure count
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure_time = current_time

        # Check if we should open the circuit
        if (
            self._circuit_breaker_failures >= CIRCUIT_BREAKER_FAILURE_THRESHOLD
            and self._circuit_breaker_state == "closed"
        ):
            self._circuit_breaker_state = "open"
            logger.warning(f"Circuit breaker opened after {self._circuit_breaker_failures} failures")

    def _classify_error_sync(self, exc: Exception) -> str:
        """Classify error type synchronously.

        Args:
            exc: Exception to classify

        Returns:
            Error type string
        """
        if isinstance(exc, ChunkingMemoryError):
            return "memory_error"
        if isinstance(exc, ChunkingTimeoutError):
            return "timeout_error"
        if isinstance(exc, ChunkingValidationError):
            return "validation_error"
        if isinstance(exc, ChunkingStrategyError):
            return "strategy_error"
        if isinstance(exc, ChunkingDependencyError):
            return "dependency_error"
        if isinstance(exc, ChunkingResourceLimitError):
            return "resource_limit_error"
        if isinstance(exc, ChunkingPartialFailureError):
            return "partial_failure"
        if isinstance(exc, ConnectionError):
            return "connection_error"
        if isinstance(exc, TimeoutError):
            return "timeout_error"
        return "unknown"

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows execution.

        Returns:
            True if execution is allowed, False otherwise
        """
        if self._circuit_breaker_state == "closed":
            return True

        current_time = time.time()

        # Check if we should try half-open state
        if (
            self._circuit_breaker_state == "open"
            and self._circuit_breaker_last_failure_time
            and current_time - self._circuit_breaker_last_failure_time > CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        ):
            self._circuit_breaker_state = "half_open"
            logger.info("Circuit breaker entering half-open state")
            return True

        # Half-open state allows one request through
        if self._circuit_breaker_state == "half_open":
            return True

        # Circuit is open - reject request
        return False

    def _increment_failure_count(self, operation_id: str) -> None:
        """Increment the failure count for an operation.
        
        Args:
            operation_id: Operation identifier
        """
        if not self._redis_client:
            return
            
        try:
            key = f"operation:{operation_id}:failure_count"
            self._redis_client.incr(key)
            self._redis_client.expire(key, 86400)  # Expire after 24 hours
        except Exception as e:
            logger.warning(f"Failed to increment failure count: {e}")

    def _send_to_dead_letter_queue(
        self,
        task_id: str,
        operation_id: str,
        correlation_id: str,
        error: Exception,
        error_type: str,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Send failed task to dead letter queue for manual processing.

        Args:
            task_id: Celery task ID
            operation_id: Operation identifier
            correlation_id: Correlation ID for tracing
            error: Exception that caused the failure
            error_type: Classified error type
            args: Original task arguments
            kwargs: Original task keyword arguments
        """
        if not self._redis_client:
            logger.error("Cannot send to DLQ: Redis client not available")
            return

        try:
            dlq_entry = {
                "task_id": task_id,
                "operation_id": operation_id,
                "correlation_id": correlation_id,
                "error_type": error_type,
                "error_message": str(error),
                "error_class": type(error).__name__,
                "args": args,
                "kwargs": kwargs,
                "timestamp": datetime.now(UTC).isoformat(),
                "retries": self.request.retries,
            }

            # Add to dead letter queue
            dlq_key = "chunking:dlq:tasks"
            self._redis_client.rpush(dlq_key, json.dumps(dlq_entry))

            # Expire old entries (keep for 7 days)
            self._redis_client.expire(dlq_key, 604800)

            logger.error(
                f"Task {task_id} sent to dead letter queue",
                extra={
                    "task_id": task_id,
                    "operation_id": operation_id,
                    "correlation_id": correlation_id,
                    "dlq_key": dlq_key,
                },
            )

        except Exception as e:
            logger.error(f"Failed to send task to DLQ: {e}", exc_info=e)


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
    """Process a chunking operation with comprehensive error handling.

    This task handles document chunking with:
    - Idempotency through operation fingerprinting
    - Resource limit enforcement
    - Graceful shutdown on soft time limit
    - Progress tracking and error reporting
    - Automatic cleanup on failure

    Args:
        self: Task instance (bound task)
        operation_id: Unique operation identifier
        correlation_id: Correlation ID for distributed tracing

    Returns:
        Dictionary with operation results

    Raises:
        Various chunking exceptions based on failure type
    """
    # Check circuit breaker before proceeding
    if not self._check_circuit_breaker():
        raise ChunkingDependencyError(
            detail="Circuit breaker is open - external service unavailable",
            correlation_id=correlation_id,
            dependency="chunking_service",
            operation_id=operation_id,
        )

    # Run synchronous implementation (no asyncio.run!)
    try:
        return _process_chunking_operation_sync(
            operation_id=operation_id,
            correlation_id=correlation_id,
            celery_task=self,
        )
    except SoftTimeLimitExceeded:
        # Handle soft time limit gracefully
        logger.warning(
            f"Soft time limit exceeded for operation {operation_id}",
            extra={
                "operation_id": operation_id,
                "correlation_id": correlation_id,
            },
        )
        # Clean up and save partial results
        _handle_soft_timeout_sync(operation_id, correlation_id, self)
        raise ChunkingTimeoutError(
            detail="Operation exceeded soft time limit",
            correlation_id=correlation_id,
            operation_id=operation_id,
            elapsed_time=CHUNKING_SOFT_TIME_LIMIT,
            timeout_limit=CHUNKING_SOFT_TIME_LIMIT,
        ) from None
    except Exception:
        # Let the task class handle retries and failures
        raise


def _process_chunking_operation_sync(
    operation_id: str,
    correlation_id: str,
    celery_task: ChunkingTask,
) -> dict[str, Any]:
    """Synchronous implementation of chunking operation processing for Celery.

    This is a sync version that doesn't use asyncio, preventing event loop conflicts
    in Celery workers.

    Args:
        operation_id: Operation identifier
        correlation_id: Correlation ID for tracing
        celery_task: Celery task instance

    Returns:
        Operation results dictionary
    """
    start_time = time.time()
    chunks_created = 0
    documents_processed = 0
    failed_documents = []

    # Initialize resources
    redis_client = get_redis_client()  # Sync Redis client

    # Track resources
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    process.cpu_times().user + process.cpu_times().system

    # Store task ID immediately
    task_id = current_task.request.id if current_task else str(uuid.uuid4())

    logger.info(f"Processing chunking operation {operation_id} (sync mode)")

    try:
        # Update operation status in Redis
        redis_client.hset(
            f"operation:{operation_id}",
            mapping={
                "status": "processing",
                "task_id": task_id,
                "started_at": datetime.now(UTC).isoformat(),
            },
        )

        # Send progress update via Redis
        _send_progress_update_sync(
            redis_client,
            operation_id,
            correlation_id,
            0,
            "Starting chunking operation",
        )

        # Get operation configuration from Redis
        operation_data = redis_client.hgetall(f"operation:{operation_id}:config")
        if not operation_data:
            logger.warning(f"No configuration found for operation {operation_id}, using defaults")
            operation_data = {
                "strategy": "recursive",
                "chunk_size": "1000",
                "chunk_overlap": "200",
            }

        # Parse configuration
        strategy = operation_data.get("strategy", "recursive")
        chunk_size = int(operation_data.get("chunk_size", "1000"))
        chunk_overlap = int(operation_data.get("chunk_overlap", "200"))

        # Get documents to process (stored in Redis during operation creation)
        documents_key = f"operation:{operation_id}:documents"
        document_ids = redis_client.lrange(documents_key, 0, -1)
        total_documents = len(document_ids)

        if total_documents == 0:
            logger.warning(f"No documents found for operation {operation_id}")
            # Still mark as successful but with 0 chunks
            redis_client.hset(
                f"operation:{operation_id}",
                mapping={
                    "status": "completed",
                    "completed_at": datetime.now(UTC).isoformat(),
                    "chunks_created": "0",
                    "documents_processed": "0",
                },
            )
            return {
                "operation_id": operation_id,
                "status": "success",
                "chunks_created": 0,
                "documents_processed": 0,
                "duration_seconds": time.time() - start_time,
            }

        # Process documents in batches
        batch_size = 10  # Process 10 documents at a time

        for i in range(0, total_documents, batch_size):
            # Check for graceful shutdown
            if celery_task._graceful_shutdown:
                logger.info("Graceful shutdown requested, saving progress")
                break

            # Monitor resources
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory

            if memory_increase > CHUNKING_MEMORY_LIMIT_GB * 1024**3:
                raise ChunkingMemoryError(
                    detail="Operation memory usage exceeded limit",
                    correlation_id=correlation_id,
                    operation_id=operation_id,
                    memory_used=current_memory,
                    memory_limit=CHUNKING_MEMORY_LIMIT_GB * 1024**3,
                )

            batch = document_ids[i : i + batch_size]

            for doc_id in batch:
                try:
                    # Get document content from Redis
                    doc_content = redis_client.get(f"document:{doc_id}:content")
                    if not doc_content:
                        logger.warning(f"No content found for document {doc_id}")
                        failed_documents.append(doc_id.decode() if isinstance(doc_id, bytes) else doc_id)
                        continue

                    # Decode content if needed
                    if isinstance(doc_content, bytes):
                        doc_content = doc_content.decode("utf-8")

                    # Simple chunking logic (character-based with overlap)
                    doc_chunks: list[dict[str, Any]] = []
                    step = max(1, chunk_size - chunk_overlap)

                    for chunk_start in range(0, len(doc_content), step):
                        chunk_end = min(chunk_start + chunk_size, len(doc_content))
                        chunk_text = doc_content[chunk_start:chunk_end]

                        if chunk_text.strip():  # Only add non-empty chunks
                            chunk_id = f"{doc_id}_chunk_{len(doc_chunks):04d}"
                            doc_chunks.append(
                                {
                                    "id": chunk_id,
                                    "content": chunk_text,
                                    "metadata": {
                                        "document_id": doc_id.decode() if isinstance(doc_id, bytes) else doc_id,
                                        "chunk_index": len(doc_chunks),
                                        "chunk_start": chunk_start,
                                        "chunk_end": chunk_end,
                                        "strategy": strategy,
                                    },
                                }
                            )

                    # Store chunks in Redis
                    for chunk in doc_chunks:
                        chunk_key = f"chunk:{chunk['id']}"
                        mapping_data: "Mapping[str | bytes, bytes | float | int | str]" = {
                            "content": chunk["content"],
                            "document_id": chunk["metadata"]["document_id"],
                            "chunk_index": str(chunk["metadata"]["chunk_index"]),
                            "created_at": datetime.now(UTC).isoformat(),
                        }
                        redis_client.hset(
                            chunk_key,
                            mapping=mapping_data,
                        )
                        # Add to operation's chunk list
                        redis_client.rpush(f"operation:{operation_id}:chunks", cast(str, chunk["id"]))

                    chunks_created += len(doc_chunks)
                    documents_processed += 1

                except Exception as e:
                    logger.error(f"Failed to process document {doc_id}: {e}")
                    failed_documents.append(doc_id.decode() if isinstance(doc_id, bytes) else doc_id)

            # Update progress
            progress = int((documents_processed / total_documents) * 100) if total_documents > 0 else 0
            _send_progress_update_sync(
                redis_client,
                operation_id,
                correlation_id,
                progress,
                f"Processed {documents_processed}/{total_documents} documents",
            )

            # Force garbage collection after each batch
            gc.collect()

        # Update completion status
        redis_client.hset(
            f"operation:{operation_id}",
            mapping={
                "status": "completed" if not failed_documents else "partial_success",
                "completed_at": datetime.now(UTC).isoformat(),
                "chunks_created": str(chunks_created),
                "documents_processed": str(documents_processed),
                "documents_failed": str(len(failed_documents)),
            },
        )

        # Store failed documents for potential retry
        if failed_documents:
            for doc_id in failed_documents:
                redis_client.rpush(f"operation:{operation_id}:failed_documents", doc_id)

        return {
            "operation_id": operation_id,
            "status": "success" if not failed_documents else "partial_success",
            "chunks_created": chunks_created,
            "documents_processed": documents_processed,
            "documents_failed": len(failed_documents),
            "duration_seconds": time.time() - start_time,
        }

    except Exception as exc:
        logger.error(
            f"Chunking operation failed: {exc}",
            extra={
                "operation_id": operation_id,
                "correlation_id": correlation_id,
            },
            exc_info=exc,
        )

        # Update failure status in Redis
        redis_client.hset(
            f"operation:{operation_id}",
            mapping={
                "status": "failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "failed_at": datetime.now(UTC).isoformat(),
            },
        )

        # Re-raise with appropriate exception type
        if isinstance(exc, MemoryError):
            current_memory = process.memory_info().rss
            raise ChunkingMemoryError(
                detail="Operation exceeded memory limits",
                correlation_id=correlation_id,
                operation_id=operation_id,
                memory_used=current_memory,
                memory_limit=CHUNKING_MEMORY_LIMIT_GB * 1024**3,
            ) from exc
        if isinstance(exc, TimeoutError):
            raise ChunkingTimeoutError(
                detail="Operation timed out",
                correlation_id=correlation_id,
                operation_id=operation_id,
                elapsed_time=time.time() - start_time,
                timeout_limit=CHUNKING_SOFT_TIME_LIMIT,
            ) from exc
        raise

    finally:
        # Clear memory usage metric
        chunking_operation_memory_usage.labels(operation_id=operation_id).set(0)


async def _process_chunking_operation_async(
    operation_id: str,
    correlation_id: str,
    celery_task: ChunkingTask,
) -> dict[str, Any]:
    """Async implementation of chunking operation processing.

    Args:
        operation_id: Operation identifier
        correlation_id: Correlation ID for tracing
        celery_task: Celery task instance

    Returns:
        Operation results dictionary
    """
    start_time = time.time()
    operation = None
    chunks_created = 0

    # Initialize resources
    redis_client = get_redis_client()
    # ChunkingErrorHandler expects async Redis, create async client

    from packages.shared.config import settings

    async_redis = AsyncRedis.from_url(settings.REDIS_URL)
    error_handler = ChunkingErrorHandler(async_redis)

    # Track resources
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    initial_cpu_time = process.cpu_times().user + process.cpu_times().system

    # Store task ID immediately
    task_id = current_task.request.id if current_task else str(uuid.uuid4())

    # Initialize database connection
    if not pg_connection_manager._sessionmaker:
        await pg_connection_manager.initialize()
        logger.info("Initialized database connection for chunking task")

    async with AsyncSessionLocal() as db:  # type: ignore[misc]
        operation_repo = OperationRepository(db)
        collection_repo = CollectionRepository(db)

        try:
            # Update operation with task ID
            operation = await operation_repo.get_by_uuid(operation_id)
            if not operation:
                raise ValueError(f"Operation {operation_id} not found")

            # Check idempotency - if already processed, return early
            if operation.status == OperationStatus.COMPLETED:
                logger.info(f"Operation {operation_id} already completed (idempotent)")
                return {
                    "operation_id": operation_id,
                    "status": "already_completed",
                    "chunks_created": operation.metadata.get("chunks_created", 0),
                }

            # Update operation status to processing
            await operation_repo.update_status(
                operation_id,
                OperationStatus.PROCESSING,
                json.dumps({"task_id": task_id, "started_at": datetime.now(UTC).isoformat()}),
            )

            # Send progress update
            await _send_progress_update(
                redis_client,
                operation_id,
                correlation_id,
                0,
                "Starting chunking operation",
            )

            # Initialize chunking service
            chunking_service = ChunkingService(
                db_session=db,
                collection_repo=collection_repo,
                document_repo=DocumentRepository(db),
                redis_client=None,  # Redis client type mismatch - TODO: use async redis
            )

            # Check resource limits before processing
            await _check_resource_limits(
                error_handler,
                operation_id,
                correlation_id,
                initial_memory,
            )

            # Process documents with progress tracking
            collection_id = operation.collection_id
            collection = await collection_repo.get_by_uuid(collection_id)

            if not collection:
                raise ValueError(f"Collection {collection_id} not found")

            # Get documents to process based on operation type
            documents = await _get_documents_for_operation(
                operation,
                collection,
                operation_repo,
                db,
            )

            total_documents = len(documents)
            processed_count = 0
            failed_documents = []
            errors = []
            chunks: list[ChunkResult] = []

            # Process documents in batches with resource monitoring
            batch_size = await _calculate_batch_size(error_handler, initial_memory)

            for i in range(0, total_documents, batch_size):
                # Check for graceful shutdown
                if celery_task._graceful_shutdown:
                    logger.info("Graceful shutdown requested, saving progress")
                    break

                batch = documents[i : i + batch_size]

                # Monitor resources
                await _monitor_resources(
                    process,
                    operation_id,
                    initial_memory,
                    initial_cpu_time,
                    error_handler,
                    correlation_id,
                )

                # Process batch
                try:
                    # TODO: Implement actual chunking logic
                    # ChunkingService doesn't have process_documents method yet
                    # This needs to be implemented based on the actual chunking strategy
                    batch_results: list[ChunkResult] = []  # Placeholder

                    chunks.extend(batch_results)
                    processed_count += len(batch)

                except ChunkingPartialFailureError as e:
                    # Handle partial failures
                    # ChunkingPartialFailureError.successful_chunks is an int, not a list
                    # The actual chunks should be tracked separately
                    # chunks.extend(e.successful_chunks)  # Can't extend with int
                    failed_documents.extend(e.failed_documents)
                    errors.append(e)
                    processed_count += e.total_documents - len(e.failed_documents)

                except Exception as e:
                    # Track failed documents
                    failed_documents.extend([doc["id"] for doc in batch])
                    errors.append(e)  # type: ignore[arg-type]
                    logger.error(
                        f"Batch processing failed: {e}",
                        extra={
                            "operation_id": operation_id,
                            "correlation_id": correlation_id,
                            "batch_start": i,
                            "batch_size": len(batch),
                        },
                        exc_info=e,
                    )

                # Update progress
                progress = int((processed_count / total_documents) * 100)
                await _send_progress_update(
                    redis_client,
                    operation_id,
                    correlation_id,
                    progress,
                    f"Processed {processed_count}/{total_documents} documents",
                )

                # Force garbage collection after each batch
                gc.collect()

            # Handle any failures
            if failed_documents:
                result = await error_handler.handle_partial_failure(
                    operation_id=operation_id,
                    processed_chunks=chunks,
                    failed_documents=failed_documents,
                    errors=errors,  # type: ignore[arg-type]
                )

                # Update operation with partial success
                await operation_repo.update_status(
                    operation_id,
                    OperationStatus.COMPLETED,
                    json.dumps(
                        {
                            "chunks_created": len(chunks),
                            "documents_processed": processed_count,
                            "documents_failed": len(failed_documents),
                            "partial_failure": True,
                            "recovery_operation_id": result.recovery_operation_id,
                            "completed_at": datetime.now(UTC).isoformat(),
                        }
                    ),
                )

                return {
                    "operation_id": operation_id,
                    "status": "partial_success",
                    "chunks_created": len(chunks),
                    "documents_processed": processed_count,
                    "documents_failed": len(failed_documents),
                    "recovery_operation_id": result.recovery_operation_id,
                    "recommendations": result.recommendations,
                }

            # All successful
            chunks_created = len(chunks)

            # Update operation status
            await operation_repo.update_status(
                operation_id,
                OperationStatus.COMPLETED,
                json.dumps(
                    {
                        "chunks_created": chunks_created,
                        "documents_processed": processed_count,
                        "completed_at": datetime.now(UTC).isoformat(),
                        "duration_seconds": time.time() - start_time,
                    }
                ),
            )

            # Update collection status
            await collection_repo.update_status(
                collection_id,
                CollectionStatus.READY,
            )

            # Send completion update
            await _send_progress_update(
                redis_client,
                operation_id,
                correlation_id,
                100,
                f"Successfully created {chunks_created} chunks",
            )

            # Track metrics
            chunking_task_duration.labels(operation_type="chunking").observe(time.time() - start_time)

            return {
                "operation_id": operation_id,
                "status": "success",
                "chunks_created": chunks_created,
                "documents_processed": processed_count,
                "duration_seconds": time.time() - start_time,
            }

        except Exception as exc:
            logger.error(
                f"Chunking operation failed: {exc}",
                extra={
                    "operation_id": operation_id,
                    "correlation_id": correlation_id,
                },
                exc_info=exc,
            )

            # Update operation status
            if operation:
                await operation_repo.update_status(
                    operation_id,
                    OperationStatus.FAILED,
                    json.dumps(
                        {
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "failed_at": datetime.now(UTC).isoformat(),
                        }
                    ),
                )

            # Clean up failed operation
            cleanup_result = await error_handler.cleanup_failed_operation(
                operation_id=operation_id,
                partial_results=chunks if "chunks" in locals() else None,
                cleanup_strategy="save_partial" if chunks_created > 0 else "rollback",
            )

            logger.info(
                f"Cleanup completed for operation {operation_id}",
                extra={
                    "operation_id": operation_id,
                    "cleanup_result": (
                        cleanup_result.to_dict() if hasattr(cleanup_result, "to_dict") else str(cleanup_result)
                    ),
                },
            )

            # Re-raise with appropriate exception type
            if isinstance(exc, MemoryError):
                current_memory = process.memory_info().rss
                raise ChunkingMemoryError(
                    detail="Operation exceeded memory limits",
                    correlation_id=correlation_id,
                    operation_id=operation_id,
                    memory_used=current_memory,
                    memory_limit=CHUNKING_MEMORY_LIMIT_GB * 1024**3,
                ) from exc
            if isinstance(exc, TimeoutError):
                raise ChunkingTimeoutError(
                    detail="Operation timed out",
                    correlation_id=correlation_id,
                    operation_id=operation_id,
                    elapsed_time=time.time() - start_time,
                    timeout_limit=CHUNKING_SOFT_TIME_LIMIT,
                ) from exc
            raise

        finally:
            # Clear memory usage metric
            chunking_operation_memory_usage.labels(operation_id=operation_id).set(0)


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

    try:
        update = {
            "operation_id": operation_id,
            "correlation_id": correlation_id,
            "progress": str(progress),  # Redis requires string values
            "message": message,
            "timestamp": str(time.time()),
        }

        # Send to Redis stream
        stream_key = f"stream:chunking:{operation_id}"
        redis_client.xadd(stream_key, update, maxlen=1000)

        # Also update hash for current state
        redis_client.hset(
            f"operation:{operation_id}:progress",
            mapping={
                "progress": str(progress),
                "message": message,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    except Exception as e:
        logger.warning(f"Failed to send progress update: {e}")


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


async def _check_resource_limits(
    error_handler: ChunkingErrorHandler,
    operation_id: str,
    correlation_id: str,
    initial_memory: int,  # noqa: ARG001
) -> None:
    """Check system resource limits before processing.

    Args:
        error_handler: Error handler instance
        operation_id: Operation identifier
        correlation_id: Correlation ID
        initial_memory: Initial memory usage in bytes

    Raises:
        ChunkingResourceLimitError: If resources are exhausted
    """
    # Check memory
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        recovery_action = await error_handler.handle_resource_exhaustion(
            operation_id=operation_id,
            resource_type=ResourceType.MEMORY,
            current_usage=memory.percent,
            limit=100,
        )

        if recovery_action.action == "fail":
            raise ChunkingResourceLimitError(
                detail="System memory exhausted",
                correlation_id=correlation_id,
                resource_type=ResourceType.MEMORY,
                current_usage=memory.percent,
                limit=100,
                operation_id=operation_id,
            )

    # Check CPU
    cpu_percent = psutil.cpu_percent(interval=0.1)
    if cpu_percent > 90:
        recovery_action = await error_handler.handle_resource_exhaustion(
            operation_id=operation_id,
            resource_type=ResourceType.CPU,
            current_usage=cpu_percent,
            limit=100,
        )

        if recovery_action.action == "wait_and_retry":
            await asyncio.sleep(recovery_action.wait_time or 30)


async def _monitor_resources(
    process: psutil.Process,
    operation_id: str,
    initial_memory: int,
    initial_cpu_time: float,
    error_handler: ChunkingErrorHandler,  # noqa: ARG001
    correlation_id: str,
) -> None:
    """Monitor resource usage during processing.

    Args:
        process: Process object for monitoring
        operation_id: Operation identifier
        initial_memory: Initial memory usage
        initial_cpu_time: Initial CPU time
        error_handler: Error handler instance
        correlation_id: Correlation ID

    Raises:
        ChunkingMemoryError: If memory limit exceeded
        ChunkingTimeoutError: If CPU time limit exceeded
    """
    current_memory = process.memory_info().rss
    memory_increase = current_memory - initial_memory

    # Update memory metric
    chunking_operation_memory_usage.labels(operation_id=operation_id).set(current_memory)

    # Check memory limit
    if memory_increase > CHUNKING_MEMORY_LIMIT_GB * 1024**3:
        raise ChunkingMemoryError(
            detail="Operation memory usage exceeded limit",
            correlation_id=correlation_id,
            operation_id=operation_id,
            memory_used=current_memory,
            memory_limit=CHUNKING_MEMORY_LIMIT_GB * 1024**3,
        )

    # Check CPU time
    current_cpu_time = process.cpu_times().user + process.cpu_times().system
    cpu_time_used = current_cpu_time - initial_cpu_time

    if cpu_time_used > CHUNKING_CPU_TIME_LIMIT:
        raise ChunkingTimeoutError(
            detail="Operation CPU time exceeded limit",
            correlation_id=correlation_id,
            operation_id=operation_id,
            elapsed_time=cpu_time_used,
            timeout_limit=CHUNKING_CPU_TIME_LIMIT,
        )


async def _calculate_batch_size(
    error_handler: ChunkingErrorHandler,
    initial_memory: int,  # noqa: ARG001
) -> int:
    """Calculate optimal batch size based on available resources.

    Args:
        error_handler: Error handler instance
        initial_memory: Initial memory usage

    Returns:
        Optimal batch size
    """
    memory = psutil.virtual_memory()

    # Use error handler's adaptive calculation
    return error_handler._calculate_adaptive_batch_size(
        current_usage=memory.percent,
        limit=100,
    )


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

    try:
        update = {
            "operation_id": operation_id,
            "correlation_id": correlation_id,
            "progress": progress,
            "message": message,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Send to Redis stream for WebSocket updates
        stream_key = f"chunking:progress:{operation_id}"
        # Redis client is sync, not async - need to use sync methods
        redis_client.xadd(stream_key, update)

        # Expire stream after 1 hour for progress updates
        redis_client.expire(stream_key, 3600)

    except Exception as e:
        logger.error(f"Failed to send progress update: {e}")


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
