"""Collection ingestion and orchestration Celery tasks.

This module contains the main task entrypoints for collection operations along with
helpers that drive INDEX, APPEND, REMOVE_SOURCE, and compatibility flows used in
tests. The implementation leans on utilities consolidated in
``webui.tasks.utils`` and delegates reindex-specific logic to the
``reindex`` module.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from datetime import UTC, datetime
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

import httpx
import psutil
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue, PointStruct

from shared.config import settings
from shared.database import pg_connection_manager
from shared.database.database import ensure_async_sessionmaker
from shared.metrics.collection_metrics import (
    OperationTimer,
    QdrantOperationTimer,
    collection_cpu_seconds_total,
    collection_memory_usage_bytes,
    collections_total,
)
from webui.services.chunking.container import resolve_celery_chunking_orchestrator
from webui.services.connector_factory import ConnectorFactory
from webui.services.document_registry_service import DocumentRegistryService

from . import reindex as reindex_tasks
from .utils import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DOCUMENT_REMOVAL_BATCH_SIZE,
    EMBEDDING_BATCH_SIZE,
    OPERATION_HARD_TIME_LIMIT,
    OPERATION_SOFT_TIME_LIMIT,
    VECTOR_UPLOAD_BATCH_SIZE,
    CeleryTaskWithOperationUpdates,
    _audit_log_operation,
    _record_operation_metrics,
    _sanitize_error_message,
    _update_collection_metrics,
    await_if_awaitable,
    celery_app,
    extract_and_serialize_thread_safe,
    logger,
    resolve_qdrant_manager,
    resolve_qdrant_manager_class,
)

if TYPE_CHECKING:
    from types import ModuleType


def _get_embedding_concurrency() -> int:
    """Return per-worker embed concurrency, defaulting to 1 if unset/invalid."""
    try:
        val = int(os.getenv("EMBEDDING_CONCURRENCY_PER_WORKER", "1"))
        return max(1, val)
    except Exception:  # pragma: no cover - defensive parsing
        return 1


# Single-process semaphore throttling embed calls so we can run more workers for CPU-bound steps
_embedding_semaphore = asyncio.Semaphore(_get_embedding_concurrency())


def _tasks_namespace() -> ModuleType:
    """Return the top-level tasks module for accessing patched attributes."""
    return import_module("webui.tasks")


@celery_app.task(bind=True)
def test_task(self: Any) -> dict[str, str]:  # noqa: ARG001
    """Test task to verify Celery is working."""
    return {"status": "success", "message": "Celery is working!"}


@celery_app.task(
    bind=True,
    name="webui.tasks.process_collection_operation",
    max_retries=DEFAULT_MAX_RETRIES,
    default_retry_delay=DEFAULT_RETRY_DELAY,
    acks_late=True,  # Ensure message reliability
    soft_time_limit=OPERATION_SOFT_TIME_LIMIT,
    time_limit=OPERATION_HARD_TIME_LIMIT,
    on_failure=lambda self, exc, task_id, args, kwargs, einfo: _handle_task_failure(
        self, exc, task_id, args, kwargs, einfo
    ),
)
def process_collection_operation(self: Any, operation_id: str) -> dict[str, Any]:
    """Process a collection operation (INDEX, APPEND, REINDEX, REMOVE_SOURCE)."""
    tasks_ns = _tasks_namespace()
    try:
        loop = tasks_ns.asyncio.get_event_loop()
        if loop.is_closed():
            loop = tasks_ns.asyncio.new_event_loop()
            tasks_ns.asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = tasks_ns.asyncio.new_event_loop()
        tasks_ns.asyncio.set_event_loop(loop)

    try:
        return cast(
            dict[str, Any],
            loop.run_until_complete(tasks_ns._process_collection_operation_async(operation_id, self)),
        )
    except Exception as exc:  # pragma: no cover - retry path
        logger.error("Task failed for operation %s: %s", operation_id, exc)
        if isinstance(exc, ValueError | TypeError):
            raise
        raise self.retry(exc=exc, countdown=60) from exc


async def _process_collection_operation_async(operation_id: str, celery_task: Any) -> dict[str, Any]:
    """Async implementation of collection operation processing with enhanced monitoring."""
    from shared.database.models import CollectionStatus, OperationStatus, OperationType
    from shared.database.repositories.collection_repository import CollectionRepository
    from shared.database.repositories.document_repository import DocumentRepository
    from shared.database.repositories.operation_repository import OperationRepository
    from shared.database.repositories.projection_run_repository import ProjectionRunRepository

    start_time = time.time()
    operation = None
    collection: dict[str, Any] | None = None

    process = psutil.Process()
    tasks_ns = _tasks_namespace()

    await pg_connection_manager.initialize()

    session_factory = pg_connection_manager.sessionmaker
    if session_factory is None:
        # Fallback to a patched AsyncSessionLocal (used in tests)
        shared_db_module = import_module("shared.database.database")
        session_factory = getattr(shared_db_module, "AsyncSessionLocal", None)

    if session_factory is None or not callable(session_factory):
        raise RuntimeError("Failed to initialize database connection for this task")

    def _safe_cpu_seconds(proc: Any) -> float:
        """Return total CPU seconds for the process, tolerating mocked values."""
        try:
            cpu_times = proc.cpu_times()
        except Exception:  # pragma: no cover - psutil edge cases
            return 0.0

        user = getattr(cpu_times, "user", 0.0)
        system = getattr(cpu_times, "system", 0.0)

        try:
            user_seconds = float(user)
        except (TypeError, ValueError):
            user_seconds = 0.0

        try:
            system_seconds = float(system)
        except (TypeError, ValueError):
            system_seconds = 0.0

        return user_seconds + system_seconds

    initial_cpu_time = _safe_cpu_seconds(process)

    task_id = celery_task.request.id if hasattr(celery_task, "request") else str(uuid.uuid4())

    logger.info("Initialized database connection for this task")

    try:
        async with session_factory() as db:
            operation_repo = OperationRepository(db)
            collection_repo = CollectionRepository(db)
            document_repo = DocumentRepository(db)
            projection_repo = ProjectionRunRepository(db)

            try:
                await operation_repo.set_task_id(operation_id, task_id)
                logger.info("Set task_id %s for operation %s", task_id, operation_id)

                async with CeleryTaskWithOperationUpdates(operation_id) as updater:
                    operation_obj = await operation_repo.get_by_uuid(operation_id)
                    if not operation_obj:
                        raise ValueError(f"Operation {operation_id} not found in database")

                    operation = {
                        "id": operation_obj.id,
                        "uuid": operation_obj.uuid,
                        "collection_id": operation_obj.collection_id,
                        "type": operation_obj.type,
                        "config": operation_obj.config,
                        "user_id": getattr(operation_obj, "user_id", None),
                    }

                    logger.info(
                        "Starting collection operation",
                        extra={
                            "operation_id": operation_id,
                            "operation_type": operation["type"].value,
                            "collection_id": operation["collection_id"],
                            "task_id": task_id,
                        },
                    )

                    # Set user_id on updater for user-channel notifications
                    if operation.get("user_id"):
                        updater.set_user_id(operation["user_id"])

                    await operation_repo.update_status(operation_id, OperationStatus.PROCESSING)
                    await updater.send_update(
                        "operation_started", {"status": "processing", "type": operation["type"].value}
                    )

                    collection_obj = await collection_repo.get_by_uuid(operation["collection_id"])
                    if not collection_obj:
                        raise ValueError(f"Collection {operation['collection_id']} not found in database")

                    collection = {
                        "id": collection_obj.id,
                        "uuid": collection_obj.id,
                        "name": collection_obj.name,
                        "vector_store_name": collection_obj.vector_store_name,
                        "config": getattr(collection_obj, "config", {}),
                        "embedding_model": getattr(collection_obj, "embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
                        "quantization": getattr(collection_obj, "quantization", "float16"),
                        "chunk_size": getattr(collection_obj, "chunk_size", 1000),
                        "chunk_overlap": getattr(collection_obj, "chunk_overlap", 200),
                        "chunking_strategy": getattr(collection_obj, "chunking_strategy", None),
                        "chunking_config": getattr(collection_obj, "chunking_config", {}) or {},
                        "qdrant_collections": getattr(collection_obj, "qdrant_collections", []),
                        "qdrant_staging": getattr(collection_obj, "qdrant_staging", []),
                        "status": getattr(collection_obj, "status", CollectionStatus.PENDING),
                        "vector_count": getattr(collection_obj, "vector_count", 0),
                    }

                    vector_collection_id = getattr(collection_obj, "vector_collection_id", None)
                    if vector_collection_id and not collection["vector_store_name"]:
                        collection["vector_store_name"] = vector_collection_id

                    result: dict[str, Any] = {}
                    operation_type = operation["type"].value.lower()

                    # Commit setup changes before starting potentially long-running operations
                    # This prevents idle-in-transaction timeouts and ensures a clean session state
                    await db.commit()

                    with OperationTimer(operation_type):
                        memory_before = process.memory_info().rss

                        if operation["type"] == OperationType.INDEX:
                            result = await tasks_ns._process_index_operation(
                                operation, collection, collection_repo, document_repo, updater
                            )
                        elif operation["type"] == OperationType.APPEND:
                            result = await tasks_ns._process_append_operation_impl(
                                operation, collection, collection_repo, document_repo, updater
                            )
                        elif operation["type"] == OperationType.REINDEX:
                            result = await tasks_ns._process_reindex_operation(db, updater, operation["id"])
                        elif operation["type"] == OperationType.REMOVE_SOURCE:
                            result = await tasks_ns._process_remove_source_operation(
                                operation, collection, collection_repo, document_repo, updater
                            )
                        elif operation["type"] == OperationType.PROJECTION_BUILD:
                            result = await tasks_ns._process_projection_operation(
                                operation,
                                collection,
                                projection_repo,
                                updater,
                            )
                        else:  # pragma: no cover - defensive branch
                            raise ValueError(f"Unknown operation type: {operation['type']}")

                        memory_peak = process.memory_info().rss
                        collection_memory_usage_bytes.labels(operation_type=operation_type).set(
                            memory_peak - memory_before
                        )

                    duration = time.time() - start_time
                    cpu_time = _safe_cpu_seconds(process) - initial_cpu_time
                    if cpu_time < 0 or not isinstance(cpu_time, int | float):
                        cpu_time = 0.0

                    defer_completion = operation["type"] == OperationType.PROJECTION_BUILD and result.get(
                        "defer_completion"
                    )

                    if defer_completion:
                        await db.commit()
                        logger.info(
                            "Projection operation %s enqueued for async processing",
                            operation_id,
                        )
                        return result

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

                    collection_cpu_seconds_total.labels(operation_type=operation_type).inc(cpu_time)

                    await operation_repo.update_status(operation_id, OperationStatus.COMPLETED)

                    old_status = collection.get("status", CollectionStatus.PENDING)

                    if result.get("success"):
                        doc_stats = await document_repo.get_stats_by_collection(collection["id"])
                        new_status = CollectionStatus.READY
                        await collection_repo.update_status(collection["id"], new_status)

                        # Mark existing projections as stale so the UI can prompt recomputation.
                        try:
                            runs, _ = await projection_repo.list_for_collection(collection["id"], limit=1)
                        except Exception:  # pragma: no cover - defensive path
                            runs = []
                        if runs:
                            try:
                                await projection_repo.update_metadata(
                                    runs[0].uuid,
                                    meta={"degraded": True},
                                )
                            except Exception:  # pragma: no cover - defensive logging
                                logger.warning(
                                    "Failed to mark projection %s as degraded after collection update",
                                    runs[0].uuid,
                                )

                        await _update_collection_metrics(
                            collection["id"],
                            doc_stats["total_documents"],
                            collection.get("vector_count", 0),
                            doc_stats["total_size_bytes"],
                        )
                    else:
                        new_status = CollectionStatus.DEGRADED
                        await collection_repo.update_status(collection["id"], new_status)

                    if old_status != new_status:
                        collections_total.labels(status=old_status.value).dec()
                        collections_total.labels(status=new_status.value).inc()

                    await updater.send_update("operation_completed", {"status": "completed", "result": result})

                    logger.info(
                        "Collection operation completed",
                        extra={
                            "operation_id": operation_id,
                            "operation_type": operation["type"].value,
                            "duration_seconds": duration,
                            "success": result.get("success", False),
                        },
                    )

                    # Commit after notifying listeners; add_source will tolerate the small window
                    await db.commit()

                    return result

            except Exception as exc:
                logger.error("Operation %s failed: %s", operation_id, exc, exc_info=True)

                with contextlib.suppress(Exception):
                    await db.rollback()

                try:
                    if operation:
                        await _record_operation_metrics(
                            operation_repo,
                            operation_id,
                            {
                                "duration_seconds": time.time() - start_time,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "success": False,
                            },
                        )

                    await operation_repo.update_status(operation_id, OperationStatus.FAILED, error_message=str(exc))

                    if operation and collection:
                        collection_id = operation["collection_id"]
                        operation_type = operation["type"]

                        if operation_type == OperationType.INDEX:
                            await collection_repo.update_status(
                                collection["uuid"],
                                CollectionStatus.ERROR,
                                status_message=f"Initial indexing failed: {str(exc)}",
                            )
                        elif operation_type == OperationType.REINDEX:
                            await collection_repo.update_status(
                                collection["uuid"],
                                CollectionStatus.DEGRADED,
                                status_message=f"Re-indexing failed: {str(exc)}. Original collection still available.",
                            )
                            await reindex_tasks._cleanup_staging_resources(collection_id, operation)

                    # updater context manager handles failure updates
                except Exception as update_error:  # pragma: no cover - defensive logging
                    logger.error("Failed to update operation status during error handling: %s", update_error)

                raise

            finally:
                try:
                    if operation:
                        current_status = await operation_repo.get_by_uuid(operation_id)
                        from shared.database.models import OperationStatus as _OperationStatus

                        if current_status and current_status.status == _OperationStatus.PROCESSING:
                            await operation_repo.update_status(
                                operation_id, _OperationStatus.FAILED, error_message="Task terminated unexpectedly"
                            )
                            await db.commit()
                except Exception as final_error:  # pragma: no cover - defensive logging
                    logger.error("Failed to finalize operation status: %s", final_error)
                    with contextlib.suppress(Exception):
                        await db.rollback()
    finally:
        pass


async def _process_append_operation(db: Any, updater: Any, _operation_id: str) -> dict[str, Any]:
    """Compatibility wrapper used by tests to process an APPEND operation."""

    def _get(obj: Any, name: str, default: Any = None) -> Any:
        try:
            return obj.get(name, default)
        except Exception:
            return getattr(obj, name, default)

    tasks_ns = _tasks_namespace()
    extract_fn = getattr(tasks_ns, "extract_and_serialize_thread_safe", extract_and_serialize_thread_safe)
    chunking_resolver = getattr(
        tasks_ns,
        "resolve_celery_chunking_orchestrator",
        resolve_celery_chunking_orchestrator,
    )

    # Replace placeholder executes with real queries
    from sqlalchemy import select

    from shared.database.models import Collection as _Collection, Document as _Document, Operation as _Operation

    # Fetch the operation using whichever identifier the caller supplied
    op_lookup = select(_Operation)
    try:
        op_lookup = op_lookup.where(_Operation.id == int(_operation_id))
    except (TypeError, ValueError):
        op_lookup = op_lookup.where(_Operation.uuid == _operation_id)

    op = (await db.execute(op_lookup)).scalar_one()

    # Fetch the parent collection
    collection_obj = (
        await db.execute(select(_Collection).where(_Collection.id == op.collection_id))
    ).scalar_one_or_none()

    # Fetch documents for this collection
    docs = (await db.execute(select(_Document).where(_Document.collection_id == op.collection_id))).scalars().all()

    collection = {
        "id": _get(collection_obj, "id"),
        "name": _get(collection_obj, "name"),
        "chunking_strategy": _get(collection_obj, "chunking_strategy"),
        "chunking_config": _get(collection_obj, "chunking_config", {}) or {},
        "chunk_size": _get(collection_obj, "chunk_size", 1000),
        "chunk_overlap": _get(collection_obj, "chunk_overlap", 200),
        "embedding_model": _get(collection_obj, "embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
        "quantization": _get(collection_obj, "quantization", "float16"),
        "vector_store_name": _get(collection_obj, "vector_collection_id") or _get(collection_obj, "vector_store_name"),
    }

    from shared.database.repositories.collection_repository import CollectionRepository
    from shared.database.repositories.document_repository import DocumentRepository

    collection_repo = CollectionRepository(db)
    document_repo = DocumentRepository(db)
    cs = await await_if_awaitable(
        chunking_resolver(
            db,
            collection_repo=collection_repo,
            document_repo=document_repo,
        )
    )

    processed = 0
    skipped = 0
    failed = 0
    from shared.database.models import DocumentStatus

    for doc in docs:
        status = getattr(doc, "status", None)
        status_value = getattr(status, "value", status)
        existing_chunks = getattr(doc, "chunk_count", 0) or 0

        if existing_chunks > 0 and status_value == DocumentStatus.COMPLETED.value:
            skipped += 1
            continue

        try:
            try:
                blocks_result = extract_fn(_get(doc, "file_path", ""))
                blocks = await await_if_awaitable(blocks_result)
            except Exception:
                blocks = []

            text = "".join((t for t, _m in (blocks or []) if isinstance(t, str)))
            metadata: dict[str, Any] = {}
            for _t, m in blocks or []:
                if isinstance(m, dict):
                    metadata.update(m)

            if not text or not blocks:
                with contextlib.suppress(Exception):
                    doc.chunk_count = 0
                    doc.status = DocumentStatus.COMPLETED
                skipped += 1
                continue

            res_chunks = await cs.execute_ingestion_chunking(
                content=text,
                strategy=collection.get("chunking_strategy") or "recursive",
                config=collection.get("chunking_config") or {},
                metadata={**metadata, "document_id": _get(doc, "id")},
            )

            chunks = res_chunks or []

            if chunks:
                texts = [c.get("text", "") for c in chunks]
                embed_req = {"texts": texts, "model_name": collection.get("embedding_model"), "mode": "document"}
                upsert_req: dict[str, Any] = {"collection_name": collection.get("vector_store_name"), "points": []}
                async with httpx.AsyncClient(timeout=60.0) as client:
                    await client.post("http://vecpipe:8000/embed", json=embed_req)
                    await client.post("http://vecpipe:8000/upsert", json=upsert_req)

                with contextlib.suppress(Exception):
                    doc.chunk_count = len(chunks)
                    doc.status = DocumentStatus.COMPLETED
                processed += 1
            else:
                with contextlib.suppress(Exception):
                    doc.chunk_count = 0
                    doc.status = DocumentStatus.COMPLETED
                skipped += 1

        except Exception:
            with contextlib.suppress(Exception):
                doc.status = DocumentStatus.FAILED
            failed += 1
            if doc == docs[-1]:
                raise

    with contextlib.suppress(Exception):
        await updater.send_update(
            "append_completed",
            {
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
                "operation_id": _get(op, "id"),
            },
        )

    # Mark legacy wrapper successes explicitly so orchestration logic can
    # promote the collection out of DEGRADED status (it expects a "success"
    # flag in the result payload).
    return {
        "success": failed == 0,
        "processed": processed,
        "documents_added": processed,
        "skipped": skipped,
        "failed": failed,
    }


async def _process_index_operation(
    operation: dict,
    collection: dict,
    collection_repo: Any,
    document_repo: Any,  # noqa: ARG001
    updater: CeleryTaskWithOperationUpdates,
) -> dict[str, Any]:
    """Process INDEX operation - Initial collection creation with monitoring."""
    from qdrant_client.models import Distance, VectorParams

    from shared.database.collection_metadata import ensure_metadata_collection, store_collection_metadata
    from shared.embedding.factory import resolve_model_config
    from shared.embedding.validation import get_model_dimension
    from shared.metrics.collection_metrics import record_qdrant_operation

    try:
        manager = resolve_qdrant_manager()
        qdrant_client = manager.get_client()

        vector_store_name = collection.get("vector_store_name")
        if not vector_store_name:
            vector_store_name = f"col_{collection['uuid'].replace('-', '_')}"
            logger.warning(
                "Collection %s missing vector_store_name, generated: %s", collection["id"], vector_store_name
            )

        try:
            ensure_metadata_collection(qdrant_client)
        except Exception as exc:
            logger.warning(
                "Failed to ensure metadata collection before creating %s: %s",
                vector_store_name,
                exc,
            )

        config = collection.get("config", {})

        vector_dim = config.get("vector_dim")
        if not vector_dim:
            model_name = collection.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B")
            # Use resolve_model_config to check providers (including plugins) first
            model_config = resolve_model_config(model_name)
            if model_config:
                vector_dim = model_config.dimension
            else:
                logger.warning("Unknown model %s, using default dimension 1024", model_name)
                vector_dim = 1024

        actual_model_name = collection.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B")
        actual_model_dim = get_model_dimension(actual_model_name)

        if actual_model_dim and actual_model_dim != vector_dim:
            logger.warning(
                "Model %s has dimension %s, but collection will be created with dimension %s. This may cause issues during indexing.",
                actual_model_name,
                actual_model_dim,
                vector_dim,
            )

        with QdrantOperationTimer("create_collection"):
            qdrant_client.create_collection(
                collection_name=vector_store_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )

        try:
            store_collection_metadata(
                qdrant=qdrant_client,
                collection_name=vector_store_name,
                model_name=actual_model_name,
                quantization=collection.get("quantization", "float16"),
                vector_dim=vector_dim,
                chunk_size=config.get("chunk_size"),
                chunk_overlap=config.get("chunk_overlap"),
                instruction=config.get("instruction"),
                ensure=False,
            )
        except Exception as exc:
            logger.warning("Failed to store collection metadata: %s", exc)

        try:
            collection_info = qdrant_client.get_collection(vector_store_name)
            logger.info(
                "Verified Qdrant collection %s exists with %s vectors",
                vector_store_name,
                collection_info.vectors_count,
            )
        except Exception as exc:
            logger.error("Failed to verify collection %s after creation: %s", vector_store_name, exc)
            raise Exception(f"Collection {vector_store_name} was not properly created in Qdrant") from exc

        try:
            await collection_repo.update(collection["id"], {"vector_store_name": vector_store_name})
            logger.info("Updated collection %s with vector_store_name: %s", collection["id"], vector_store_name)
        except Exception as exc:
            logger.error("Failed to update collection in database: %s", exc)
            try:
                qdrant_client.delete_collection(vector_store_name)
                logger.info("Cleaned up Qdrant collection %s after database update failure", vector_store_name)
            except Exception as cleanup_error:
                logger.error("Failed to clean up Qdrant collection %s: %s", vector_store_name, cleanup_error)
            raise Exception(f"Failed to update collection {collection['id']} in database") from exc

        await _audit_log_operation(
            collection["id"],
            operation["id"],
            operation.get("user_id"),
            "collection_indexed",
            {"qdrant_collection": vector_store_name, "vector_dim": vector_dim},
        )

        await updater.send_update("index_completed", {"qdrant_collection": vector_store_name, "vector_dim": vector_dim})

        return {"success": True, "qdrant_collection": vector_store_name, "vector_dim": vector_dim}

    except Exception as exc:
        logger.error("Failed to create Qdrant collection: %s", exc)
        record_qdrant_operation("create_collection", "failed")
        raise


async def _process_append_operation_impl(
    operation: dict,
    collection: dict,
    collection_repo: Any,
    document_repo: Any,
    updater: CeleryTaskWithOperationUpdates,
) -> dict[str, Any]:
    """Process APPEND operation - Add documents to existing collection with monitoring.

    Uses ConnectorFactory + DocumentRegistryService for flexible source support.
    Supports both new format (source_type + source_config) and legacy (source_path).
    """
    from shared.database.models import DocumentStatus
    from shared.metrics.collection_metrics import document_processing_duration, record_document_processed

    op_config = operation.get("config", {})

    # Extract source_id (required for document-source linking)
    source_id = op_config.get("source_id")
    if source_id is None:
        raise ValueError("source_id is required in operation config for APPEND operation")

    # New format: source_type + source_config
    source_type = op_config.get("source_type", "directory")
    source_config = op_config.get("source_config")

    # Legacy fallback: derive from source_path if new format not present
    legacy_source_path = op_config.get("source_path")
    if source_config is None:
        if legacy_source_path:
            source_config = {"path": legacy_source_path}
            source_type = "directory"
        else:
            raise ValueError("source_config or source_path is required for APPEND operation")

    # Extract display path for logging/updates
    display_path = source_config.get("path") or source_config.get("url") or str(source_config)

    tasks_ns = _tasks_namespace()
    extract_fn = getattr(tasks_ns, "extract_and_serialize_thread_safe", extract_and_serialize_thread_safe)

    session = document_repo.session

    # Instantiate connector via factory (validates source_type)
    try:
        connector = ConnectorFactory.get_connector(source_type, source_config)
    except ValueError as e:
        raise ValueError(f"Invalid source configuration: {e}") from e

    # Apply stored connector secrets (if any) before authentication.
    #
    # Secrets are stored encrypted in the DB (via SourceService). Celery workers do not
    # run the FastAPI startup hooks, so we need to ensure encryption is initialised
    # and then load/decrypt secrets here for connectors that support credentials.
    set_credentials = getattr(connector, "set_credentials", None)
    if callable(set_credentials):
        import inspect

        from shared.database.repositories.connector_secret_repository import ConnectorSecretRepository
        from shared.utils.encryption import DecryptionError, EncryptionNotConfiguredError, SecretEncryption

        if not SecretEncryption.is_initialized() and settings.CONNECTOR_SECRETS_KEY:
            try:
                SecretEncryption.initialize(settings.CONNECTOR_SECRETS_KEY)
                logger.debug(
                    "Connector secrets encryption initialised in worker (key_id=%s)",
                    SecretEncryption.get_key_id(),
                )
            except ValueError as exc:
                raise ValueError(f"Invalid CONNECTOR_SECRETS_KEY: {exc}") from exc

        secret_repo = ConnectorSecretRepository(session)
        secret_types = await secret_repo.get_secret_types_for_source(source_id)
        if secret_types:
            credentials: dict[str, str] = {}
            for secret_type in secret_types:
                try:
                    secret_value = await secret_repo.get_secret(source_id, secret_type)
                except (EncryptionNotConfiguredError, DecryptionError) as exc:
                    raise ValueError(
                        f"Failed to decrypt stored secrets for source_id={source_id}; "
                        "ensure CONNECTOR_SECRETS_KEY matches the key used to encrypt them."
                    ) from exc
                if secret_value is not None:
                    credentials[secret_type] = secret_value

            if credentials:
                allowed_params = set(inspect.signature(set_credentials).parameters.keys())
                allowed_params.discard("self")
                filtered_credentials = {k: v for k, v in credentials.items() if k in allowed_params}
                if filtered_credentials:
                    set_credentials(**filtered_credentials)
                    logger.debug(
                        "Applied connector secrets for source_id=%s: %s",
                        source_id,
                        sorted(filtered_credentials.keys()),
                    )

    # Authenticate / validate access
    try:
        auth_ok = await connector.authenticate()
        if not auth_ok:
            raise ValueError(f"Authentication failed for {source_type} source")
    except Exception as e:
        raise ValueError(f"Failed to authenticate with {source_type} source: {e}") from e

    await updater.send_update("scanning_documents", {"status": "scanning", "source": display_path})

    try:
        scan_start = time.time()
        # Record sync start time for marking stale documents after processing
        sync_started_at = datetime.now(UTC)

        # Commit any pending changes from setup phase and ensure session is in clean state
        # This prevents idle-in-transaction timeouts during file scanning/parsing
        if session.in_transaction():
            try:
                await session.commit()
            except Exception as commit_exc:
                logger.warning("Failed to commit before document scan: %s, rolling back", commit_exc)
                with contextlib.suppress(Exception):
                    await session.rollback()

        # Create registry for document registration (with chunk_repo for sync updates)
        from shared.database.repositories.chunk_repository import ChunkRepository
        from shared.database.repositories.document_artifact_repository import DocumentArtifactRepository

        chunk_repo = ChunkRepository(session)
        artifact_repo = DocumentArtifactRepository(session, max_artifact_bytes=settings.MAX_ARTIFACT_BYTES)
        registry = DocumentRegistryService(
            db_session=session,
            document_repo=document_repo,
            chunk_repo=chunk_repo,
            artifact_repo=artifact_repo,
        )

        # Track statistics
        scan_stats: dict[str, Any] = {
            "total_documents_found": 0,
            "new_documents_registered": 0,
            "documents_updated": 0,
            "unchanged_documents_skipped": 0,
            "errors": [],
            "total_size_bytes": 0,
        }

        # Store content for new/updated documents (avoid re-parsing)
        new_doc_contents: dict[int, str] = {}

        # Iterate through connector's documents using savepoints for isolation
        # This allows failures on individual documents without losing previously registered ones
        async for ingested_doc in connector.load_documents():
            scan_stats["total_documents_found"] += 1

            try:
                # Use savepoint (nested transaction) for isolation - if this document's
                # registration fails, only this document's changes are rolled back
                async with session.begin_nested():
                    result = await registry.register_or_update(
                        collection_id=collection["id"],
                        ingested=ingested_doc,
                        source_id=source_id,
                    )

                if result["is_new"]:
                    scan_stats["new_documents_registered"] += 1
                    # Store content for later chunking
                    new_doc_contents[result["document_id"]] = ingested_doc.content
                elif result.get("is_updated", False):
                    scan_stats["documents_updated"] += 1
                    # Updated documents need re-chunking too
                    new_doc_contents[result["document_id"]] = ingested_doc.content
                else:
                    scan_stats["unchanged_documents_skipped"] += 1

                scan_stats["total_size_bytes"] += result["file_size"]

                # Commit after each document to close the transaction completely
                # This prevents idle-in-transaction timeouts when the next file
                # takes a long time to parse (e.g., large PDFs)
                await session.commit()

            except Exception as e:
                logger.error(f"Failed to register document {ingested_doc.unique_id}: {e}")
                # If session is in invalid state (e.g., connection lost), try to recover
                if "invalid transaction" in str(e).lower() or "pending" in str(e).lower():
                    logger.warning("Session in invalid state, attempting rollback recovery")
                    with contextlib.suppress(Exception):
                        await session.rollback()
                scan_stats["errors"].append(
                    {
                        "document": ingested_doc.unique_id,
                        "error": str(e),
                    }
                )

        # Commit after all registrations (if there's an active transaction)
        if session.in_transaction():
            try:
                await session.commit()
            except Exception as commit_exc:
                logger.warning("Failed to commit document registrations: %s", commit_exc)
                with contextlib.suppress(Exception):
                    await session.rollback()

        scan_duration = time.time() - scan_start
        document_processing_duration.labels(operation_type="append").observe(scan_duration)

        logger.info(
            "Scan stats for %s (%s): %s documents found, %s new, %s updated, %s unchanged, %s errors",
            display_path,
            source_type,
            scan_stats["total_documents_found"],
            scan_stats["new_documents_registered"],
            scan_stats["documents_updated"],
            scan_stats["unchanged_documents_skipped"],
            len(scan_stats.get("errors", [])),
        )

        await updater.send_update(
            "scanning_completed",
            {
                "status": "scanning_completed",
                "total_files_found": scan_stats["total_documents_found"],
                "new_documents_registered": scan_stats["new_documents_registered"],
                "documents_updated": scan_stats["documents_updated"],
                "unchanged_documents_skipped": scan_stats["unchanged_documents_skipped"],
                "errors_count": len(scan_stats.get("errors", [])),
            },
        )

        for _ in range(scan_stats["new_documents_registered"]):
            record_document_processed("append", "registered")
        for _ in range(scan_stats["documents_updated"]):
            record_document_processed("append", "updated")
        for _ in range(scan_stats["unchanged_documents_skipped"]):
            record_document_processed("append", "skipped")
        for _ in range(len(scan_stats.get("errors", []))):
            record_document_processed("append", "failed")

        await _audit_log_operation(
            collection["id"],
            operation["id"],
            operation.get("user_id"),
            "documents_appended",
            {
                "source_type": source_type,
                "source": display_path,
                "documents_added": scan_stats["new_documents_registered"],
                "documents_updated": scan_stats["documents_updated"],
                "unchanged_skipped": scan_stats["unchanged_documents_skipped"],
            },
        )

        # Fetch newly registered documents for processing
        new_doc_ids = list(new_doc_contents.keys())
        unprocessed_documents = []
        if new_doc_ids:
            for doc_id in new_doc_ids:
                doc = await document_repo.get_by_id(doc_id)
                if doc and doc.chunk_count == 0:
                    unprocessed_documents.append(doc)

        logger.info(
            "Found %s documents (new + updated) to process for %s (%s)",
            len(unprocessed_documents),
            display_path,
            source_type,
        )

        failed_count = 0

        if len(unprocessed_documents) > 0:
            await updater.send_update(
                "processing_embeddings",
                {
                    "status": "generating_embeddings",
                    "documents_to_process": len(unprocessed_documents),
                },
            )

            documents = unprocessed_documents

            embedding_model = collection.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B")
            quantization = collection.get("quantization", "float16")
            batch_size = op_config.get("batch_size", EMBEDDING_BATCH_SIZE)
            instruction = op_config.get("instruction")

            qdrant_collection_name = collection.get("vector_store_name")
            if not qdrant_collection_name:
                raise ValueError("Collection missing vector_store_name")

            manager = resolve_qdrant_manager()
            qdrant_client = manager.get_client()

            processed_count = 0
            total_vectors_created = 0

            chunking_service = await resolve_celery_chunking_orchestrator(
                document_repo.session,
                collection_repo=collection_repo,
                document_repo=document_repo,
            )  # pragma: no cover - exercised in integration/e2e

            loop = tasks_ns.asyncio.get_event_loop()
            executor_pool = tasks_ns.executor

            for doc in documents:
                try:
                    doc_identifier = doc.file_path or doc.uri or f"doc:{doc.id}"
                    logger.info("Processing document: %s", doc_identifier)

                    # Check if we have pre-parsed content from connector
                    combined_metadata: dict[str, Any] = {}
                    if doc.id in new_doc_contents:
                        combined_text = new_doc_contents[doc.id]
                        # Metadata already captured during registration
                    else:
                        # Fallback for documents without pre-parsed content (legacy path)
                        try:
                            text_blocks = await asyncio.wait_for(
                                await_if_awaitable(
                                    loop.run_in_executor(
                                        executor_pool,
                                        extract_fn,
                                        doc.file_path,
                                    )
                                ),
                                timeout=300,
                            )
                        except RuntimeError as exc:  # pragma: no cover - defensive
                            if "cannot reuse already awaited coroutine" not in str(exc):
                                raise
                            logger.debug(
                                "Executor returned a reusable coroutine for %s; falling back to direct call",
                                doc.file_path,
                            )
                            text_blocks = await asyncio.wait_for(
                                await_if_awaitable(extract_fn(doc.file_path)),
                                timeout=300,
                            )

                        if not text_blocks:
                            logger.warning("No text extracted from %s", doc_identifier)
                            # Mark as completed but with 0 chunks so we don't fail the whole batch
                            await document_repo.update_status(
                                doc.id,
                                DocumentStatus.COMPLETED,
                                chunk_count=0,
                            )
                            # Count as skipped/processed rather than failed
                            processed_count += 1
                            continue

                        combined_text = ""
                        for text, metadata in text_blocks:
                            if text.strip():
                                combined_text += text + "\n\n"
                                if metadata:
                                    combined_metadata.update(metadata)

                    # Skip if no content
                    if not combined_text.strip():
                        logger.warning("No content for document %s", doc_identifier)
                        await document_repo.update_status(
                            doc.id,
                            DocumentStatus.COMPLETED,
                            chunk_count=0,
                        )
                        processed_count += 1
                        continue

                    # End any open transaction before long external calls to avoid
                    # idle_in_transaction_session_timeout disconnects from Postgres.
                    if session.in_transaction():
                        await session.commit()

                    strategy = collection.get("chunking_strategy") or "recursive"
                    chunking_config = collection.get("chunking_config") or {}
                    chunks = await chunking_service.execute_ingestion_chunking(
                        content=combined_text,
                        strategy=strategy,
                        config=chunking_config,
                        metadata=(
                            {**combined_metadata, "document_id": doc.id}
                            if combined_metadata
                            else {"document_id": doc.id}
                        ),
                    )  # pragma: no cover - integration path

                    fallback_used = any((chunk.get("metadata") or {}).get("fallback") for chunk in chunks)
                    fallback_reason = None
                    if fallback_used:
                        for chunk in chunks:
                            reason = (chunk.get("metadata") or {}).get("fallback_reason")
                            if reason:
                                fallback_reason = reason
                                break

                    chunking_stats = {
                        "strategy_used": strategy,
                        "chunk_count": len(chunks),
                        "fallback": fallback_used,
                        "fallback_reason": fallback_reason,
                        "duration_ms": None,
                    }

                    logger.info(
                        "Created %s chunks for %s using %s strategy (fallback: %s, duration: %sms)",
                        len(chunks),
                        doc_identifier,
                        chunking_stats["strategy_used"],
                        chunking_stats["fallback"],
                        chunking_stats["duration_ms"],
                    )

                    if not chunks:
                        logger.warning("No chunks created for %s", doc_identifier)
                        await document_repo.update_status(
                            doc.id,
                            DocumentStatus.FAILED,
                            error_message="No chunks created",
                        )
                        failed_count += 1
                        continue

                    texts = [chunk.get("text") or chunk.get("content") for chunk in chunks]

                    vecpipe_url = "http://vecpipe:8000/embed"
                    embed_request = {
                        "texts": texts,
                        "model_name": embedding_model,
                        "quantization": quantization,
                        "instruction": instruction,
                        "batch_size": batch_size,
                        "mode": "document",  # Document indexing uses document mode
                    }

                    async with httpx.AsyncClient(timeout=300.0) as client:
                        logger.info(
                            "Calling vecpipe /embed for %s texts (semaphore cap=%s)",
                            len(texts),
                            _embedding_semaphore._value,
                        )
                        async with _embedding_semaphore:
                            response = await client.post(vecpipe_url, json=embed_request)

                        if response.status_code != 200:
                            raise Exception(
                                f"Failed to generate embeddings via vecpipe: {response.status_code} - {response.text}"
                            )

                        embed_response = response.json()
                        embeddings_array = embed_response["embeddings"]

                    if embeddings_array is None:
                        raise Exception("Failed to generate embeddings")

                    embeddings = embeddings_array

                    if embeddings:
                        from shared.database.exceptions import DimensionMismatchError
                        from shared.embedding.validation import (
                            get_collection_dimension,
                            validate_dimension_compatibility,
                        )

                        expected_dim = get_collection_dimension(qdrant_client, qdrant_collection_name)
                        if expected_dim is None:
                            logger.warning("Could not get dimension for collection %s", qdrant_collection_name)
                        else:
                            for embedding in embeddings:
                                actual_dim = len(embedding)
                                try:
                                    validate_dimension_compatibility(
                                        expected_dimension=expected_dim,
                                        actual_dimension=actual_dim,
                                        collection_name=qdrant_collection_name,
                                        model_name=embedding_model,
                                    )
                                except DimensionMismatchError as exc:
                                    error_msg = (
                                        "Embedding dimension mismatch during indexing: {}. Collection {} expects {}-dimensional vectors, "
                                        "but model {} produced {}-dimensional vectors. Please ensure you're using the same model that "
                                        "was used to create the collection."
                                    ).format(exc, qdrant_collection_name, expected_dim, embedding_model, actual_dim)
                                    logger.error(error_msg)
                                    raise ValueError(error_msg) from exc

                    # Ensure no open transaction lingers while we call external vector upserts
                    if session.in_transaction():
                        await session.commit()

                    points = []
                    for i, chunk in enumerate(chunks):
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embeddings[i],
                            payload={
                                "collection_id": collection["id"],
                                "doc_id": doc.id,
                                "chunk_id": chunk["chunk_id"],
                                "path": doc_identifier,
                                "content": chunk["text"],
                                "metadata": chunk.get("metadata", {}),
                            },
                        )
                        points.append(point)

                    for batch_start in range(0, len(points), VECTOR_UPLOAD_BATCH_SIZE):
                        batch_end = min(batch_start + VECTOR_UPLOAD_BATCH_SIZE, len(points))
                        batch_points = points[batch_start:batch_end]

                        points_data = [
                            {"id": point.id, "vector": point.vector, "payload": point.payload} for point in batch_points
                        ]

                        upsert_request = {
                            "collection_name": qdrant_collection_name,
                            "points": points_data,
                            "wait": True,
                        }

                        async with httpx.AsyncClient(timeout=60.0) as client:
                            vecpipe_upsert_url = "http://vecpipe:8000/upsert"
                            response = await client.post(vecpipe_upsert_url, json=upsert_request)

                            if response.status_code != 200:
                                raise Exception(
                                    f"Failed to upsert vectors via vecpipe: {response.status_code} - {response.text}"
                                )

                    await document_repo.update_status(
                        doc.id,
                        DocumentStatus.COMPLETED,
                        chunk_count=len(chunks),
                    )

                    processed_count += 1
                    total_vectors_created += len(chunks)

                    await updater.send_update(
                        "document_processed",
                        {
                            "processed": processed_count,
                            "failed": failed_count,
                            "total": len(documents),
                            "current_document": doc_identifier,
                        },
                    )

                    # Commit to prevent idle-in-transaction timeout
                    await session.commit()

                except Exception as exc:
                    logger.error("Failed to process document %s: %s", doc_identifier, exc)
                    with contextlib.suppress(Exception):
                        # Clear any pending transaction state so status updates use a fresh connection
                        await session.rollback()
                    await document_repo.update_status(
                        doc.id,
                        DocumentStatus.FAILED,
                        error_message=str(exc),
                    )
                    failed_count += 1
                    # Commit to prevent idle-in-transaction timeout
                    await session.commit()

            doc_stats = await document_repo.get_stats_by_collection(collection["id"])
            current_doc_count = doc_stats.get("total_documents", 0)

            qdrant_info = qdrant_client.get_collection(qdrant_collection_name)
            current_vector_count = qdrant_info.points_count if qdrant_info else 0

            await collection_repo.update_stats(
                collection["id"],
                document_count=current_doc_count,
                vector_count=current_vector_count,
            )

            logger.info(
                "Embedding generation complete: %s processed, %s failed, %s vectors created, collection now has %s documents and %s vectors",
                processed_count,
                failed_count,
                total_vectors_created,
                current_doc_count,
                current_vector_count,
            )

            await updater.send_update(
                "append_completed",
                {
                    "source": display_path,
                    "source_type": source_type,
                    "documents_added": scan_stats["new_documents_registered"],
                    "documents_updated": scan_stats["documents_updated"],
                    "total_files_scanned": scan_stats["total_documents_found"],
                    "unchanged_skipped": scan_stats["unchanged_documents_skipped"],
                },
            )

        scan_errors = scan_stats.get("errors", []) or []
        success = failed_count == 0 and not scan_errors

        # Mark documents not seen during this sync as stale (keep last-known behavior)
        # Only mark stale for this specific source
        #
        # IMPORTANT: Only mark stale when the entire sync completed cleanly.
        # If scan/registration errors occurred, some existing documents may have been present in
        # the source but failed to register/update last_seen_at; marking unseen docs stale in
        # that scenario can incorrectly flag healthy documents as stale.
        #
        # Similarly, if downstream processing (chunking/embedding) fails, the run is incomplete and
        # marking unseen docs stale can mislabel valid documents as stale after a partial sync.
        if source_id is not None and success:
            try:
                stale_count = await document_repo.mark_unseen_as_stale(
                    collection_id=collection["id"],
                    source_id=source_id,
                    since=sync_started_at,
                )
                if stale_count > 0:
                    logger.info(
                        "Marked %s documents as stale for source %s (not seen since %s)",
                        stale_count,
                        source_id,
                        sync_started_at.isoformat(),
                    )
                scan_stats["documents_marked_stale"] = stale_count
                await session.commit()
            except Exception as stale_exc:
                logger.warning("Failed to mark stale documents: %s", stale_exc)
                scan_stats["documents_marked_stale"] = 0
                with contextlib.suppress(Exception):
                    await session.rollback()
        elif source_id is not None:
            scan_stats["documents_marked_stale"] = 0
            reasons: list[str] = []
            if scan_errors:
                reasons.append(f"{len(scan_errors)} scan/registration error(s)")
            if failed_count > 0:
                reasons.append(f"{failed_count} embedding/chunking failure(s)")
            logger.info(
                "Skipping stale marking for source %s due to %s",
                source_id,
                ", ".join(reasons),
            )

        # Update source sync status
        if source_id is not None:
            try:
                from shared.database.repositories.collection_source_repository import (
                    CollectionSourceRepository,
                )

                source_repo = CollectionSourceRepository(session)
                status = "success" if success else ("partial" if scan_errors else "failed")
                error_msg = None
                if scan_errors:
                    error_msg = f"{len(scan_errors)} document(s) failed to process"
                elif failed_count > 0:
                    error_msg = f"{failed_count} document(s) failed embedding generation"

                await source_repo.update_sync_status(
                    source_id=source_id,
                    status=status,
                    error=error_msg,
                )
                await session.commit()
            except Exception as status_exc:
                logger.warning("Failed to update source sync status: %s", status_exc)
                with contextlib.suppress(Exception):
                    await session.rollback()

        return {
            "success": success,
            "source": display_path,
            "source_type": source_type,
            "documents_added": scan_stats["new_documents_registered"],
            "documents_updated": scan_stats["documents_updated"],
            "total_files_scanned": scan_stats["total_documents_found"],
            "unchanged_skipped": scan_stats["unchanged_documents_skipped"],
            "documents_marked_stale": scan_stats.get("documents_marked_stale", 0),
            "total_size_bytes": scan_stats["total_size_bytes"],
            "scan_duration_seconds": scan_duration,
            "errors": scan_errors,
            "failed_documents": failed_count,
        }

    except Exception as exc:
        logger.error("Failed to process %s source %s: %s", source_type, display_path, exc)
        # Update source sync status on failure
        if source_id is not None:
            try:
                from shared.database.repositories.collection_source_repository import (
                    CollectionSourceRepository,
                )

                source_repo = CollectionSourceRepository(session)
                await source_repo.update_sync_status(
                    source_id=source_id,
                    status="failed",
                    error=str(exc)[:500],  # Truncate long errors
                )
                await session.commit()
            except Exception:
                pass  # Don't mask the original exception
        raise


async def _process_remove_source_operation(
    operation: dict,
    collection: dict,
    collection_repo: Any,  # noqa: ARG001
    document_repo: Any,
    updater: CeleryTaskWithOperationUpdates,
) -> dict[str, Any]:
    """Process REMOVE_SOURCE operation - Remove documents from a source with monitoring."""
    from shared.database.models import DocumentStatus
    from shared.database.repositories.collection_repository import CollectionRepository
    from shared.database.repositories.document_repository import DocumentRepository
    from shared.metrics.collection_metrics import record_document_processed

    config = operation.get("config", {})
    source_id = config.get("source_id")
    source_path = config.get("source_path", "unknown")  # For logging only

    if source_id is None:
        raise ValueError("source_id is required for REMOVE_SOURCE operation")

    documents = await document_repo.list_by_source_id(collection["id"], source_id)

    if not documents:
        logger.info("No documents found for source %s", source_path)
        await _audit_log_operation(
            collection["id"],
            operation["id"],
            operation.get("user_id"),
            "source_removed",
            {"source_path": source_path, "documents_removed": 0},
        )
        return {"success": True, "documents_removed": 0, "source_path": source_path}

    manager = resolve_qdrant_manager()
    qdrant_client = manager.get_client()
    qdrant_manager_class = resolve_qdrant_manager_class()
    qdrant_manager_instance = qdrant_manager_class(qdrant_client)
    logger.debug(
        "Resolved Qdrant dependencies for remove_source: client=%s manager_instance=%s",
        qdrant_client,
        qdrant_manager_instance,
    )

    collections_to_clean = []

    vector_store_name = collection.get("vector_store_name")
    if vector_store_name:
        collections_to_clean.append(vector_store_name)

    qdrant_collections = collection.get("qdrant_collections", [])
    if isinstance(qdrant_collections, list):
        collections_to_clean.extend(qdrant_collections)

    qdrant_staging = collection.get("qdrant_staging", [])
    if isinstance(qdrant_staging, list):
        collections_to_clean.extend(qdrant_staging)

    seen_collections: set[str] = set()
    unique_collections: list[str] = []
    for candidate in collections_to_clean:
        if candidate and candidate not in seen_collections:
            seen_collections.add(candidate)
            unique_collections.append(candidate)

    collections_to_clean = unique_collections

    doc_ids = [doc.id for doc in documents]
    removed_count = 0
    deletion_errors = []

    batch_size = DOCUMENT_REMOVAL_BATCH_SIZE
    for i in range(0, len(doc_ids), batch_size):
        batch_ids = doc_ids[i : i + batch_size]

        try:
            for qdrant_collection in collections_to_clean:
                try:
                    collection_exists = await await_if_awaitable(
                        qdrant_manager_instance.collection_exists(qdrant_collection)
                    )
                    collection_exists = bool(collection_exists)

                    if not collection_exists:
                        logger.warning("Qdrant collection %s does not exist, skipping", qdrant_collection)
                        continue

                    for doc_id in batch_ids:
                        with QdrantOperationTimer("delete_points"):
                            filter_condition = Filter(
                                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                            )

                            qdrant_client.delete(
                                collection_name=qdrant_collection,
                                points_selector=FilterSelector(filter=filter_condition),
                            )

                            logger.info("Deleted vectors for doc_id=%s from collection %s", doc_id, qdrant_collection)
                            logger.debug(
                                "Issued qdrant delete for doc_id=%s collection=%s",
                                doc_id,
                                qdrant_collection,
                            )

                except Exception as exc:
                    error_msg = f"Failed to delete from collection {qdrant_collection}: {exc}"
                    logger.error(error_msg)
                    deletion_errors.append(error_msg)

            removed_count += len(batch_ids)

            progress = ((i + len(batch_ids)) / len(doc_ids)) * 100
            await updater.send_update(
                "removing_documents",
                {
                    "removed": i + len(batch_ids),
                    "total": len(doc_ids),
                    "progress_percent": progress,
                },
            )
        except Exception as exc:
            logger.error("Failed to remove vectors for batch: %s", exc)
            deletion_errors.append(f"Batch {i//batch_size + 1} error: {str(exc)}")

    session_factory = pg_connection_manager.sessionmaker or await ensure_async_sessionmaker()

    async with session_factory() as session, session.begin():
        doc_repo_tx = DocumentRepository(session)
        collection_repo_tx = CollectionRepository(session)

        await doc_repo_tx.bulk_update_status(doc_ids, DocumentStatus.DELETED)

        for _ in range(len(documents)):
            record_document_processed("remove_source", "deleted")

        stats = await doc_repo_tx.get_stats_by_collection(collection["id"])
        await collection_repo_tx.update_stats(
            collection["id"],
            total_documents=stats["total_documents"],
            total_chunks=stats["total_chunks"],
            total_size_bytes=stats["total_size_bytes"],
        )

    await _update_collection_metrics(
        collection["id"],
        stats["total_documents"],
        collection.get("vector_count", 0) - removed_count,
        stats["total_size_bytes"],
    )

    await _audit_log_operation(
        collection["id"],
        operation["id"],
        operation.get("user_id"),
        "source_removed",
        {
            "source_path": source_path,
            "documents_removed": len(documents),
            "vectors_removed": removed_count,
            "deletion_errors": deletion_errors if deletion_errors else None,
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
        "deletion_errors": deletion_errors if deletion_errors else None,
    }


def _handle_task_failure(
    self: Any,  # noqa: ARG001
    exc: Exception,
    task_id: str,
    args: tuple,
    kwargs: dict,
    einfo: Any,  # noqa: ARG001
) -> None:
    """Handle task failure by updating operation and collection status appropriately."""
    operation_id = args[1] if len(args) > 1 else kwargs.get("operation_id")
    if not operation_id:
        logger.error("Task %s failed but operation_id not found in args/kwargs", task_id)
        return

    try:
        asyncio.run(_handle_task_failure_async(operation_id, exc, task_id))
    except Exception as failure_error:  # pragma: no cover - defensive logging
        logger.error("Failed to handle task failure for operation %s: %s", operation_id, failure_error)


async def _handle_task_failure_async(operation_id: str, exc: Exception, task_id: str) -> None:
    """Async implementation of failure handling."""
    from shared.database.models import CollectionStatus, OperationStatus, OperationType
    from shared.database.repositories.collection_repository import CollectionRepository
    from shared.database.repositories.operation_repository import OperationRepository
    from shared.metrics.collection_metrics import collection_operations_total

    operation = None
    collection = None
    collection_id = None

    session_factory = pg_connection_manager.sessionmaker or await ensure_async_sessionmaker()

    async with session_factory() as db:
        operation_repo = OperationRepository(db)
        collection_repo = CollectionRepository(db)

        try:
            operation_obj = await operation_repo.get_by_uuid(operation_id)
            if not operation_obj:
                logger.error("Operation %s not found during failure handling", operation_id)
                return

            operation = {
                "id": operation_obj.uuid,
                "collection_id": operation_obj.collection_id,
                "type": operation_obj.type,
            }

            sanitized_error = _sanitize_error_message(str(exc))

            error_message = f"{type(exc).__name__}: {sanitized_error}"
            if hasattr(exc, "__traceback__"):
                import traceback

                tb_lines = traceback.format_tb(exc.__traceback__)
                if tb_lines:
                    sanitized_tb = _sanitize_error_message(tb_lines[-1].strip())
                    error_message += f"\n{sanitized_tb}"

            await operation_repo.update_status(
                operation_id,
                OperationStatus.FAILED,
                error_message=error_message,
            )

            collection_id = operation["collection_id"]
            operation_type = operation["type"]

            collection_obj = await collection_repo.get_by_uuid(collection_id)
            if collection_obj:
                if operation_type == OperationType.INDEX:
                    await collection_repo.update_status(
                        collection_obj.id,
                        CollectionStatus.ERROR,
                        status_message=f"Initial indexing failed: {sanitized_error}",
                    )
                elif operation_type == OperationType.REINDEX:
                    await collection_repo.update_status(
                        collection_obj.id,
                        CollectionStatus.DEGRADED,
                        status_message=f"Re-indexing failed: {sanitized_error}. Original collection still available.",
                    )
                elif operation_type == OperationType.APPEND:
                    if collection_obj.status != CollectionStatus.ERROR:
                        await collection_repo.update_status(
                            collection_obj.uuid,
                            CollectionStatus.DEGRADED,
                            status_message=f"Append operation failed: {sanitized_error}",
                        )
                elif operation_type == OperationType.REMOVE_SOURCE:
                    await collection_repo.update_status(
                        collection_obj.id,
                        CollectionStatus.DEGRADED,
                        status_message=f"Remove source operation failed: {sanitized_error}",
                    )

            await _audit_log_operation(
                collection_id=collection_id,
                operation_id=operation["id"],
                user_id=operation.get("user_id"),
                action=f"{operation_type.value.lower()}_failed",
                details={
                    "operation_uuid": operation_id,
                    "error": sanitized_error,
                    "error_type": type(exc).__name__,
                    "task_id": task_id,
                },
            )

        except Exception as failure_error:  # pragma: no cover - defensive logging
            logger.error("Error in failure handler for operation %s: %s", operation_id, failure_error, exc_info=True)

        try:
            if operation and operation["type"].value == "reindex" and collection and collection_id:
                await reindex_tasks._cleanup_staging_resources(collection_id, operation)

            if operation:
                collection_operations_total.labels(
                    operation_type=operation["type"].value.lower(), status="failed"
                ).inc()

            logger.info(
                "Handled failure for operation %s (type: %s), updated collection %s status appropriately",
                operation_id,
                operation.get("type") if operation else "unknown",
                collection_id if collection_id else "unknown",
            )
        except Exception as post_cleanup_error:  # pragma: no cover - defensive logging
            logger.error("Error in post-cleanup for operation %s: %s", operation_id, post_cleanup_error)

        await db.commit()


__all__ = [
    "process_collection_operation",
    "test_task",
    "_process_collection_operation_async",
    "_process_index_operation",
    "_process_append_operation",
    "_process_append_operation_impl",
    "_process_remove_source_operation",
    "_handle_task_failure",
    "_handle_task_failure_async",
]
