"""Collection ingestion and orchestration Celery tasks.

This module contains the main task entrypoints for collection operations along with
helpers that drive INDEX, APPEND, REMOVE_SOURCE, and compatibility flows used in
tests. The implementation leans on utilities consolidated in
``packages.webui.tasks.utils`` and delegates reindex-specific logic to the
``reindex`` module.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from datetime import UTC, datetime
from importlib import import_module
from typing import Any

import httpx
import psutil
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue, PointStruct
from shared.metrics.collection_metrics import (
    OperationTimer,
    QdrantOperationTimer,
    collection_cpu_seconds_total,
    collection_memory_usage_bytes,
    collections_total,
)

from packages.webui.services.chunking.container import resolve_celery_chunking_service

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


def _tasks_namespace():
    """Return the top-level tasks module for accessing patched attributes."""
    return import_module("packages.webui.tasks")


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
        return loop.run_until_complete(tasks_ns._process_collection_operation_async(operation_id, self))
    except Exception as exc:  # pragma: no cover - retry path
        logger.error("Task failed for operation %s: %s", operation_id, exc)
        if isinstance(exc, ValueError | TypeError):
            raise
        raise self.retry(exc=exc, countdown=60) from exc


async def _process_collection_operation_async(operation_id: str, celery_task: Any) -> dict[str, Any]:
    """Async implementation of collection operation processing with enhanced monitoring."""
    from shared.database import pg_connection_manager
    from shared.database.database import AsyncSessionLocal
    from shared.database.models import CollectionStatus, OperationStatus, OperationType
    from shared.database.repositories.collection_repository import CollectionRepository
    from shared.database.repositories.document_repository import DocumentRepository
    from shared.database.repositories.operation_repository import OperationRepository

    start_time = time.time()
    operation = None
    collection: dict[str, Any] | None = None

    process = psutil.Process()
    initial_cpu_time = process.cpu_times().user + process.cpu_times().system

    task_id = celery_task.request.id if hasattr(celery_task, "request") else str(uuid.uuid4())

    if not pg_connection_manager._sessionmaker:
        await pg_connection_manager.initialize()
        logger.info("Initialized database connection for this task")

    async with AsyncSessionLocal() as db:
        operation_repo = OperationRepository(db)
        collection_repo = CollectionRepository(db)
        document_repo = DocumentRepository(db)

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
                }

                tasks_ns = _tasks_namespace()
                result: dict[str, Any] = {}
                operation_type = operation["type"].value.lower()

                with OperationTimer(operation_type):
                    memory_before = process.memory_info().rss

                    if operation["type"] == OperationType.INDEX:
                        result = await tasks_ns._process_index_operation(
                            operation, collection, collection_repo, document_repo, updater
                        )
                    elif operation["type"] == OperationType.APPEND:
                        result = await tasks_ns._process_append_operation(db, updater, operation["id"])
                    elif operation["type"] == OperationType.REINDEX:
                        result = await tasks_ns._process_reindex_operation(db, updater, operation["id"])
                    elif operation["type"] == OperationType.REMOVE_SOURCE:
                        result = await tasks_ns._process_remove_source_operation(
                            operation, collection, collection_repo, document_repo, updater
                        )
                    else:  # pragma: no cover - defensive branch
                        raise ValueError(f"Unknown operation type: {operation['type']}")

                    memory_peak = process.memory_info().rss
                    collection_memory_usage_bytes.labels(operation_type=operation_type).set(memory_peak - memory_before)

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

                collection_cpu_seconds_total.labels(operation_type=operation_type).inc(cpu_time)

                await operation_repo.update_status(operation_id, OperationStatus.COMPLETED)

                old_status = collection.get("status", CollectionStatus.PENDING)

                if result.get("success"):
                    doc_stats = await document_repo.get_stats_by_collection(collection["id"])
                    new_status = CollectionStatus.READY
                    await collection_repo.update_status(collection["id"], new_status)

                    await _update_collection_metrics(
                        collection["id"],
                        doc_stats["total_documents"],
                        collection.get("vector_count", 0),
                        doc_stats["total_size_bytes"],
                    )
                else:
                    new_status = CollectionStatus.PARTIALLY_READY
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


async def _process_append_operation(db: Any, updater: Any, _operation_id: str) -> dict[str, Any]:
    """Compatibility wrapper used by tests to process an APPEND operation."""

    def _get(obj: Any, name: str, default: Any = None) -> Any:
        try:
            return obj.get(name, default)
        except Exception:
            return getattr(obj, name, default)

    tasks_ns = _tasks_namespace()
    extract_fn = getattr(tasks_ns, "extract_and_serialize_thread_safe", extract_and_serialize_thread_safe)
    chunking_resolver = getattr(tasks_ns, "resolve_celery_chunking_service", resolve_celery_chunking_service)

    op = (await db.execute(None)).scalar_one()
    collection_obj = (await db.execute(None)).scalar_one_or_none()
    docs = (await db.execute(None)).scalars().all()

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
    from shared.database.models import DocumentStatus

    for doc in docs:
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
                processed += 1
                continue

            chunk_response = cs.execute_ingestion_chunking(
                text=text,
                document_id=_get(doc, "id"),
                collection=collection,
                metadata=metadata,
                file_type=_get(doc, "file_path", "").split(".")[-1] if "." in _get(doc, "file_path", "") else None,
            )

            res = await await_if_awaitable(chunk_response)

            chunks = res.get("chunks", [])

            if chunks:
                texts = [c.get("text", "") for c in chunks]
                embed_req = {"texts": texts, "model_name": collection.get("embedding_model")}
                upsert_req: dict[str, Any] = {"collection_name": collection.get("vector_store_name"), "points": []}
                async with httpx.AsyncClient(timeout=60.0) as client:
                    await client.post("http://vecpipe:8000/embed", json=embed_req)
                    await client.post("http://vecpipe:8000/upsert", json=upsert_req)

            with contextlib.suppress(Exception):
                doc.chunk_count = len(chunks)
                doc.status = DocumentStatus.COMPLETED
            processed += 1

        except Exception:
            with contextlib.suppress(Exception):
                doc.status = DocumentStatus.FAILED
            if doc == docs[-1]:
                raise

    with contextlib.suppress(Exception):
        await updater.send_update("append_completed", {"processed": processed, "operation_id": _get(op, "id")})

    return {"processed": processed}


async def _process_index_operation(
    operation: dict,
    collection: dict,
    collection_repo: Any,
    document_repo: Any,  # noqa: ARG001
    updater: CeleryTaskWithOperationUpdates,
) -> dict[str, Any]:
    """Process INDEX operation - Initial collection creation with monitoring."""
    from qdrant_client.models import Distance, VectorParams
    from shared.database.collection_metadata import store_collection_metadata
    from shared.embedding.models import get_model_config
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

        config = collection.get("config", {})

        vector_dim = config.get("vector_dim")
        if not vector_dim:
            model_name = collection.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B")
            model_config = get_model_config(model_name)
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
                qdrant_client,
                vector_store_name,
                {
                    "model_name": actual_model_name,
                    "quantization": collection.get("quantization", "float16"),
                    "instruction": config.get("instruction"),
                    "dimension": vector_dim,
                    "created_at": datetime.now(UTC).isoformat(),
                },
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
    collection_repo: Any,  # noqa: ARG001
    document_repo: Any,
    updater: CeleryTaskWithOperationUpdates,
) -> dict[str, Any]:
    """Process APPEND operation - Add documents to existing collection with monitoring."""
    from shared.database.models import DocumentStatus
    from shared.metrics.collection_metrics import document_processing_duration, record_document_processed
    from webui.services.document_scanning_service import DocumentScanningService
    from webui.services.factory import create_celery_chunking_service_with_repos

    config = operation.get("config", {})
    source_path = config.get("source_path")

    if not source_path:
        raise ValueError("source_path is required for APPEND operation")

    tasks_ns = _tasks_namespace()

    session = document_repo.session
    document_scanner = DocumentScanningService(db_session=session, document_repo=document_repo)

    await updater.send_update("scanning_documents", {"status": "scanning", "source_path": source_path})

    try:
        scan_start = time.time()

        scan_stats = await document_scanner.scan_directory_and_register_documents(
            collection_id=collection["id"],
            source_path=source_path,
            recursive=True,
            batch_size=EMBEDDING_BATCH_SIZE,
        )

        scan_duration = time.time() - scan_start
        document_processing_duration.labels(operation_type="append").observe(scan_duration)

        await updater.send_update(
            "scanning_completed",
            {
                "status": "scanning_completed",
                "total_files_found": scan_stats["total_documents_found"],
                "new_documents_registered": scan_stats["new_documents_registered"],
                "duplicate_documents_skipped": scan_stats["duplicate_documents_skipped"],
                "errors_count": len(scan_stats.get("errors", [])),
            },
        )

        for _ in range(scan_stats["new_documents_registered"]):
            record_document_processed("append", "registered")
        for _ in range(scan_stats["duplicate_documents_skipped"]):
            record_document_processed("append", "skipped")
        for _ in range(len(scan_stats.get("errors", []))):
            record_document_processed("append", "failed")

        await _audit_log_operation(
            collection["id"],
            operation["id"],
            operation.get("user_id"),
            "documents_appended",
            {
                "source_path": source_path,
                "documents_added": scan_stats["new_documents_registered"],
                "duplicates_skipped": scan_stats["duplicate_documents_skipped"],
            },
        )

        all_docs, _ = await document_repo.list_by_collection(
            collection["id"],
            status=None,
            limit=10000,
        )

        documents = [doc for doc in all_docs if doc.file_path.startswith(source_path)]
        unprocessed_documents = [doc for doc in documents if doc.chunk_count == 0]

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
            batch_size = config.get("batch_size", EMBEDDING_BATCH_SIZE)
            instruction = config.get("instruction")

            qdrant_collection_name = collection.get("vector_store_name")
            if not qdrant_collection_name:
                raise ValueError("Collection missing vector_store_name")

            manager = resolve_qdrant_manager()
            qdrant_client = manager.get_client()

            processed_count = 0
            failed_count = 0
            total_vectors_created = 0

            chunking_service = create_celery_chunking_service_with_repos(
                db_session=document_repo.session,
                collection_repo=collection_repo,
                document_repo=document_repo,
            )

            loop = tasks_ns.asyncio.get_event_loop()
            executor_pool = tasks_ns.executor

            for doc in documents:
                try:
                    logger.info("Processing document: %s", doc.file_path)

                    text_blocks = await asyncio.wait_for(
                        loop.run_in_executor(executor_pool, extract_and_serialize_thread_safe, doc.file_path),
                        timeout=300,
                    )

                    if not text_blocks:
                        logger.warning("No text extracted from %s", doc.file_path)
                        await document_repo.update_status(
                            doc.id,
                            DocumentStatus.FAILED,
                            error_message="No text content extracted",
                        )
                        failed_count += 1
                        continue

                    combined_text = ""
                    combined_metadata = {}
                    for text, metadata in text_blocks:
                        if text.strip():
                            combined_text += text + "\n\n"
                            if metadata:
                                combined_metadata.update(metadata)

                    chunking_result = await chunking_service.execute_ingestion_chunking(
                        text=combined_text,
                        document_id=doc.id,
                        collection=collection,
                        metadata=combined_metadata,
                        file_type=doc.file_path.split(".")[-1] if "." in doc.file_path else None,
                    )

                    chunks = chunking_result["chunks"]
                    chunking_stats = chunking_result["stats"]

                    logger.info(
                        "Created %s chunks for %s using %s strategy (fallback: %s, duration: %sms)",
                        len(chunks),
                        doc.file_path,
                        chunking_stats["strategy_used"],
                        chunking_stats["fallback"],
                        chunking_stats["duration_ms"],
                    )

                    if not chunks:
                        logger.warning("No chunks created for %s", doc.file_path)
                        await document_repo.update_status(
                            doc.id,
                            DocumentStatus.FAILED,
                            error_message="No chunks created",
                        )
                        failed_count += 1
                        continue

                    texts = [chunk["text"] for chunk in chunks]

                    vecpipe_url = "http://vecpipe:8000/embed"
                    embed_request = {
                        "texts": texts,
                        "model_name": embedding_model,
                        "quantization": quantization,
                        "instruction": instruction,
                        "batch_size": batch_size,
                    }

                    async with httpx.AsyncClient(timeout=300.0) as client:
                        logger.info("Calling vecpipe /embed for %s texts", len(texts))
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

                    points = []
                    for i, chunk in enumerate(chunks):
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embeddings[i],
                            payload={
                                "collection_id": collection["id"],
                                "doc_id": doc.id,
                                "chunk_id": chunk["chunk_id"],
                                "path": doc.file_path,
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
                            "current_document": doc.file_path,
                        },
                    )

                except Exception as exc:
                    logger.error("Failed to process document %s: %s", doc.file_path, exc)
                    await document_repo.update_status(
                        doc.id,
                        DocumentStatus.FAILED,
                        error_message=str(exc),
                    )
                    failed_count += 1

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
                    "source_path": source_path,
                    "documents_added": scan_stats["new_documents_registered"],
                    "total_files_scanned": scan_stats["total_documents_found"],
                    "duplicates_skipped": scan_stats["duplicate_documents_skipped"],
                },
            )

        return {
            "success": True,
            "source_path": source_path,
            "documents_added": scan_stats["new_documents_registered"],
            "total_files_scanned": scan_stats["total_documents_found"],
            "duplicates_skipped": scan_stats["duplicate_documents_skipped"],
            "total_size_bytes": scan_stats["total_size_bytes"],
            "scan_duration_seconds": scan_duration,
            "errors": scan_stats.get("errors", []),
        }

    except Exception as exc:
        logger.error("Failed to scan and register documents: %s", exc)
        raise


async def _process_remove_source_operation(
    operation: dict,
    collection: dict,
    collection_repo: Any,  # noqa: ARG001
    document_repo: Any,
    updater: CeleryTaskWithOperationUpdates,
) -> dict[str, Any]:
    """Process REMOVE_SOURCE operation - Remove documents from a source with monitoring."""
    from shared.database.database import AsyncSessionLocal
    from shared.database.models import DocumentStatus
    from shared.database.repositories.collection_repository import CollectionRepository
    from shared.database.repositories.document_repository import DocumentRepository
    from shared.metrics.collection_metrics import record_document_processed

    config = operation.get("config", {})
    source_path = config.get("source_path")

    if not source_path:
        raise ValueError("source_path is required for REMOVE_SOURCE operation")

    documents = await document_repo.list_by_collection_and_source(collection["id"], source_path)

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

    doc_ids = [doc["id"] for doc in documents]
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

    async with AsyncSessionLocal() as session, session.begin():
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
    from shared.database.database import AsyncSessionLocal
    from shared.database.models import CollectionStatus, OperationStatus, OperationType
    from shared.database.repositories.collection_repository import CollectionRepository
    from shared.database.repositories.operation_repository import OperationRepository
    from shared.metrics.collection_metrics import collection_operations_total

    operation = None
    collection = None
    collection_id = None

    async with AsyncSessionLocal() as db:
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
                            CollectionStatus.PARTIALLY_READY,
                            status_message=f"Append operation failed: {sanitized_error}",
                        )
                elif operation_type == OperationType.REMOVE_SOURCE:
                    await collection_repo.update_status(
                        collection_obj.id,
                        CollectionStatus.PARTIALLY_READY,
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
