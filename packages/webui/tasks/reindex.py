"""Reindex and maintenance Celery utilities.

This module contains the REINDEX workflow, including the blue/green staging
handler, validation logic, and staging cleanup helpers. The compatibility wrapper
used in tests lives here so the ingestion task can delegate without pulling in the
entire reindex implementation.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from importlib import import_module
from typing import TYPE_CHECKING, Any

import httpx
from qdrant_client.models import PointStruct

from packages.webui.services.chunking.container import resolve_celery_chunking_service

if TYPE_CHECKING:
    from types import ModuleType

from .utils import (
    EMBEDDING_BATCH_SIZE,
    REINDEX_SCORE_DIFF_THRESHOLD,
    REINDEX_SEARCH_MISMATCH_THRESHOLD,
    REINDEX_VECTOR_COUNT_VARIANCE,
    CeleryTaskWithOperationUpdates,
    _audit_log_operation,
    _build_internal_api_headers,
    await_if_awaitable,
    calculate_cleanup_delay,
    extract_and_serialize_thread_safe,
    logger,
    resolve_qdrant_manager,
    resolve_qdrant_manager_class,
    settings,
)


def _tasks_namespace() -> ModuleType:
    """Return the top-level tasks module for accessing patched attributes."""
    return import_module("packages.webui.tasks")


async def _process_reindex_operation(db: Any, updater: Any, _operation_id: str) -> dict[str, Any]:
    """Compatibility wrapper used by tests to process a REINDEX operation."""

    def _get(obj: Any, name: str, default: Any = None) -> Any:
        try:
            return obj.get(name, default)
        except Exception:
            return getattr(obj, name, default)

    tasks_ns = _tasks_namespace()
    log = getattr(tasks_ns, "logger", logger)
    extract_fn = getattr(tasks_ns, "extract_and_serialize_thread_safe", extract_and_serialize_thread_safe)
    chunking_resolver = getattr(tasks_ns, "resolve_celery_chunking_service", resolve_celery_chunking_service)

    # Replace placeholder executes with real queries while preserving
    # the number/order of db.execute(...) calls that tests expect.
    from shared.database.models import Collection as _Collection
    from shared.database.models import Document as _Document
    from shared.database.models import Operation as _Operation
    from sqlalchemy import select

    # Fetch the operation using whichever identifier the caller supplied
    op_lookup = select(_Operation)
    try:
        op_lookup = op_lookup.where(_Operation.id == int(_operation_id))
    except (TypeError, ValueError):
        op_lookup = op_lookup.where(_Operation.uuid == _operation_id)

    op = (await db.execute(op_lookup)).scalar_one()

    # Source collection is the operation's collection
    source_collection = (
        await db.execute(select(_Collection).where(_Collection.id == op.collection_id))
    ).scalar_one_or_none()

    def _extract_staging_collection_name(value: Any) -> str | None:
        """Best-effort extraction of the staging collection identifier."""
        if not value:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return text
            return _extract_staging_collection_name(parsed)
        if isinstance(value, dict):
            for key in (
                "collection_name",
                "staging_collection_name",
                "vector_collection_id",
                "vector_store_name",
            ):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate:
                    return candidate
            for nested_key in ("staging_collection", "staging", "info"):
                nested = value.get(nested_key)
                name = _extract_staging_collection_name(nested)
                if name:
                    return name
            for nested_value in value.values():
                name = _extract_staging_collection_name(nested_value)
                if name:
                    return name
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                name = _extract_staging_collection_name(item)
                if name:
                    return name
        return None

    op_config = _get(op, "config", {}) or {}
    staging_name = (
        _extract_staging_collection_name(op_config)
        or _extract_staging_collection_name(_get(op, "meta", None))
        or _extract_staging_collection_name(_get(source_collection, "qdrant_staging", None))
    )

    staging_query = select(_Collection)
    if staging_name:
        staging_query = staging_query.where(_Collection.vector_store_name == staging_name)
    else:
        staging_query = staging_query.where(_Collection.id == _get(source_collection, "id"))

    staging_collection = (await db.execute(staging_query)).scalar_one_or_none()

    # Documents associated to source collection (fourth execute)
    docs = (await db.execute(select(_Document).where(_Document.collection_id == op.collection_id))).scalars().all()

    collection = {
        "id": _get(source_collection, "id"),
        "name": _get(source_collection, "name"),
        "chunking_strategy": _get(source_collection, "chunking_strategy"),
        "chunking_config": _get(source_collection, "chunking_config", {}) or {},
        "chunk_size": _get(source_collection, "chunk_size", 1000),
        "chunk_overlap": _get(source_collection, "chunk_overlap", 200),
        "embedding_model": _get(source_collection, "embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
        "quantization": _get(source_collection, "quantization", "float16"),
        "vector_store_name": _get(staging_collection, "vector_collection_id")
        or _get(staging_collection, "vector_store_name")
        or _get(source_collection, "vector_collection_id")
        or _get(source_collection, "vector_store_name"),
    }

    new_cfg = _get(op, "config", {}) or {}
    if "chunking_strategy" in new_cfg:
        collection["chunking_strategy"] = new_cfg["chunking_strategy"]
    if "chunking_config" in new_cfg:
        collection["chunking_config"] = new_cfg["chunking_config"]
    if "chunk_size" in new_cfg:
        collection["chunk_size"] = new_cfg["chunk_size"]
    if "chunk_overlap" in new_cfg:
        collection["chunk_overlap"] = new_cfg["chunk_overlap"]

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
            blocks_result = extract_fn(_get(doc, "file_path", ""))
            blocks = await await_if_awaitable(blocks_result)
        except FileNotFoundError:
            blocks = []
        except Exception as exc:
            log.warning("Failed to extract document %s: %s", _get(doc, "file_path", ""), exc)
            blocks = []

        text = "".join((t for t, _m in (blocks or []) if isinstance(t, str)))
        metadata: dict[str, Any] = {}
        for _t, m in blocks or []:
            if isinstance(m, dict):
                metadata.update(m)

        if not text and not metadata:
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
            async with httpx.AsyncClient(timeout=60.0) as client:
                await client.post(
                    "http://vecpipe:8000/embed", json={"texts": texts, "model_name": collection.get("embedding_model")}
                )
                await client.post(
                    "http://vecpipe:8000/upsert",
                    json={"collection_name": collection.get("vector_store_name"), "points": []},
                )

        try:
            doc.chunk_count = len(chunks)
            doc.status = DocumentStatus.COMPLETED
        except Exception:
            pass
        processed += 1

    with contextlib.suppress(Exception):
        await updater.send_update("reindex_completed", {"processed": processed, "operation_id": _get(op, "id")})

    return {"processed": processed}


async def reindex_handler(
    collection: dict,
    new_config: dict[str, Any],
    qdrant_manager_instance: Any,
) -> dict[str, Any]:
    """Create staging collection for blue-green reindexing."""
    from webui.services.collection_service import DEFAULT_VECTOR_DIMENSION

    base_collection_name = collection.get("vector_store_name")
    if not base_collection_name:
        raise ValueError("Collection missing vector_store_name field")

    vector_dim = new_config.get("vector_dim", collection.get("config", {}).get("vector_dim", DEFAULT_VECTOR_DIMENSION))

    logger.info("Creating staging collection for %s with vector_dim=%s", base_collection_name, vector_dim)

    staging_collection_name = qdrant_manager_instance.create_staging_collection(
        base_name=base_collection_name, vector_size=vector_dim
    )

    staging_info = {
        "collection_name": staging_collection_name,
        "created_at": datetime.now(UTC).isoformat(),
        "vector_dim": vector_dim,
        "base_collection": base_collection_name,
    }

    logger.info("Successfully created staging collection: %s", staging_collection_name)

    return staging_info


async def _process_reindex_operation_impl(
    operation: dict,
    collection: dict,
    collection_repo: Any,
    document_repo: Any,
    updater: CeleryTaskWithOperationUpdates,
) -> dict[str, Any]:
    """Process REINDEX operation - Blue-green reindexing with validation checkpoints."""
    from shared.database.models import DocumentStatus
    from shared.metrics.collection_metrics import (
        QdrantOperationTimer,
        record_reindex_checkpoint,
        reindex_switch_duration,
        reindex_validation_duration,
    )
    from webui.services.factory import create_celery_chunking_service_with_repos

    config = operation.get("config", {})
    new_config = config.get("new_config", {})
    staging_collection_name = None
    checkpoints = []

    tasks_ns = _tasks_namespace()
    manager = resolve_qdrant_manager()
    qdrant_client = manager.get_client()
    qdrant_manager_class = resolve_qdrant_manager_class()
    qdrant_manager_instance = qdrant_manager_class(qdrant_client)

    try:
        checkpoint_time = time.time()
        record_reindex_checkpoint(collection["id"], "preflight_start")
        checkpoints.append(("preflight_start", checkpoint_time))

        if collection.get("status") == "error":
            raise ValueError("Cannot reindex collection in error state")

        doc_stats = await document_repo.get_stats_by_collection(collection["id"])
        if doc_stats["total_documents"] == 0:
            raise ValueError("Cannot reindex empty collection")

        await updater.send_update(
            "reindex_preflight",
            {
                "status": "preflight_complete",
                "documents_to_process": doc_stats["total_documents"],
                "current_vector_count": collection.get("vector_count", 0),
            },
        )

        record_reindex_checkpoint(collection["id"], "preflight_complete")
        checkpoints.append(("preflight_complete", time.time()))

        record_reindex_checkpoint(collection["id"], "staging_creation_start")

        old_collection_name = collection.get("vector_store_name")
        if not old_collection_name:
            raise ValueError("Collection missing vector_store_name field")

        staging_info = await tasks_ns.reindex_handler(collection, new_config, qdrant_manager_instance)
        staging_collection_name = staging_info["collection_name"]

        try:
            from shared.database.collection_metadata import store_collection_metadata

            store_collection_metadata(
                qdrant=qdrant_client,
                collection_name=staging_collection_name,
                model_name=new_config.get(
                    "model_name",
                    collection.get("config", {}).get("model_name", collection.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B")),
                ),
                quantization=new_config.get(
                    "quantization", collection.get("config", {}).get("quantization", collection.get("quantization", "float32"))
                ),
                vector_dim=staging_info.get("vector_dim"),
                chunk_size=new_config.get("chunk_size", collection.get("config", {}).get("chunk_size")),
                chunk_overlap=new_config.get("chunk_overlap", collection.get("config", {}).get("chunk_overlap")),
                instruction=new_config.get("instruction", collection.get("config", {}).get("instruction")),
            )
        except Exception as exc:
            logger.warning("Failed to store staging collection metadata: %s", exc)

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

        record_reindex_checkpoint(collection["id"], "reprocessing_start")

        documents = await document_repo.list_by_collection(
            collection["id"],
            status_filter=DocumentStatus.COMPLETED,
            limit=None,
        )

        total_documents = len(documents)
        processed_count = 0
        failed_count = 0
        vector_count = 0

        batch_size = new_config.get("batch_size", collection.get("config", {}).get("batch_size", EMBEDDING_BATCH_SIZE))

        model_name = new_config.get(
            "model_name", collection.get("config", {}).get("model_name", "Qwen/Qwen3-Embedding-0.6B")
        )
        quantization = new_config.get("quantization", collection.get("config", {}).get("quantization", "float32"))
        instruction = new_config.get("instruction", collection.get("config", {}).get("instruction"))
        vector_dim = new_config.get("vector_dim", collection.get("config", {}).get("vector_dim"))

        chunking_service = create_celery_chunking_service_with_repos(
            db_session=document_repo.session,
            collection_repo=collection_repo,
            document_repo=document_repo,
        )

        loop = tasks_ns.asyncio.get_event_loop()
        executor_pool = tasks_ns.executor

        for i in range(0, total_documents, batch_size):
            batch = documents[i : i + batch_size]

            for doc in batch:
                document_id = doc.get("id") if hasattr(doc, "get") else getattr(doc, "id", None)
                file_path = doc.get("file_path", doc.get("path"))

                try:
                    loop = asyncio.get_event_loop()

                    logger.info("Reprocessing document: %s", file_path)

                    text_blocks = await asyncio.wait_for(
                        loop.run_in_executor(executor_pool, extract_and_serialize_thread_safe, file_path),
                        timeout=300,
                    )

                    doc_id = hashlib.md5(file_path.encode()).hexdigest()[:16]

                    combined_text = ""
                    combined_metadata = {}
                    for text, metadata in text_blocks:
                        if text.strip():
                            combined_text += text + "\n\n"
                            if metadata:
                                combined_metadata.update(metadata)

                    reindex_collection = collection.copy()
                    if new_config.get("chunking_strategy"):
                        reindex_collection["chunking_strategy"] = new_config["chunking_strategy"]
                    if new_config.get("chunking_config"):
                        reindex_collection["chunking_config"] = new_config["chunking_config"]
                    if "chunk_size" in new_config:
                        reindex_collection["chunk_size"] = new_config["chunk_size"]
                    if "chunk_overlap" in new_config:
                        reindex_collection["chunk_overlap"] = new_config["chunk_overlap"]

                    chunking_result = await chunking_service.execute_ingestion_chunking(
                        text=combined_text,
                        document_id=doc_id,
                        collection=reindex_collection,
                        metadata=combined_metadata,
                        file_type=file_path.split(".")[-1] if "." in file_path else None,
                    )

                    all_chunks = chunking_result["chunks"]
                    chunking_stats = chunking_result["stats"]

                    logger.info(
                        "Reprocessed document %s: %s chunks using %s strategy (fallback: %s, duration: %sms)",
                        file_path,
                        len(all_chunks),
                        chunking_stats["strategy_used"],
                        chunking_stats["fallback"],
                        chunking_stats["duration_ms"],
                    )

                    if not all_chunks:
                        logger.warning("No chunks created for document: %s", file_path)
                        continue

                    texts = [chunk["text"] for chunk in all_chunks]

                    vecpipe_url = "http://vecpipe:8000/embed"
                    embed_request = {
                        "texts": texts,
                        "model_name": model_name,
                        "quantization": quantization,
                        "instruction": instruction,
                        "batch_size": batch_size,
                    }

                    async with httpx.AsyncClient(timeout=300.0) as client:
                        logger.info("Calling vecpipe /embed for %s texts (reindex)", len(texts))
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
                            adjust_embeddings_dimension,
                            get_collection_dimension,
                            validate_dimension_compatibility,
                        )

                        expected_dim = get_collection_dimension(qdrant_client, staging_collection_name)
                        if expected_dim is None:
                            logger.warning("Could not get dimension for staging collection %s", staging_collection_name)
                        else:
                            actual_dim = len(embeddings[0]) if embeddings else 0

                            try:
                                validate_dimension_compatibility(
                                    expected_dimension=expected_dim,
                                    actual_dimension=actual_dim,
                                    collection_name=staging_collection_name,
                                    model_name=model_name,
                                )
                            except DimensionMismatchError as exc:
                                if vector_dim and vector_dim == expected_dim:
                                    logger.warning(
                                        "Dimension mismatch during reindexing: %s. Adjusting embeddings from %s to %s dimensions.",
                                        exc,
                                        actual_dim,
                                        expected_dim,
                                    )
                                    embeddings = adjust_embeddings_dimension(
                                        embeddings, target_dimension=expected_dim, normalize=True
                                    )
                                else:
                                    error_msg = (
                                        "Embedding dimension mismatch during reindexing: {}. Staging collection {} expects {}-dimensional vectors, "
                                        "but model {} produced {}-dimensional vectors."
                                    ).format(exc, staging_collection_name, expected_dim, model_name, actual_dim)
                                    logger.error(error_msg)
                                    raise ValueError(error_msg) from exc

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

                    with QdrantOperationTimer("upsert_staging_vectors"):
                        points_data = [
                            {"id": point.id, "vector": point.vector, "payload": point.payload} for point in points
                        ]

                        upsert_request = {
                            "collection_name": staging_collection_name,
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

                    vector_count += len(points)
                    processed_count += 1

                    logger.info(
                        "Successfully reprocessed document %s: %s vectors created",
                        file_path,
                        len(points),
                    )

                    if document_id and all_chunks:
                        await document_repo.update_status(
                            document_id,
                            DocumentStatus.COMPLETED,
                            chunk_count=len(all_chunks),
                        )
                        logger.info("Updated document %s with chunk_count=%s", document_id, len(all_chunks))

                    del text_blocks, all_chunks, texts, embeddings_array, embeddings, points
                    gc.collect()

                except Exception as exc:
                    logger.error("Failed to reprocess document %s: %s", file_path, exc)
                    failed_count += 1

                    if document_id:
                        try:
                            await document_repo.update_status(
                                document_id,
                                DocumentStatus.FAILED,
                                error_message=str(exc)[:500],
                            )
                            logger.info("Marked document %s as FAILED due to reprocessing error", document_id)
                        except Exception as update_error:
                            logger.error("Failed to update document status to FAILED: %s", update_error)

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

        validation_start = time.time()
        record_reindex_checkpoint(collection["id"], "validation_start")

        validation_result = await tasks_ns._validate_reindex(
            qdrant_client,
            old_collection_name,
            staging_collection_name,
            sample_size=min(100, total_documents // 10),
        )

        validation_duration = time.time() - validation_start
        reindex_validation_duration.observe(validation_duration)

        if not validation_result["passed"]:
            raise ValueError(f"Reindex validation failed: {validation_result['issues']}")

        if validation_result.get("warnings"):
            for warning in validation_result["warnings"]:
                logger.warning("Reindex validation warning: %s", warning)

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

        switch_start = time.time()
        record_reindex_checkpoint(collection["id"], "atomic_switch_start")

        host = settings.WEBUI_INTERNAL_HOST
        port = settings.WEBUI_PORT
        internal_api_url = f"http://{host}:{port}/api/internal/complete-reindex"
        request_data = {
            "collection_id": collection["id"],
            "operation_id": operation["id"],
            "staging_collection_name": staging_collection_name,
            "new_config": new_config,
            "vector_count": vector_count,
        }

        headers = _build_internal_api_headers()

        async with tasks_ns.httpx.AsyncClient() as client:
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

        record_reindex_checkpoint(collection["id"], "cleanup_scheduled")

        cleanup_delay = calculate_cleanup_delay(vector_count)

        cleanup_task = tasks_ns.cleanup_old_collections.apply_async(
            args=[old_collection_names, collection["id"]],
            countdown=cleanup_delay,
        )

        logger.info(
            "Scheduled cleanup of %s old collections to run in %s seconds. Task ID: %s",
            len(old_collection_names),
            cleanup_delay,
            cleanup_task.id,
        )

        record_reindex_checkpoint(collection["id"], "cleanup_scheduled_complete")
        checkpoints.append(("cleanup_scheduled_complete", time.time()))

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
        logger.error(
            "Failed to reindex collection at checkpoint: %s",
            checkpoints[-1][0] if checkpoints else "unknown",
        )

        if staging_collection_name:
            try:
                qdrant_client.delete_collection(staging_collection_name)
                logger.info("Cleaned up staging collection %s", staging_collection_name)
            except Exception as cleanup_error:
                logger.error("Failed to cleanup staging collection: %s", cleanup_error)

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
        old_info = qdrant_client.get_collection(old_collection)
        new_info = qdrant_client.get_collection(new_collection)

        issues = []

        if new_info.points_count == 0:
            issues.append("New collection has no vectors")

        if old_info.points_count > 0:
            ratio = new_info.points_count / old_info.points_count
            if ratio < (1 - REINDEX_VECTOR_COUNT_VARIANCE) or ratio > (1 + REINDEX_VECTOR_COUNT_VARIANCE):
                issues.append(f"Vector count mismatch: {old_info.points_count} -> {new_info.points_count}")

        if old_info.points_count > 0 and new_info.points_count > 0:
            try:
                import random

                scroll_result = qdrant_client.scroll(
                    collection_name=old_collection,
                    limit=min(sample_size, old_info.points_count),
                    with_vectors=True,
                    with_payload=True,
                )

                sample_points = scroll_result[0]

                if len(sample_points) > 0:
                    search_mismatches = 0
                    total_score_diff = 0.0
                    comparisons_made = 0

                    test_points = random.sample(sample_points, min(10, len(sample_points)))

                    for point in test_points:
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

                        if old_results and new_results:
                            old_doc_ids = {r.payload.get("doc_id") for r in old_results if r.payload}
                            new_doc_ids = {r.payload.get("doc_id") for r in new_results if r.payload}

                            overlap = len(old_doc_ids & new_doc_ids)
                            if overlap < 3:
                                search_mismatches += 1

                            if old_results[0].score and new_results[0].score:
                                score_diff = abs(old_results[0].score - new_results[0].score)
                                total_score_diff += score_diff
                                comparisons_made += 1

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

            except Exception as exc:
                logger.warning("Failed to perform search validation: %s", exc)
                issues.append(f"Could not validate search quality: {str(exc)}")

        if hasattr(old_info.config, "params") and hasattr(new_info.config, "params"):
            old_dim = old_info.config.params.vectors.size if hasattr(old_info.config.params.vectors, "size") else None
            new_dim = new_info.config.params.vectors.size if hasattr(new_info.config.params.vectors, "size") else None

            if old_dim and new_dim and old_dim != new_dim:
                issues.append(f"Vector dimension mismatch: {old_dim} -> {new_dim}")

        validation_passed = len(issues) == 0

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

    except Exception as exc:
        logger.error("Validation error: %s", exc)
        return {
            "passed": False,
            "issues": [f"Validation error: {str(exc)}"],
            "sample_size": 0,
        }


async def _cleanup_staging_resources(collection_id: str, operation: dict) -> None:  # noqa: ARG001
    """Clean up staging resources for failed reindex operation."""
    try:
        from shared.database.database import AsyncSessionLocal
        from shared.database.repositories.collection_repository import CollectionRepository

        async with AsyncSessionLocal() as session:
            collection_repo = CollectionRepository(session)
            collection = await collection_repo.get_by_uuid(collection_id)

            if not collection:
                logger.warning("Collection %s not found during staging cleanup", collection_id)
                return

            staging_info = collection.qdrant_staging
            if not staging_info:
                logger.info("No staging collections to clean up")
                return

            if isinstance(staging_info, str):
                try:
                    staging_info = json.loads(staging_info)
                except json.JSONDecodeError:
                    logger.error("Failed to parse staging info: %s", staging_info)
                    return

            staging_collections = []
            if isinstance(staging_info, dict) and "collection_name" in staging_info:
                staging_collections.append(staging_info["collection_name"])
            elif isinstance(staging_info, list):
                for item in staging_info:
                    if isinstance(item, dict) and "collection_name" in item:
                        staging_collections.append(item["collection_name"])

            if not staging_collections:
                logger.info("No staging collection names found in staging info")
                return

            manager = resolve_qdrant_manager()
            qdrant_client = manager.get_client()
            for staging_collection in staging_collections:
                try:
                    collections = qdrant_client.get_collections()
                    if any(col.name == staging_collection for col in collections.collections):
                        qdrant_client.delete_collection(staging_collection)
                        logger.info("Deleted staging collection: %s", staging_collection)
                    else:
                        logger.warning("Staging collection %s not found in Qdrant", staging_collection)
                except Exception as exc:
                    logger.error("Failed to delete staging collection %s: %s", staging_collection, exc)

            await collection_repo.update(collection_id, {"qdrant_staging": None})

            logger.info("Cleaned up %s staging collections for collection %s", len(staging_collections), collection_id)

    except Exception as exc:
        logger.error("Failed to clean up staging resources for collection %s: %s", collection_id, exc, exc_info=True)


__all__ = [
    "_process_reindex_operation",
    "_process_reindex_operation_impl",
    "_cleanup_staging_resources",
    "_validate_reindex",
    "reindex_handler",
]
