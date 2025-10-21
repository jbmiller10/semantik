"""Projection Celery tasks for computing and tracking embedding projections."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

import numpy as np
from shared.database import pg_connection_manager
from shared.database.database import AsyncSessionLocal
from shared.database.models import OperationStatus, ProjectionRunStatus
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.operation_repository import OperationRepository
from shared.database.repositories.projection_run_repository import ProjectionRunRepository

from packages.webui.tasks.utils import (
    CeleryTaskWithOperationUpdates,
    celery_app,
    logger,
    resolve_awaitable_sync,
    resolve_qdrant_manager,
    settings,
    _sanitize_error_message,
)


DEFAULT_SAMPLE_LIMIT = 10_000
QDRANT_SCROLL_BATCH = 1_000


def _ensure_float32(array: np.ndarray) -> np.ndarray:
    """Return ``array`` cast to ``np.float32`` without unnecessary copies."""

    return array.astype(np.float32, copy=False)


def _compute_pca_projection(vectors: np.ndarray) -> dict[str, np.ndarray]:
    """Compute a 2D PCA projection using NumPy's SVD implementation."""

    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D array for PCA, got {vectors.shape!r}")
    if vectors.shape[0] < 2:
        raise ValueError("At least two samples are required for PCA projection")
    if vectors.shape[1] < 2:
        raise ValueError("Vectors must have at least two dimensions for PCA projection")

    mean = vectors.mean(axis=0, keepdims=True)
    centered = vectors - mean

    # Compute SVD on centered data; full_matrices=False keeps outputs minimal.
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    top_components = vt[:2]

    projection = centered @ top_components.T

    total_variance = float(np.square(singular_values).sum())
    top_singular_values = singular_values[:2]
    if total_variance > 0:
        explained_variance_ratio = np.square(top_singular_values) / total_variance
    else:  # Degenerate case where all vectors are identical
        explained_variance_ratio = np.zeros_like(top_singular_values)

    return {
        "projection": _ensure_float32(projection),
        "components": _ensure_float32(top_components),
        "mean": _ensure_float32(mean.squeeze(axis=0)),
        "singular_values": _ensure_float32(top_singular_values),
        "explained_variance_ratio": _ensure_float32(explained_variance_ratio),
    }


def _write_binary(path: Path, array: np.ndarray, dtype: np.dtype) -> None:
    """Write ``array`` to ``path`` as raw binary in the specified dtype."""

    array.astype(dtype, copy=False).tofile(path)


def _write_meta(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@asynccontextmanager
async def _operation_updates(operation_id: str | None) -> AsyncIterator[CeleryTaskWithOperationUpdates | None]:
    """Yield an update publisher when an operation ID is available."""

    if not operation_id:
        yield None
        return

    async with CeleryTaskWithOperationUpdates(operation_id) as updater:
        yield updater


@celery_app.task(bind=True, name="webui.tasks.compute_projection")
def compute_projection(self: Any, projection_id: str) -> dict[str, Any]:  # noqa: ANN401
    """Compute a 2D PCA projection for the requested projection run."""

    logger.info("compute_projection task invoked for projection_id=%s", projection_id)

    try:
        result = resolve_awaitable_sync(_compute_projection_async(projection_id))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Projection computation failed for %s", projection_id)
        return {
            "projection_id": projection_id,
            "status": "failed",
            "message": str(exc),
        }

    return result


async def _compute_projection_async(projection_id: str) -> dict[str, Any]:
    """Async implementation for ``compute_projection``."""

    if not pg_connection_manager._sessionmaker:
        await pg_connection_manager.initialize()
        logger.info("Initialized database connections for projection task")

    async with AsyncSessionLocal() as session:
        projection_repo = ProjectionRunRepository(session)
        operation_repo = OperationRepository(session)
        collection_repo = CollectionRepository(session)

        run = await projection_repo.get_by_uuid(projection_id)
        if not run:
            raise ValueError(f"Projection run {projection_id} not found")

        operation_uuid = getattr(run, "operation_uuid", None)
        collection = await collection_repo.get_by_uuid(run.collection_id)
        if not collection:
            raise ValueError(f"Collection {run.collection_id} for projection {projection_id} not found")

        vector_collection_name = getattr(collection, "vector_store_name", None) or getattr(
            collection, "vector_collection_id", None
        )
        if not vector_collection_name:
            raise ValueError("Collection is missing a vector store name for projection computation")

        config = run.config or {}
        sample_limit = int(
            config.get("sample_size")
            or config.get("sample_limit")
            or config.get("sample_n")
            or DEFAULT_SAMPLE_LIMIT
        )
        sample_limit = max(sample_limit, 1)

        run_dir = settings.data_dir / "semantik" / "projections" / run.collection_id / run.uuid
        run_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(UTC)

        async with _operation_updates(operation_uuid) as updater:
            try:
                await projection_repo.update_status(
                    run.uuid,
                    status=ProjectionRunStatus.RUNNING,
                    started_at=now,
                )
                if operation_uuid:
                    await operation_repo.update_status(
                        operation_uuid,
                        OperationStatus.PROCESSING,
                        started_at=now,
                        error_message=None,
                    )
                await session.commit()

                if updater:
                    await updater.send_update(
                        "projection_started",
                        {
                            "projection_id": run.uuid,
                            "collection_id": run.collection_id,
                            "sample_limit": sample_limit,
                        },
                    )

                vectors: List[np.ndarray] = []
                original_ids: List[str] = []
                categories: List[int] = []
                doc_category_map: Dict[str, int] = {}
                overflow_logged = False

                manager = resolve_qdrant_manager()
                qdrant_client = getattr(manager, "client", None)
                if qdrant_client is None and hasattr(manager, "get_client"):
                    qdrant_client = manager.get_client()
                if qdrant_client is None:
                    raise RuntimeError("Unable to acquire Qdrant client from manager")

                offset: Any = None
                while len(vectors) < sample_limit:
                    remaining = sample_limit - len(vectors)
                    batch_limit = max(1, min(QDRANT_SCROLL_BATCH, remaining))
                    records, offset = qdrant_client.scroll(
                        collection_name=vector_collection_name,
                        offset=offset,
                        limit=batch_limit,
                        with_payload=True,
                        with_vectors=True,
                    )

                    if not records:
                        break

                    for record in records:
                        vector_values = getattr(record, "vector", None)
                        if isinstance(vector_values, dict):
                            # Use the first available vector when multiple are present.
                            vector_values = next((val for val in vector_values.values() if val is not None), None)
                        if vector_values is None:
                            continue

                        vector_array = np.asarray(vector_values, dtype=np.float32)
                        if vector_array.ndim != 1:
                            continue

                        vectors.append(vector_array)
                        original_ids.append(str(record.id))

                        payload = getattr(record, "payload", {}) or {}
                        category_idx = 0
                        if isinstance(payload, dict):
                            doc_identifier = (
                                payload.get("doc_id")
                                or payload.get("document_id")
                                or payload.get("chunk_id")
                                or payload.get("source_id")
                            )
                            if doc_identifier is not None:
                                doc_key = str(doc_identifier)
                                if doc_key not in doc_category_map:
                                    if len(doc_category_map) < 255:
                                        doc_category_map[doc_key] = len(doc_category_map)
                                    else:
                                        if not overflow_logged:
                                            logger.warning(
                                                "Projection run %s has more than 255 categories; "
                                                "additional categories will be grouped together",
                                                projection_id,
                                            )
                                            overflow_logged = True
                                        doc_category_map[doc_key] = 255
                                category_idx = doc_category_map[doc_key]

                        categories.append(int(category_idx))

                    if updater:
                        await updater.send_update(
                            "projection_fetch_progress",
                            {
                                "projection_id": run.uuid,
                                "fetched": len(vectors),
                                "sample_limit": sample_limit,
                            },
                        )

                    if offset is None:
                        break

                point_count = len(vectors)
                if point_count < 2:
                    raise ValueError("Not enough vectors available to compute projection (need at least 2)")

                vectors_array = np.stack(vectors, axis=0)
                pca_result = _compute_pca_projection(vectors_array)

                projection_array = pca_result["projection"]
                x_path = run_dir / "x.f32.bin"
                y_path = run_dir / "y.f32.bin"
                ids_path = run_dir / "ids.i32.bin"
                cat_path = run_dir / "cat.u8.bin"
                meta_path = run_dir / "meta.json"

                x_values = projection_array[:, 0]
                y_values = projection_array[:, 1]
                ids_array = []
                warned_fallback = False
                int32_info = np.iinfo(np.int32)
                for idx, point_id in enumerate(original_ids):
                    use_sequential = False
                    try:
                        numeric = int(point_id)
                    except (TypeError, ValueError):
                        use_sequential = True
                    else:
                        if numeric < int32_info.min or numeric > int32_info.max:
                            use_sequential = True

                    if use_sequential:
                        if not warned_fallback:
                            logger.warning(
                                "Projection %s has point IDs that are non-integer or outside int32 bounds; "
                                "using sequential fallback",
                                projection_id,
                            )
                            warned_fallback = True
                        numeric = idx

                    ids_array.append(numeric)

                ids_array = np.asarray(ids_array, dtype=np.int32)
                categories_array = np.array(categories, dtype=np.uint8)

                _write_binary(x_path, x_values, np.float32)
                _write_binary(y_path, y_values, np.float32)
                _write_binary(ids_path, ids_array, np.int32)
                _write_binary(cat_path, categories_array, np.uint8)

                meta_payload: Dict[str, Any] = {
                    "projection_id": run.uuid,
                    "collection_id": run.collection_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "point_count": point_count,
                    "reducer": "pca",
                    "dimensionality": 2,
                    "source_vector_collection": vector_collection_name,
                    "sample_limit": sample_limit,
                    "files": {
                        "x": x_path.name,
                        "y": y_path.name,
                        "ids": ids_path.name,
                        "categories": cat_path.name,
                    },
                    "explained_variance_ratio": pca_result["explained_variance_ratio"].tolist(),
                    "singular_values": pca_result["singular_values"].tolist(),
                    "original_ids": original_ids,
                    "category_map": {key: int(val) for key, val in doc_category_map.items()},
                }

                _write_meta(meta_path, meta_payload)

                await projection_repo.update_metadata(
                    run.uuid,
                    storage_path=str(run_dir),
                    point_count=point_count,
                    meta={"projection_artifacts": meta_payload},
                )
                await projection_repo.update_status(
                    run.uuid,
                    status=ProjectionRunStatus.COMPLETED,
                    completed_at=datetime.now(UTC),
                )

                if operation_uuid:
                    await operation_repo.update_status(
                        operation_uuid,
                        OperationStatus.COMPLETED,
                        completed_at=datetime.now(UTC),
                        error_message=None,
                    )

                await session.commit()

                if updater:
                    await updater.send_update(
                        "projection_completed",
                        {
                            "projection_id": run.uuid,
                            "point_count": point_count,
                            "storage_path": str(run_dir),
                        },
                    )

                return {
                    "projection_id": projection_id,
                    "status": "completed",
                    "message": None,
                    "point_count": point_count,
                    "storage_path": str(run_dir),
                }

            except Exception as exc:
                sanitized_error = _sanitize_error_message(str(exc))
                logger.exception("Projection computation failed for %s", projection_id)

                await session.rollback()

                try:
                    await projection_repo.update_status(
                        run.uuid,
                        status=ProjectionRunStatus.FAILED,
                        error_message=sanitized_error,
                        completed_at=datetime.now(UTC),
                    )
                    if operation_uuid:
                        await operation_repo.update_status(
                            operation_uuid,
                            OperationStatus.FAILED,
                            error_message=sanitized_error,
                            completed_at=datetime.now(UTC),
                        )
                    await session.commit()
                except Exception as status_exc:  # pragma: no cover - defensive logging
                    logger.error("Failed to update projection status after error: %s", status_exc)
                    await session.rollback()

                if updater:
                    await updater.send_update(
                        "projection_failed",
                        {
                            "projection_id": run.uuid,
                            "error": sanitized_error,
                        },
                    )

                return {
                    "projection_id": projection_id,
                    "status": "failed",
                    "message": sanitized_error,
                }


async def _process_projection_operation(
    operation: dict[str, Any],
    collection: dict[str, Any],
    projection_repo: Any,
    updater: CeleryTaskWithOperationUpdates,
) -> dict[str, Any]:
    """Async handler invoked from the ingestion dispatcher (placeholder)."""

    logger.info(
        "Processing projection operation (stub) operation_id=%s collection_id=%s",
        operation.get("uuid"),
        collection.get("id"),
    )

    await updater.send_update(
        "projection_stub",
        {
            "status": "pending",
            "message": "Projection processing not yet implemented",
        },
    )

    # TODO: create projection run records, enqueue compute_projection task, update progress
    return {"success": True, "message": "Projection processing not yet implemented"}
