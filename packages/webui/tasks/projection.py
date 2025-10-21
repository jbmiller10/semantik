"""Projection Celery tasks for computing and tracking embedding projections."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Mapping, Tuple

import numpy as np

try:  # Optional UMAP dependency
    import umap  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    umap = None  # type: ignore[assignment]
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
OVERFLOW_CATEGORY_INDEX = 255
OVERFLOW_LEGEND_LABEL = "Other"
UNKNOWN_CATEGORY_LABEL = "unknown"
ALLOWED_COLOR_BY = {"document_id", "source_dir", "filetype", "age_bucket"}


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse assorted timestamp representations into timezone-aware datetimes."""

    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (ValueError, OSError):  # pragma: no cover - defensive
            return None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _bucket_age(timestamp: datetime, now: datetime) -> str:
    """Bucket timestamp deltas into coarse age groups."""

    delta = now - timestamp
    if delta.total_seconds() < 0:
        return "future"
    days = delta.total_seconds() / 86_400
    if days <= 1:
        return "≤1d"
    if days <= 7:
        return "≤7d"
    if days <= 30:
        return "≤30d"
    if days <= 90:
        return "≤90d"
    if days <= 180:
        return "≤180d"
    if days <= 365:
        return "≤1y"
    return ">1y"


def _extract_source_dir(payload: Mapping[str, Any]) -> str:
    source_path = payload.get("source_path")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    if not source_path:
        source_path = payload.get("path") or metadata.get("source_path")
    if not source_path or not isinstance(source_path, str):
        return UNKNOWN_CATEGORY_LABEL
    try:
        path_obj = Path(source_path)
    except Exception:  # pragma: no cover - defensive
        return str(source_path) or UNKNOWN_CATEGORY_LABEL
    if path_obj.suffix:
        parent = path_obj.parent
        if parent and parent.name:
            return parent.name
    if path_obj.name:
        return path_obj.name
    if path_obj.parent and path_obj.parent.name:
        return path_obj.parent.name
    return str(path_obj) or UNKNOWN_CATEGORY_LABEL


def _extract_filetype(payload: Mapping[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    mime = payload.get("mime_type") or metadata.get("mime_type")
    if isinstance(mime, str) and mime:
        return mime.lower()
    path_value = payload.get("path") or payload.get("source_path") or metadata.get("source_path")
    if isinstance(path_value, str) and path_value:
        try:
            suffix = Path(path_value).suffix.lower()
        except Exception:  # pragma: no cover - defensive
            suffix = ""
        if suffix:
            return suffix.lstrip(".") or UNKNOWN_CATEGORY_LABEL
    return UNKNOWN_CATEGORY_LABEL


def _extract_age_bucket(payload: Mapping[str, Any], now: datetime) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    timestamp_value = (
        payload.get("ingested_at")
        or payload.get("created_at")
        or payload.get("updated_at")
        or payload.get("timestamp")
        or metadata.get("ingested_at")
        or metadata.get("created_at")
        or metadata.get("updated_at")
        or metadata.get("timestamp")
    )
    parsed = _parse_timestamp(timestamp_value)
    if parsed is None:
        return UNKNOWN_CATEGORY_LABEL
    return _bucket_age(parsed, now)


def _derive_category_label(
    payload: Mapping[str, Any] | None,
    color_by: str,
    now: datetime,
) -> Tuple[str, str | None]:
    """Return the category label and optional document identifier."""

    if not isinstance(payload, Mapping):
        return UNKNOWN_CATEGORY_LABEL, None

    if color_by == "document_id":
        doc_identifier = (
            payload.get("doc_id")
            or payload.get("document_id")
            or payload.get("chunk_id")
            or payload.get("source_id")
        )
        if doc_identifier is None:
            return UNKNOWN_CATEGORY_LABEL, None
        return str(doc_identifier), str(doc_identifier)

    if color_by == "source_dir":
        return _extract_source_dir(payload), None

    if color_by == "filetype":
        return _extract_filetype(payload), None

    if color_by == "age_bucket":
        return _extract_age_bucket(payload, now), None

    # Fallback to document_id semantics for unexpected values
    doc_identifier = (
        payload.get("doc_id")
        or payload.get("document_id")
        or payload.get("chunk_id")
        or payload.get("source_id")
    )
    if doc_identifier is None:
        return UNKNOWN_CATEGORY_LABEL, None
    return str(doc_identifier), str(doc_identifier)


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


def _compute_umap_projection(
    vectors: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
) -> dict[str, np.ndarray]:
    """Compute a 2D UMAP projection using umap-learn."""

    if umap is None:
        raise RuntimeError("umap-learn is not installed")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=42,
    )
    embedding = reducer.fit_transform(vectors)
    return {
        "projection": _ensure_float32(embedding),
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
        color_by = str(config.get("color_by") or "document_id").lower()
        if color_by not in ALLOWED_COLOR_BY:
            color_by = "document_id"

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

                vectors: List[np.ndarray] = []
                original_ids: List[str] = []
                categories: List[int] = []

                if updater:
                    await updater.send_update(
                        "projection_started",
                        {
                            "projection_id": run.uuid,
                            "collection_id": run.collection_id,
                            "sample_limit": sample_limit,
                            "color_by": color_by,
                        },
                    )

                category_index_map: Dict[str, int] = {}
                label_for_index: Dict[int, str] = {}
                category_counts: Counter[int] = Counter()
                doc_category_map: Dict[str, int] = {} if color_by == "document_id" else {}
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
                        category_label, doc_identifier = _derive_category_label(payload, color_by, now)

                        if not category_label:
                            category_label = UNKNOWN_CATEGORY_LABEL

                        category_idx = category_index_map.get(category_label)
                        if category_idx is None:
                            if len(category_index_map) < OVERFLOW_CATEGORY_INDEX:
                                category_idx = len(category_index_map)
                                category_index_map[category_label] = category_idx
                                label_for_index[category_idx] = category_label
                            else:
                                category_idx = OVERFLOW_CATEGORY_INDEX
                        else:
                            if category_idx != OVERFLOW_CATEGORY_INDEX:
                                label_for_index.setdefault(category_idx, category_label)

                        if category_idx == OVERFLOW_CATEGORY_INDEX:
                            label_for_index.setdefault(OVERFLOW_CATEGORY_INDEX, OVERFLOW_LEGEND_LABEL)
                            if not overflow_logged:
                                logger.warning(
                                    "Projection %s exceeded 255 categories; using overflow bucket",
                                    projection_id,
                                )
                                overflow_logged = True

                        categories.append(int(category_idx))
                        category_counts[category_idx] += 1

                        if color_by == "document_id" and doc_identifier is not None:
                            doc_key = str(doc_identifier)
                            if doc_key not in doc_category_map:
                                doc_category_map[doc_key] = int(category_idx)

                    if updater:
                        await updater.send_update(
                            "projection_fetch_progress",
                            {
                                "projection_id": run.uuid,
                                "fetched": len(vectors),
                                "sample_limit": sample_limit,
                                "color_by": color_by,
                            },
                        )

                    if offset is None:
                        break

                point_count = len(vectors)
                if point_count < 2:
                    raise ValueError("Not enough vectors available to compute projection (need at least 2)")

                vectors_array = np.stack(vectors, axis=0)

                requested_reducer = (run.reducer or "pca").lower()
                reducer_used = requested_reducer
                reducer_params: Dict[str, Any] = {}
                fallback_reason: str | None = None

                try:
                    if requested_reducer == "umap":
                        params = config if isinstance(config, dict) else {}
                        n_neighbors = int(params.get("n_neighbors", 15))
                        min_dist = float(params.get("min_dist", 0.1))
                        metric = str(params.get("metric", "cosine"))
                        reducer_params = {
                            "n_neighbors": n_neighbors,
                            "min_dist": min_dist,
                            "metric": metric,
                        }
                        projection_result = _compute_umap_projection(
                            vectors_array,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            metric=metric,
                        )
                        reducer_used = "umap"
                    elif requested_reducer == "pca":
                        projection_result = _compute_pca_projection(vectors_array)
                        reducer_used = "pca"
                    else:
                        fallback_reason = f"Unsupported reducer '{requested_reducer}'"
                        logger.warning(
                            "Reducer %s not supported for projection %s; falling back to PCA",
                            requested_reducer,
                            projection_id,
                        )
                        projection_result = _compute_pca_projection(vectors_array)
                        reducer_used = "pca"
                        reducer_params = config if isinstance(config, dict) else {}
                except Exception as exc:
                    fallback_reason = str(exc)
                    logger.warning(
                        "Reducer %s failed for projection %s; falling back to PCA: %s",
                        requested_reducer,
                        projection_id,
                        exc,
                    )
                    projection_result = _compute_pca_projection(vectors_array)
                    reducer_used = "pca"
                    if requested_reducer == "umap":
                        params = config if isinstance(config, dict) else {}
                        reducer_params = {
                            "n_neighbors": int(params.get("n_neighbors", 15)),
                            "min_dist": float(params.get("min_dist", 0.1)),
                            "metric": str(params.get("metric", "cosine")),
                        }
                    else:
                        reducer_params = config if isinstance(config, dict) else {}

                projection_array = projection_result["projection"]
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

                legend_entries = [
                    {
                        "index": int(idx),
                        "label": label,
                        "count": int(category_counts.get(idx, 0)),
                    }
                    for idx, label in sorted(label_for_index.items())
                ]
                if category_counts.get(OVERFLOW_CATEGORY_INDEX) and not any(
                    entry["index"] == OVERFLOW_CATEGORY_INDEX for entry in legend_entries
                ):
                    legend_entries.append(
                        {
                            "index": OVERFLOW_CATEGORY_INDEX,
                            "label": OVERFLOW_LEGEND_LABEL,
                            "count": int(category_counts[OVERFLOW_CATEGORY_INDEX]),
                        }
                    )

                _write_binary(x_path, x_values, np.float32)
                _write_binary(y_path, y_values, np.float32)
                _write_binary(ids_path, ids_array, np.int32)
                _write_binary(cat_path, categories_array, np.uint8)

                meta_payload: Dict[str, Any] = {
                    "projection_id": run.uuid,
                    "collection_id": run.collection_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "point_count": point_count,
                    "reducer_requested": requested_reducer,
                    "reducer_used": reducer_used,
                    "reducer_params": reducer_params,
                    "dimensionality": 2,
                    "source_vector_collection": vector_collection_name,
                    "sample_limit": sample_limit,
                    "files": {
                        "x": x_path.name,
                        "y": y_path.name,
                        "ids": ids_path.name,
                        "categories": cat_path.name,
                    },
                    "color_by": color_by,
                    "legend": legend_entries,
                }
                if "explained_variance_ratio" in projection_result:
                    meta_payload["explained_variance_ratio"] = (
                        projection_result["explained_variance_ratio"].tolist()
                        if isinstance(projection_result["explained_variance_ratio"], np.ndarray)
                        else projection_result["explained_variance_ratio"]
                    )
                if "singular_values" in projection_result:
                    meta_payload["singular_values"] = (
                        projection_result["singular_values"].tolist()
                        if isinstance(projection_result["singular_values"], np.ndarray)
                        else projection_result["singular_values"]
                    )
                if fallback_reason:
                    meta_payload["fallback_reason"] = fallback_reason
                meta_payload["original_ids"] = original_ids
                if color_by == "document_id":
                    meta_payload["category_map"] = {key: int(val) for key, val in doc_category_map.items()}
                meta_payload["category_counts"] = {
                    str(int(idx)): int(count) for idx, count in category_counts.items()
                }

                _write_meta(meta_path, meta_payload)

                await projection_repo.update_metadata(
                    run.uuid,
                    storage_path=str(run_dir),
                    point_count=point_count,
                    meta={
                        "projection_artifacts": meta_payload,
                        "color_by": color_by,
                        "legend": legend_entries,
                    },
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
                            "reducer": reducer_used,
                            "storage_path": str(run_dir),
                            "color_by": color_by,
                            "legend": legend_entries,
                        },
                    )

                return {
                    "projection_id": projection_id,
                    "status": "completed",
                    "reducer": reducer_used,
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
                            "color_by": color_by,
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
    """Async handler invoked from the ingestion dispatcher for projections."""
    logger.info(
        "Processing projection operation operation_id=%s collection_id=%s",
        operation.get("uuid"),
        collection.get("id"),
    )

    operation_config = operation.get("config") or {}
    projection_id = (
        operation_config.get("projection_run_id")
        or operation_config.get("projection_id")
        or operation_config.get("projection_uuid")
    )
    if not projection_id:
        raise ValueError("Projection operation missing projection run identifier")

    projection_id = str(projection_id)

    session = getattr(projection_repo, "session", None)
    if session is None:
        raise RuntimeError("Projection repository is missing bound session")

    operation_repo = OperationRepository(session)

    run = await projection_repo.get_by_uuid(projection_id)
    if not run:
        raise ValueError(f"Projection run {projection_id} not found")

    try:
        if getattr(run, "operation_uuid", None) != operation.get("uuid"):
            await projection_repo.set_operation_uuid(projection_id, operation.get("uuid"))

        await projection_repo.update_status(
            projection_id,
            status=ProjectionRunStatus.RUNNING,
            error_message=None,
            started_at=datetime.now(UTC),
        )

        await session.commit()
    except Exception as exc:
        await session.rollback()
        logger.exception("Failed preparing projection %s before enqueue: %s", projection_id, exc)
        raise

    if updater:
        await updater.send_update(
            "projection_enqueued",
            {
                "projection_id": projection_id,
                "operation_id": operation.get("uuid"),
                "status": ProjectionRunStatus.RUNNING.value,
            },
        )

    try:
        compute_projection.apply_async(args=(projection_id,), task_id=str(uuid.uuid4()))
    except Exception as exc:  # pragma: no cover - broker failure path
        sanitized_error = _sanitize_error_message(str(exc))
        logger.error(
            "Failed to enqueue compute_projection for %s: %s",
            projection_id,
            sanitized_error,
        )

        await projection_repo.update_status(
            projection_id,
            status=ProjectionRunStatus.FAILED,
            error_message=sanitized_error,
            completed_at=datetime.now(UTC),
        )
        await operation_repo.update_status(
            operation.get("uuid"),
            OperationStatus.FAILED,
            error_message=sanitized_error,
            completed_at=datetime.now(UTC),
        )
        await session.commit()

        if updater:
            await updater.send_update(
                "projection_failed",
                {
                    "projection_id": projection_id,
                    "error": sanitized_error,
                },
            )

        return {"success": False, "message": sanitized_error}

    return {
        "success": True,
        "defer_completion": True,
        "projection_id": projection_id,
        "message": "Projection compute enqueued",
    }
