"""Projection Celery tasks for computing and tracking embedding projections.

Artifact layout
---------------

For each successfully completed ``ProjectionRun`` this module writes a
canonical set of artifacts under a per-run directory::

    <settings.data_dir>/semantik/projections/<collection_id>/<projection_uuid>/

The directory contains the following files:

* ``x.f32.bin``  – float32 array of length ``point_count`` with X coordinates
* ``y.f32.bin``  – float32 array of length ``point_count`` with Y coordinates
* ``ids.i32.bin`` – int32 array of length ``point_count`` with stable point IDs
* ``cat.u8.bin`` – uint8 array of length ``point_count`` with category indices
* ``meta.json``  – JSON payload describing the projection run and artifacts

All four binary arrays MUST share the same ``point_count``; this invariant is
validated before artifacts are written.

``meta.json`` schema
--------------------

The ``meta.json`` payload (and the value stored in
``ProjectionRun.meta['projection_artifacts']``) has the following core shape:

* ``projection_id`` (str) – projection UUID
* ``collection_id`` (str) – parent collection UUID
* ``created_at`` (ISO 8601 str) – metadata creation time
* ``point_count`` (int) – number of projected points
* ``total_count`` (int) – total vectors available in the collection
* ``shown_count`` (int) – number of points shown in this run
* ``sampled`` (bool) – whether the run used sampling
* ``reducer_requested`` / ``reducer_used`` (str) – reducer names
* ``reducer_params`` (dict) – effective reducer configuration
* ``dimensionality`` (int) – currently always ``2``
* ``source_vector_collection`` (str) – backing Qdrant collection name
* ``sample_limit`` (int) – sampling cap used when scrolling vectors
* ``files`` (dict) – filenames for ``x``, ``y``, ``ids`` and ``categories``
* ``color_by`` (str) – active color-by mode
* ``legend`` (list[dict]) – entries with ``index``, ``label`` and ``count``
* ``original_ids`` (list[str]) – original Qdrant point identifiers
* ``category_counts`` (dict[str, int]) – counts per category index

Optional, reducer-specific fields (e.g. ``explained_variance_ratio``,
``singular_values``, ``kl_divergence``, ``fallback_reason``) may be included
for diagnostics but are always backwards compatible.
"""

from __future__ import annotations

import inspect
import json
import sys
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:  # Optional UMAP dependency
    import umap  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    umap = None  # type: ignore[assignment]

try:  # Optional scikit-learn dependency for t-SNE
    from sklearn.manifold import TSNE  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    TSNE = None  # type: ignore[assignment]
from shared.database.models import OperationStatus, ProjectionRunStatus
from shared.database.postgres_database import PostgresConnectionManager
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.operation_repository import OperationRepository
from shared.database.repositories.projection_run_repository import ProjectionRunRepository

from webui.tasks.utils import (
    CeleryTaskWithOperationUpdates,
    _sanitize_error_message,
    celery_app,
    logger,
    resolve_awaitable_sync,
    resolve_qdrant_manager,
    settings,
)

DEFAULT_SAMPLE_LIMIT = 200_000
PCA_SVD_SAMPLE_LIMIT = 50_000
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
    if isinstance(value, int | float):
        try:
            timestamp = float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

        abs_ts = abs(timestamp)
        # Heuristic conversions: treat large magnitudes as milliseconds/microseconds.
        if abs_ts >= 1e14:  # microseconds since epoch
            timestamp /= 1_000_000
        elif abs_ts >= 1e12:  # milliseconds since epoch
            timestamp /= 1_000

        try:
            return datetime.fromtimestamp(timestamp, tz=UTC)
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
) -> tuple[str, str | None]:
    """Return the category label and optional document identifier."""

    if not isinstance(payload, Mapping):
        return UNKNOWN_CATEGORY_LABEL, None

    if color_by == "document_id":
        doc_identifier = (
            payload.get("doc_id") or payload.get("document_id") or payload.get("chunk_id") or payload.get("source_id")
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
        payload.get("doc_id") or payload.get("document_id") or payload.get("chunk_id") or payload.get("source_id")
    )
    if doc_identifier is None:
        return UNKNOWN_CATEGORY_LABEL, None
    return str(doc_identifier), str(doc_identifier)


def _ensure_float32(array: np.ndarray) -> np.ndarray:
    """Return ``array`` cast to ``np.float32`` without unnecessary copies."""

    return array.astype(np.float32, copy=False)


def _compute_pca_projection(vectors: np.ndarray) -> dict[str, np.ndarray]:
    """Compute a 2D PCA projection while guarding against oversized SVD workloads."""

    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D array for PCA, got {vectors.shape!r}")
    if vectors.shape[0] < 2:
        raise ValueError("At least two samples are required for PCA projection")
    if vectors.shape[1] < 2:
        raise ValueError("Vectors must have at least two dimensions for PCA projection")

    sample_limit = PCA_SVD_SAMPLE_LIMIT
    num_rows = vectors.shape[0]

    mean = vectors.mean(axis=0, keepdims=True)

    if num_rows > sample_limit:
        # Down-sample deterministically to keep SVD cost manageable while remaining reproducible.
        sample_indices = np.linspace(0, num_rows - 1, sample_limit, dtype=np.int64)
        centered_sample = vectors[sample_indices] - mean
        _, singular_values, vt = np.linalg.svd(centered_sample, full_matrices=False)
        centered_full = vectors - mean
    else:
        centered_full = vectors - mean
        _, singular_values, vt = np.linalg.svd(centered_full, full_matrices=False)

    top_components = vt[:2]

    projection = centered_full @ top_components.T

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


def _compute_tsne_projection(
    vectors: np.ndarray,
    *,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    metric: str,
    init: str = "pca",
) -> dict[str, np.ndarray]:
    """Compute a 2D t-SNE projection using scikit-learn."""

    if TSNE is None:
        raise RuntimeError("scikit-learn is not installed; t-SNE reducer unavailable")

    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D array for t-SNE, got {vectors.shape!r}")
    n_samples = vectors.shape[0]
    if n_samples < 2:
        raise ValueError("At least two samples are required for t-SNE projection")

    # Adjust perplexity to remain within valid bounds for the dataset size.
    max_perplexity = max(1.0, min(perplexity, n_samples - 1))
    effective_perplexity = max(1.0, min(perplexity, max_perplexity))

    effective_learning_rate = max(10.0, float(learning_rate))
    effective_n_iter = max(250, int(n_iter))
    init_mode = init if init in {"pca", "random"} else "pca"

    tsne_signature = inspect.signature(TSNE.__init__)
    tsne_params = tsne_signature.parameters

    tsne_kwargs: dict[str, Any] = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "metric": metric,
        "init": init_mode,
        "random_state": 42,
    }

    if "learning_rate" in tsne_params:
        tsne_kwargs["learning_rate"] = effective_learning_rate
    elif "learning_rate_init" in tsne_params:
        tsne_kwargs["learning_rate_init"] = effective_learning_rate

    iteration_param: str | None = None
    if "n_iter" in tsne_params:
        tsne_kwargs["n_iter"] = effective_n_iter
        iteration_param = "n_iter"
    elif "max_iter" in tsne_params:
        tsne_kwargs["max_iter"] = effective_n_iter
        iteration_param = "max_iter"

    if "square_distances" in tsne_params:
        tsne_kwargs["square_distances"] = True

    tsne = TSNE(**tsne_kwargs)

    if iteration_param is None:
        # As a fallback for very old sklearn versions, try to set the attribute directly.
        with suppress(Exception):
            tsne.n_iter = effective_n_iter

    embedding = tsne.fit_transform(vectors)

    kl_divergence = float(getattr(tsne, "kl_divergence_", float("nan")))
    iterations_run = getattr(tsne, "n_iter_", getattr(tsne, "n_iter", effective_n_iter))

    return {
        "projection": _ensure_float32(embedding),
        "perplexity": float(effective_perplexity),
        "learning_rate": float(effective_learning_rate),
        "n_iter": int(iterations_run),
        "kl_divergence": kl_divergence,
    }


def _write_binary(path: Path, array: np.ndarray, dtype: np.dtype) -> None:
    """Write ``array`` to ``path`` as raw binary in the specified dtype."""

    array.astype(dtype, copy=False).tofile(path)


def _write_meta(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@asynccontextmanager
async def _operation_updates(
    operation_id: str | None, user_id: int | None = None
) -> AsyncIterator[CeleryTaskWithOperationUpdates | None]:
    """Yield an update publisher when an operation ID is available."""

    if not operation_id:
        yield None
        return

    async with CeleryTaskWithOperationUpdates(operation_id) as updater:
        if user_id is not None:
            updater.set_user_id(user_id)
        yield updater


@celery_app.task(bind=True, name="webui.tasks.compute_projection")
def compute_projection(self: Any, projection_id: str) -> dict[str, Any]:  # noqa: ANN401, ARG001
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

    pg_manager = PostgresConnectionManager()
    await pg_manager.initialize()

    session_factory = getattr(pg_manager, "sessionmaker", None) or getattr(pg_manager, "_sessionmaker", None)
    if session_factory is None:
        await pg_manager.close()
        raise RuntimeError("Failed to initialize projection session maker")
    session: Any | None = None
    session_guard: Callable[[], Any] | None = None

    try:
        async with session_factory() as session:
            ensure_open_guard = getattr(session, "ensure_open", None)
            if callable(ensure_open_guard):
                session_guard = ensure_open_guard

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

            configured_sample = config.get("sample_size")
            if configured_sample is None:
                configured_sample = config.get("sample_limit")
            if configured_sample is None:
                configured_sample = config.get("sample_n")
            try:
                sample_limit = int(configured_sample) if configured_sample is not None else DEFAULT_SAMPLE_LIMIT
            except (TypeError, ValueError):
                sample_limit = DEFAULT_SAMPLE_LIMIT
            sample_limit = max(sample_limit, 1)

            run_dir = settings.data_dir / "semantik" / "projections" / run.collection_id / run.uuid
            run_dir.mkdir(parents=True, exist_ok=True)

            now = datetime.now(UTC)

            user_id = None
            if operation_uuid and hasattr(operation_repo, "get_by_uuid"):
                try:
                    op_obj = operation_repo.get_by_uuid(operation_uuid)
                    operation = await op_obj if inspect.isawaitable(op_obj) else op_obj
                except Exception as exc:  # pragma: no cover - fail open for optional enrichment
                    logger.warning(
                        "Failed to fetch operation %s for projection %s: %s",
                        operation_uuid,
                        projection_id,
                        exc,
                    )
                    operation = None

                if operation:
                    user_id = getattr(operation, "user_id", None)

            operation_updates_fn = _operation_updates
            supports_user_id = "user_id" in inspect.signature(operation_updates_fn).parameters
            updates_ctx = (
                operation_updates_fn(operation_uuid, user_id=user_id)
                if supports_user_id
                else operation_updates_fn(operation_uuid)
            )

            async with updates_ctx as updater:
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

                    vectors: list[np.ndarray] = []
                    original_ids: list[str] = []
                    categories: list[int] = []

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

                    # Category indices for ``cat.u8.bin`` are assigned
                    # sequentially starting at 0 for each distinct category
                    # label observed while scrolling Qdrant. Once indices
                    # 0–254 are exhausted, any additional categories are
                    # mapped into the overflow bucket at
                    # ``OVERFLOW_CATEGORY_INDEX`` (255), which is exposed as a
                    # single "Other" entry in the legend. This ensures every
                    # value stored in the categories array either has a
                    # corresponding legend label or is explicitly grouped into
                    # the overflow bucket.
                    category_index_map: dict[str, int] = {}
                    label_for_index: dict[int, str] = {}
                    category_counts: Counter[int] = Counter()
                    doc_category_map: dict[str, int] = {} if color_by == "document_id" else {}
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
                    total_vectors = getattr(collection, "vector_count", None)
                    if isinstance(total_vectors, int) and total_vectors > 0:
                        total_vectors = max(total_vectors, point_count)
                    else:
                        total_vectors = point_count
                    sampled_flag = point_count < total_vectors

                    if point_count < 2:
                        raise ValueError("Not enough vectors available to compute projection (need at least 2)")

                    vectors_array = np.stack(vectors, axis=0)

                    requested_reducer = (run.reducer or "pca").lower()
                    reducer_used = requested_reducer
                    reducer_params: dict[str, Any] = {}
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
                        elif requested_reducer == "tsne":
                            params = config if isinstance(config, dict) else {}
                            perplexity = float(params.get("perplexity", 30.0))
                            learning_rate = float(params.get("learning_rate", 200.0))
                            n_iter = int(params.get("n_iter", 1_000))
                            metric = str(params.get("metric", "euclidean"))
                            init = str(params.get("init", "pca"))
                            projection_result = _compute_tsne_projection(
                                vectors_array,
                                perplexity=perplexity,
                                learning_rate=learning_rate,
                                n_iter=n_iter,
                                metric=metric,
                                init=init,
                            )
                            reducer_used = "tsne"
                            reducer_params = {
                                "perplexity": float(projection_result.get("perplexity", perplexity)),
                                "learning_rate": float(projection_result.get("learning_rate", learning_rate)),
                                "n_iter": int(projection_result.get("n_iter", n_iter)),
                                "metric": metric,
                                "init": init,
                            }
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
                        elif requested_reducer == "tsne":
                            params = config if isinstance(config, dict) else {}
                            reducer_params = {
                                "perplexity": float(params.get("perplexity", 30.0)),
                                "learning_rate": float(params.get("learning_rate", 200.0)),
                                "n_iter": int(params.get("n_iter", 1_000)),
                                "metric": str(params.get("metric", "euclidean")),
                                "init": str(params.get("init", "pca")),
                            }
                        else:
                            reducer_params = config if isinstance(config, dict) else {}

                    projection_array = projection_result["projection"]
                    if projection_array.ndim != 2 or projection_array.shape[1] < 2:
                        raise ValueError(
                            f"Projection reducer returned invalid shape {projection_array.shape!r}; "
                            "expected (point_count, 2)"
                        )

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

                    if not (
                        len(x_values) == len(y_values) == ids_array.shape[0] == categories_array.shape[0] == point_count
                    ):
                        raise ValueError(
                            "Projection artifact length mismatch: "
                            f"x={len(x_values)}, y={len(y_values)}, ids={ids_array.shape[0]}, "
                            f"cat={categories_array.shape[0]}, point_count={point_count}"
                        )

                    # Legend entries provide a stable mapping from category
                    # index → label → count that mirrors ``cat.u8.bin``:
                    # - For indices < OVERFLOW_CATEGORY_INDEX, there is a
                    #   one-to-one mapping between each distinct category
                    #   label and its index.
                    # - When the overflow bucket is used, index 255 is
                    #   represented once with label ``OVERFLOW_LEGEND_LABEL``
                    #   and the aggregated count of all overflowed points.
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

                    meta_payload: dict[str, Any] = {
                        "projection_id": run.uuid,
                        "collection_id": run.collection_id,
                        "created_at": datetime.now(UTC).isoformat(),
                        "point_count": point_count,
                        "total_count": total_vectors,
                        "shown_count": point_count,
                        "sampled": sampled_flag,
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
                    if "kl_divergence" in projection_result and projection_result["kl_divergence"] is not None:
                        try:
                            meta_payload["kl_divergence"] = float(projection_result["kl_divergence"])
                        except (TypeError, ValueError):
                            meta_payload["kl_divergence"] = projection_result["kl_divergence"]
                    if fallback_reason:
                        meta_payload["fallback_reason"] = fallback_reason
                    meta_payload["original_ids"] = original_ids
                    if color_by == "document_id":
                        meta_payload["category_map"] = {key: int(val) for key, val in doc_category_map.items()}
                    meta_payload["category_counts"] = {
                        str(int(idx)): int(count) for idx, count in category_counts.items()
                    }

                    # Mark runs as degraded when the reducer had to fall back
                    # to PCA or other non‑requested behaviour. This flag is
                    # persisted both in the projection_artifacts payload and at
                    # the top level run.meta so that API consumers can rely on
                    # a single degraded boolean when deciding whether to offer
                    # a recompute action.
                    degraded_flag = bool(fallback_reason)
                    if degraded_flag:
                        meta_payload["degraded"] = True

                    _write_meta(meta_path, meta_payload)

                    try:
                        storage_path_value = str(run_dir.relative_to(settings.data_dir))
                    except ValueError:
                        storage_path_value = str(run_dir)

                    await projection_repo.update_metadata(
                        run.uuid,
                        storage_path=storage_path_value,
                        point_count=point_count,
                        meta={
                            "projection_artifacts": meta_payload,
                            "color_by": color_by,
                            "legend": legend_entries,
                            "sampled": sampled_flag,
                            "shown_count": point_count,
                            "total_count": total_vectors,
                            "degraded": degraded_flag,
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
                                "sampled": sampled_flag,
                                "shown_count": point_count,
                                "total_count": total_vectors,
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

                    if operation_uuid:
                        await operation_repo.update_status(
                            operation_uuid,
                            OperationStatus.FAILED,
                            error_message=sanitized_error,
                        )
                    await projection_repo.update_status(
                        run.uuid,
                        status=ProjectionRunStatus.FAILED,
                        error_message=sanitized_error,
                    )
                    await session.commit()

                    if updater:
                        await updater.send_update(
                            "projection_failed",
                            {
                                "projection_id": run.uuid,
                                "error": sanitized_error,
                            },
                        )

                    raise

                finally:
                    if updater:
                        await updater.close()
    finally:
        guard_exception: Exception | None = None
        active_exc_type = sys.exc_info()[0]
        if session_guard is not None:
            try:
                session_guard()
            except Exception as exc:  # pragma: no cover - surface cleanup errors
                guard_exception = exc

        await pg_manager.close()

        if guard_exception is not None and active_exc_type is None:
            raise guard_exception


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

        # Treat the operation as actively processing as soon as the projection
        # run has been prepared and the compute task is about to be enqueued so
        # that API consumers can distinguish between queued and idle runs.
        await operation_repo.update_status(
            operation.get("uuid"),
            OperationStatus.PROCESSING,
            started_at=datetime.now(UTC),
            error_message=None,
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
        logger.info(
            "Enqueuing compute_projection task for projection_id=%s operation_id=%s",
            projection_id,
            operation.get("uuid"),
        )
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
