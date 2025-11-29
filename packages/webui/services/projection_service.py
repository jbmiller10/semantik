"""Service layer for embedding projection operations and artifact access."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from array import array
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from shared.config import settings
from shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from shared.database.models import OperationStatus, OperationType, ProjectionRun, ProjectionRunStatus
from shared.database.repositories.chunk_repository import ChunkRepository
from shared.database.repositories.document_repository import DocumentRepository
from webui.celery_app import celery_app
from webui.qdrant import qdrant_manager

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.repositories.collection_repository import CollectionRepository
    from shared.database.repositories.operation_repository import OperationRepository
    from shared.database.repositories.projection_run_repository import ProjectionRunRepository

logger = logging.getLogger(__name__)


def compute_projection_metadata_hash(
    *,
    collection_id: str,
    embedding_model: str,
    collection_vector_count: int,
    collection_updated_at: datetime | None,
    reducer: str,
    dimensionality: int,
    color_by: str,
    config: dict[str, Any] | None,
    sample_limit: int | None,
) -> str:
    """Compute a deterministic metadata hash for projection idempotency.

    The hash is derived from immutable inputs to the projection:

    - collection identity and embedding model
    - collection vector statistics (count + last update timestamp)
    - reducer name and dimensionality
    - normalised reducer config (sorted JSON)
    - colour mode and sampling limit
    """

    import hashlib

    def _normalise_config(raw: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        return {key: raw[key] for key in sorted(raw.keys())}

    updated_at_str: str | None
    if isinstance(collection_updated_at, datetime):
        updated_at_str = collection_updated_at.astimezone(UTC).replace(microsecond=0).isoformat()
    else:
        updated_at_str = None

    payload: dict[str, Any] = {
        "collection_id": collection_id,
        "embedding_model": embedding_model,
        "collection_vector_count": int(collection_vector_count),
        "collection_updated_at": updated_at_str,
        "reducer": reducer.lower(),
        "dimensionality": int(dimensionality),
        "color_by": color_by.lower(),
        "sample_limit": int(sample_limit) if sample_limit is not None else None,
        "config": _normalise_config(config),
    }

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ProjectionService:
    """Facade for projection run orchestration.

    The current implementation only provides scaffolding so that API routes and
    workers can be wired without the full projection pipeline. Each method
    returns placeholder payloads and records TODO markers where the real logic
    will eventually live.
    """

    def __init__(
        self,
        *,
        db_session: AsyncSession,
        projection_repo: ProjectionRunRepository,
        operation_repo: OperationRepository,
        collection_repo: CollectionRepository,
    ) -> None:
        self.db_session = db_session
        self.projection_repo = projection_repo
        self.operation_repo = operation_repo
        self.collection_repo = collection_repo

    @staticmethod
    def _extract_sample_limit(config: dict[str, Any] | None) -> int | None:
        """Derive a canonical sample_limit from config aliases."""
        if not isinstance(config, dict):
            return None
        for key in ("sample_size", "sample_limit", "sample_n"):
            value = config.get(key)
            if value is None:
                continue
            try:
                as_int = int(value)
            except (TypeError, ValueError):
                continue
            if as_int <= 0:
                continue
            return as_int
        return None

    @staticmethod
    def _is_run_degraded(run: ProjectionRun) -> bool:
        """Return True when the run meta marks the projection as degraded."""
        meta_raw = run.meta if isinstance(run.meta, dict) else None
        if not meta_raw:
            return False

        if bool(meta_raw.get("degraded")):
            return True

        projection_meta = meta_raw.get("projection_artifacts")
        if isinstance(projection_meta, dict) and bool(projection_meta.get("degraded")):
            return True

        return False

    @staticmethod
    def _is_metadata_compatible(
        run: ProjectionRun,
        *,
        reducer: str,
        dimensionality: int,
        color_by: str,
        sample_limit: int | None,
    ) -> bool:
        """Validate that a candidate run matches the requested inputs for idempotent reuse."""

        if run.reducer.lower() != reducer.lower():
            return False
        if run.dimensionality != dimensionality:
            return False

        config = run.config if isinstance(run.config, dict) else {}
        run_color_raw = config.get("color_by")
        run_color = str(run_color_raw).lower() if run_color_raw is not None else None
        if run_color != color_by.lower():
            return False

        run_sample_limit = ProjectionService._extract_sample_limit(config)
        if (run_sample_limit or None) != (sample_limit or None):
            return False

        return True

    @staticmethod
    def _encode_projection(
        run: ProjectionRun, *, operation: Any | None = None, message: str | None = None
    ) -> dict[str, Any]:
        """Convert a ProjectionRun ORM instance into a serialisable payload.

        Args:
            run: The ProjectionRun database model to encode
            operation: Optional Operation model to include status from
            message: Optional status message to include

        Returns:
            Dictionary with projection metadata including operation_status if operation provided
        """

        created_at = run.created_at if isinstance(run.created_at, datetime) else None
        config = run.config if isinstance(run.config, dict) else None
        meta_raw = run.meta if isinstance(run.meta, dict) else None
        meta: dict[str, Any] = dict(meta_raw) if meta_raw is not None else {}

        projection_meta = meta.get("projection_artifacts")
        if isinstance(projection_meta, dict) and "color_by" in projection_meta and "color_by" not in meta:
            meta["color_by"] = projection_meta["color_by"]
        if config and "color_by" in config and "color_by" not in meta:
            meta["color_by"] = config["color_by"]

        # Surface degraded flag in the top-level meta payload when present either
        # on the run or within the projection_artifacts payload so API consumers
        # can rely on a single boolean field.
        if not meta.get("degraded"):
            projection_degraded = False
            if isinstance(projection_meta, dict):
                projection_degraded = bool(projection_meta.get("degraded"))
            if meta_raw and isinstance(meta_raw, dict):
                projection_degraded = bool(meta_raw.get("degraded")) or projection_degraded
            if projection_degraded:
                meta["degraded"] = True

        meta_for_response: dict[str, Any] | None = meta or None

        # Build base response
        response = {
            "collection_id": run.collection_id,
            "projection_id": run.uuid,
            "status": run.status.value,
            "reducer": run.reducer,
            "dimensionality": run.dimensionality,
            "created_at": created_at,
            "operation_id": run.operation_uuid,
            "config": config,
            "meta": meta_for_response,
            "message": message,
        }

        # Include operation status if operation provided
        if operation is not None:
            response["operation_status"] = operation.status.value
            # Override message with error message if operation failed
            if operation.error_message and not message:
                response["message"] = operation.error_message

        return response

    @staticmethod
    def _normalise_reducer_config(reducer: str, config: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate and normalise reducer-specific configuration."""

        reducer_key = reducer.lower()
        if reducer_key == "umap":
            if config is None:
                cfg: dict[str, Any] = {}
            elif isinstance(config, dict):
                cfg = dict(config)
            else:
                raise HTTPException(status_code=400, detail="config must be an object")
            try:
                n_neighbors = int(cfg.get("n_neighbors", 15))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="n_neighbors must be an integer") from None
            if n_neighbors < 2:
                raise HTTPException(status_code=400, detail="n_neighbors must be >= 2")

            try:
                min_dist = float(cfg.get("min_dist", 0.1))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="min_dist must be a number") from None
            if not 0.0 <= min_dist <= 1.0:
                raise HTTPException(status_code=400, detail="min_dist must be between 0 and 1")

            metric = str(cfg.get("metric", "cosine"))
            if not metric:
                raise HTTPException(status_code=400, detail="metric must be a non-empty string")

            cfg["n_neighbors"] = n_neighbors
            cfg["min_dist"] = min_dist
            cfg["metric"] = metric
            return cfg

        if reducer_key == "tsne":
            if config is None:
                cfg = {}
            elif isinstance(config, dict):
                cfg = dict(config)
            else:
                raise HTTPException(status_code=400, detail="config must be an object")

            try:
                perplexity = float(cfg.get("perplexity", 30.0))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="perplexity must be a number") from None
            if perplexity <= 0:
                raise HTTPException(status_code=400, detail="perplexity must be > 0")

            try:
                learning_rate = float(cfg.get("learning_rate", 200.0))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="learning_rate must be a number") from None
            if learning_rate <= 0:
                raise HTTPException(status_code=400, detail="learning_rate must be > 0")

            try:
                n_iter = int(cfg.get("n_iter", 1_000))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="n_iter must be an integer") from None
            if n_iter < 250:
                raise HTTPException(status_code=400, detail="n_iter must be >= 250")

            metric = str(cfg.get("metric", "euclidean"))
            if not metric:
                raise HTTPException(status_code=400, detail="metric must be a non-empty string")

            init = str(cfg.get("init", "pca")).lower()
            if init not in {"pca", "random"}:
                init = "pca"

            return {
                "perplexity": perplexity,
                "learning_rate": learning_rate,
                "n_iter": n_iter,
                "metric": metric,
                "init": init,
            }

        if config is None:
            return None
        if not isinstance(config, dict):
            raise HTTPException(status_code=400, detail="config must be an object")
        return config

    async def start_projection_build(
        self,
        collection_id: str,
        user_id: int,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Kick off a projection run for a collection.

        A projection build is always modelled as a *new* ProjectionRun and a
        backing Operation record:

        - Existing runs are never overwritten; recompute requests simply create
          additional runs for the same collection.
        - The ProjectionRun starts in ``PENDING`` while the Operation starts in
          ``PENDING`` and is later advanced to ``PROCESSING`` / ``COMPLETED`` /
          ``FAILED`` by the Celery task pipeline.

        The returned payload exposes both the projection run status and the
        operation status so that callers can treat projections as long‑running
        jobs and track progress via WebSocket channels.
        """

        logger.info(
            "Scheduling projection build for collection %s (user=%s) with params=%s",
            collection_id,
            user_id,
            parameters,
        )

        # Validate collection existence and permissions prior to acknowledgement
        collection = await self.collection_repo.get_by_uuid_with_permission_check(collection_id, user_id)

        reducer = str(parameters.get("reducer") or "umap").lower()
        dimensionality = int(parameters.get("dimensionality") or 2)
        if dimensionality != 2:
            raise HTTPException(status_code=400, detail="Only 2D projections are currently supported")

        raw_config = parameters.get("config") if isinstance(parameters.get("config"), dict) else None
        normalised_config = self._normalise_reducer_config(reducer, raw_config)

        colour_by = str(parameters.get("color_by") or "document_id").lower()
        run_config: dict[str, Any] = dict(normalised_config or {})

        # Persist colour mode and sampling controls as part of the immutable run
        # configuration so that recompute and selection calls can faithfully
        # describe how a particular projection was produced.
        run_config["color_by"] = colour_by

        # Sampling knobs may be provided as top‑level parameters; normalise
        # common aliases into the stored config so the worker can derive a
        # single sample_limit and the idempotency hash can treat sampling
        # consistently across clients.
        sample_limit: int | None = None
        sample_aliases = ("sample_size", "sample_limit", "sample_n")
        for alias in sample_aliases:
            if alias in parameters and parameters[alias] is not None:
                try:
                    value = int(parameters[alias])
                except (TypeError, ValueError):
                    raise HTTPException(status_code=400, detail=f"{alias} must be an integer") from None
                if value <= 0:
                    raise HTTPException(status_code=400, detail=f"{alias} must be > 0") from None
                run_config[alias] = value
                sample_limit = value
                break

        # Compute a deterministic idempotency hash so repeated recompute
        # requests with identical inputs can reuse an existing completed run.
        raw_client_hash = parameters.get("metadata_hash")
        client_metadata_hash = raw_client_hash.strip() if isinstance(raw_client_hash, str) else None

        computed_metadata_hash = compute_projection_metadata_hash(
            collection_id=collection.id,
            embedding_model=getattr(collection, "embedding_model", "") or "",
            collection_vector_count=getattr(collection, "vector_count", 0) or 0,
            collection_updated_at=getattr(collection, "updated_at", None),
            reducer=reducer,
            dimensionality=dimensionality,
            color_by=colour_by,
            config=run_config,
            sample_limit=sample_limit,
        )

        metadata_hash = computed_metadata_hash
        if client_metadata_hash:
            if client_metadata_hash != computed_metadata_hash:
                logger.warning(
                    "Client-supplied projection metadata_hash mismatch for collection %s: client=%s, computed=%s",
                    collection.id,
                    client_metadata_hash,
                    computed_metadata_hash,
                )
            else:
                metadata_hash = client_metadata_hash

        # Idempotent shortcut: when an identical, non-degraded completed run
        # already exists for this collection and metadata hash, return it
        # instead of creating a new ProjectionRun/Operation pair.
        if metadata_hash:
            existing = await self.projection_repo.find_latest_completed_by_metadata_hash(collection.id, metadata_hash)
            if existing is not None:
                if not self._is_run_degraded(existing) and self._is_metadata_compatible(
                    existing,
                    reducer=reducer,
                    dimensionality=dimensionality,
                    color_by=colour_by,
                    sample_limit=sample_limit,
                ):
                    operation = getattr(existing, "operation", None)
                    logger.info(
                        "Reusing completed projection run %s for collection %s via metadata_hash",
                        existing.uuid,
                        collection.id,
                    )
                    payload = self._encode_projection(
                        existing,
                        operation=operation,
                        message="Reused completed projection for identical parameters",
                    )
                    payload["idempotent_reuse"] = True
                    return payload

                logger.warning(
                    "Projection metadata_hash collision or degraded run for collection %s (hash=%s); "
                    "ignoring idempotent reuse",
                    collection.id,
                    metadata_hash,
                )

        run = await self.projection_repo.create(
            collection_id=collection.id,
            reducer=reducer,
            dimensionality=dimensionality,
            config=run_config,
            meta={"initiated_by": user_id},
            metadata_hash=metadata_hash,
        )
        operation = await self.operation_repo.create(
            collection_id=collection.id,
            user_id=user_id,
            operation_type=OperationType.PROJECTION_BUILD,
            config={
                "projection_run_id": run.uuid,
                "reducer": reducer,
                "dimensionality": dimensionality,
                "config": run_config,
            },
            meta={"projection_run_uuid": run.uuid},
        )

        run.operation_uuid = operation.uuid

        await self.db_session.flush()

        # Commit transaction BEFORE dispatching celery task
        await self.db_session.commit()

        try:
            celery_app.send_task(
                "webui.tasks.process_collection_operation",
                args=[operation.uuid],
                task_id=str(uuid.uuid4()),
            )
        except Exception as exc:  # pragma: no cover - broker failures
            logger.error("Failed to dispatch projection operation %s: %s", operation.uuid, exc)

            await self.operation_repo.update_status(
                operation.uuid,
                OperationStatus.FAILED,
                error_message=str(exc),
            )
            await self.projection_repo.update_status(
                run.uuid,
                status=ProjectionRunStatus.FAILED,
                error_message=str(exc),
            )
            await self.db_session.commit()
            raise HTTPException(status_code=503, detail="Failed to enqueue projection build task") from exc

        return self._encode_projection(run, operation=operation)

    async def list_projections(self, collection_id: str, user_id: int) -> list[dict[str, Any]]:
        """List projection runs for a collection (placeholder)."""

        logger.debug("Listing projections for collection %s", collection_id)
        await self.collection_repo.get_by_uuid_with_permission_check(collection_id, user_id)
        runs, _total = await self.projection_repo.list_for_collection(collection_id)
        projections: list[dict[str, Any]] = []
        for run in runs:
            operation = getattr(run, "operation", None)
            payload = self._encode_projection(run, operation=operation)
            projections.append(payload)
        return projections

    async def get_projection_metadata(self, collection_id: str, projection_id: str, user_id: int) -> dict[str, Any]:
        """Fetch metadata for a projection run (placeholder)."""

        logger.debug("Fetching projection metadata collection=%s projection=%s", collection_id, projection_id)
        collection = await self.collection_repo.get_by_uuid_with_permission_check(collection_id, user_id)
        owner_id = getattr(collection, "owner_id", None) or getattr(collection, "user_id", None)
        if owner_id is not None and owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection", collection_id)
        run = await self.projection_repo.get_by_uuid(projection_id)
        if not run or run.collection_id != collection_id:
            raise EntityNotFoundError("projection_run", projection_id)

        operation = None
        if run.operation_uuid:
            operation = await self.operation_repo.get_by_uuid(run.operation_uuid)

        return self._encode_projection(run, operation=operation)

    _ALLOWED_ARTIFACTS: dict[str, str] = {
        "x": "x.f32.bin",
        "y": "y.f32.bin",
        "ids": "ids.i32.bin",
        "cat": "cat.u8.bin",
    }
    _MAX_SELECTION_IDS = 5000

    async def _resolve_storage_directory(self, run: ProjectionRun, storage_path_raw: str) -> Path:
        """Resolve the storage directory for a projection run across environments."""

        data_dir = settings.data_dir.resolve()
        raw_path = Path(storage_path_raw)

        def _projection_suffix(path: Path) -> Path | None:
            parts = path.parts
            for idx in range(len(parts) - 1):
                if parts[idx] == "semantik" and parts[idx + 1] == "projections":
                    return Path(*parts[idx:])
            return None

        candidates: list[Path] = []

        def _add_candidate(path: Path) -> None:
            resolved = path if path.is_absolute() else data_dir / path
            resolved = resolved.resolve(strict=False)
            if resolved not in candidates:
                candidates.append(resolved)

        if raw_path.is_absolute():
            _add_candidate(raw_path)
            try:
                relative_path = raw_path.relative_to(data_dir)
            except ValueError:
                suffix = _projection_suffix(raw_path)
                if suffix is not None:
                    _add_candidate(suffix)
            else:
                _add_candidate(relative_path)
        else:
            _add_candidate(raw_path)

        resolved_dir: Path | None = None
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                resolved_dir = candidate
                break

        if resolved_dir is None:
            raise FileNotFoundError("Projection artifacts directory not found")

        try:
            normalized_relative = resolved_dir.relative_to(data_dir)
        except ValueError as exc:  # pragma: no cover - defensive
            raise PermissionError("Attempted access outside projection storage root") from exc

        normalized_storage = str(normalized_relative)
        if normalized_storage != storage_path_raw:
            run.storage_path = normalized_storage
            await self.db_session.flush()

        return resolved_dir

    async def resolve_artifact_path(
        self,
        collection_id: str,
        projection_id: str,
        artifact_name: str,
        user_id: int,
    ) -> Path:
        """Resolve and validate the on-disk path for a projection artifact."""

        normalized_name = artifact_name.strip().lower()
        if normalized_name not in self._ALLOWED_ARTIFACTS:
            raise ValueError(f"Unsupported projection artifact '{artifact_name}'")

        run = await self.projection_repo.get_by_uuid(projection_id)
        if not run or run.collection_id != collection_id:
            raise EntityNotFoundError("projection_run", projection_id)

        await self.collection_repo.get_by_uuid_with_permission_check(collection_id, user_id)

        storage_path_raw = getattr(run, "storage_path", None)
        if not storage_path_raw:
            raise HTTPException(status_code=409, detail="Projection artifacts are not yet available")

        try:
            resolved_dir = await self._resolve_storage_directory(run, storage_path_raw)
        except FileNotFoundError:
            # Artifacts directory has gone missing; mark the run as degraded so
            # callers can be prompted to recompute before surfacing the error.
            try:
                await self.projection_repo.update_metadata(run.uuid, meta={"degraded": True})
                await self.db_session.flush()
            except Exception:  # pragma: no cover - defensive logging
                logger.warning("Failed to mark projection %s as degraded after missing artifacts directory", run.uuid)
            raise

        file_path = (resolved_dir / self._ALLOWED_ARTIFACTS[normalized_name]).resolve()

        if resolved_dir not in file_path.parents and file_path != resolved_dir:
            raise PermissionError("Attempted access outside projection storage root")

        if not file_path.is_file():
            # Individual artifact is missing; treat the run as degraded so
            # metadata consumers can recommend recomputation.
            try:
                await self.projection_repo.update_metadata(run.uuid, meta={"degraded": True})
                await self.db_session.flush()
            except Exception:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to mark projection %s as degraded after missing artifact %s", run.uuid, artifact_name
                )
            raise FileNotFoundError(f"Projection artifact '{artifact_name}' not found")

        return file_path

    async def select_projection_region(
        self,
        collection_id: str,
        projection_id: str,
        selection: dict[str, Any],
        user_id: int,
    ) -> dict[str, Any]:
        """Resolve selection requests over a projection to chunk/document metadata."""

        logger.debug(
            "Selecting projection region collection=%s projection=%s selection=%s",
            collection_id,
            projection_id,
            selection,
        )

        ids = selection.get("ids")
        if not isinstance(ids, list) or not ids:
            raise HTTPException(status_code=400, detail="ids list is required")

        try:
            ordered_ids: list[int] = []
            seen_ids: set[int] = set()
            for value in ids:
                int_value = int(value)
                if int_value not in seen_ids:
                    seen_ids.add(int_value)
                    ordered_ids.append(int_value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="ids must be integers") from exc

        if len(ordered_ids) > self._MAX_SELECTION_IDS:
            raise HTTPException(status_code=413, detail=f"Too many ids; max {self._MAX_SELECTION_IDS}")

        await self.collection_repo.get_by_uuid_with_permission_check(collection_id, user_id)
        run = await self.projection_repo.get_by_uuid(projection_id)
        if not run or run.collection_id != collection_id:
            raise EntityNotFoundError("projection_run", projection_id)

        storage_path = getattr(run, "storage_path", None)
        if not storage_path:
            raise FileNotFoundError("Projection artifacts have not been generated yet")

        try:
            artifacts_dir = await self._resolve_storage_directory(run, storage_path)
        except FileNotFoundError as exc:
            # If the artifacts directory cannot be resolved anymore, consider
            # the run degraded so future metadata calls can surface a recompute
            # recommendation.
            try:
                await self.projection_repo.update_metadata(run.uuid, meta={"degraded": True})
                await self.db_session.flush()
            except Exception:  # pragma: no cover - defensive logging
                logger.warning("Failed to mark projection %s as degraded after missing artifacts directory", run.uuid)
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        ids_path = artifacts_dir / self._ALLOWED_ARTIFACTS["ids"]
        meta_path = artifacts_dir / "meta.json"

        if not ids_path.is_file():
            # Missing ids array indicates the on-disk projection is incomplete;
            # mark the run degraded while surfacing a 404 to the caller.
            try:
                await self.projection_repo.update_metadata(run.uuid, meta={"degraded": True})
                await self.db_session.flush()
            except Exception:  # pragma: no cover - defensive logging
                logger.warning("Failed to mark projection %s as degraded after missing ids artifact", run.uuid)
            raise HTTPException(status_code=404, detail="Projection ids artifact is missing")

        meta_payload: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:  # pragma: no cover - defensive
                # Corrupt metadata should not break selection entirely, but the
                # run should be marked degraded so the UI can suggest a
                # recompute.
                try:
                    await self.projection_repo.update_metadata(run.uuid, meta={"degraded": True})
                    await self.db_session.flush()
                except Exception:  # pragma: no cover - defensive logging
                    logger.warning("Failed to mark projection %s as degraded after invalid meta.json", run.uuid)
                meta_payload = {}

        run_meta: dict[str, Any] = run.meta if isinstance(run.meta, dict) else {}
        projection_meta_value = run_meta.get("projection_artifacts")
        projection_meta: dict[str, Any] = projection_meta_value if isinstance(projection_meta_value, dict) else {}
        if not projection_meta and meta_payload:
            projection_meta = meta_payload

        degraded_flag = bool(run_meta.get("degraded") or projection_meta.get("degraded"))

        # Qdrant vector collection name (if available) for optional metadata
        # resolution directly from the vector store when chunk/document
        # mappings are not present in the database.
        vector_collection_name: str | None = None
        if projection_meta:
            source_vector_collection = projection_meta.get("source_vector_collection")
            if isinstance(source_vector_collection, str) and source_vector_collection:
                vector_collection_name = source_vector_collection

        original_ids: list[str] | None = None
        if projection_meta:
            original_ids = (
                projection_meta.get("original_ids") if isinstance(projection_meta.get("original_ids"), list) else None
            )

        id_array = array("i")
        with ids_path.open("rb") as buffer:
            id_array.frombytes(buffer.read())

        requested_ids_set = set(ordered_ids)
        id_to_index: dict[int, int] = {}
        for index, value in enumerate(id_array):
            if value in requested_ids_set and value not in id_to_index:
                id_to_index[value] = index
                if len(id_to_index) == len(requested_ids_set):
                    break

        chunk_repo = ChunkRepository(self.db_session)
        document_repo = DocumentRepository(self.db_session)

        items: list[dict[str, Any]] = []
        missing_ids: list[int] = []

        # Lazy-initialise Qdrant client if we need to fall back to vector
        # store metadata. This avoids adding a hard dependency for tests or
        # environments without Qdrant.
        qdrant_client = None
        if vector_collection_name:
            try:  # pragma: no cover - exercised via integration, not unit tests
                qdrant_client = qdrant_manager.get_client()
            except Exception:
                qdrant_client = None

        for selected_id in ordered_ids:
            retrieved_index = id_to_index.get(selected_id)
            if retrieved_index is None:
                missing_ids.append(selected_id)
                continue

            original_identifier: str | None = None
            if original_ids and 0 <= retrieved_index < len(original_ids):
                raw_identifier = original_ids[retrieved_index]
                original_identifier = str(raw_identifier)
            else:
                raw_identifier = None

            chunk_data: dict[str, Any] | None = None
            document_data: dict[str, Any] | None = None

            # First, attempt to interpret the original identifier as a numeric
            # chunk primary key for backward compatibility with earlier runs
            # and tests that store chunk ids directly in original_ids.
            chunk_id: int | None = None
            if isinstance(raw_identifier, str):
                try:
                    chunk_id = int(raw_identifier)
                except ValueError:
                    chunk_id = None
            elif isinstance(raw_identifier, int):
                chunk_id = raw_identifier

            chunk = None
            if chunk_id is not None:
                try:
                    chunk = await chunk_repo.get_chunk_by_id(chunk_id, collection_id)
                except Exception:
                    chunk = None

            # If no chunk was found via primary key mapping, fall back to
            # resolving by embedding_vector_id when the original identifier
            # is a non-empty string (for example, a Qdrant point ID).
            if chunk is None and isinstance(raw_identifier, str) and raw_identifier:
                try:
                    chunk = await chunk_repo.get_chunk_by_embedding_vector_id(raw_identifier, collection_id)
                except Exception:
                    chunk = None

            # If we still cannot resolve a chunk from the relational database,
            # and a Qdrant client is available, attempt to pull minimal
            # metadata (doc_id, chunk_id, content) directly from the vector
            # store so tooltips and selection remain useful even when
            # embedding_vector_id has not been backfilled.
            if chunk is None and qdrant_client is not None and isinstance(raw_identifier, str) and raw_identifier:
                try:  # pragma: no cover - relies on live Qdrant in integration environments
                    records = qdrant_client.retrieve(
                        collection_name=vector_collection_name,
                        ids=[raw_identifier],
                        with_payload=True,
                    )
                except Exception:
                    records = None

                if records:
                    record = records[0]
                    payload = getattr(record, "payload", {}) or {}
                    doc_id = payload.get("doc_id") or None
                    content_value = payload.get("content")
                    content_preview = content_value[:200] if isinstance(content_value, str) else None

                    # Best-effort parsing of chunk_index from chunk_id-style strings.
                    chunk_index_value: int | None = None
                    chunk_id_str = payload.get("chunk_id")
                    if isinstance(chunk_id_str, str):
                        import re

                        match = re.search(r"(\\d+)$", chunk_id_str)
                        if match:
                            try:
                                chunk_index_value = int(match.group(1))
                            except ValueError:
                                chunk_index_value = None

                    if doc_id or content_preview:
                        chunk_data = {
                            "chunk_id": None,
                            "document_id": doc_id,
                            "chunk_index": chunk_index_value,
                            "content_preview": content_preview,
                        }
                        if doc_id:
                            try:
                                document = await document_repo.get_by_id(str(doc_id))
                            except Exception:
                                document = None
                            if document:
                                document_data = {
                                    "document_id": document.id,
                                    "file_name": document.file_name,
                                    "source_id": document.source_id,
                                    "mime_type": document.mime_type,
                                }

            if chunk:
                chunk_data = {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "content_preview": (chunk.content or "")[:200] if chunk.content else None,
                }
                if chunk.document_id:
                    try:
                        document = await document_repo.get_by_id(chunk.document_id)
                    except Exception:
                        document = None
                    if document:
                        document_data = {
                            "document_id": document.id,
                            "file_name": document.file_name,
                            "source_id": document.source_id,
                            "mime_type": document.mime_type,
                        }

            items.append(
                {
                    "selected_id": selected_id,
                    "index": index,
                    "original_id": original_identifier,
                    "chunk_id": chunk_data.get("chunk_id") if chunk_data else None,
                    "document_id": chunk_data.get("document_id") if chunk_data else None,
                    "chunk_index": chunk_data.get("chunk_index") if chunk_data else None,
                    "content_preview": chunk_data.get("content_preview") if chunk_data else None,
                }
            )

            if document_data:
                items[-1]["document"] = document_data

        return {
            "items": items,
            "missing_ids": missing_ids,
            "degraded": degraded_flag,
        }

    async def delete_projection(
        self,
        collection_id: str,
        projection_id: str,
        user_id: int,
    ) -> None:
        """Delete a projection run and associated on-disk artifacts."""

        await self.collection_repo.get_by_uuid_with_permission_check(collection_id, user_id)
        run = await self.projection_repo.get_by_uuid(projection_id)
        if not run or run.collection_id != collection_id:
            raise EntityNotFoundError("projection_run", projection_id)

        # Avoid deleting runs that are still being computed to prevent races
        # with the worker writing artifacts. Callers should wait for the
        # associated operation to reach a terminal state instead.
        if run.status in {ProjectionRunStatus.PENDING, ProjectionRunStatus.RUNNING}:
            raise HTTPException(status_code=409, detail="Projection is still in progress and cannot be deleted")

        storage_path = getattr(run, "storage_path", None)
        if storage_path:
            try:
                artifacts_dir = await self._resolve_storage_directory(run, storage_path)
            except FileNotFoundError:
                artifacts_dir = None
            except PermissionError as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Projection storage path for %s is outside data dir: %s", projection_id, exc)
                artifacts_dir = None

            if artifacts_dir:
                try:
                    shutil.rmtree(artifacts_dir, ignore_errors=False)
                except FileNotFoundError:
                    pass
                except Exception as exc:  # pragma: no cover - defensive cleanup
                    logger.warning("Failed to delete projection artifacts %s: %s", artifacts_dir, exc)

        await self.projection_repo.delete(projection_id)
        await self.db_session.commit()
