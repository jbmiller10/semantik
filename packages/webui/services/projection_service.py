"""Service layer for embedding projection operations and artifact access."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from array import array
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from shared.config import settings
from shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from shared.database.models import OperationStatus, OperationType, ProjectionRun, ProjectionRunStatus
from shared.database.repositories.chunk_repository import ChunkRepository
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.document_repository import DocumentRepository
from shared.database.repositories.operation_repository import OperationRepository
from shared.database.repositories.projection_run_repository import ProjectionRunRepository
from sqlalchemy.ext.asyncio import AsyncSession

from packages.webui.celery_app import celery_app

logger = logging.getLogger(__name__)


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
    def _encode_projection(run: ProjectionRun, *, operation: Any | None = None, message: str | None = None) -> dict[str, Any]:
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
        meta = dict(meta_raw) if meta_raw is not None else {}

        if config and "color_by" in config and "color_by" not in meta:
            meta["color_by"] = config["color_by"]

        if not meta:
            meta = None

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
            "meta": meta,
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

        Returns a placeholder response until the compute pipeline is implemented.
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
        run_config["color_by"] = colour_by

        run = await self.projection_repo.create(
            collection_id=collection.id,
            reducer=reducer,
            dimensionality=dimensionality,
            config=run_config,
            meta={"initiated_by": user_id},
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

        return self._encode_projection(run, operation=operation, message="Projection scheduling not yet implemented")

    async def list_projections(self, collection_id: str, user_id: int) -> list[dict[str, Any]]:
        """List projection runs for a collection (placeholder)."""

        logger.debug("Listing projections for collection %s", collection_id)
        await self.collection_repo.get_by_uuid_with_permission_check(collection_id, user_id)
        runs, _total = await self.projection_repo.list_for_collection(collection_id)
        projections: list[dict[str, Any]] = []
        for run in runs:
            operation = None
            if run.operation_uuid:
                operation = await self.operation_repo.get_by_uuid(run.operation_uuid)
            payload = self._encode_projection(run, operation=operation)
            projections.append(payload)
        return projections

    async def get_projection_metadata(self, collection_id: str, projection_id: str, user_id: int) -> dict[str, Any]:
        """Fetch metadata for a projection run (placeholder)."""

        logger.debug(
            "Fetching projection metadata collection=%s projection=%s", collection_id, projection_id
        )
        collection = await self.collection_repo.get_by_uuid_with_permission_check(collection_id, user_id)
        owner_id = getattr(collection, "owner_id", None) or getattr(collection, "user_id", None)
        if owner_id is not None and owner_id != user_id:
            raise AccessDeniedError("collection", collection_id)
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

        resolved_dir = await self._resolve_storage_directory(run, storage_path_raw)

        file_path = (resolved_dir / self._ALLOWED_ARTIFACTS[normalized_name]).resolve()

        if resolved_dir not in file_path.parents and file_path != resolved_dir:
            raise PermissionError("Attempted access outside projection storage root")

        if not file_path.is_file():
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
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        ids_path = artifacts_dir / self._ALLOWED_ARTIFACTS["ids"]
        meta_path = artifacts_dir / "meta.json"

        if not ids_path.is_file():
            raise HTTPException(status_code=404, detail="Projection ids artifact is missing")

        meta_payload: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:  # pragma: no cover - defensive
                meta_payload = {}

        run_meta = run.meta if isinstance(run.meta, dict) else {}
        projection_meta = run_meta.get("projection_artifacts") if isinstance(run_meta.get("projection_artifacts"), dict) else {}
        if not projection_meta and meta_payload:
            projection_meta = meta_payload

        degraded_flag = bool(run_meta.get("degraded") or projection_meta.get("degraded"))

        original_ids: list[str] | None = None
        if projection_meta:
            original_ids = projection_meta.get("original_ids") if isinstance(projection_meta.get("original_ids"), list) else None

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

        for selected_id in ordered_ids:
            index = id_to_index.get(selected_id)
            if index is None:
                missing_ids.append(selected_id)
                continue

            original_identifier: str | None = None
            if original_ids and 0 <= index < len(original_ids):
                raw_identifier = original_ids[index]
                original_identifier = str(raw_identifier)
            else:
                raw_identifier = None

            chunk_data: dict[str, Any] | None = None
            document_data: dict[str, Any] | None = None

            chunk_id: int | None = None
            if isinstance(raw_identifier, str):
                try:
                    chunk_id = int(raw_identifier)
                except ValueError:
                    chunk_id = None
            elif isinstance(raw_identifier, int):
                chunk_id = raw_identifier

            if chunk_id is not None:
                try:
                    chunk = await chunk_repo.get_chunk_by_id(chunk_id, collection_id)
                except Exception:
                    chunk = None

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
