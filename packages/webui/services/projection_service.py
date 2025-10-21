"""Service layer for embedding projection operations and artifact access."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from shared.database.exceptions import EntityNotFoundError
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.operation_repository import OperationRepository
from shared.database.repositories.projection_run_repository import ProjectionRunRepository
from shared.database.models import OperationStatus, OperationType, ProjectionRun, ProjectionRunStatus
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
    def _encode_projection(run: ProjectionRun, *, message: str | None = None) -> dict[str, Any]:
        """Convert a ProjectionRun ORM instance into a serialisable payload."""

        created_at = run.created_at if isinstance(run.created_at, datetime) else None
        config = run.config if isinstance(run.config, dict) else None
        meta = run.meta if isinstance(run.meta, dict) else None

        return {
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

        run = await self.projection_repo.create(
            collection_id=collection.id,
            reducer=reducer,
            dimensionality=dimensionality,
            config=normalised_config,
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
                "config": normalised_config or {},
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

        response = self._encode_projection(run, message="Projection scheduling not yet implemented")
        response["operation_status"] = operation.status.value
        return response

    async def list_projections(self, collection_id: str, user_id: int) -> list[dict[str, Any]]:
        """List projection runs for a collection (placeholder)."""

        logger.debug("Listing projections for collection %s", collection_id)
        await self.collection_repo.get_by_uuid_with_permission_check(collection_id, user_id)
        runs, _total = await self.projection_repo.list_for_collection(collection_id)
        projections: list[dict[str, Any]] = []
        for run in runs:
            payload = self._encode_projection(run)
            if run.operation_uuid:
                operation = await self.operation_repo.get_by_uuid(run.operation_uuid)
                if operation:
                    payload["operation_status"] = operation.status.value
                    if operation.error_message:
                        payload["message"] = operation.error_message
            projections.append(payload)
        return projections

    async def get_projection_metadata(self, collection_id: str, projection_id: str, user_id: int) -> dict[str, Any]:
        """Fetch metadata for a projection run (placeholder)."""

        logger.debug(
            "Fetching projection metadata collection=%s projection=%s", collection_id, projection_id
        )
        await self.collection_repo.get_by_uuid_with_permission_check(collection_id, user_id)
        run = await self.projection_repo.get_by_uuid(projection_id)
        if not run or run.collection_id != collection_id:
            raise EntityNotFoundError("projection_run", projection_id)

        message = None
        message: str | None = None
        response = self._encode_projection(run, message=message)

        if run.operation_uuid:
            operation = await self.operation_repo.get_by_uuid(run.operation_uuid)
            if operation:
                response["operation_status"] = operation.status.value
                if operation.error_message:
                    response["message"] = operation.error_message

        return response

    _ALLOWED_ARTIFACTS: dict[str, str] = {
        "x": "x.f32.bin",
        "y": "y.f32.bin",
        "ids": "ids.i32.bin",
        "cat": "cat.u8.bin",
    }

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

        storage_path = getattr(run, "storage_path", None)
        if not storage_path:
            raise FileNotFoundError("Projection artifacts have not been generated yet")

        base_dir = Path(storage_path).resolve()
        file_path = (base_dir / self._ALLOWED_ARTIFACTS[normalized_name]).resolve()

        if base_dir not in file_path.parents and file_path != base_dir:
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
        """Resolve selection requests over a projection (placeholder)."""

        logger.debug(
            "Selecting projection region collection=%s projection=%s selection=%s",
            collection_id,
            projection_id,
            selection,
        )
        await self.collection_repo.get_by_uuid_with_permission_check(collection_id, user_id)
        run = await self.projection_repo.get_by_uuid(projection_id)
        if not run or run.collection_id != collection_id:
            raise EntityNotFoundError("projection_run", projection_id)

        # TODO: map screen selection to documents/chunks
        return {
            "collection_id": collection_id,
            "projection_id": projection_id,
            "chunks": [],
            "message": "Projection selection not yet implemented",
        }
