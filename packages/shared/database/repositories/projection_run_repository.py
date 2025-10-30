"""Repository implementation for ProjectionRun model."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from shared.database.exceptions import DatabaseOperationError, EntityNotFoundError, ValidationError
from shared.database.models import Collection, ProjectionRun, ProjectionRunStatus
from sqlalchemy import Select, func, select
from sqlalchemy.orm import selectinload

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class ProjectionRunRepository:
    """Repository providing CRUD helpers for projection runs."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        collection_id: str,
        reducer: str,
        dimensionality: int,
        config: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ProjectionRun:
        """Create a new projection run stub for a collection."""
        if dimensionality <= 0:
            raise ValidationError("dimensionality must be greater than zero", "dimensionality")
        if not reducer:
            raise ValidationError("reducer is required", "reducer")

        collection_exists = await self.session.scalar(
            select(func.count()).select_from(Collection).where(Collection.id == collection_id)
        )
        if not collection_exists:
            raise EntityNotFoundError("collection", collection_id)

        run = ProjectionRun(
            uuid=str(uuid4()),
            collection_id=collection_id,
            reducer=reducer,
            dimensionality=dimensionality,
            config=config or {},
            meta=meta or {},
        )

        try:
            self.session.add(run)
            await self.session.flush()
            logger.info("Created projection run %s for collection %s", run.uuid, collection_id)
            return run
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to create projection run for collection %s: %s", collection_id, exc)
            raise DatabaseOperationError("create", "projection_run", str(exc)) from exc

    async def get_by_uuid(self, projection_uuid: str) -> ProjectionRun | None:
        """Fetch a projection run by its external UUID."""
        stmt: Select[tuple[ProjectionRun]] = (
            select(ProjectionRun)
            .where(ProjectionRun.uuid == projection_uuid)
            .options(selectinload(ProjectionRun.collection), selectinload(ProjectionRun.operation))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_for_collection(
        self,
        collection_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        statuses: Sequence[ProjectionRunStatus] | None = None,
    ) -> tuple[list[ProjectionRun], int]:
        """List projection runs for a collection with optional status filter."""
        stmt: Select[tuple[ProjectionRun]] = (
            select(ProjectionRun)
            .where(ProjectionRun.collection_id == collection_id)
            .order_by(ProjectionRun.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if statuses:
            stmt = stmt.where(ProjectionRun.status.in_(tuple(statuses)))

        runs = list((await self.session.execute(stmt)).scalars().all())

        count_stmt = select(func.count(ProjectionRun.id)).where(ProjectionRun.collection_id == collection_id)
        if statuses:
            count_stmt = count_stmt.where(ProjectionRun.status.in_(tuple(statuses)))
        total = await self.session.scalar(count_stmt) or 0
        return runs, total

    async def update_status(
        self,
        projection_uuid: str,
        *,
        status: ProjectionRunStatus,
        error_message: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> ProjectionRun:
        """Update lifecycle status for a projection run."""
        run = await self.get_by_uuid(projection_uuid)
        if not run:
            raise EntityNotFoundError("projection_run", projection_uuid)

        now = datetime.now(UTC)
        run.status = status
        run.updated_at = now

        if status == ProjectionRunStatus.RUNNING and run.started_at is None:
            run.started_at = started_at or now
        elif status in {ProjectionRunStatus.COMPLETED, ProjectionRunStatus.FAILED, ProjectionRunStatus.CANCELLED}:
            run.completed_at = completed_at or now
            if status == ProjectionRunStatus.COMPLETED:
                run.error_message = None

        if error_message is not None:
            run.error_message = error_message

        await self.session.flush()
        return run

    async def set_operation_uuid(self, projection_uuid: str, operation_uuid: str | None) -> ProjectionRun:
        """Link the projection run to a backing operation."""
        run = await self.get_by_uuid(projection_uuid)
        if not run:
            raise EntityNotFoundError("projection_run", projection_uuid)

        run.operation_uuid = operation_uuid
        await self.session.flush()
        return run

    async def update_metadata(
        self,
        projection_uuid: str,
        *,
        storage_path: str | None = None,
        point_count: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ProjectionRun:
        """Update storage attributes associated with a projection run."""
        run = await self.get_by_uuid(projection_uuid)
        if not run:
            raise EntityNotFoundError("projection_run", projection_uuid)

        if point_count is not None and point_count < 0:
            raise ValidationError("point_count must be non-negative", "point_count")

        if storage_path is not None:
            run.storage_path = storage_path
        if point_count is not None:
            run.point_count = point_count
        if meta:
            merged_meta = dict(run.meta or {})
            merged_meta.update(meta)
            run.meta = merged_meta

        run.updated_at = datetime.now(UTC)
        await self.session.flush()
        return run

    async def delete(self, projection_uuid: str) -> None:
        """Delete a projection run by UUID."""
        run = await self.get_by_uuid(projection_uuid)
        if not run:
            raise EntityNotFoundError("projection_run", projection_uuid)

        await self.session.delete(run)
        await self.session.flush()
