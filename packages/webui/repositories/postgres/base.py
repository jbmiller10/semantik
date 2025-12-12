"""Base PostgreSQL repository with common operations and optimizations."""

import logging
from typing import Any, TypeVar

from asyncpg.exceptions import ForeignKeyViolationError, UniqueViolationError
from sqlalchemy import func, insert, select, update as sql_update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import Select

from shared.database.exceptions import DatabaseOperationError, EntityAlreadyExistsError

T = TypeVar("T", bound=DeclarativeBase)

logger = logging.getLogger(__name__)


class PostgreSQLBaseRepository:
    """Base repository with PostgreSQL-specific optimizations.

    This class provides common database operations with PostgreSQL-specific
    features like ON CONFLICT, RETURNING, and bulk operations.
    """

    def __init__(self, session: AsyncSession, model: type[T]):
        """Initialize the repository.

        Args:
            session: AsyncSession instance for database operations
            model: SQLAlchemy model class
        """
        self.session = session
        self.model = model
        self.model_name = model.__name__.lower()

    async def bulk_insert(self, records: list[dict[str, Any]], on_conflict_update: bool = False) -> list[T]:
        """Bulk insert records using PostgreSQL optimizations.

        Args:
            records: List of dictionaries containing record data
            on_conflict_update: If True, update on conflict instead of raising error

        Returns:
            List of created/updated model instances

        Raises:
            EntityAlreadyExistsError: If record exists and on_conflict_update is False
            DatabaseOperationError: For database errors
        """
        if not records:
            return []

        try:
            if on_conflict_update:
                # Use PostgreSQL's INSERT ... ON CONFLICT
                stmt = pg_insert(self.model).values(records)

                # Get primary key columns
                pk_columns = [col.name for col in self.model.__table__.primary_key.columns]  # type: ignore[attr-defined]

                # Update all non-primary key columns on conflict
                update_columns = {col.name: col for col in stmt.excluded if col.name not in pk_columns}

                stmt = stmt.on_conflict_do_update(index_elements=pk_columns, set_=update_columns).returning(self.model)  # type: ignore[assignment]  # type: ignore[assignment]

                result = await self.session.execute(stmt)
                return list(result.scalars().all())
            # Regular bulk insert
            stmt = insert(self.model).values(records).returning(self.model)  # type: ignore[assignment]
            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        except IntegrityError as e:
            if isinstance(e.orig, UniqueViolationError):
                logger.error(f"Unique constraint violation during bulk insert: {e}")
                raise EntityAlreadyExistsError(self.model_name, "multiple records") from e
            if isinstance(e.orig, ForeignKeyViolationError):
                logger.error(f"Foreign key violation during bulk insert: {e}")
                raise DatabaseOperationError("bulk_insert", self.model_name, str(e)) from e
            logger.error(f"Integrity error during bulk insert: {e}")
            raise DatabaseOperationError("bulk_insert", self.model_name, str(e)) from e
        except Exception as e:
            logger.error(f"Failed to bulk insert {self.model_name} records: {e}")
            raise DatabaseOperationError("bulk_insert", self.model_name, str(e)) from e

    async def upsert(self, **kwargs: Any) -> T:
        """Insert or update a record using PostgreSQL's ON CONFLICT.

        Args:
            **kwargs: Field values for the record

        Returns:
            Created or updated model instance

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            stmt = pg_insert(self.model).values(**kwargs)

            # Get primary key columns
            pk_columns = [col.name for col in self.model.__table__.primary_key.columns]  # type: ignore[attr-defined]

            # Update all non-primary key columns on conflict
            update_columns = {col.name: col for col in stmt.excluded if col.name not in pk_columns}

            stmt = stmt.on_conflict_do_update(index_elements=pk_columns, set_=update_columns).returning(self.model)  # type: ignore[assignment]

            result = await self.session.execute(stmt)
            return result.scalar_one()  # type: ignore[no-any-return]

        except Exception as e:
            logger.error(f"Failed to upsert {self.model_name}: {e}")
            raise DatabaseOperationError("upsert", self.model_name, str(e)) from e

    async def bulk_update(self, updates: list[dict[str, Any]], key_field: str = "id") -> int:
        """Bulk update records using PostgreSQL optimizations.

        Args:
            updates: List of dictionaries containing updates (must include key_field)
            key_field: Field to match records by (default: "id")

        Returns:
            Number of records updated

        Raises:
            ValueError: If key_field is missing from any update
            DatabaseOperationError: For database errors
        """
        if not updates:
            return 0

        try:
            # Validate all updates have the key field
            for update in updates:
                if key_field not in update:
                    raise ValueError(f"Missing {key_field} in update: {update}")

            # Use PostgreSQL's UPDATE ... FROM VALUES
            updated_count = 0
            for update_data in updates:
                key_value = update_data.pop(key_field)
                stmt = sql_update(self.model).where(getattr(self.model, key_field) == key_value).values(**update_data)
                result = await self.session.execute(stmt)
                updated_count += result.rowcount or 0

            await self.session.flush()
            logger.info(f"Bulk updated {updated_count} {self.model_name} records")
            return updated_count

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to bulk update {self.model_name} records: {e}")
            raise DatabaseOperationError("bulk_update", self.model_name, str(e)) from e

    async def count(self, filters: Select | None = None) -> int:
        """Count records with optional filters.

        Args:
            filters: SQLAlchemy select statement with filters

        Returns:
            Number of records matching filters
        """
        try:
            if filters is not None:
                # Use the provided select statement but only count
                count_query = filters.with_only_columns(func.count()).select_from(self.model)
            else:
                count_query = select(func.count()).select_from(self.model)

            result = await self.session.scalar(count_query)
            return result or 0
        except Exception as e:
            logger.error(f"Failed to count {self.model_name} records: {e}")
            raise DatabaseOperationError("count", self.model_name, str(e)) from e

    async def exists(self, **kwargs: Any) -> bool:
        """Check if a record exists with given criteria.

        Args:
            **kwargs: Field values to match

        Returns:
            True if record exists, False otherwise
        """
        try:
            from sqlalchemy import exists, select

            stmt = select(exists().where(*[getattr(self.model, k) == v for k, v in kwargs.items()]))
            result = await self.session.scalar(stmt)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to check existence of {self.model_name}: {e}")
            raise DatabaseOperationError("exists", self.model_name, str(e)) from e

    def handle_integrity_error(self, error: IntegrityError, operation: str) -> None:
        """Handle PostgreSQL-specific integrity errors.

        Args:
            error: The IntegrityError to handle
            operation: The operation that caused the error

        Raises:
            EntityAlreadyExistsError: For unique constraint violations
            DatabaseOperationError: For other integrity errors
        """
        if isinstance(error.orig, UniqueViolationError):
            # Extract constraint name if available
            constraint_name = getattr(error.orig, "constraint_name", "unknown")
            logger.error(f"Unique constraint violation ({constraint_name}) during {operation}")
            raise EntityAlreadyExistsError(self.model_name, f"constraint: {constraint_name}") from error
        if isinstance(error.orig, ForeignKeyViolationError):
            # Extract foreign key info if available
            detail = getattr(error.orig, "detail", "unknown")
            logger.error(f"Foreign key violation during {operation}: {detail}")
            raise DatabaseOperationError(operation, self.model_name, f"Foreign key violation: {detail}") from error
        logger.error(f"Integrity error during {operation}: {error}")
        raise DatabaseOperationError(operation, self.model_name, str(error)) from error
