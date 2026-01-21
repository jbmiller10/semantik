"""Source Service for managing collection source operations."""

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.exceptions import AccessDeniedError, EntityNotFoundError, InvalidStateError, ValidationError
from shared.database.models import CollectionSource, OperationType
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.collection_source_repository import CollectionSourceRepository
from shared.database.repositories.connector_secret_repository import ConnectorSecretRepository
from shared.database.repositories.operation_repository import OperationRepository
from shared.utils.encryption import EncryptionNotConfiguredError
from webui.celery_app import celery_app
from webui.services.connector_factory import ConnectorFactory

logger = logging.getLogger(__name__)


class SourceService:
    """Service for managing collection source lifecycle operations.

    This service handles:
    - Creating, updating, and deleting sources
    - Triggering sync runs (manual and scheduled)
    - Pausing and resuming continuous syncs
    - Listing sources with sync status

    Uses the existing operation infrastructure for triggering ingestion.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        collection_repo: CollectionRepository,
        source_repo: CollectionSourceRepository,
        operation_repo: OperationRepository,
        secret_repo: ConnectorSecretRepository | None = None,
    ):
        """Initialize the source service.

        Args:
            db_session: Database session for transactions
            collection_repo: Collection repository for access checks
            source_repo: Source repository for CRUD operations
            operation_repo: Operation repository for dispatching operations
            secret_repo: Secret repository for encrypted credential storage (optional)
        """
        self.db_session = db_session
        self.collection_repo = collection_repo
        self.source_repo = source_repo
        self.operation_repo = operation_repo
        self.secret_repo = secret_repo

    async def create_source(
        self,
        user_id: int,
        collection_id: str,
        source_type: str,
        source_path: str,
        source_config: dict[str, Any],
        _sync_mode: str = "one_time",
        _interval_minutes: int | None = None,
        secrets: dict[str, str] | None = None,
    ) -> tuple[CollectionSource, list[str]]:
        """Create a new source for a collection.

        Args:
            user_id: ID of the user creating the source
            collection_id: UUID of the parent collection
            source_type: Type of source (directory, git, imap)
            source_path: Display path or identifier for the source
            source_config: Connector-specific configuration
            sync_mode: 'one_time' or 'continuous' (default: 'one_time')
            interval_minutes: Sync interval for continuous mode (min 15)
            secrets: Dict mapping secret_type to plaintext value (optional)

        Returns:
            Tuple of (CollectionSource instance, list of secret types stored)

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have access
            ValidationError: If validation fails
            EncryptionNotConfiguredError: If secrets provided but encryption not configured
        """
        normalized_type = (source_type or "").strip().lower()
        available_types = ConnectorFactory.list_available_types()
        if not normalized_type or normalized_type not in available_types:
            available = ", ".join(sorted(available_types)) if available_types else "none registered"
            raise ValidationError(
                f"Invalid source_type: '{source_type}'. Must be one of: {available}",
                field="source_type",
            )
        source_type = normalized_type

        # Verify user has access to collection
        collection = await self.collection_repo.get_by_uuid(collection_id)
        if not collection:
            raise EntityNotFoundError("collection", collection_id)

        if collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection", collection_id)

        secrets_to_store = {secret_type: value for secret_type, value in (secrets or {}).items() if value}
        if secrets_to_store and not self.secret_repo:
            logger.warning(
                "Secrets provided for source but encryption not configured "
                f"(collection_id={collection_id}, source_type={source_type}, source_path={source_path})"
            )
            raise EncryptionNotConfiguredError(
                "Encryption not configured - set CONNECTOR_SECRETS_KEY environment variable"
            )

        # Create the source
        # Note: sync_mode and interval_minutes are now managed at collection level
        source = await self.source_repo.create(
            collection_id=collection_id,
            source_type=source_type,
            source_path=source_path,
            source_config=source_config,
        )

        # Store secrets if provided
        secret_types: list[str] = []
        if secrets_to_store and self.secret_repo:
            try:
                for secret_type, value in secrets_to_store.items():
                    await self.secret_repo.set_secret(source.id, secret_type, value)
                    secret_types.append(secret_type)
            except EncryptionNotConfiguredError:
                logger.warning(
                    "Secrets provided for source %s but encryption not configured",
                    source.id,
                    exc_info=True,
                )
                raise

        logger.info(
            f"Created source {source.id} for collection {collection_id} "
            f"(type={source_type}, secrets={secret_types})"
        )

        return source, secret_types

    async def update_source(
        self,
        user_id: int,
        source_id: int,
        source_config: dict[str, Any] | None = None,
        _sync_mode: str | None = None,
        _interval_minutes: int | None = None,
        secrets: dict[str, str] | None = None,
    ) -> tuple[CollectionSource, list[str]]:
        """Update a source's configuration.

        Args:
            user_id: ID of the user updating the source
            source_id: ID of the source to update
            source_config: New connector-specific configuration
            _sync_mode: Deprecated - sync mode is now managed at collection level
            _interval_minutes: Deprecated - sync interval is now managed at collection level
            secrets: Dict mapping secret_type to new value (empty string deletes)

        Returns:
            Tuple of (Updated CollectionSource instance, list of current secret types)

        Raises:
            EntityNotFoundError: If source not found
            AccessDeniedError: If user doesn't have access
            ValidationError: If validation fails
            EncryptionNotConfiguredError: If secrets provided but encryption not configured
        """
        # Get source and verify access
        source = await self.source_repo.get_by_id(source_id)
        if not source:
            raise EntityNotFoundError("collection_source", str(source_id))

        collection = await self.collection_repo.get_by_uuid(source.collection_id)
        if not collection or collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection_source", str(source_id))

        if secrets and not self.secret_repo:
            logger.warning(
                "Secrets update requested for source %s but encryption not configured",
                source_id,
            )
            raise EncryptionNotConfiguredError(
                "Encryption not configured - set CONNECTOR_SECRETS_KEY environment variable"
            )

        # Update the source
        # Note: sync_mode and interval_minutes are now managed at collection level
        updated_source = await self.source_repo.update(
            source_id=source_id,
            source_config=source_config,
        )

        # Update secrets if provided
        if secrets and self.secret_repo:
            try:
                for secret_type, value in secrets.items():
                    if value:  # Non-empty: store/update
                        await self.secret_repo.set_secret(source_id, secret_type, value)
                    else:  # Empty string: delete
                        await self.secret_repo.delete_secret(source_id, secret_type)
            except EncryptionNotConfiguredError:
                logger.warning(
                    "Secrets update requested for source %s but encryption not configured",
                    source_id,
                    exc_info=True,
                )
                raise

        # Get current secret types
        secret_types: list[str] = []
        if self.secret_repo:
            secret_types = await self.secret_repo.get_secret_types_for_source(source_id)

        logger.info(f"Updated source {source_id}")
        return updated_source, secret_types

    async def delete_source(
        self,
        user_id: int,
        source_id: int,
    ) -> dict[str, Any]:
        """Delete a source and its documents.

        This triggers a REMOVE_SOURCE operation to delete the source's
        documents and vectors before removing the source record.

        Args:
            user_id: ID of the user deleting the source
            source_id: ID of the source to delete

        Returns:
            Operation dictionary for the REMOVE_SOURCE operation

        Raises:
            EntityNotFoundError: If source not found
            AccessDeniedError: If user doesn't have access
            InvalidStateError: If collection has active operation
        """
        # Get source and verify access
        source = await self.source_repo.get_by_id(source_id)
        if not source:
            raise EntityNotFoundError("collection_source", str(source_id))

        collection = await self.collection_repo.get_by_uuid(source.collection_id)
        if not collection or collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection_source", str(source_id))

        # Check for active operations
        active_ops = await self.operation_repo.get_active_operations(source.collection_id)
        if active_ops:
            raise InvalidStateError(
                "Collection has active operation(s). Please wait for them to complete before deleting sources."
            )

        # Create REMOVE_SOURCE operation
        operation = await self.operation_repo.create(
            collection_id=source.collection_id,
            user_id=user_id,
            operation_type=OperationType.REMOVE_SOURCE,
            config={"source_id": source_id},
        )

        # Commit before dispatching Celery task
        await self.db_session.commit()

        # Dispatch the task
        celery_app.send_task(
            "webui.tasks.process_collection_operation",
            args=[operation.uuid],
            queue="default",
        )

        logger.info(f"Dispatched REMOVE_SOURCE operation {operation.uuid} for source {source_id}")

        return {
            "id": operation.id,
            "uuid": operation.uuid,
            "type": operation.type.value,
            "status": operation.status.value,
        }

    async def get_source(
        self,
        user_id: int,
        source_id: int,
        include_secret_types: bool = False,
    ) -> CollectionSource | tuple[CollectionSource, list[str]]:
        """Get a source by ID.

        Args:
            user_id: ID of the user requesting the source
            source_id: ID of the source
            include_secret_types: If True, return tuple with secret types list

        Returns:
            CollectionSource instance, or tuple of (source, secret_types) if include_secret_types

        Raises:
            EntityNotFoundError: If source not found
            AccessDeniedError: If user doesn't have access
        """
        source = await self.source_repo.get_by_id(source_id)
        if not source:
            raise EntityNotFoundError("collection_source", str(source_id))

        collection = await self.collection_repo.get_by_uuid(source.collection_id)
        if not collection or collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection_source", str(source_id))

        if include_secret_types:
            secret_types: list[str] = []
            if self.secret_repo:
                secret_types = await self.secret_repo.get_secret_types_for_source(source_id)
            return source, secret_types

        return source

    async def list_sources(
        self,
        user_id: int,
        collection_id: str,
        offset: int = 0,
        limit: int = 50,
        include_secret_types: bool = False,
    ) -> tuple[list[CollectionSource], int] | tuple[list[tuple[CollectionSource, list[str]]], int]:
        """List sources for a collection.

        Args:
            user_id: ID of the user requesting sources
            collection_id: UUID of the collection
            offset: Pagination offset
            limit: Maximum results
            include_secret_types: If True, return (source, secret_types) tuples

        Returns:
            Tuple of (sources list, total count) or ((source, secret_types) list, total count)

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have access
        """
        # Verify user has access to collection
        collection = await self.collection_repo.get_by_uuid(collection_id)
        if not collection:
            raise EntityNotFoundError("collection", collection_id)

        if collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection", collection_id)

        sources, total = await self.source_repo.list_by_collection(
            collection_id=collection_id,
            offset=offset,
            limit=limit,
        )

        if include_secret_types:
            # Get secret types for each source
            sources_with_secrets: list[tuple[CollectionSource, list[str]]] = []
            for source in sources:
                secret_types: list[str] = []
                if self.secret_repo:
                    secret_types = await self.secret_repo.get_secret_types_for_source(source.id)
                sources_with_secrets.append((source, secret_types))
            return sources_with_secrets, total

        return sources, total

    async def run_now(
        self,
        user_id: int,
        source_id: int,
    ) -> dict[str, Any]:
        """Trigger an immediate sync run for a source.

        Creates an APPEND operation for the source and dispatches it.
        For continuous sync sources, also updates next_run_at.

        Args:
            user_id: ID of the user triggering the run
            source_id: ID of the source to sync

        Returns:
            Operation dictionary

        Raises:
            EntityNotFoundError: If source not found
            AccessDeniedError: If user doesn't have access
            InvalidStateError: If collection has active operation
        """
        # Get source and verify access
        source = await self.source_repo.get_by_id(source_id)
        if not source:
            raise EntityNotFoundError("collection_source", str(source_id))

        collection = await self.collection_repo.get_by_uuid(source.collection_id)
        if not collection or collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection_source", str(source_id))

        # Check for active operations
        active_ops = await self.operation_repo.get_active_operations(source.collection_id)
        if active_ops:
            raise InvalidStateError(
                "Collection has active operation(s). Please wait for them to complete before running sync."
            )

        # Mark sync as started
        await self.source_repo.update_sync_status(
            source_id=source_id,
            status="partial",  # Will be updated on completion
            started_at=datetime.now(UTC),
        )

        # Create APPEND operation
        operation = await self.operation_repo.create(
            collection_id=source.collection_id,
            user_id=user_id,
            operation_type=OperationType.APPEND,
            config={
                "source_id": source_id,
                "source_type": source.source_type,
                "source_config": source.source_config,
                "source_path": source.source_path,
            },
        )

        # Update next_run_at for continuous sync (now at collection level)
        if collection.sync_mode == "continuous" and collection.sync_interval_minutes:
            await self.collection_repo.set_next_sync_run(source.collection_id)

        # Commit before dispatching Celery task
        await self.db_session.commit()

        # Dispatch the task
        celery_app.send_task(
            "webui.tasks.process_collection_operation",
            args=[operation.uuid],
            queue="default",
        )

        logger.info(f"Dispatched APPEND operation {operation.uuid} for source {source_id}")

        return {
            "id": operation.id,
            "uuid": operation.uuid,
            "type": operation.type.value,
            "status": operation.status.value,
        }

    async def pause(
        self,
        user_id: int,
        source_id: int,
    ) -> CollectionSource:
        """Pause a source's continuous sync.

        Args:
            user_id: ID of the user pausing the source
            source_id: ID of the source to pause

        Returns:
            Updated CollectionSource instance

        Raises:
            EntityNotFoundError: If source not found
            AccessDeniedError: If user doesn't have access
            ValidationError: If source is not continuous sync
        """
        # Get source and verify access
        source = await self.source_repo.get_by_id(source_id)
        if not source:
            raise EntityNotFoundError("collection_source", str(source_id))

        collection = await self.collection_repo.get_by_uuid(source.collection_id)
        if not collection or collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection_source", str(source_id))

        return await self.source_repo.pause(source_id)

    async def resume(
        self,
        user_id: int,
        source_id: int,
    ) -> CollectionSource:
        """Resume a paused source's continuous sync.

        Args:
            user_id: ID of the user resuming the source
            source_id: ID of the source to resume

        Returns:
            Updated CollectionSource instance

        Raises:
            EntityNotFoundError: If source not found
            AccessDeniedError: If user doesn't have access
            ValidationError: If source is not continuous sync
        """
        # Get source and verify access
        source = await self.source_repo.get_by_id(source_id)
        if not source:
            raise EntityNotFoundError("collection_source", str(source_id))

        collection = await self.collection_repo.get_by_uuid(source.collection_id)
        if not collection or collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection_source", str(source_id))

        return await self.source_repo.resume(source_id)
