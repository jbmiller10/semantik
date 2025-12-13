"""Collection Service for managing collection operations."""

import asyncio
import logging
import re
import uuid
from typing import Any, cast

from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from shared.chunking.infrastructure.exceptions import ChunkingStrategyError
from shared.database.exceptions import (
    AccessDeniedError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidStateError,
    ValidationError,
)
from shared.database.models import Collection, CollectionStatus, CollectionSyncRun, OperationType
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.collection_source_repository import CollectionSourceRepository
from shared.database.repositories.collection_sync_run_repository import CollectionSyncRunRepository
from shared.database.repositories.document_repository import DocumentRepository
from shared.database.repositories.operation_repository import OperationRepository
from shared.managers import QdrantManager
from webui.celery_app import celery_app
from webui.qdrant import get_qdrant_manager
from webui.services.chunking_config_builder import ChunkingConfigBuilder
from webui.services.chunking_strategy_factory import ChunkingStrategyFactory

logger = logging.getLogger(__name__)

# Configuration constants
QDRANT_COLLECTION_PREFIX = "collection_"
DEFAULT_VECTOR_DIMENSION = 1536  # Default vector dimension for embeddings

# Backward compatibility for tests that monkeypatch the module-level manager


class CollectionService:
    """Service for managing collection operations."""

    def __init__(
        self,
        db_session: AsyncSession,
        collection_repo: CollectionRepository,
        operation_repo: OperationRepository,
        document_repo: DocumentRepository,
        collection_source_repo: CollectionSourceRepository,
        qdrant_manager: QdrantManager | None,
        sync_run_repo: CollectionSyncRunRepository | None = None,
    ):
        """Initialize the collection service."""
        self.db_session = db_session
        self.collection_repo = collection_repo
        self.operation_repo = operation_repo
        self.document_repo = document_repo
        self.collection_source_repo = collection_source_repo
        self.qdrant_manager = qdrant_manager
        self.sync_run_repo = sync_run_repo

    def _ensure_qdrant_manager(self) -> QdrantManager | None:
        """Lazily resolve a Qdrant manager, honoring test monkeypatches."""
        if self.qdrant_manager is not None:
            return self.qdrant_manager

        try:
            self.qdrant_manager = get_qdrant_manager()
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Qdrant manager unavailable: %s", exc)
            return None

        return self.qdrant_manager

    async def create_collection(
        self,
        user_id: int,
        name: str,
        description: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create a new collection and dispatch indexing operation.

        Args:
            user_id: ID of the user creating the collection
            name: Name of the collection
            description: Optional description
            config: Optional configuration (embedding model, chunk settings, etc.)

        Returns:
            Tuple of (collection, operation) dictionaries

        Raises:
            ValueError: If validation fails
            AccessDeniedError: If user doesn't have permission
        """
        # Validate inputs
        if not name or not name.strip():
            raise ValueError("Collection name is required")

        # Create collection in database
        try:
            # Apply expected defaults for legacy chunking fields
            # Pull values from config while treating explicit None as "unspecified"
            embedding_model = (config.get("embedding_model") if config else None) or "Qwen/Qwen3-Embedding-0.6B"
            quantization = (config.get("quantization") if config else None) or "float16"
            # If client sends null for legacy fields, fall back to safe defaults
            chunk_size = (config.get("chunk_size") if config else None) or 1000
            chunk_overlap = (config.get("chunk_overlap") if config else None) or 200
            chunking_strategy = config.get("chunking_strategy") if config else None
            chunking_config = config.get("chunking_config") if config else None
            is_public = (config.get("is_public") if config else None) or False

            meta = config.get("metadata") if config else None

            # Validate chunking strategy if provided
            if chunking_strategy is not None:
                try:
                    # Validate that the strategy exists and is supported
                    ChunkingStrategyFactory.create_strategy(
                        strategy_name=chunking_strategy,
                        config=chunking_config or {},
                    )
                    # Normalize the strategy name to internal format for persistence
                    chunking_strategy = ChunkingStrategyFactory.normalize_strategy_name(chunking_strategy)
                except ChunkingStrategyError as e:
                    # Use the structured error fields for a user-friendly response
                    if "Unknown strategy" in e.reason:
                        available = ChunkingStrategyFactory.get_available_strategies()
                        raise ValueError(
                            f"Invalid chunking_strategy '{e.strategy}'. "
                            f"Available strategies: {', '.join(available)}"
                        ) from None
                    raise ValueError(f"Invalid chunking_strategy: Strategy {e.strategy} failed: {e.reason}") from None
                except Exception as e:
                    # Catch any other unexpected errors
                    raise ValueError(f"Invalid chunking_strategy: {str(e)}") from None

            # Validate chunking config if provided
            if chunking_config is not None:
                # Only validate config if we have a strategy
                if chunking_strategy is not None:
                    config_builder = ChunkingConfigBuilder()
                    result = config_builder.build_config(
                        strategy=chunking_strategy,
                        user_config=chunking_config,
                    )
                    if result.validation_errors:
                        errors = "; ".join(result.validation_errors)
                        raise ValueError(f"Invalid chunking_config for strategy '{chunking_strategy}': {errors}")
                    # Use the validated and normalized config
                    chunking_config = result.config
                else:
                    # Config without strategy is not allowed
                    raise ValueError("chunking_config requires chunking_strategy to be specified")

            # Create with new chunking fields
            collection = await self.collection_repo.create(
                owner_id=user_id,
                name=name,
                description=description,
                embedding_model=embedding_model,
                quantization=quantization,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunking_strategy=chunking_strategy,
                chunking_config=chunking_config,
                is_public=is_public,
                meta=meta,
            )
        except EntityAlreadyExistsError:
            # Re-raise EntityAlreadyExistsError to be handled by the API endpoint
            raise
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

        # Create operation record
        operation = await self.operation_repo.create(
            collection_id=collection.id,
            user_id=user_id,
            operation_type=OperationType.INDEX,
            config={
                "sources": [],  # Initial creation has no sources
                "collection_config": config or {},
            },
        )

        # Commit the transaction BEFORE dispatching the task
        await self.db_session.commit()

        # Dispatch Celery task AFTER commit to avoid race condition
        celery_app.send_task(
            "webui.tasks.process_collection_operation",
            args=[operation.uuid],
            task_id=str(uuid.uuid4()),
        )

        # Convert ORM object to dictionary via shared serializer
        collection_dict = self._collection_to_dict(collection)

        operation_dict = {
            "uuid": operation.uuid,
            "collection_id": operation.collection_id,
            "type": operation.type.value,
            "status": operation.status.value,
            "config": operation.config,
            "created_at": operation.created_at,
            "started_at": operation.started_at,
            "completed_at": operation.completed_at,
            "error_message": operation.error_message,
        }

        return collection_dict, operation_dict

    async def add_source(
        self,
        collection_id: str,
        user_id: int,
        source_type: str = "directory",
        source_config: dict[str, Any] | None = None,
        legacy_source_path: str | None = None,
        additional_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a source to an existing collection.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user performing the operation
            source_type: Type of source (e.g., "directory", "web", "slack")
            source_config: Connector-specific configuration
            legacy_source_path: Deprecated - for backward compatibility
            additional_config: Additional config (chunk settings, metadata)

        Returns:
            Operation dictionary

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
            InvalidStateError: If collection is in invalid state
        """
        # Normalize: derive/augment source_config from legacy_source_path if needed
        if legacy_source_path is not None:
            if not source_config:
                source_config = {"path": legacy_source_path}
                source_type = "directory"
            elif source_type == "directory" and "path" not in source_config:
                # Some clients send directory options (e.g., recursive) in source_config
                # while still providing the actual path separately. Merge the path in.
                source_config = {**source_config, "path": legacy_source_path}

        # Derive source_path for the operation config (for display/audit)
        # For directory sources, use "path"; for web sources, use "url"; otherwise first value
        source_path = self._derive_source_path(source_type, source_config)
        if not source_path:
            raise ValidationError(
                "source_config must include a path/url/identifier (e.g. source_config={'path': ...}) or provide source_path",
                "source_path",
            )

        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Validate collection state
        if collection.status not in [CollectionStatus.PENDING, CollectionStatus.READY, CollectionStatus.DEGRADED]:
            raise InvalidStateError(
                f"Cannot add source to collection in {collection.status} state. "
                f"Collection must be in {CollectionStatus.PENDING}, {CollectionStatus.READY} or {CollectionStatus.DEGRADED} state."
            )

        # Check if there's already an active operation
        active_operations = await self.operation_repo.get_active_operations(collection.id)
        if active_operations:
            # Allow a short grace period for recently-finished operations to commit their status.
            for attempt in range(5):
                await asyncio.sleep(0.1 * (attempt + 1))
                active_operations = await self.operation_repo.get_active_operations(collection.id)
                if not active_operations:
                    break

            if active_operations:
                raise InvalidStateError(
                    "Cannot add source while another operation is in progress. "
                    "Please wait for the current operation to complete."
                )

        # Create or get existing CollectionSource
        collection_source, is_new_source = await self.collection_source_repo.get_or_create(
            collection_id=collection.id,
            source_type=source_type,
            source_path=source_path,
            source_config=source_config,
        )

        if is_new_source:
            logger.info(f"Created new CollectionSource {collection_source.id} for collection {collection.id}")
        else:
            logger.info(f"Reusing existing CollectionSource {collection_source.id} for collection {collection.id}")

        # Create operation record with source_id included
        operation = await self.operation_repo.create(
            collection_id=collection.id,
            user_id=user_id,
            operation_type=OperationType.APPEND,
            config={
                "source_id": collection_source.id,  # Link to CollectionSource for task
                "source_type": source_type,
                "source_config": source_config or {},
                "source_path": source_path,  # Keep for backward compatibility/audit
                "additional_config": additional_config or {},
            },
        )

        # Update collection status to processing
        await self.collection_repo.update_status(collection.id, CollectionStatus.PROCESSING)

        # Commit the transaction BEFORE dispatching the task
        await self.db_session.commit()

        # Dispatch Celery task AFTER commit to avoid race condition
        celery_app.send_task(
            "webui.tasks.process_collection_operation",
            args=[operation.uuid],
            task_id=str(uuid.uuid4()),
        )

        # Convert ORM object to dictionary
        return {
            "uuid": operation.uuid,
            "collection_id": operation.collection_id,
            "type": operation.type.value,
            "status": operation.status.value,
            "config": operation.config,
            "created_at": operation.created_at,
            "started_at": operation.started_at,
            "completed_at": operation.completed_at,
            "error_message": operation.error_message,
        }

    def _derive_source_path(self, source_type: str, source_config: dict[str, Any] | None) -> str:
        """Derive display source path from source type and config.

        Args:
            source_type: Type of source (directory, web, slack, etc.)
            source_config: Connector-specific configuration

        Returns:
            String path/identifier for display and deduplication
        """
        if not source_config:
            return ""

        # For directory sources, use "path"
        if source_type == "directory":
            path = source_config.get("path")
            return str(path) if path is not None else ""

        # For web sources, use "url"
        if source_type == "web":
            url = source_config.get("url")
            return str(url) if url is not None else ""

        # For other sources, try common keys or return first string value
        for key in ["path", "url", "channel", "identifier"]:
            if key in source_config and isinstance(source_config[key], str):
                return str(source_config[key])

        # Fallback: return first string value found
        for value in source_config.values():
            if isinstance(value, str):
                return value

        return ""

    async def reindex_collection(
        self, collection_id: str, user_id: int, config_updates: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Reindex a collection with optional configuration updates.

        Implements blue-green reindexing for zero downtime.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user performing the operation
            config_updates: Optional configuration updates to apply

        Returns:
            Operation dictionary

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
            InvalidStateError: If collection is in invalid state
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Validate collection state
        if collection.status == CollectionStatus.PROCESSING:
            raise InvalidStateError(
                "Cannot reindex collection that is currently processing. "
                "Please wait for the current operation to complete."
            )

        if collection.status == CollectionStatus.ERROR:
            raise InvalidStateError(
                "Cannot reindex failed collection. Please delete and recreate the collection instead."
            )

        # Check if there's already an active operation
        active_ops = await self.operation_repo.get_active_operations_count(collection.id)
        if active_ops > 0:
            raise InvalidStateError(
                "Cannot reindex while another operation is in progress. "
                "Please wait for the current operation to complete."
            )

        # Merge config updates with existing config
        new_config = {
            "embedding_model": collection.embedding_model,
            "quantization": collection.quantization,
            "chunk_size": collection.chunk_size,
            "chunk_overlap": collection.chunk_overlap,
            "is_public": collection.is_public,
            "metadata": collection.meta,
        }
        if config_updates:
            new_config.update(config_updates)

        # Create operation record
        operation = await self.operation_repo.create(
            collection_id=collection.id,
            user_id=user_id,
            operation_type=OperationType.REINDEX,
            config={
                "previous_config": {
                    "embedding_model": collection.embedding_model,
                    "quantization": collection.quantization,
                    "chunk_size": collection.chunk_size,
                    "chunk_overlap": collection.chunk_overlap,
                    "is_public": collection.is_public,
                    "metadata": collection.meta,
                },
                "new_config": new_config,
                "blue_green": True,  # Always use blue-green for zero downtime
            },
        )

        # Update collection status to processing
        await self.collection_repo.update_status(collection.id, CollectionStatus.PROCESSING)

        # Commit the transaction BEFORE dispatching the task
        await self.db_session.commit()

        # Dispatch Celery task AFTER commit to avoid race condition
        celery_app.send_task(
            "webui.tasks.process_collection_operation",
            args=[operation.uuid],
            task_id=str(uuid.uuid4()),
        )

        # Convert ORM object to dictionary
        return {
            "uuid": operation.uuid,
            "collection_id": operation.collection_id,
            "type": operation.type.value,
            "status": operation.status.value,
            "config": operation.config,
            "created_at": operation.created_at,
            "started_at": operation.started_at,
            "completed_at": operation.completed_at,
            "error_message": operation.error_message,
        }

    async def delete_collection(self, collection_id: str, user_id: int) -> None:
        """Delete a collection and all associated data.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user performing the operation

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
            InvalidStateError: If collection cannot be deleted
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Only owner can delete
        if collection.owner_id != user_id:
            raise AccessDeniedError(user_id=str(user_id), resource_type="Collection", resource_id=collection_id)

        # Check if there's an active operation
        active_ops = await self.operation_repo.get_active_operations_count(collection.id)
        if active_ops > 0:
            raise InvalidStateError(
                "Cannot delete collection while operations are in progress. "
                "Please cancel or wait for operations to complete."
            )

        try:
            manager = self._ensure_qdrant_manager()

            # Delete from Qdrant if collection exists
            if collection.vector_store_name and manager is not None:
                try:
                    collection_names = manager.list_collections()
                    if collection.vector_store_name in collection_names:
                        manager.client.delete_collection(collection.vector_store_name)
                        logger.info(f"Deleted Qdrant collection: {collection.vector_store_name}")
                except Exception as e:
                    logger.error(f"Failed to delete Qdrant collection: {e}")
                    # Continue with database deletion even if Qdrant deletion fails

            # Delete from database (cascade will handle operations, documents, etc.)
            await self.collection_repo.delete(collection_id, user_id)

            # Commit the transaction to persist the deletion
            await self.db_session.commit()

            logger.info(f"Deleted collection {collection_id} and all associated data")

        except Exception as e:
            logger.error(f"Failed to delete collection {collection_id}: {e}")
            raise

    async def remove_source(self, collection_id: str, user_id: int, source_path: str) -> dict[str, Any]:
        """Remove documents from a specific source.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user performing the operation
            source_path: Path of the source to remove

        Returns:
            Operation dictionary

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
            InvalidStateError: If collection is in invalid state
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Validate collection state
        if collection.status not in [CollectionStatus.READY, CollectionStatus.DEGRADED]:
            raise InvalidStateError(
                f"Cannot remove source from collection in {collection.status} state. "
                f"Collection must be in {CollectionStatus.READY} or {CollectionStatus.DEGRADED} state."
            )

        # Check if there's already an active operation
        active_ops = await self.operation_repo.get_active_operations_count(collection.id)
        if active_ops > 0:
            raise InvalidStateError(
                "Cannot remove source while another operation is in progress. "
                "Please wait for the current operation to complete."
            )

        # Look up the collection source to get source_id
        collection_source = await self.collection_source_repo.get_by_collection_and_path(
            collection_id=collection.id, source_path=source_path
        )
        if not collection_source:
            raise EntityNotFoundError("collection_source", source_path)

        # Create operation record
        operation = await self.operation_repo.create(
            collection_id=collection.id,
            user_id=user_id,
            operation_type=OperationType.REMOVE_SOURCE,
            config={
                "source_id": collection_source.id,
                "source_path": source_path,
            },
        )

        # Update collection status
        await self.collection_repo.update_status(collection.id, CollectionStatus.PROCESSING)

        # Commit the transaction BEFORE dispatching the task
        await self.db_session.commit()

        # Dispatch Celery task AFTER commit to avoid race condition
        celery_app.send_task(
            "webui.tasks.process_collection_operation",
            args=[operation.uuid],
            task_id=str(uuid.uuid4()),
        )

        # Convert ORM object to dictionary
        return {
            "uuid": operation.uuid,
            "collection_id": operation.collection_id,
            "type": operation.type.value,
            "status": operation.status.value,
            "config": operation.config,
            "created_at": operation.created_at,
            "started_at": operation.started_at,
            "completed_at": operation.completed_at,
            "error_message": operation.error_message,
        }

    async def list_for_user(
        self, user_id: int, offset: int = 0, limit: int = 50, include_public: bool = True
    ) -> tuple[list[Collection], int]:
        """List collections accessible to a user.

        Args:
            user_id: ID of the user
            offset: Pagination offset
            limit: Pagination limit
            include_public: Whether to include public collections

        Returns:
            Tuple of (collections list, total count)
        """
        result = await self.collection_repo.list_for_user(
            user_id=user_id,
            offset=offset,
            limit=limit,
            include_public=include_public,
        )
        return cast(tuple[list[Collection], int], result)

    async def update(self, collection_id: str, user_id: int, updates: dict[str, Any]) -> Collection:
        """Update collection metadata with chunking validation.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user performing the update
            updates: Dictionary of fields to update

        Returns:
            Updated collection object

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
            EntityAlreadyExistsError: If new name already exists
            ValueError: If chunking strategy/config validation fails
        """
        # Get collection with permission check (only owner can update)
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Only the owner can update the collection
        if collection.owner_id != user_id:
            raise AccessDeniedError(user_id=str(user_id), resource_type="Collection", resource_id=collection_id)

        # Handle chunking strategy update with validation
        if "chunking_strategy" in updates:
            strategy = updates["chunking_strategy"]
            if strategy:
                try:
                    # Normalize and validate strategy
                    factory = ChunkingStrategyFactory()
                    normalized_strategy = factory.normalize_strategy_name(strategy)
                    # Validate by attempting to create (but don't actually use it)
                    factory.create_strategy(
                        strategy_name=normalized_strategy,
                        config={},
                        correlation_id=str(uuid.uuid4()),
                    )
                    updates["chunking_strategy"] = normalized_strategy
                except ChunkingStrategyError as e:
                    raise ValueError(f"Invalid chunking strategy: {e}") from e

        # Handle chunking config update with validation
        if "chunking_config" in updates:
            config = updates["chunking_config"]
            if config:
                # Require a strategy to be present
                strategy = updates.get("chunking_strategy") or collection.chunking_strategy
                if not strategy:
                    raise ValueError("chunking_config requires chunking_strategy to be set")

                # Validate and normalize config
                try:
                    builder = ChunkingConfigBuilder()
                    config_result = builder.build_config(
                        strategy=strategy,
                        user_config=config,
                    )
                    if config_result.validation_errors:
                        raise ValueError(f"Invalid chunking config: {', '.join(config_result.validation_errors)}")
                    updates["chunking_config"] = config_result.config
                except Exception as e:
                    raise ValueError(f"Invalid chunking config: {e}") from e

        requires_qdrant_sync = (
            "name" in updates
            and updates["name"]
            and updates["name"] != collection.name
            and getattr(collection, "vector_store_name", None)
        )

        new_vector_store_name: str | None = None
        old_vector_store_name = getattr(collection, "vector_store_name", None)
        qdrant_manager_for_rename: QdrantManager | None = None
        if requires_qdrant_sync:
            manager = self._ensure_qdrant_manager()
            if manager is None or not hasattr(manager, "rename_collection"):
                raise RuntimeError("Qdrant manager is not available to rename collection")
            qdrant_manager_for_rename = cast(QdrantManager, manager)
            new_vector_store_name = self._build_vector_store_name(str(collection.id), updates["name"])
            updates["vector_store_name"] = new_vector_store_name

        qdrant_rename_performed = False

        try:
            updated_collection = await self.collection_repo.update(str(collection.id), updates)

            if requires_qdrant_sync and new_vector_store_name and old_vector_store_name:
                assert qdrant_manager_for_rename is not None  # mypy assurance
                await qdrant_manager_for_rename.rename_collection(
                    old_name=old_vector_store_name,
                    new_name=new_vector_store_name,
                )
                qdrant_rename_performed = True

            await self.db_session.commit()
            return cast(Collection, updated_collection)
        except Exception as exc:  # pragma: no cover - covered via explicit tests
            await self.db_session.rollback()
            if qdrant_rename_performed and requires_qdrant_sync and new_vector_store_name and old_vector_store_name:
                try:
                    assert qdrant_manager_for_rename is not None  # mypy assurance
                    await qdrant_manager_for_rename.rename_collection(
                        old_name=new_vector_store_name,
                        new_name=old_vector_store_name,
                    )
                except Exception as revert_exc:  # best effort rollback; log and continue raising
                    logger.error(
                        "Failed to revert Qdrant rename from %s back to %s after DB rollback: %s",
                        new_vector_store_name,
                        old_vector_store_name,
                        revert_exc,
                    )
            raise exc

    @staticmethod
    def _build_vector_store_name(collection_id: str, new_name: str) -> str:
        base = collection_id.replace("-", "_")
        slug = re.sub(r"[^a-z0-9]+", "_", new_name.lower()).strip("_")
        candidate = f"col_{base}_{slug}" if slug else f"col_{base}"

        # Qdrant currently allows up to 255 chars; keep buffer for safety
        if len(candidate) > 120:
            candidate = candidate[:120].rstrip("_")

        return candidate

    async def list_documents(
        self, collection_id: str, user_id: int, offset: int = 0, limit: int = 50
    ) -> tuple[list[Any], int]:
        """List documents in a collection.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user requesting documents
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (documents list, total count)

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Get documents for the collection
        documents, total = await self.document_repo.list_by_collection(
            collection_id=collection.id,
            offset=offset,
            limit=limit,
        )

        return documents, total

    async def list_operations(
        self, collection_id: str, user_id: int, offset: int = 0, limit: int = 50
    ) -> tuple[list[Any], int]:
        """List operations for a collection.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user requesting operations
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (operations list, total count)

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Get operations for the collection
        operations, total = await self.operation_repo.list_for_collection(
            collection_id=collection.id,
            user_id=user_id,
            offset=offset,
            limit=limit,
        )

        return operations, total

    async def list_operations_filtered(
        self,
        collection_id: str,
        user_id: int,
        status: str | None = None,
        operation_type: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Any], int]:
        """List operations for a collection with filtering.

        This method contains the filtering logic that was previously
        in the router, ensuring proper separation of concerns.

        Args:
            collection_id: Collection UUID
            user_id: User ID for permission check
            status: Optional status filter
            operation_type: Optional type filter
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (filtered operations, total count)

        Raises:
            ValueError: If invalid filter values provided
        """
        from shared.database.models import OperationStatus, OperationType

        # Validate filters first
        if status:
            try:
                OperationStatus(status)
            except ValueError:
                raise ValueError(
                    f"Invalid status: {status}. Valid values are: {[st.value for st in OperationStatus]}"
                ) from None

        if operation_type:
            try:
                OperationType(operation_type)
            except ValueError:
                raise ValueError(
                    f"Invalid operation type: {operation_type}. Valid values are: {[t.value for t in OperationType]}"
                ) from None

        # Get all operations
        operations, total = await self.list_operations(
            collection_id=collection_id,
            user_id=user_id,
            offset=offset,
            limit=limit,
        )

        # Apply filters if specified
        if status or operation_type:
            filtered_operations = operations

            if status:
                status_enum = OperationStatus(status)
                filtered_operations = [op for op in filtered_operations if op.status == status_enum]

            if operation_type:
                type_enum = OperationType(operation_type)
                filtered_operations = [op for op in filtered_operations if op.type == type_enum]

            return filtered_operations, len(filtered_operations)

        return operations, total

    async def list_documents_filtered(
        self,
        collection_id: str,
        user_id: int,
        status: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Any], int]:
        """List documents in a collection with filtering.

        This method contains the filtering logic that was previously
        in the router, ensuring proper separation of concerns.

        Args:
            collection_id: Collection UUID
            user_id: User ID for permission check
            status: Optional status filter
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (filtered documents, total count)

        Raises:
            ValueError: If invalid filter values provided
        """
        from shared.database.models import DocumentStatus

        # Validate status filter first
        if status:
            try:
                DocumentStatus(status)
            except ValueError:
                raise ValueError(
                    f"Invalid status: {status}. Valid values are: {[st.value for st in DocumentStatus]}"
                ) from None

        # Get all documents
        documents, total = await self.list_documents(
            collection_id=collection_id,
            user_id=user_id,
            offset=offset,
            limit=limit,
        )

        # Apply filter if specified
        if status:
            status_enum = DocumentStatus(status)
            documents = [doc for doc in documents if doc.status == status_enum]
            total = len(documents)  # Update total after filtering

        return documents, total

    async def create_operation(
        self,
        collection_id: str,
        operation_type: str,
        config: dict[str, Any],
        user_id: int,
    ) -> dict[str, Any]:
        """Create a new operation for a collection.

        Args:
            collection_id: Collection UUID
            operation_type: Type of operation
            config: Operation configuration
            user_id: User ID

        Returns:
            Created operation data
        """
        # Get collection
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id,
            user_id=user_id,
        )

        if not collection:
            raise EntityNotFoundError("Collection", collection_id)

        # Map operation type string to enum
        operation_type_enum = {
            "chunking": OperationType.INDEX,  # Initial chunking uses INDEX type
            "rechunking": OperationType.REINDEX,  # Re-chunking uses REINDEX type
            "index": OperationType.INDEX,
            "reindex": OperationType.REINDEX,
        }.get(operation_type, OperationType.INDEX)

        # Create operation
        operation = await self.operation_repo.create(
            collection_id=collection.id,
            user_id=user_id,
            operation_type=operation_type_enum,
            config=config,
            meta={"operation_type": operation_type},  # Store original operation type in meta
        )

        await self.db_session.commit()

        return {
            "uuid": operation.uuid,
            "collection_id": collection_id,
            "type": operation.type.value,
            "status": operation.status.value,
            "meta": operation.meta,
            "created_at": operation.created_at.isoformat() if operation.created_at else None,
        }

    async def update_collection(
        self,
        collection_id: str,
        updates: dict[str, Any],
        user_id: int,
    ) -> dict[str, Any]:
        """Update collection settings (alias for update that returns dict).

        This method exists for backward compatibility with tests.

        Args:
            collection_id: Collection UUID
            updates: Fields to update
            user_id: User ID

        Returns:
            Updated collection as dictionary
        """
        # Use the main update method
        updated_collection = await self.update(
            collection_id=collection_id,
            user_id=user_id,
            updates=updates,
        )
        # Build a stable dictionary representation (avoid repo.to_dict dependency)
        return self._collection_to_dict(updated_collection)

    def _collection_to_dict(self, collection: Collection) -> dict[str, Any]:
        """Serialize a Collection ORM object into a response dictionary."""
        status_value = collection.status.value if hasattr(collection.status, "value") else collection.status
        base = {
            "id": collection.id,
            "name": collection.name,
            "description": getattr(collection, "description", None),
            "owner_id": getattr(collection, "owner_id", None),
            "vector_store_name": getattr(collection, "vector_store_name", None),
            "embedding_model": getattr(collection, "embedding_model", None),
            "quantization": getattr(collection, "quantization", None),
            "chunk_size": getattr(collection, "chunk_size", None),
            "chunk_overlap": getattr(collection, "chunk_overlap", None),
            "chunking_strategy": getattr(collection, "chunking_strategy", None),
            "chunking_config": getattr(collection, "chunking_config", None),
            "is_public": getattr(collection, "is_public", None),
            "metadata": getattr(collection, "meta", None),
            "created_at": getattr(collection, "created_at", None),
            "updated_at": getattr(collection, "updated_at", None),
            "document_count": getattr(collection, "document_count", 0),
            "vector_count": getattr(collection, "vector_count", 0),
            "status": status_value,
            "status_message": getattr(collection, "status_message", None),
        }
        base["config"] = {
            "embedding_model": base["embedding_model"],
            "quantization": base["quantization"],
            "chunk_size": base["chunk_size"],
            "chunk_overlap": base["chunk_overlap"],
            "chunking_strategy": base["chunking_strategy"],
            "chunking_config": base["chunking_config"],
            "is_public": base["is_public"],
            "metadata": base["metadata"],
        }
        return base

    def _ensure_sync_run_repo(self) -> CollectionSyncRunRepository:
        """Ensure sync run repository is available."""
        if self.sync_run_repo is None:
            self.sync_run_repo = CollectionSyncRunRepository(self.db_session)
        return self.sync_run_repo

    # =========================================================================
    # Collection Sync Methods
    # =========================================================================

    async def run_collection_sync(
        self,
        collection_id: str,
        user_id: int,
        triggered_by: str = "manual",
    ) -> CollectionSyncRun:
        """Trigger a sync run for all sources in a collection.

        Fans out APPEND operations for each source and creates a sync run record
        to track completion aggregation.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user triggering the sync
            triggered_by: How the sync was triggered ('manual' or 'scheduler')

        Returns:
            Created CollectionSyncRun instance

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
            InvalidStateError: If collection is not in valid state or has active operations
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Validate collection state - must be READY or DEGRADED
        if collection.status not in [CollectionStatus.READY, CollectionStatus.DEGRADED]:
            raise InvalidStateError(
                f"Cannot sync collection in {collection.status} state. "
                f"Collection must be in READY or DEGRADED state."
            )

        # Check for active operations (collection-level gating)
        active_operations = await self.operation_repo.get_active_operations(collection.id)
        if active_operations:
            raise InvalidStateError(
                "Cannot start sync while another operation is in progress. "
                "Please wait for all current operations to complete."
            )

        # Get all sources for this collection
        sources, total = await self.collection_source_repo.list_by_collection(
            collection_id=collection.id,
            offset=0,
            limit=1000,  # Assume reasonable upper bound
        )

        if not sources:
            raise InvalidStateError("Cannot sync collection with no sources.")

        # Create sync run record
        sync_run_repo = self._ensure_sync_run_repo()
        sync_run = await sync_run_repo.create(
            collection_id=collection.id,
            triggered_by=triggered_by,
            expected_sources=len(sources),
        )

        # Update collection sync status
        now = datetime.now(UTC)
        await self.collection_repo.update_sync_status(
            collection_uuid=collection.id,
            status="running",
            started_at=now,
        )

        # Create APPEND operations for each source
        operations = []
        for source in sources:
            operation = await self.operation_repo.create(
                collection_id=collection.id,
                user_id=user_id,
                operation_type=OperationType.APPEND,
                config={
                    "source_id": source.id,
                    "source_type": source.source_type,
                    "source_path": source.source_path,
                    "source_config": source.source_config or {},
                    "sync_run_id": sync_run.id,  # CRITICAL: Link to sync run for aggregation
                    "triggered_by": triggered_by,
                },
            )
            operations.append(operation)

        # Update collection status to PROCESSING
        await self.collection_repo.update_status(collection.id, CollectionStatus.PROCESSING)

        # Calculate and set next sync run if continuous mode
        if collection.sync_mode == "continuous" and collection.sync_interval_minutes:
            next_run = now + timedelta(minutes=collection.sync_interval_minutes)
            await self.collection_repo.set_next_sync_run(collection.id, next_run)

        # Commit the transaction BEFORE dispatching tasks
        await self.db_session.commit()

        # Dispatch Celery tasks AFTER commit to avoid race condition
        for operation in operations:
            celery_app.send_task(
                "webui.tasks.process_collection_operation",
                args=[operation.uuid],
                task_id=str(uuid.uuid4()),
            )

        logger.info(
            f"Started sync run {sync_run.id} for collection {collection_id} "
            f"with {len(sources)} sources (triggered_by={triggered_by})"
        )

        return sync_run

    async def pause_collection_sync(
        self,
        collection_id: str,
        user_id: int,
    ) -> Collection:
        """Pause continuous sync for a collection.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user

        Returns:
            Updated Collection instance

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
            ValidationError: If collection is not in continuous sync mode
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Validate - must be continuous mode
        if collection.sync_mode != "continuous":
            raise ValidationError(
                "Cannot pause sync - collection is not in continuous sync mode",
                "sync_mode",
            )

        # Pause via repository
        updated_collection = await self.collection_repo.pause_sync(collection.id)

        await self.db_session.commit()

        logger.info(f"Paused sync for collection {collection_id}")
        return cast(Collection, updated_collection)

    async def resume_collection_sync(
        self,
        collection_id: str,
        user_id: int,
    ) -> Collection:
        """Resume continuous sync for a collection.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user

        Returns:
            Updated Collection instance

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
            ValidationError: If collection is not paused or not in continuous mode
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Validate - must be continuous mode and paused
        if collection.sync_mode != "continuous":
            raise ValidationError(
                "Cannot resume sync - collection is not in continuous sync mode",
                "sync_mode",
            )

        if collection.sync_paused_at is None:
            raise ValidationError(
                "Cannot resume sync - collection is not paused",
                "sync_paused_at",
            )

        # Resume via repository
        updated_collection = await self.collection_repo.resume_sync(collection.id)

        await self.db_session.commit()

        logger.info(f"Resumed sync for collection {collection_id}")
        return cast(Collection, updated_collection)

    async def list_collection_sync_runs(
        self,
        collection_id: str,
        user_id: int,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[CollectionSyncRun], int]:
        """List sync runs for a collection.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (sync runs list, total count)

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # List sync runs
        sync_run_repo = self._ensure_sync_run_repo()
        sync_runs, total = await sync_run_repo.list_for_collection(
            collection_id=collection.id,
            offset=offset,
            limit=limit,
        )

        return sync_runs, total
