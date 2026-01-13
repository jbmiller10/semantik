"""Collection Service for managing collection operations."""

import asyncio
import logging
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import select
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
from webui.services.connector_factory import ConnectorFactory

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
            logger.warning("Qdrant manager unavailable: %s", exc, exc_info=True)
            return None

        return self.qdrant_manager

    async def _delete_sparse_collection_if_exists(self, vector_store_name: str) -> None:
        """Delete sparse collection and config if they exist.

        This is called during collection deletion to clean up any sparse index.

        Args:
            vector_store_name: Name of the main dense collection
        """
        from qdrant_client import AsyncQdrantClient

        from shared.config import settings
        from shared.database.collection_metadata import delete_sparse_index_config, get_sparse_index_config
        from vecpipe.sparse import delete_sparse_collection

        async_qdrant = AsyncQdrantClient(
            url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
            api_key=settings.QDRANT_API_KEY,
        )

        try:
            # Check if sparse index config exists
            sparse_config = await get_sparse_index_config(async_qdrant, vector_store_name)

            if sparse_config and sparse_config.get("enabled"):
                sparse_collection_name = sparse_config.get("sparse_collection_name")
                if sparse_collection_name:
                    try:
                        await delete_sparse_collection(sparse_collection_name, async_qdrant)
                        logger.info("Deleted sparse collection: %s", sparse_collection_name)
                    except Exception as e:
                        logger.warning(
                            "Failed to delete sparse collection %s: %s",
                            sparse_collection_name,
                            e,
                        )

                # Delete sparse config from metadata
                await delete_sparse_index_config(async_qdrant, vector_store_name)
                logger.info("Deleted sparse index config for: %s", vector_store_name)
        except Exception as e:
            logger.warning(
                "Failed to cleanup sparse index for %s: %s",
                vector_store_name,
                e,
            )
        finally:
            await async_qdrant.close()

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
            sync_mode = (config.get("sync_mode") if config else None) or "one_time"
            sync_interval_minutes = config.get("sync_interval_minutes") if config else None
            sync_next_run_at = None

            if sync_mode not in {"one_time", "continuous"}:
                raise ValueError("sync_mode must be one_time or continuous")

            if sync_mode == "continuous":
                if sync_interval_minutes is None:
                    raise ValueError("sync_interval_minutes is required for continuous sync mode")
                if sync_interval_minutes < 15:
                    raise ValueError("sync_interval_minutes must be at least 15 minutes for continuous sync mode")
                sync_next_run_at = datetime.now(UTC) + timedelta(minutes=sync_interval_minutes)
            else:
                sync_interval_minutes = None

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

            # Validate sparse indexing config if provided
            sparse_index_config = config.get("sparse_index_config") if config else None
            if sparse_index_config and sparse_index_config.get("enabled"):
                from shared.plugins import load_plugins, plugin_registry

                load_plugins(plugin_types={"sparse_indexer"})
                plugin_id = sparse_index_config.get("plugin_id")
                plugin_record = plugin_registry.find_by_id(plugin_id)
                if plugin_record is None or plugin_record.plugin_type != "sparse_indexer":
                    raise ValueError(f"Sparse indexer plugin '{plugin_id}' not found")

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
                sync_mode=sync_mode,
                sync_interval_minutes=sync_interval_minutes,
                sync_next_run_at=sync_next_run_at,
            )
        except EntityAlreadyExistsError:
            # Re-raise EntityAlreadyExistsError to be handled by the API endpoint
            raise
        except Exception as e:
            logger.error("Failed to create collection: %s", e, exc_info=True)
            raise

        # Create operation record
        operation_config: dict[str, Any] = {
            "sources": [],  # Initial creation has no sources
            "collection_config": config or {},
        }
        # Include sparse_index_config at top level for easier access in INDEX task
        if sparse_index_config and sparse_index_config.get("enabled"):
            operation_config["sparse_index_config"] = sparse_index_config

        operation = await self.operation_repo.create(
            collection_id=collection.id,
            user_id=user_id,
            operation_type=OperationType.INDEX,
            config=operation_config,
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
        source_type = (source_type or "").strip().lower()

        # Normalize: derive/augment source_config from legacy_source_path if needed
        if legacy_source_path is not None:
            if not source_config:
                source_config = {"path": legacy_source_path}
                source_type = "directory"
            elif source_type == "directory" and "path" not in source_config:
                # Some clients send directory options (e.g., recursive) in source_config
                # while still providing the actual path separately. Merge the path in.
                source_config = {**source_config, "path": legacy_source_path}

        available_types = ConnectorFactory.list_available_types()
        if not source_type or source_type not in available_types:
            available = ", ".join(sorted(available_types)) if available_types else "none registered"
            raise ValidationError(
                f"Invalid source_type: '{source_type}'. Must be one of: {available}",
                "source_type",
            )

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

        # Lock the collection row to prevent race conditions between checking
        # for active operations and creating a new one (TOCTOU protection)
        await self.db_session.execute(select(Collection).where(Collection.id == collection.id).with_for_update())

        # Check if there's already an active operation (now safe due to row lock)
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
                        logger.info("Deleted Qdrant collection: %s", collection.vector_store_name)
                except Exception as e:
                    logger.error("Failed to delete Qdrant collection: %s", e, exc_info=True)
                    # Continue with database deletion even if Qdrant deletion fails

                # Cascade delete: Also delete sparse collection if it exists
                await self._delete_sparse_collection_if_exists(collection.vector_store_name)

            # Delete from database (cascade will handle operations, documents, etc.)
            await self.collection_repo.delete(collection_id, user_id)

            # Commit the transaction to persist the deletion
            await self.db_session.commit()

            logger.info("Deleted collection %s and all associated data", collection_id)

        except Exception as e:
            logger.error("Failed to delete collection %s: %s", collection_id, e, exc_info=True)
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

        # Handle sync policy updates
        current_sync_mode = getattr(collection, "sync_mode", "one_time") or "one_time"
        current_sync_interval_minutes = getattr(collection, "sync_interval_minutes", None)

        if "sync_mode" in updates or "sync_interval_minutes" in updates:
            requested_sync_mode = updates.get("sync_mode", current_sync_mode) or current_sync_mode
            requested_sync_mode = str(requested_sync_mode)
            requested_sync_interval_minutes = updates.get("sync_interval_minutes", current_sync_interval_minutes)

            mode_changed = "sync_mode" in updates and requested_sync_mode != current_sync_mode
            interval_changed = (
                "sync_interval_minutes" in updates and requested_sync_interval_minutes != current_sync_interval_minutes
            )

            if requested_sync_mode not in {"one_time", "continuous"}:
                raise ValidationError("sync_mode must be one_time or continuous", "sync_mode")

            if requested_sync_mode == "one_time":
                updates["sync_mode"] = "one_time"
                updates["sync_interval_minutes"] = None
                updates["sync_paused_at"] = None
                updates["sync_next_run_at"] = None
            else:
                if requested_sync_interval_minutes is None:
                    raise ValidationError(
                        "sync_interval_minutes is required for continuous sync mode",
                        "sync_interval_minutes",
                    )
                if requested_sync_interval_minutes < 15:
                    raise ValidationError(
                        "sync_interval_minutes must be at least 15 minutes for continuous sync mode",
                        "sync_interval_minutes",
                    )

                updates["sync_mode"] = "continuous"
                updates["sync_interval_minutes"] = requested_sync_interval_minutes

                paused_at = updates.get("sync_paused_at", getattr(collection, "sync_paused_at", None))
                if mode_changed and current_sync_mode != "continuous":
                    updates["sync_paused_at"] = None
                    paused_at = None
                if paused_at is None and (mode_changed or interval_changed):
                    updates["sync_next_run_at"] = datetime.now(UTC) + timedelta(minutes=requested_sync_interval_minutes)

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

        original_values: dict[str, Any] | None = None
        if requires_qdrant_sync:
            original_values = {key: getattr(collection, key, None) for key in updates}

        try:
            updated_collection = await self.collection_repo.update(str(collection.id), updates)
            await self.db_session.commit()
        except Exception as exc:  # pragma: no cover - covered via explicit tests
            await self.db_session.rollback()
            raise exc

        if requires_qdrant_sync and new_vector_store_name and old_vector_store_name:
            try:
                assert qdrant_manager_for_rename is not None  # mypy assurance
                await qdrant_manager_for_rename.rename_collection(
                    old_name=old_vector_store_name,
                    new_name=new_vector_store_name,
                )
            except Exception as qdrant_exc:  # pragma: no cover - covered via explicit tests
                logger.error(
                    "Qdrant rename failed for collection %s (old=%s new=%s): %s",
                    collection_id,
                    old_vector_store_name,
                    new_vector_store_name,
                    qdrant_exc,
                    exc_info=True,
                )
                if original_values is not None:
                    try:
                        await self.collection_repo.update(str(collection.id), original_values)
                        await self.db_session.commit()
                        logger.info("Successfully reverted collection %s after Qdrant rename failure", collection_id)
                    except Exception as revert_exc:  # pragma: no cover - defensive logging
                        logger.critical(
                            "CRITICAL: Failed to revert DB after Qdrant rename failure. "
                            "Collection ID: %s, DB has: %s, Qdrant has: %s, Revert error: %s",
                            collection_id,
                            new_vector_store_name,
                            old_vector_store_name,
                            revert_exc,
                            exc_info=True,
                        )
                raise qdrant_exc

        return cast(Collection, updated_collection)

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
            "total_size_bytes": getattr(collection, "total_size_bytes", 0) or 0,
            "status": status_value,
            "status_message": getattr(collection, "status_message", None),
            # Sync policy fields
            "sync_mode": getattr(collection, "sync_mode", "one_time") or "one_time",
            "sync_interval_minutes": getattr(collection, "sync_interval_minutes", None),
            "sync_paused_at": getattr(collection, "sync_paused_at", None),
            "sync_next_run_at": getattr(collection, "sync_next_run_at", None),
            # Sync run tracking
            "sync_last_run_started_at": getattr(collection, "sync_last_run_started_at", None),
            "sync_last_run_completed_at": getattr(collection, "sync_last_run_completed_at", None),
            "sync_last_run_status": getattr(collection, "sync_last_run_status", None),
            "sync_last_error": getattr(collection, "sync_last_error", None),
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

    # =========================================================================
    # Sparse Index Management Methods (Phase 3)
    # =========================================================================

    async def get_sparse_index_config(
        self,
        collection_id: str,
        user_id: int,
    ) -> dict[str, Any] | None:
        """Get sparse index configuration for a collection.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user

        Returns:
            Sparse index config dict or None if not enabled

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
        """
        from qdrant_client import AsyncQdrantClient

        from shared.config import settings
        from shared.database.collection_metadata import get_sparse_index_config

        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Get sparse config from collection metadata
        async_qdrant = AsyncQdrantClient(
            url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
            api_key=settings.QDRANT_API_KEY,
        )
        try:
            result = await get_sparse_index_config(async_qdrant, collection.vector_store_name)
            return cast(dict[str, Any] | None, result)
        finally:
            await async_qdrant.close()

    async def enable_sparse_index(
        self,
        collection_id: str,
        user_id: int,
        plugin_id: str,
        model_config: dict[str, Any] | None = None,
        reindex_existing: bool = False,
    ) -> dict[str, Any]:
        """Enable sparse indexing for a collection.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user
            plugin_id: Sparse indexer plugin ID
            model_config: Plugin-specific configuration
            reindex_existing: Whether to reindex existing documents

        Returns:
            Sparse index config dict

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
            InvalidStateError: If sparse indexing already enabled
            ValidationError: If plugin not found
        """
        from qdrant_client import AsyncQdrantClient

        from shared.config import settings
        from shared.database.collection_metadata import get_sparse_index_config, store_sparse_index_config
        from shared.plugins import load_plugins, plugin_registry
        from vecpipe.sparse import ensure_sparse_collection, generate_sparse_collection_name

        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        async_qdrant = AsyncQdrantClient(
            url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
            api_key=settings.QDRANT_API_KEY,
        )

        try:
            # Check if sparse indexing is already enabled
            existing_config = await get_sparse_index_config(async_qdrant, collection.vector_store_name)
            if existing_config and existing_config.get("enabled"):
                raise InvalidStateError(f"Sparse indexing already enabled for collection '{collection_id}'")

            # Ensure sparse indexer plugins are loaded before validation.
            load_plugins(plugin_types={"sparse_indexer"})

            # Validate plugin exists
            plugin_record = plugin_registry.find_by_id(plugin_id)
            if plugin_record is None or plugin_record.plugin_type != "sparse_indexer":
                raise ValidationError(f"Sparse indexer plugin '{plugin_id}' not found", "plugin_id")

            # Get sparse type from plugin (e.g., "bm25" or "splade")
            plugin_cls = plugin_record.plugin_class
            sparse_type = getattr(plugin_cls, "SPARSE_TYPE", "bm25")

            # Generate sparse collection name
            sparse_collection_name = generate_sparse_collection_name(collection.vector_store_name, sparse_type)

            # Create sparse Qdrant collection
            await ensure_sparse_collection(sparse_collection_name, async_qdrant)

            # Build sparse config
            now = datetime.now(UTC).isoformat()
            sparse_config = {
                "enabled": True,
                "plugin_id": plugin_id,
                "sparse_collection_name": sparse_collection_name,
                "model_config": model_config or {},
                "created_at": now,
                "document_count": 0,
                "last_indexed_at": None,
            }

            # Store sparse config in collection metadata
            await store_sparse_index_config(async_qdrant, collection.vector_store_name, sparse_config)

            # Optionally trigger reindex
            if reindex_existing:
                job = celery_app.send_task(
                    "sparse.reindex_collection",
                    args=[collection_id, plugin_id, model_config or {}],
                )
                logger.info(f"Triggered sparse reindex job {job.id} for collection {collection_id}")

            return sparse_config
        finally:
            await async_qdrant.close()

    async def disable_sparse_index(
        self,
        collection_id: str,
        user_id: int,
    ) -> None:
        """Disable sparse indexing for a collection.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
        """
        from qdrant_client import AsyncQdrantClient

        from shared.config import settings
        from shared.database.collection_metadata import delete_sparse_index_config, get_sparse_index_config
        from vecpipe.sparse import delete_sparse_collection

        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        async_qdrant = AsyncQdrantClient(
            url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
            api_key=settings.QDRANT_API_KEY,
        )

        try:
            # Get current sparse config
            sparse_config = await get_sparse_index_config(async_qdrant, collection.vector_store_name)

            if sparse_config:
                # Delete the sparse Qdrant collection
                sparse_collection_name = sparse_config.get("sparse_collection_name")
                if sparse_collection_name:
                    await delete_sparse_collection(sparse_collection_name, async_qdrant)
                    logger.info(f"Deleted sparse collection '{sparse_collection_name}'")

                # Remove sparse config from metadata
                await delete_sparse_index_config(async_qdrant, collection.vector_store_name)
                logger.info(f"Removed sparse index config for collection {collection_id}")
        finally:
            await async_qdrant.close()

    async def trigger_sparse_reindex(
        self,
        collection_id: str,
        user_id: int,
    ) -> dict[str, Any]:
        """Trigger a full sparse reindex of the collection.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user

        Returns:
            Dict with job_id, status, and plugin_id

        Raises:
            EntityNotFoundError: If collection or sparse config not found
            AccessDeniedError: If user doesn't have permission
            InvalidStateError: If sparse indexing not enabled
        """
        from qdrant_client import AsyncQdrantClient

        from shared.config import settings
        from shared.database.collection_metadata import get_sparse_index_config

        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        async_qdrant = AsyncQdrantClient(
            url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
            api_key=settings.QDRANT_API_KEY,
        )

        try:
            # Get sparse config
            sparse_config = await get_sparse_index_config(async_qdrant, collection.vector_store_name)

            if not sparse_config or not sparse_config.get("enabled"):
                raise EntityNotFoundError(
                    f"Sparse indexing not enabled for collection '{collection_id}'",
                    entity_id=collection_id,
                )

            plugin_id = sparse_config["plugin_id"]
            model_config = sparse_config.get("model_config", {})

            # Dispatch Celery task
            job = celery_app.send_task(
                "sparse.reindex_collection",
                args=[collection_id, plugin_id, model_config],
            )

            logger.info(f"Triggered sparse reindex job {job.id} for collection {collection_id}")

            return {
                "job_id": job.id,
                "status": "queued",
                "plugin_id": plugin_id,
            }
        finally:
            await async_qdrant.close()

    async def get_sparse_reindex_progress(
        self,
        collection_id: str,
        user_id: int,
        job_id: str,
    ) -> dict[str, Any]:
        """Get progress of a sparse reindex job.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user
            job_id: Celery task ID

        Returns:
            Dict with status and progress info

        Raises:
            EntityNotFoundError: If collection or job not found
            AccessDeniedError: If user doesn't have permission
        """
        from celery.result import AsyncResult

        # Get collection with permission check (just for access control)
        await self.collection_repo.get_by_uuid_with_permission_check(collection_uuid=collection_id, user_id=user_id)

        # Get Celery task result
        result = AsyncResult(job_id, app=celery_app)

        progress_info: dict[str, Any] = {
            "status": result.state,
        }

        if result.state == "PROGRESS":
            info = result.info or {}
            progress_info.update(
                {
                    "progress": info.get("progress", 0),
                    "documents_processed": info.get("documents_processed"),
                    "total_documents": info.get("total_documents"),
                }
            )
        elif result.state == "FAILURE":
            progress_info["error"] = str(result.result) if result.result else "Unknown error"
        elif result.state == "SUCCESS":
            info = result.result or {}
            progress_info.update(
                {
                    "progress": 100.0,
                    "documents_processed": info.get("documents_processed"),
                    "total_documents": info.get("total_documents"),
                }
            )

        return progress_info
