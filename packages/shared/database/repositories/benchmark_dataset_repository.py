"""Repository implementation for BenchmarkDataset and related models."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import Select, delete, func, select
from sqlalchemy.orm import selectinload

from shared.database.exceptions import (
    AccessDeniedError,
    DatabaseOperationError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    ValidationError,
)
from shared.database.models import (
    BenchmarkDataset,
    BenchmarkDatasetMapping,
    BenchmarkQuery,
    BenchmarkRelevance,
    Collection,
    MappingStatus,
    User,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class BenchmarkDatasetRepository:
    """Repository for BenchmarkDataset and related models."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session."""
        self.session = session

    # =========================================================================
    # Dataset CRUD
    # =========================================================================

    async def create(
        self,
        *,
        name: str,
        owner_id: int,
        query_count: int = 0,
        description: str | None = None,
        raw_file_path: str | None = None,
        schema_version: str = "1.0",
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkDataset:
        """Create a new benchmark dataset.

        Args:
            name: Dataset name
            owner_id: ID of the user creating the dataset
            query_count: Number of queries in the dataset
            description: Optional description
            raw_file_path: Path to the original dataset file
            schema_version: Version of the dataset schema
            metadata: Optional metadata

        Returns:
            Created BenchmarkDataset instance

        Raises:
            ValidationError: If validation fails
            DatabaseOperationError: For database errors
        """
        if not name or not name.strip():
            raise ValidationError("Dataset name is required", "name")

        # Verify owner exists
        user_exists = await self.session.scalar(select(func.count()).select_from(User).where(User.id == owner_id))
        if not user_exists:
            raise EntityNotFoundError("user", str(owner_id))

        dataset = BenchmarkDataset(
            id=str(uuid4()),
            name=name.strip(),
            description=description,
            owner_id=owner_id,
            query_count=query_count,
            raw_file_path=raw_file_path,
            schema_version=schema_version,
            meta=metadata or {},
        )

        try:
            self.session.add(dataset)
            await self.session.flush()
            logger.info("Created benchmark dataset %s for user %d", dataset.id, owner_id)
            return dataset
        except Exception as exc:
            logger.error("Failed to create benchmark dataset: %s", exc)
            raise DatabaseOperationError("create", "benchmark_dataset", str(exc)) from exc

    async def get_by_uuid(self, dataset_uuid: str) -> BenchmarkDataset | None:
        """Get a dataset by UUID.

        Args:
            dataset_uuid: UUID of the dataset

        Returns:
            BenchmarkDataset instance or None if not found
        """
        stmt: Select[tuple[BenchmarkDataset]] = (
            select(BenchmarkDataset)
            .where(BenchmarkDataset.id == dataset_uuid)
            .options(
                selectinload(BenchmarkDataset.owner),
                selectinload(BenchmarkDataset.mappings),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_uuid_for_user(
        self,
        dataset_uuid: str,
        user_id: int,
    ) -> BenchmarkDataset:
        """Get a dataset by UUID with ownership check.

        Args:
            dataset_uuid: UUID of the dataset
            user_id: ID of the user requesting access

        Returns:
            BenchmarkDataset instance

        Raises:
            EntityNotFoundError: If dataset not found
            AccessDeniedError: If user doesn't own the dataset
        """
        dataset = await self.get_by_uuid(dataset_uuid)

        if not dataset:
            raise EntityNotFoundError("benchmark_dataset", dataset_uuid)

        if dataset.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "benchmark_dataset", dataset_uuid)

        return dataset

    async def list_for_user(
        self,
        user_id: int,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[BenchmarkDataset], int]:
        """List datasets owned by a user.

        Args:
            user_id: ID of the user
            offset: Pagination offset
            limit: Maximum number of results

        Returns:
            Tuple of (datasets list, total count)
        """
        try:
            # Build query
            stmt: Select[tuple[BenchmarkDataset]] = (
                select(BenchmarkDataset)
                .where(BenchmarkDataset.owner_id == user_id)
                .options(selectinload(BenchmarkDataset.mappings))
                .order_by(BenchmarkDataset.created_at.desc())
                .offset(offset)
                .limit(limit)
            )

            datasets = list((await self.session.execute(stmt)).scalars().all())

            # Get total count
            count_stmt = select(func.count(BenchmarkDataset.id)).where(BenchmarkDataset.owner_id == user_id)
            total = await self.session.scalar(count_stmt) or 0

            return datasets, total
        except Exception as exc:
            logger.error("Failed to list benchmark datasets: %s", exc)
            raise DatabaseOperationError("list", "benchmark_datasets", str(exc)) from exc

    async def update(
        self,
        dataset_uuid: str,
        *,
        name: str | None = None,
        description: str | None = None,
        query_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkDataset:
        """Update a dataset.

        Args:
            dataset_uuid: UUID of the dataset to update
            name: New name (optional)
            description: New description (optional)
            query_count: New query count (optional)
            metadata: New metadata (merged with existing)

        Returns:
            Updated BenchmarkDataset instance

        Raises:
            EntityNotFoundError: If dataset not found
        """
        dataset = await self.get_by_uuid(dataset_uuid)
        if not dataset:
            raise EntityNotFoundError("benchmark_dataset", dataset_uuid)

        if name is not None:
            if not name.strip():
                raise ValidationError("Dataset name cannot be empty", "name")
            dataset.name = name.strip()

        if description is not None:
            dataset.description = description

        if query_count is not None:
            dataset.query_count = query_count

        if metadata is not None:
            merged = dict(dataset.meta or {})
            merged.update(metadata)
            dataset.meta = merged

        dataset.updated_at = datetime.now(UTC)
        await self.session.flush()
        return dataset

    async def delete(
        self,
        dataset_uuid: str,
        user_id: int,
    ) -> None:
        """Delete a dataset.

        Args:
            dataset_uuid: UUID of the dataset to delete
            user_id: ID of the user requesting deletion

        Raises:
            EntityNotFoundError: If dataset not found
            AccessDeniedError: If user doesn't own the dataset
        """
        dataset = await self.get_by_uuid_for_user(dataset_uuid, user_id)

        stmt = delete(BenchmarkDataset).where(BenchmarkDataset.id == dataset.id)
        await self.session.execute(stmt)
        await self.session.flush()
        logger.info("Deleted benchmark dataset %s", dataset_uuid)

    # =========================================================================
    # Mapping CRUD
    # =========================================================================

    async def create_mapping(
        self,
        dataset_id: str,
        collection_id: str,
    ) -> BenchmarkDatasetMapping:
        """Create a mapping between a dataset and a collection.

        Args:
            dataset_id: UUID of the dataset
            collection_id: UUID of the collection

        Returns:
            Created BenchmarkDatasetMapping instance

        Raises:
            EntityNotFoundError: If dataset or collection not found
            EntityAlreadyExistsError: If mapping already exists
        """
        # Verify dataset exists
        dataset = await self.get_by_uuid(dataset_id)
        if not dataset:
            raise EntityNotFoundError("benchmark_dataset", dataset_id)

        # Verify collection exists
        collection_exists = await self.session.scalar(
            select(func.count()).select_from(Collection).where(Collection.id == collection_id)
        )
        if not collection_exists:
            raise EntityNotFoundError("collection", collection_id)

        # Check for existing mapping
        existing = await self.session.scalar(
            select(func.count())
            .select_from(BenchmarkDatasetMapping)
            .where(
                BenchmarkDatasetMapping.dataset_id == dataset_id,
                BenchmarkDatasetMapping.collection_id == collection_id,
            )
        )
        if existing:
            raise EntityAlreadyExistsError(
                "benchmark_dataset_mapping",
                f"dataset={dataset_id}, collection={collection_id}",
            )

        mapping = BenchmarkDatasetMapping(
            dataset_id=dataset_id,
            collection_id=collection_id,
            mapping_status=MappingStatus.PENDING.value,
            mapped_count=0,
            total_count=0,
        )

        try:
            self.session.add(mapping)
            await self.session.flush()
            logger.info(
                "Created mapping %d: dataset %s -> collection %s",
                mapping.id,
                dataset_id,
                collection_id,
            )
            return mapping
        except Exception as exc:
            logger.error("Failed to create dataset mapping: %s", exc)
            raise DatabaseOperationError("create", "benchmark_dataset_mapping", str(exc)) from exc

    async def get_mapping(self, mapping_id: int) -> BenchmarkDatasetMapping | None:
        """Get a mapping by ID.

        Args:
            mapping_id: ID of the mapping

        Returns:
            BenchmarkDatasetMapping instance or None
        """
        stmt: Select[tuple[BenchmarkDatasetMapping]] = (
            select(BenchmarkDatasetMapping)
            .where(BenchmarkDatasetMapping.id == mapping_id)
            .options(
                selectinload(BenchmarkDatasetMapping.dataset),
                selectinload(BenchmarkDatasetMapping.collection),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_mappings_for_dataset(
        self,
        dataset_id: str,
    ) -> list[BenchmarkDatasetMapping]:
        """List all mappings for a dataset.

        Args:
            dataset_id: UUID of the dataset

        Returns:
            List of BenchmarkDatasetMapping instances
        """
        stmt: Select[tuple[BenchmarkDatasetMapping]] = (
            select(BenchmarkDatasetMapping)
            .where(BenchmarkDatasetMapping.dataset_id == dataset_id)
            .options(selectinload(BenchmarkDatasetMapping.collection))
            .order_by(BenchmarkDatasetMapping.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_mapping_status(
        self,
        mapping_id: int,
        status: MappingStatus,
        mapped_count: int | None = None,
        total_count: int | None = None,
    ) -> BenchmarkDatasetMapping:
        """Update a mapping's resolution status.

        Args:
            mapping_id: ID of the mapping
            status: New status
            mapped_count: Number of resolved document references
            total_count: Total number of document references

        Returns:
            Updated BenchmarkDatasetMapping instance

        Raises:
            EntityNotFoundError: If mapping not found
        """
        mapping = await self.get_mapping(mapping_id)
        if not mapping:
            raise EntityNotFoundError("benchmark_dataset_mapping", str(mapping_id))

        mapping.mapping_status = status.value

        if mapped_count is not None:
            mapping.mapped_count = mapped_count
        if total_count is not None:
            mapping.total_count = total_count

        if status in (MappingStatus.RESOLVED, MappingStatus.PARTIAL):
            mapping.resolved_at = datetime.now(UTC)

        await self.session.flush()
        return mapping

    # =========================================================================
    # Query CRUD
    # =========================================================================

    async def add_query(
        self,
        dataset_id: str,
        query_key: str,
        query_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkQuery:
        """Add a query to a dataset.

        Args:
            dataset_id: UUID of the dataset
            query_key: Unique key for the query within the dataset
            query_text: The query text
            metadata: Optional query metadata

        Returns:
            Created BenchmarkQuery instance

        Raises:
            EntityNotFoundError: If dataset not found
            ValidationError: If validation fails
        """
        if not query_key or not query_key.strip():
            raise ValidationError("Query key is required", "query_key")
        if not query_text or not query_text.strip():
            raise ValidationError("Query text is required", "query_text")

        # Verify dataset exists
        dataset = await self.get_by_uuid(dataset_id)
        if not dataset:
            raise EntityNotFoundError("benchmark_dataset", dataset_id)

        query = BenchmarkQuery(
            dataset_id=dataset_id,
            query_key=query_key.strip(),
            query_text=query_text.strip(),
            query_metadata=metadata,
        )

        try:
            self.session.add(query)
            await self.session.flush()
            return query
        except Exception as exc:
            logger.error("Failed to add query to dataset %s: %s", dataset_id, exc)
            raise DatabaseOperationError("create", "benchmark_query", str(exc)) from exc

    async def get_queries_for_dataset(
        self,
        dataset_id: str,
    ) -> list[BenchmarkQuery]:
        """Get all queries for a dataset.

        Args:
            dataset_id: UUID of the dataset

        Returns:
            List of BenchmarkQuery instances
        """
        stmt: Select[tuple[BenchmarkQuery]] = (
            select(BenchmarkQuery).where(BenchmarkQuery.dataset_id == dataset_id).order_by(BenchmarkQuery.id)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # =========================================================================
    # Relevance CRUD
    # =========================================================================

    async def add_relevance(
        self,
        query_id: int,
        mapping_id: int,
        doc_ref: dict[str, Any],
        relevance_grade: int,
        doc_ref_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkRelevance:
        """Add a relevance judgment.

        Args:
            query_id: ID of the benchmark query
            mapping_id: ID of the dataset-collection mapping
            doc_ref: Document reference (e.g., {"path": "...", "uri": "..."})
            relevance_grade: Relevance grade (0-3)
            doc_ref_hash: Hash of the document reference (computed if not provided)
            metadata: Optional metadata

        Returns:
            Created BenchmarkRelevance instance

        Raises:
            ValidationError: If validation fails
            EntityNotFoundError: If query or mapping not found
        """
        if relevance_grade < 0 or relevance_grade > 3:
            raise ValidationError(
                "Relevance grade must be between 0 and 3",
                "relevance_grade",
            )

        # Compute hash if not provided
        if doc_ref_hash is None:
            import hashlib
            import json

            doc_ref_hash = hashlib.sha256(json.dumps(doc_ref, sort_keys=True).encode()).hexdigest()

        relevance = BenchmarkRelevance(
            benchmark_query_id=query_id,
            mapping_id=mapping_id,
            doc_ref_hash=doc_ref_hash,
            doc_ref=doc_ref,
            relevance_grade=relevance_grade,
            relevance_metadata=metadata,
        )

        try:
            self.session.add(relevance)
            await self.session.flush()
            return relevance
        except Exception as exc:
            logger.error("Failed to add relevance judgment: %s", exc)
            raise DatabaseOperationError("create", "benchmark_relevance", str(exc)) from exc

    async def resolve_relevance(
        self,
        relevance_id: int,
        document_id: str,
    ) -> BenchmarkRelevance:
        """Resolve a relevance judgment to an actual document.

        Args:
            relevance_id: ID of the relevance judgment
            document_id: UUID of the resolved document

        Returns:
            Updated BenchmarkRelevance instance

        Raises:
            EntityNotFoundError: If relevance not found
        """
        stmt: Select[tuple[BenchmarkRelevance]] = select(BenchmarkRelevance).where(
            BenchmarkRelevance.id == relevance_id
        )
        result = await self.session.execute(stmt)
        relevance = result.scalar_one_or_none()

        if not relevance:
            raise EntityNotFoundError("benchmark_relevance", str(relevance_id))

        relevance.resolved_document_id = document_id
        await self.session.flush()
        return relevance

    async def get_relevance_for_mapping(
        self,
        mapping_id: int,
    ) -> list[BenchmarkRelevance]:
        """Get all relevance judgments for a mapping.

        Args:
            mapping_id: ID of the mapping

        Returns:
            List of BenchmarkRelevance instances
        """
        stmt: Select[tuple[BenchmarkRelevance]] = (
            select(BenchmarkRelevance)
            .where(BenchmarkRelevance.mapping_id == mapping_id)
            .options(selectinload(BenchmarkRelevance.query))
            .order_by(BenchmarkRelevance.benchmark_query_id)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_relevance_for_mapping(self, mapping_id: int) -> int:
        """Count total relevance judgments for a mapping."""
        try:
            total = await self.session.scalar(
                select(func.count(BenchmarkRelevance.id)).where(BenchmarkRelevance.mapping_id == mapping_id)
            )
            return int(total or 0)
        except Exception as exc:
            logger.error("Failed to count relevance for mapping %s: %s", mapping_id, exc, exc_info=True)
            raise DatabaseOperationError("count", "benchmark_relevance", str(exc)) from exc

    async def count_resolved_relevance_for_mapping(self, mapping_id: int) -> int:
        """Count resolved relevance judgments for a mapping."""
        try:
            total = await self.session.scalar(
                select(func.count(BenchmarkRelevance.id)).where(
                    BenchmarkRelevance.mapping_id == mapping_id,
                    BenchmarkRelevance.resolved_document_id.is_not(None),
                )
            )
            return int(total or 0)
        except Exception as exc:
            logger.error("Failed to count resolved relevance for mapping %s: %s", mapping_id, exc, exc_info=True)
            raise DatabaseOperationError("count", "benchmark_relevance", str(exc)) from exc

    async def list_unresolved_relevance_for_mapping(
        self,
        mapping_id: int,
        *,
        after_id: int = 0,
        limit: int = 1000,
    ) -> list[BenchmarkRelevance]:
        """List unresolved relevance judgments for a mapping in a stable order.

        This is intended for batch processing; it returns only rows with
        resolved_document_id IS NULL.
        """
        stmt: Select[tuple[BenchmarkRelevance]] = (
            select(BenchmarkRelevance)
            .where(
                BenchmarkRelevance.mapping_id == mapping_id,
                BenchmarkRelevance.resolved_document_id.is_(None),
                BenchmarkRelevance.id > after_id,
            )
            .order_by(BenchmarkRelevance.id)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_relevance_for_query(
        self,
        query_id: int,
        mapping_id: int,
    ) -> list[BenchmarkRelevance]:
        """Get relevance judgments for a specific query in a mapping.

        Args:
            query_id: ID of the query
            mapping_id: ID of the mapping

        Returns:
            List of BenchmarkRelevance instances for the query
        """
        stmt: Select[tuple[BenchmarkRelevance]] = (
            select(BenchmarkRelevance)
            .where(
                BenchmarkRelevance.benchmark_query_id == query_id,
                BenchmarkRelevance.mapping_id == mapping_id,
            )
            .order_by(BenchmarkRelevance.relevance_grade.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
