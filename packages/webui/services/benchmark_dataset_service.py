"""Benchmark Dataset Service for managing benchmark datasets and mappings."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, cast

from shared.database.exceptions import AccessDeniedError, EntityNotFoundError, ValidationError
from shared.database.models import BenchmarkDataset, BenchmarkDatasetMapping, MappingStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.repositories.benchmark_dataset_repository import BenchmarkDatasetRepository
    from shared.database.repositories.collection_repository import CollectionRepository
    from shared.database.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class BenchmarkDatasetService:
    """Service for managing benchmark datasets and mappings."""

    # Supported dataset file formats
    SUPPORTED_FORMATS = {"json"}

    def __init__(
        self,
        db_session: AsyncSession,
        benchmark_dataset_repo: BenchmarkDatasetRepository,
        collection_repo: CollectionRepository,
        document_repo: DocumentRepository,
    ):
        """Initialize the service.

        Args:
            db_session: AsyncSession for database operations
            benchmark_dataset_repo: Repository for benchmark dataset operations
            collection_repo: Repository for collection operations
            document_repo: Repository for document operations
        """
        self.db_session = db_session
        self.benchmark_dataset_repo = benchmark_dataset_repo
        self.collection_repo = collection_repo
        self.document_repo = document_repo

    async def upload_dataset(
        self,
        user_id: int,
        name: str,
        description: str | None,
        file_content: bytes,
    ) -> dict[str, Any]:
        """Upload and parse a benchmark dataset.

        Args:
            user_id: ID of the user uploading the dataset
            name: Name for the dataset
            description: Optional description
            file_content: Raw file content (JSON format)

        Returns:
            Dictionary with dataset summary including id, name, query_count

        Raises:
            ValidationError: If file format is invalid or parsing fails
        """
        # Parse the file content
        try:
            data = json.loads(file_content.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValidationError(f"Invalid JSON format: {e}", "file") from e

        # Validate schema
        if not isinstance(data, dict):
            raise ValidationError("Dataset must be a JSON object", "file")

        schema_version = data.get("schema_version", "1.0")
        queries_data = data.get("queries", [])

        if not isinstance(queries_data, list):
            raise ValidationError("Dataset 'queries' must be an array", "file")

        if not queries_data:
            raise ValidationError("Dataset must contain at least one query", "file")

        # Create the dataset record
        dataset = await self.benchmark_dataset_repo.create(
            name=name,
            owner_id=user_id,
            query_count=len(queries_data),
            description=description,
            schema_version=schema_version,
            metadata=data.get("metadata"),
        )

        # Add queries and their relevance judgments
        for query_data in queries_data:
            if not isinstance(query_data, dict):
                raise ValidationError("Each query must be a JSON object", "file")

            query_key = query_data.get("query_key") or query_data.get("query_id")
            query_text = query_data.get("query_text") or query_data.get("query")

            if not query_key or not query_text:
                raise ValidationError(
                    "Each query must have 'query_key' and 'query_text' fields",
                    "file",
                )

            # Build pending relevance judgments to store in metadata
            pending_relevance: list[dict[str, Any]] = []
            judgments = query_data.get("relevant_docs", [])
            for judgment in judgments:
                if isinstance(judgment, dict):
                    doc_ref = judgment.get("doc_ref", judgment)
                    grade = judgment.get("relevance_grade", 2)  # Default to "relevant"
                else:
                    # Simple string reference
                    doc_ref = {"uri": str(judgment)}
                    grade = 2
                pending_relevance.append({"doc_ref": doc_ref, "relevance_grade": grade})

            # Store relevance judgments in metadata (will be resolved when mapping is created)
            query_metadata = query_data.get("metadata") or {}
            if pending_relevance:
                query_metadata["_pending_relevance"] = pending_relevance

            await self.benchmark_dataset_repo.add_query(
                dataset_id=str(dataset.id),
                query_key=str(query_key),
                query_text=str(query_text),
                metadata=query_metadata,
            )

        logger.info(
            "Uploaded benchmark dataset %s with %d queries for user %d",
            dataset.id,
            len(queries_data),
            user_id,
        )

        return {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "query_count": dataset.query_count,
            "schema_version": dataset.schema_version,
            "created_at": dataset.created_at,
        }

    async def get_dataset(
        self,
        dataset_id: str,
        user_id: int,
    ) -> BenchmarkDataset:
        """Get a dataset by ID with ownership check.

        Args:
            dataset_id: UUID of the dataset
            user_id: ID of the user requesting access

        Returns:
            BenchmarkDataset instance

        Raises:
            EntityNotFoundError: If dataset not found
            AccessDeniedError: If user doesn't own the dataset
        """
        return await self.benchmark_dataset_repo.get_by_uuid_for_user(dataset_id, user_id)

    async def list_datasets(
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
        result = await self.benchmark_dataset_repo.list_for_user(user_id, offset, limit)
        return cast(tuple[list[BenchmarkDataset], int], result)

    async def delete_dataset(
        self,
        dataset_id: str,
        user_id: int,
    ) -> None:
        """Delete a dataset.

        Args:
            dataset_id: UUID of the dataset to delete
            user_id: ID of the user requesting deletion

        Raises:
            EntityNotFoundError: If dataset not found
            AccessDeniedError: If user doesn't own the dataset
        """
        await self.benchmark_dataset_repo.delete(dataset_id, user_id)
        logger.info("Deleted benchmark dataset %s for user %d", dataset_id, user_id)

    async def create_mapping(
        self,
        dataset_id: str,
        collection_id: str,
        user_id: int,
    ) -> dict[str, Any]:
        """Create a mapping between a dataset and a collection.

        Args:
            dataset_id: UUID of the dataset
            collection_id: UUID of the collection
            user_id: ID of the user creating the mapping

        Returns:
            Dictionary with mapping details

        Raises:
            EntityNotFoundError: If dataset or collection not found
            AccessDeniedError: If user doesn't own the dataset or collection
        """
        # Verify dataset ownership
        dataset = await self.benchmark_dataset_repo.get_by_uuid_for_user(dataset_id, user_id)

        # Verify collection ownership
        try:
            collection = await self.collection_repo.get_by_uuid_with_permission_check(
                collection_uuid=collection_id,
                user_id=user_id,
            )
        except EntityNotFoundError:
            raise EntityNotFoundError("collection", collection_id) from None
        except AccessDeniedError:
            raise AccessDeniedError(str(user_id), "collection", collection_id) from None

        # Create the mapping
        dataset_id_str = str(dataset.id)
        collection_id_str = str(collection.id)

        mapping = await self.benchmark_dataset_repo.create_mapping(
            dataset_id=dataset_id_str,
            collection_id=collection_id_str,
        )

        # Copy pending relevance judgments from queries to the mapping
        queries = await self.benchmark_dataset_repo.get_queries_for_dataset(dataset_id_str)
        total_refs = 0
        mapping_id_int = int(mapping.id)

        for query in queries:
            query_metadata = query.query_metadata or {}
            pending = query_metadata.get("_pending_relevance", [])
            query_id_int = int(query.id)

            for judgment in pending:
                doc_ref = judgment.get("doc_ref", {})
                grade = judgment.get("relevance_grade", 2)

                # Compute hash for the doc ref
                doc_ref_hash = hashlib.sha256(json.dumps(doc_ref, sort_keys=True).encode()).hexdigest()

                await self.benchmark_dataset_repo.add_relevance(
                    query_id=query_id_int,
                    mapping_id=mapping_id_int,
                    doc_ref=doc_ref,
                    relevance_grade=grade,
                    doc_ref_hash=doc_ref_hash,
                )
                total_refs += 1

        # Update mapping with total count
        await self.benchmark_dataset_repo.update_mapping_status(
            mapping_id=mapping_id_int,
            status=MappingStatus.PENDING,
            total_count=total_refs,
        )

        logger.info(
            "Created mapping %d: dataset %s -> collection %s with %d references",
            mapping.id,
            dataset_id,
            collection_id,
            total_refs,
        )

        return {
            "id": mapping.id,
            "dataset_id": dataset.id,
            "collection_id": collection.id,
            "mapping_status": mapping.mapping_status,
            "mapped_count": 0,
            "total_count": total_refs,
            "created_at": mapping.created_at,
        }

    async def resolve_mapping(
        self,
        mapping_id: int,
        user_id: int,
    ) -> dict[str, Any]:
        """Resolve document references in a mapping.

        Attempts to match doc_refs to actual documents in the collection
        by URI, path, or content hash.

        Args:
            mapping_id: ID of the mapping to resolve
            user_id: ID of the user requesting resolution

        Returns:
            Dictionary with resolution results

        Raises:
            EntityNotFoundError: If mapping not found
            AccessDeniedError: If user doesn't own the dataset
        """
        mapping = await self.benchmark_dataset_repo.get_mapping(mapping_id)
        if not mapping:
            raise EntityNotFoundError("benchmark_dataset_mapping", str(mapping_id))

        # Verify dataset ownership
        mapping_dataset_id = str(mapping.dataset_id)
        mapping_collection_id = str(mapping.collection_id)
        await self.benchmark_dataset_repo.get_by_uuid_for_user(mapping_dataset_id, user_id)

        # Get all relevance judgments for this mapping
        relevances = await self.benchmark_dataset_repo.get_relevance_for_mapping(mapping_id)

        # Get all documents in the collection (paginate through all)
        documents_list, total_docs = await self.document_repo.list_by_collection(
            collection_id=mapping_collection_id,
            limit=10000,  # Get all documents for lookup
        )

        # Build lookup indices
        uri_index: dict[str, str] = {}
        hash_index: dict[str, str] = {}
        path_index: dict[str, str] = {}

        for doc in documents_list:
            doc_id = str(doc.id)
            # Safely extract string values from model attributes
            doc_uri: str | None = str(doc.uri) if doc.uri is not None else None
            doc_hash: str | None = str(doc.content_hash) if doc.content_hash is not None else None

            if doc_uri is not None:
                uri_index[doc_uri] = doc_id
                # Also index by filename portion
                if "/" in doc_uri:
                    path_index[doc_uri.split("/")[-1]] = doc_id
            if doc_hash is not None:
                hash_index[doc_hash] = doc_id

        # Resolve each relevance judgment
        resolved_count = 0
        unresolved: list[dict[str, Any]] = []

        # Build a set of valid document IDs for fast lookup
        valid_doc_ids = {str(doc.id) for doc in documents_list}

        for relevance in relevances:
            # Extract doc_ref as a dict (SQLAlchemy model attribute)
            raw_doc_ref = cast(dict[str, Any] | None, relevance.doc_ref)
            doc_ref: dict[str, Any] = raw_doc_ref if raw_doc_ref else {}
            resolved_doc_id: str | None = None

            # Try to match by various reference types
            ref_uri = doc_ref.get("uri")
            ref_path = doc_ref.get("path")
            ref_hash = doc_ref.get("content_hash")
            ref_doc_id = doc_ref.get("document_id")

            if ref_uri:
                resolved_doc_id = uri_index.get(str(ref_uri))
            if not resolved_doc_id and ref_path:
                path_str = str(ref_path)
                resolved_doc_id = uri_index.get(path_str) or path_index.get(
                    path_str.split("/")[-1] if "/" in path_str else path_str
                )
            if not resolved_doc_id and ref_hash:
                resolved_doc_id = hash_index.get(str(ref_hash))
            if not resolved_doc_id and ref_doc_id:
                # Direct document ID reference
                doc_id_str = str(ref_doc_id)
                if doc_id_str in valid_doc_ids:
                    resolved_doc_id = doc_id_str

            # Access model attributes with explicit type casting
            relevance_id = cast(int, relevance.id)
            query_id = cast(int, relevance.benchmark_query_id)
            doc_ref_hash_val = cast(str | None, relevance.doc_ref_hash)

            if resolved_doc_id:
                await self.benchmark_dataset_repo.resolve_relevance(
                    relevance_id=relevance_id,
                    document_id=resolved_doc_id,
                )
                resolved_count += 1
            else:
                unresolved.append(
                    {
                        "query_id": query_id,
                        "doc_ref": doc_ref,
                        "doc_ref_hash": doc_ref_hash_val,
                    }
                )

        # Update mapping status
        total = len(relevances)
        if resolved_count == total:
            status = MappingStatus.RESOLVED
        elif resolved_count > 0:
            status = MappingStatus.PARTIAL
        else:
            status = MappingStatus.PENDING

        await self.benchmark_dataset_repo.update_mapping_status(
            mapping_id=mapping_id,
            status=status,
            mapped_count=resolved_count,
            total_count=total,
        )

        logger.info(
            "Resolved mapping %d: %d/%d references resolved",
            mapping_id,
            resolved_count,
            total,
        )

        return {
            "id": mapping_id,
            "mapping_status": status.value,
            "mapped_count": resolved_count,
            "total_count": total,
            "unresolved": unresolved[:100],  # Limit unresolved list to 100 items
        }

    async def get_mapping(
        self,
        mapping_id: int,
        user_id: int,
    ) -> BenchmarkDatasetMapping:
        """Get a mapping by ID with ownership check.

        Args:
            mapping_id: ID of the mapping
            user_id: ID of the user requesting access

        Returns:
            BenchmarkDatasetMapping instance

        Raises:
            EntityNotFoundError: If mapping not found
            AccessDeniedError: If user doesn't own the dataset
        """
        mapping = await self.benchmark_dataset_repo.get_mapping(mapping_id)
        if not mapping:
            raise EntityNotFoundError("benchmark_dataset_mapping", str(mapping_id))

        # Verify dataset ownership
        dataset_id = cast(str, mapping.dataset_id)
        await self.benchmark_dataset_repo.get_by_uuid_for_user(dataset_id, user_id)

        return mapping

    async def list_mappings(
        self,
        dataset_id: str,
        user_id: int,
    ) -> list[BenchmarkDatasetMapping]:
        """List all mappings for a dataset.

        Args:
            dataset_id: UUID of the dataset
            user_id: ID of the user requesting access

        Returns:
            List of BenchmarkDatasetMapping instances

        Raises:
            EntityNotFoundError: If dataset not found
            AccessDeniedError: If user doesn't own the dataset
        """
        # Verify dataset ownership
        await self.benchmark_dataset_repo.get_by_uuid_for_user(dataset_id, user_id)

        result = await self.benchmark_dataset_repo.list_mappings_for_dataset(dataset_id)
        return cast(list[BenchmarkDatasetMapping], result)
