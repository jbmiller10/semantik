"""Benchmark Dataset Service for managing benchmark datasets and mappings."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from shared.config import settings
from shared.database.exceptions import AccessDeniedError, EntityNotFoundError, ValidationError
from shared.database.models import BenchmarkDataset, BenchmarkDatasetMapping, MappingStatus, OperationType

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.repositories.benchmark_dataset_repository import BenchmarkDatasetRepository
    from shared.database.repositories.collection_repository import CollectionRepository
    from shared.database.repositories.document_repository import DocumentRepository
    from shared.database.repositories.operation_repository import OperationRepository
    from webui.tasks.utils import CeleryTaskWithOperationUpdates

logger = logging.getLogger(__name__)


@dataclass
class ResolutionContext:
    """Lookup tables for document reference resolution."""

    existing_doc_ids: set[str]
    uri_to_doc_id: dict[str, str]
    hash_to_doc_ids: dict[str, list[str]]
    file_name_to_doc_ids: dict[str, list[str]]


@dataclass
class ResolutionStats:
    """Statistics tracked during resolution."""

    processed: int = 0
    resolved: int = 0
    ambiguous: int = 0
    unresolved: int = 0
    unresolved_samples: list[dict[str, Any]] = field(default_factory=list)


def resolve_single_doc_ref(
    doc_ref: dict[str, Any],
    ctx: ResolutionContext,
) -> tuple[str | None, str | None]:
    """Resolve a single document reference using 5-priority strategy.

    Resolution priority:
    1. document_id - must exist in collection
    2. uri - exact match against Document.uri
    3. content_hash - only if unique within collection
    4. path - treated as URI-like identifier
    5. file_name - only if unique within collection

    Args:
        doc_ref: Document reference dictionary with keys like document_id, uri, etc.
        ctx: Resolution context with lookup tables.

    Returns:
        Tuple of (resolved_document_id, failure_reason).
        One will always be None.
    """
    ref_doc_id = doc_ref.get("document_id")
    ref_uri = doc_ref.get("uri")
    ref_hash = doc_ref.get("content_hash")
    ref_path = doc_ref.get("path")
    ref_file_name = doc_ref.get("file_name")

    # Priority 1: document_id
    if ref_doc_id:
        doc_id_str = str(ref_doc_id)
        if doc_id_str in ctx.existing_doc_ids:
            return doc_id_str, None
        return None, "not_found"

    # Priority 2: uri
    if ref_uri:
        resolved = ctx.uri_to_doc_id.get(str(ref_uri))
        if resolved:
            return resolved, None
        return None, "not_found"

    # Priority 3: content_hash (must be unique)
    if ref_hash:
        candidates = ctx.hash_to_doc_ids.get(str(ref_hash), [])
        if len(candidates) == 1:
            return candidates[0], None
        if len(candidates) > 1:
            return None, "ambiguous"
        return None, "not_found"

    # Priority 4: path (treated as URI)
    if ref_path:
        resolved = ctx.uri_to_doc_id.get(str(ref_path))
        if resolved:
            return resolved, None
        return None, "not_found"

    # Priority 5: file_name (must be unique)
    if ref_file_name:
        candidates = ctx.file_name_to_doc_ids.get(str(ref_file_name), [])
        if len(candidates) == 1:
            return candidates[0], None
        if len(candidates) > 1:
            return None, "ambiguous"
        return None, "not_found"

    return None, "invalid_ref"


def parse_judgment(
    judgment: Any,
    query_key: str,
) -> tuple[dict[str, Any], int]:
    """Parse a relevance judgment into doc_ref and grade.

    Args:
        judgment: Raw judgment from dataset (dict, string, or scalar).
        query_key: Query key for error messages.

    Returns:
        Tuple of (doc_ref dict, relevance_grade int).

    Raises:
        ValidationError: If judgment format is invalid.
    """
    grade_raw: Any = 2

    if isinstance(judgment, dict):
        grade_raw = judgment.get("relevance_grade", 2)
        doc_ref_raw = judgment.get("doc_ref", judgment)

        if isinstance(doc_ref_raw, dict):
            doc_ref = doc_ref_raw
        elif isinstance(doc_ref_raw, str):
            doc_ref = {"uri": doc_ref_raw}
        else:
            raise ValidationError(
                f"Query {query_key!s} has invalid doc_ref (expected object or string)",
                "file",
            )
    else:
        doc_ref = {"uri": str(judgment)}

    if not doc_ref:
        raise ValidationError(f"Query {query_key!s} has empty doc_ref", "file")

    try:
        grade = int(grade_raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Query {query_key!s} has invalid relevance_grade (expected int 0-3)",
            "file",
        ) from exc

    if grade < 0 or grade > 3:
        raise ValidationError(
            f"Query {query_key!s} relevance_grade must be between 0 and 3",
            "file",
        )

    return doc_ref, grade


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
        operation_repo: OperationRepository,
    ):
        """Initialize the service.

        Args:
            db_session: AsyncSession for database operations
            benchmark_dataset_repo: Repository for benchmark dataset operations
            collection_repo: Repository for collection operations
            document_repo: Repository for document operations
            operation_repo: Repository for operation tracking
        """
        self.db_session = db_session
        self.benchmark_dataset_repo = benchmark_dataset_repo
        self.collection_repo = collection_repo
        self.document_repo = document_repo
        self.operation_repo = operation_repo

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
        max_bytes = int(getattr(settings, "BENCHMARK_DATASET_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
        if len(file_content) > max_bytes:
            raise ValidationError(
                f"Dataset file too large: {len(file_content)} bytes (max {max_bytes} bytes)",
                "file",
            )

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

        max_queries = int(getattr(settings, "BENCHMARK_DATASET_MAX_QUERIES", 1000))
        if len(queries_data) > max_queries:
            raise ValidationError(
                f"Dataset has too many queries: {len(queries_data)} (max {max_queries})",
                "file",
            )

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

            judgments = query_data.get("relevant_docs")
            if judgments is None:
                # Backwards compatibility with early frontend iterations
                judgments = query_data.get("relevant_doc_refs", [])

            if not isinstance(judgments, list):
                raise ValidationError(
                    f"Query {query_key!s} 'relevant_docs' must be an array",
                    "file",
                )

            max_judgments = int(getattr(settings, "BENCHMARK_DATASET_MAX_JUDGMENTS_PER_QUERY", 100))
            if len(judgments) > max_judgments:
                raise ValidationError(
                    f"Query {query_key!s} has too many relevance judgments: {len(judgments)} (max {max_judgments})",
                    "file",
                )

            # Build pending relevance judgments to store in metadata
            pending_relevance: list[dict[str, Any]] = []
            for judgment in judgments:
                doc_ref, grade = parse_judgment(judgment, str(query_key))
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

        await self.db_session.commit()

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
        await self.db_session.commit()
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

        await self.db_session.commit()

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

        Resolution is deterministic and uses a strict priority order:
        1) document_id (must exist in the mapped collection)
        2) uri (exact match against Document.uri)
        3) content_hash (only if unique within the collection)
        4) path (best-effort: exact match to Document.uri)
        5) file_name (best-effort: only if unique within the collection)

        For large mappings/collections, this schedules an async operation and returns
        an operation UUID for progress subscription.
        """
        mapping = await self.benchmark_dataset_repo.get_mapping(mapping_id)
        if not mapping:
            raise EntityNotFoundError("benchmark_dataset_mapping", str(mapping_id))

        mapping_dataset_id = str(mapping.dataset_id)
        mapping_collection_id = str(mapping.collection_id)
        await self.benchmark_dataset_repo.get_by_uuid_for_user(mapping_dataset_id, user_id)

        ref_count = await self.benchmark_dataset_repo.count_relevance_for_mapping(mapping_id)
        doc_count = await self.document_repo.count_by_collection(mapping_collection_id)

        max_refs = int(getattr(settings, "BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_REFS", 10_000))
        max_docs = int(getattr(settings, "BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_DOCS", 50_000))
        max_wall_ms = int(getattr(settings, "BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_WALL_MS", 8_000))

        # Route large jobs to async execution up front.
        if ref_count > max_refs or doc_count > max_docs:
            operation_uuid = await self._enqueue_mapping_resolution_operation(
                mapping_id=mapping_id,
                dataset_id=mapping_dataset_id,
                collection_id=mapping_collection_id,
                user_id=user_id,
            )
            return {
                "id": mapping_id,
                "operation_uuid": operation_uuid,
                "mapping_status": cast(str, mapping.mapping_status),
                "mapped_count": cast(int, mapping.mapped_count),
                "total_count": cast(int, mapping.total_count),
                "unresolved": [],
            }

        start_time = time.monotonic()
        stats = ResolutionStats()
        after_id = 0

        while True:
            # Enforce wall-clock budget for sync requests.
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            if elapsed_ms >= max_wall_ms:
                resolved_total = await self.benchmark_dataset_repo.count_resolved_relevance_for_mapping(mapping_id)
                if resolved_total == ref_count:
                    break

                operation_uuid = await self._enqueue_mapping_resolution_operation(
                    mapping_id=mapping_id,
                    dataset_id=mapping_dataset_id,
                    collection_id=mapping_collection_id,
                    user_id=user_id,
                )
                status = MappingStatus.PARTIAL if resolved_total > 0 else MappingStatus.PENDING
                await self.benchmark_dataset_repo.update_mapping_status(
                    mapping_id=mapping_id,
                    status=status,
                    mapped_count=resolved_total,
                    total_count=ref_count,
                )
                await self.db_session.commit()

                return {
                    "id": mapping_id,
                    "operation_uuid": operation_uuid,
                    "mapping_status": status.value,
                    "mapped_count": resolved_total,
                    "total_count": ref_count,
                    "unresolved": [],
                }

            batch = await self.benchmark_dataset_repo.list_unresolved_relevance_for_mapping(
                mapping_id,
                after_id=after_id,
                limit=1000,
            )
            if not batch:
                break

            after_id = int(cast(int, batch[-1].id))
            ctx = await self._build_resolution_context(mapping_collection_id, batch)
            self._resolve_batch(batch, ctx, stats, collect_samples=True)

            await self.db_session.flush()
            await self.db_session.commit()

        resolved_total = await self.benchmark_dataset_repo.count_resolved_relevance_for_mapping(mapping_id)
        status = self._compute_mapping_status(resolved_total, ref_count)

        await self.benchmark_dataset_repo.update_mapping_status(
            mapping_id=mapping_id,
            status=status,
            mapped_count=resolved_total,
            total_count=ref_count,
        )
        await self.db_session.commit()

        logger.info("Resolved mapping %d: %d/%d references resolved", mapping_id, resolved_total, ref_count)

        return {
            "id": mapping_id,
            "operation_uuid": None,
            "mapping_status": status.value,
            "mapped_count": resolved_total,
            "total_count": ref_count,
            "unresolved": stats.unresolved_samples,
        }

    async def resolve_mapping_with_progress(
        self,
        *,
        mapping_id: int,
        user_id: int,
        operation_uuid: str,
        progress_reporter: CeleryTaskWithOperationUpdates,
    ) -> dict[str, Any]:
        """Resolve mapping in a Celery worker with progress streaming."""
        mapping = await self.benchmark_dataset_repo.get_mapping(mapping_id)
        if not mapping:
            raise EntityNotFoundError("benchmark_dataset_mapping", str(mapping_id))

        dataset_id = str(mapping.dataset_id)
        collection_id = str(mapping.collection_id)
        await self.benchmark_dataset_repo.get_by_uuid_for_user(dataset_id, user_id)

        total_refs = await self.benchmark_dataset_repo.count_relevance_for_mapping(mapping_id)
        stats = ResolutionStats()
        after_id = 0

        progress_reporter.set_collection_id(collection_id)

        async def send_progress(stage: str) -> None:
            await progress_reporter.send_update(
                "benchmark_mapping_resolution_progress",
                {
                    "mapping_id": mapping_id,
                    "dataset_id": dataset_id,
                    "collection_id": collection_id,
                    "stage": stage,
                    "total_refs": total_refs,
                    "processed_refs": stats.processed,
                    "resolved_refs": stats.resolved,
                    "ambiguous_refs": stats.ambiguous,
                    "unresolved_refs": stats.unresolved,
                },
            )

        await send_progress("starting")

        try:
            await send_progress("loading_documents")

            while True:
                batch = await self.benchmark_dataset_repo.list_unresolved_relevance_for_mapping(
                    mapping_id,
                    after_id=after_id,
                    limit=1000,
                )
                if not batch:
                    break

                after_id = int(cast(int, batch[-1].id))
                ctx = await self._build_resolution_context(collection_id, batch)
                self._resolve_batch(batch, ctx, stats, collect_samples=False)

                await self.db_session.flush()
                await self.db_session.commit()
                await send_progress("resolving")

            resolved_total = await self.benchmark_dataset_repo.count_resolved_relevance_for_mapping(mapping_id)
            status = self._compute_mapping_status(resolved_total, total_refs)

            await send_progress("finalizing")

            await self.benchmark_dataset_repo.update_mapping_status(
                mapping_id=mapping_id,
                status=status,
                mapped_count=resolved_total,
                total_count=total_refs,
            )
            await self.db_session.commit()

            await send_progress("completed")

            return {
                "mapping_id": mapping_id,
                "operation_uuid": operation_uuid,
                "mapping_status": status.value,
                "mapped_count": resolved_total,
                "total_count": total_refs,
            }

        except Exception as original_exc:
            logger.error("Mapping resolution %s failed: %s", mapping_id, original_exc, exc_info=True)
            try:
                await send_progress("failed")
            except Exception as progress_exc:
                logger.warning("Failed to send failure progress for mapping %s: %s", mapping_id, progress_exc)
            raise original_exc

    async def _build_resolution_context(
        self,
        collection_id: str,
        batch: list[Any],
    ) -> ResolutionContext:
        """Build lookup tables for a batch of relevance records."""
        doc_ids: set[str] = set()
        uris: set[str] = set()
        hashes: set[str] = set()
        file_names: set[str] = set()

        for relevance in batch:
            raw = relevance.doc_ref
            if not isinstance(raw, dict):
                continue
            if raw.get("document_id"):
                doc_ids.add(str(raw["document_id"]))
            if raw.get("uri"):
                uris.add(str(raw["uri"]))
            if raw.get("path"):
                uris.add(str(raw["path"]))
            if raw.get("content_hash"):
                hashes.add(str(raw["content_hash"]))
            if raw.get("file_name"):
                file_names.add(str(raw["file_name"]))

        return ResolutionContext(
            existing_doc_ids=await self.document_repo.get_existing_ids_in_collection(collection_id, doc_ids),
            uri_to_doc_id=await self.document_repo.get_doc_ids_by_uri_bulk(collection_id, uris),
            hash_to_doc_ids=await self.document_repo.get_doc_ids_by_content_hash_bulk(collection_id, hashes),
            file_name_to_doc_ids=await self.document_repo.get_doc_ids_by_file_name_bulk(collection_id, file_names),
        )

    def _resolve_batch(
        self,
        batch: list[Any],
        ctx: ResolutionContext,
        stats: ResolutionStats,
        *,
        collect_samples: bool,
    ) -> None:
        """Resolve a batch of relevance records and update stats."""
        for relevance in batch:
            stats.processed += 1
            raw_doc_ref = relevance.doc_ref

            if not isinstance(raw_doc_ref, dict) or not raw_doc_ref:
                stats.unresolved += 1
                if collect_samples and len(stats.unresolved_samples) < 100:
                    stats.unresolved_samples.append({
                        "query_id": cast(int, relevance.benchmark_query_id),
                        "doc_ref": raw_doc_ref,
                        "doc_ref_hash": cast(str | None, relevance.doc_ref_hash),
                        "reason": "invalid_doc_ref",
                    })
                continue

            resolved_doc_id, reason = resolve_single_doc_ref(raw_doc_ref, ctx)

            if resolved_doc_id:
                relevance.resolved_document_id = resolved_doc_id
                stats.resolved += 1
            else:
                if reason == "ambiguous":
                    stats.ambiguous += 1
                else:
                    stats.unresolved += 1

                if collect_samples and len(stats.unresolved_samples) < 100:
                    stats.unresolved_samples.append({
                        "query_id": cast(int, relevance.benchmark_query_id),
                        "doc_ref": raw_doc_ref,
                        "doc_ref_hash": cast(str | None, relevance.doc_ref_hash),
                        "reason": reason,
                    })

    def _compute_mapping_status(self, resolved_count: int, total_count: int) -> MappingStatus:
        """Compute mapping status from resolution counts."""
        if resolved_count == total_count:
            return MappingStatus.RESOLVED
        if resolved_count > 0:
            return MappingStatus.PARTIAL
        return MappingStatus.PENDING

    async def _enqueue_mapping_resolution_operation(
        self,
        *,
        mapping_id: int,
        dataset_id: str,
        collection_id: str,
        user_id: int,
    ) -> str:
        """Create an Operation and dispatch a Celery mapping resolution task."""
        operation = await self.operation_repo.create(
            collection_id=collection_id,
            user_id=user_id,
            operation_type=OperationType.BENCHMARK,
            config={
                "kind": "mapping_resolve",
                "mapping_id": mapping_id,
                "dataset_id": dataset_id,
                "collection_id": collection_id,
            },
        )
        await self.db_session.commit()

        operation_uuid = cast(str, operation.uuid)

        from webui.celery_app import celery_app

        celery_app.send_task(
            "webui.tasks.benchmark_mapping.resolve_mapping",
            kwargs={
                "operation_uuid": operation_uuid,
                "mapping_id": mapping_id,
                "user_id": user_id,
            },
        )

        logger.info("Queued mapping resolution %d as operation %s", mapping_id, operation_uuid)
        return operation_uuid

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
