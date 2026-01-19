"""Benchmark Service for managing benchmark execution and results."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from typing import TYPE_CHECKING, Any, cast

from shared.benchmarks import parse_k_values, validate_top_k_values
from shared.database.exceptions import EntityNotFoundError, ValidationError
from shared.database.models import Benchmark, BenchmarkStatus, MappingStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.repositories.benchmark_dataset_repository import BenchmarkDatasetRepository
    from shared.database.repositories.benchmark_repository import BenchmarkRepository
    from shared.database.repositories.collection_repository import CollectionRepository
    from shared.database.repositories.operation_repository import OperationRepository

    from .search_service import SearchService

logger = logging.getLogger(__name__)


class BenchmarkService:
    """Service for managing benchmark lifecycle and execution."""

    def __init__(
        self,
        db_session: AsyncSession,
        benchmark_repo: BenchmarkRepository,
        benchmark_dataset_repo: BenchmarkDatasetRepository,
        collection_repo: CollectionRepository,
        operation_repo: OperationRepository,
        search_service: SearchService,
    ):
        """Initialize the service.

        Args:
            db_session: AsyncSession for database operations
            benchmark_repo: Repository for benchmark operations
            benchmark_dataset_repo: Repository for dataset operations
            collection_repo: Repository for collection operations
            operation_repo: Repository for operation tracking
            search_service: Service for executing searches
        """
        self.db_session = db_session
        self.benchmark_repo = benchmark_repo
        self.benchmark_dataset_repo = benchmark_dataset_repo
        self.collection_repo = collection_repo
        self.operation_repo = operation_repo
        self.search_service = search_service

    async def create_benchmark(
        self,
        user_id: int,
        mapping_id: int,
        name: str,
        description: str | None,
        config_matrix: dict[str, Any],
        top_k: int = 10,
        metrics_to_compute: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new benchmark with pre-created runs.

        Args:
            user_id: ID of the user creating the benchmark
            mapping_id: ID of the dataset-collection mapping
            name: Name for the benchmark
            description: Optional description
            config_matrix: Configuration matrix defining parameter space
            top_k: Default top-k for evaluation
            metrics_to_compute: List of metrics to compute

        Returns:
            Dictionary with benchmark details

        Raises:
            EntityNotFoundError: If mapping not found
            AccessDeniedError: If user doesn't own the mapping's dataset
            ValidationError: If validation fails
        """
        if metrics_to_compute is None:
            metrics_to_compute = ["precision", "recall", "mrr", "ndcg"]

        # Validate mapping exists and is accessible
        mapping = await self.benchmark_dataset_repo.get_mapping(mapping_id)
        if not mapping:
            raise EntityNotFoundError("benchmark_dataset_mapping", str(mapping_id))

        # Verify dataset ownership
        dataset_id = cast(str, mapping.dataset_id)
        await self.benchmark_dataset_repo.get_by_uuid_for_user(dataset_id, user_id)

        # Check mapping is resolved or partial (has some mapped documents)
        mapping_status = cast(str, mapping.mapping_status)
        if mapping_status == MappingStatus.PENDING.value:
            raise ValidationError(
                "Mapping must be resolved before running benchmarks. Call resolve_mapping first.",
                "mapping_id",
            )

        # Validate collection exists
        collection_id = cast(str, mapping.collection_id)
        collection = await self.collection_repo.get_by_uuid(collection_id)
        if not collection:
            raise EntityNotFoundError("collection", collection_id)

        # Note: Sparse/hybrid search mode validation is done at runtime during search
        # If the collection doesn't have sparse vectors, vecpipe will fall back to dense

        # Parse and validate k-values for metrics using shared validation
        k_config = parse_k_values(
            raw_primary_k=config_matrix.get("primary_k", 10),
            raw_k_values=config_matrix.get("k_values_for_metrics"),
        )
        primary_k = k_config.primary_k
        k_values_for_metrics = k_config.k_values_for_metrics

        # Validate top-k values for search configuration
        raw_top_k_values = config_matrix.get("top_k_values", [top_k])
        if not isinstance(raw_top_k_values, list) or not raw_top_k_values:
            raise ValidationError(
                "config_matrix.top_k_values must be a non-empty list",
                "config_matrix",
            )

        required_top_k = max(k_values_for_metrics)
        _, invalid_values, too_small_values = validate_top_k_values(raw_top_k_values, required_top_k)

        if invalid_values:
            raise ValidationError(
                f"config_matrix.top_k_values must be positive integers; invalid values: {invalid_values}",
                "config_matrix",
            )
        if too_small_values:
            raise ValidationError(
                f"All top_k_values must be >= max(k_values_for_metrics) ({required_top_k}); "
                f"invalid values: {too_small_values}",
                "config_matrix",
            )

        config_matrix = dict(config_matrix)
        config_matrix["primary_k"] = primary_k
        config_matrix["k_values_for_metrics"] = k_values_for_metrics

        # Compute config matrix hash for deduplication
        config_matrix_hash = hashlib.sha256(json.dumps(config_matrix, sort_keys=True).encode()).hexdigest()[:16]

        # Create the benchmark
        benchmark = await self.benchmark_repo.create(
            name=name,
            owner_id=user_id,
            mapping_id=mapping_id,
            config_matrix=config_matrix,
            config_matrix_hash=config_matrix_hash,
            metrics_to_compute=metrics_to_compute,
            top_k=top_k,
            description=description,
        )

        # Generate all configuration combinations and pre-create runs
        runs_created = await self._create_benchmark_runs(
            benchmark_id=str(benchmark.id),
            config_matrix=config_matrix,
            top_k=top_k,
        )

        # Update total runs count
        await self.benchmark_repo.set_total_runs(str(benchmark.id), runs_created)

        await self.db_session.commit()

        logger.info(
            "Created benchmark %s with %d runs for user %d",
            benchmark.id,
            runs_created,
            user_id,
        )

        return {
            "id": str(benchmark.id),
            "name": str(benchmark.name),
            "description": str(benchmark.description) if benchmark.description else None,
            "owner_id": int(benchmark.owner_id),
            "mapping_id": int(benchmark.mapping_id),
            "status": str(benchmark.status),
            "total_runs": runs_created,
            "completed_runs": 0,
            "failed_runs": 0,
            "created_at": benchmark.created_at,
        }

    async def _create_benchmark_runs(
        self,
        benchmark_id: str,
        config_matrix: dict[str, Any],
        top_k: int,
    ) -> int:
        """Create benchmark runs for all configuration combinations.

        Args:
            benchmark_id: UUID of the benchmark
            config_matrix: Configuration matrix
            top_k: Default top-k value

        Returns:
            Number of runs created
        """
        combinations = itertools.product(
            config_matrix.get("search_modes", ["dense"]),
            config_matrix.get("use_reranker", [False]),
            config_matrix.get("top_k_values", [top_k]),
            config_matrix.get("rrf_k_values", [60]),
            config_matrix.get("score_thresholds", [None]),
        )

        run_count = 0
        for run_order, (search_mode, use_reranker, k_val, rrf_k, score_thresh) in enumerate(combinations):
            config = {
                "search_mode": search_mode,
                "use_reranker": use_reranker,
                "top_k": k_val,
                "rrf_k": rrf_k,
                "score_threshold": score_thresh,
            }
            config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]

            await self.benchmark_repo.create_run(
                benchmark_id=benchmark_id,
                run_order=run_order,
                config_hash=config_hash,
                config=config,
            )
            run_count += 1

        return run_count

    async def get_benchmark(
        self,
        benchmark_id: str,
        user_id: int,
    ) -> Benchmark:
        """Get a benchmark by ID with ownership check.

        Args:
            benchmark_id: UUID of the benchmark
            user_id: ID of the user requesting access

        Returns:
            Benchmark instance

        Raises:
            EntityNotFoundError: If benchmark not found
            AccessDeniedError: If user doesn't own the benchmark
        """
        return await self.benchmark_repo.get_by_uuid_for_user(benchmark_id, user_id)

    async def list_benchmarks(
        self,
        user_id: int,
        offset: int = 0,
        limit: int = 50,
        status_filter: BenchmarkStatus | None = None,
    ) -> tuple[list[Benchmark], int]:
        """List benchmarks owned by a user.

        Args:
            user_id: ID of the user
            offset: Pagination offset
            limit: Maximum number of results
            status_filter: Optional status filter

        Returns:
            Tuple of (benchmarks list, total count)
        """
        result = await self.benchmark_repo.list_for_user(
            user_id=user_id,
            offset=offset,
            limit=limit,
            status_filter=status_filter,
        )
        return cast(tuple[list[Benchmark], int], result)

    async def start_benchmark(
        self,
        benchmark_id: str,
        user_id: int,
    ) -> dict[str, Any]:
        """Start benchmark execution.

        Creates an Operation record and dispatches the Celery task.
        Uses atomic status transition to prevent race conditions.

        Args:
            benchmark_id: UUID of the benchmark
            user_id: ID of the user starting the benchmark

        Returns:
            Dictionary with benchmark status and operation UUID

        Raises:
            EntityNotFoundError: If benchmark not found
            AccessDeniedError: If user doesn't own the benchmark
            ValidationError: If benchmark is not in PENDING status or race lost
        """
        # Verify ownership (raises EntityNotFoundError or AccessDeniedError)
        benchmark = await self.benchmark_repo.get_by_uuid_for_user(benchmark_id, user_id)

        # Get mapping to find collection (needed before operation creation)
        mapping = await self.benchmark_dataset_repo.get_mapping(cast(int, benchmark.mapping_id))
        if not mapping:
            raise EntityNotFoundError("benchmark_dataset_mapping", str(benchmark.mapping_id))

        # Create operation record first
        from shared.database.models import OperationType

        operation = await self.operation_repo.create(
            collection_id=cast(str, mapping.collection_id),
            user_id=user_id,
            operation_type=OperationType.BENCHMARK,
            config={"kind": "benchmark_run", "benchmark_id": benchmark_id},
        )

        # Atomically transition PENDING → RUNNING and link operation
        # This prevents TOCTOU race conditions where two concurrent requests
        # both check PENDING, then both try to start the benchmark
        updated_benchmark = await self.benchmark_repo.transition_status_atomically(
            benchmark_uuid=benchmark_id,
            from_status=BenchmarkStatus.PENDING,
            to_status=BenchmarkStatus.RUNNING,
            operation_uuid=operation.uuid,
        )

        if updated_benchmark is None:
            # Race lost or benchmark was not in PENDING status
            raise ValidationError(
                "Benchmark must be in PENDING status to start. It may have been started by another request.",
                "benchmark_id",
            )

        # Commit before dispatching Celery task
        await self.db_session.commit()

        # Dispatch Celery task
        from webui.celery_app import celery_app

        celery_app.send_task(
            "webui.tasks.benchmark.run_benchmark",
            kwargs={
                "operation_uuid": operation.uuid,
                "benchmark_id": benchmark_id,
            },
        )

        logger.info(
            "Started benchmark %s with operation %s",
            benchmark_id,
            operation.uuid,
        )

        return {
            "id": benchmark_id,
            "status": BenchmarkStatus.RUNNING.value,
            "operation_uuid": operation.uuid,
            "message": "Benchmark execution started",
        }

    async def cancel_benchmark(
        self,
        benchmark_id: str,
        user_id: int,
    ) -> dict[str, Any]:
        """Cancel a running benchmark.

        Args:
            benchmark_id: UUID of the benchmark
            user_id: ID of the user cancelling the benchmark

        Returns:
            Dictionary with updated benchmark status

        Raises:
            EntityNotFoundError: If benchmark not found
            AccessDeniedError: If user doesn't own the benchmark
            ValidationError: If benchmark is not in PENDING or RUNNING status
        """
        benchmark = await self.benchmark_repo.get_by_uuid_for_user(benchmark_id, user_id)

        status = cast(str, benchmark.status)
        if status not in (BenchmarkStatus.PENDING.value, BenchmarkStatus.RUNNING.value):
            raise ValidationError(
                f"Cannot cancel benchmark in {status} status",
                "benchmark_id",
            )

        await self.benchmark_repo.update_status(
            benchmark_id,
            BenchmarkStatus.CANCELLED,
        )

        await self.db_session.commit()

        logger.info("Cancelled benchmark %s", benchmark_id)

        return {
            "id": benchmark_id,
            "status": BenchmarkStatus.CANCELLED.value,
            "message": "Benchmark cancelled",
        }

    async def get_results(
        self,
        benchmark_id: str,
        user_id: int,
    ) -> dict[str, Any]:
        """Get aggregated benchmark results.

        Args:
            benchmark_id: UUID of the benchmark
            user_id: ID of the user requesting results

        Returns:
            Dictionary with aggregated results across all runs

        Raises:
            EntityNotFoundError: If benchmark not found
            AccessDeniedError: If user doesn't own the benchmark
        """
        # Verify ownership
        await self.benchmark_repo.get_by_uuid_for_user(benchmark_id, user_id)

        # Get aggregated results from repository
        result = await self.benchmark_repo.get_aggregated_results(benchmark_id)
        return cast(dict[str, Any], result)

    async def get_run_query_results(
        self,
        run_id: str,
        user_id: int,
        page: int = 1,
        per_page: int = 50,
    ) -> dict[str, Any]:
        """Get per-query results for a specific run.

        Args:
            run_id: UUID of the benchmark run
            user_id: ID of the user requesting results
            page: Page number (1-indexed)
            per_page: Results per page

        Returns:
            Dictionary with paginated query results

        Raises:
            EntityNotFoundError: If run not found
            AccessDeniedError: If user doesn't own the benchmark
        """
        # Get run and verify ownership
        run = await self.benchmark_repo.get_run(run_id)
        if not run:
            raise EntityNotFoundError("benchmark_run", run_id)

        # Verify benchmark ownership
        await self.benchmark_repo.get_by_uuid_for_user(cast(str, run.benchmark_id), user_id)

        # Get paginated query results
        offset = (page - 1) * per_page
        results, total = await self.benchmark_repo.get_query_results_for_run(
            run_id=run_id,
            offset=offset,
            limit=per_page,
        )

        # Format results
        formatted_results = []
        for r in results:
            query = r.query
            formatted_results.append(
                {
                    "query_id": cast(int, r.benchmark_query_id),
                    "query_key": str(query.query_key) if query else "unknown",
                    "query_text": str(query.query_text) if query else "unknown",
                    "retrieved_doc_ids": r.retrieved_doc_ids or [],
                    "precision_at_k": r.precision_at_k,
                    "recall_at_k": r.recall_at_k,
                    "reciprocal_rank": r.reciprocal_rank,
                    "ndcg_at_k": r.ndcg_at_k,
                    "search_time_ms": r.search_time_ms,
                    "rerank_time_ms": r.rerank_time_ms,
                }
            )

        return {
            "run_id": run_id,
            "results": formatted_results,
            "total": total,
            "page": page,
            "per_page": per_page,
        }

    async def delete_benchmark(
        self,
        benchmark_id: str,
        user_id: int,
    ) -> None:
        """Delete a benchmark.

        Args:
            benchmark_id: UUID of the benchmark to delete
            user_id: ID of the user requesting deletion

        Raises:
            EntityNotFoundError: If benchmark not found
            AccessDeniedError: If user doesn't own the benchmark
        """
        await self.benchmark_repo.delete(benchmark_id, user_id)
        await self.db_session.commit()
        logger.info("Deleted benchmark %s for user %d", benchmark_id, user_id)
