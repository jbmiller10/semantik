"""Repository implementation for Benchmark and related models."""

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
    EntityNotFoundError,
    ValidationError,
)
from shared.database.models import (
    Benchmark,
    BenchmarkDatasetMapping,
    BenchmarkQuery,
    BenchmarkQueryResult,
    BenchmarkRun,
    BenchmarkRunMetric,
    BenchmarkRunStatus,
    BenchmarkStatus,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class BenchmarkRepository:
    """Repository for Benchmark and related models."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session."""
        self.session = session

    # =========================================================================
    # Benchmark CRUD
    # =========================================================================

    async def create(
        self,
        *,
        name: str,
        owner_id: int,
        mapping_id: int,
        config_matrix: dict[str, Any],
        config_matrix_hash: str,
        metrics_to_compute: list[str],
        top_k: int = 10,
        description: str | None = None,
        evaluation_unit: str = "query",
        limits: dict[str, Any] | None = None,
        collection_snapshot_hash: str | None = None,
        reproducibility_metadata: dict[str, Any] | None = None,
    ) -> Benchmark:
        """Create a new benchmark.

        Args:
            name: Benchmark name
            owner_id: ID of the user creating the benchmark
            mapping_id: ID of the dataset-collection mapping
            config_matrix: Configuration matrix for the benchmark
            config_matrix_hash: Hash of the config matrix
            metrics_to_compute: List of metrics to compute
            top_k: Number of results to consider
            description: Optional description
            evaluation_unit: Unit of evaluation ("query" or "document")
            limits: Optional resource limits
            collection_snapshot_hash: Hash of the collection state
            reproducibility_metadata: Metadata for reproducibility

        Returns:
            Created Benchmark instance

        Raises:
            ValidationError: If validation fails
            EntityNotFoundError: If mapping not found
            DatabaseOperationError: For database errors
        """
        if not name or not name.strip():
            raise ValidationError("Benchmark name is required", "name")

        if top_k < 1 or top_k > 100:
            raise ValidationError("top_k must be between 1 and 100", "top_k")

        if not config_matrix:
            raise ValidationError("Config matrix is required", "config_matrix")

        if not metrics_to_compute:
            raise ValidationError("At least one metric is required", "metrics_to_compute")

        # Verify mapping exists
        mapping_exists = await self.session.scalar(
            select(func.count())
            .select_from(BenchmarkDatasetMapping)
            .where(BenchmarkDatasetMapping.id == mapping_id)
        )
        if not mapping_exists:
            raise EntityNotFoundError("benchmark_dataset_mapping", str(mapping_id))

        benchmark = Benchmark(
            id=str(uuid4()),
            name=name.strip(),
            description=description,
            owner_id=owner_id,
            mapping_id=mapping_id,
            evaluation_unit=evaluation_unit,
            config_matrix=config_matrix,
            config_matrix_hash=config_matrix_hash,
            limits=limits,
            collection_snapshot_hash=collection_snapshot_hash,
            reproducibility_metadata=reproducibility_metadata,
            top_k=top_k,
            metrics_to_compute=metrics_to_compute,
            status=BenchmarkStatus.PENDING.value,
            total_runs=0,
            completed_runs=0,
            failed_runs=0,
        )

        try:
            self.session.add(benchmark)
            await self.session.flush()
            logger.info("Created benchmark %s for user %d", benchmark.id, owner_id)
            return benchmark
        except Exception as exc:
            logger.error("Failed to create benchmark: %s", exc)
            raise DatabaseOperationError("create", "benchmark", str(exc)) from exc

    async def get_by_uuid(self, benchmark_uuid: str) -> Benchmark | None:
        """Get a benchmark by UUID.

        Args:
            benchmark_uuid: UUID of the benchmark

        Returns:
            Benchmark instance or None if not found
        """
        stmt: Select[tuple[Benchmark]] = (
            select(Benchmark)
            .where(Benchmark.id == benchmark_uuid)
            .options(
                selectinload(Benchmark.owner),
                selectinload(Benchmark.mapping).selectinload(BenchmarkDatasetMapping.dataset),
                selectinload(Benchmark.mapping).selectinload(BenchmarkDatasetMapping.collection),
                selectinload(Benchmark.runs),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_uuid_for_user(
        self,
        benchmark_uuid: str,
        user_id: int,
    ) -> Benchmark:
        """Get a benchmark by UUID with ownership check.

        Args:
            benchmark_uuid: UUID of the benchmark
            user_id: ID of the user requesting access

        Returns:
            Benchmark instance

        Raises:
            EntityNotFoundError: If benchmark not found
            AccessDeniedError: If user doesn't own the benchmark
        """
        benchmark = await self.get_by_uuid(benchmark_uuid)

        if not benchmark:
            raise EntityNotFoundError("benchmark", benchmark_uuid)

        if benchmark.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "benchmark", benchmark_uuid)

        return benchmark

    async def list_for_user(
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
        try:
            # Build query
            stmt: Select[tuple[Benchmark]] = (
                select(Benchmark)
                .where(Benchmark.owner_id == user_id)
                .options(
                    selectinload(Benchmark.mapping).selectinload(BenchmarkDatasetMapping.dataset),
                    selectinload(Benchmark.mapping).selectinload(BenchmarkDatasetMapping.collection),
                )
                .order_by(Benchmark.created_at.desc())
            )

            if status_filter is not None:
                stmt = stmt.where(Benchmark.status == status_filter.value)

            # Get total count
            count_stmt = select(func.count(Benchmark.id)).where(Benchmark.owner_id == user_id)
            if status_filter is not None:
                count_stmt = count_stmt.where(Benchmark.status == status_filter.value)
            total = await self.session.scalar(count_stmt) or 0

            # Apply pagination
            stmt = stmt.offset(offset).limit(limit)
            benchmarks = list((await self.session.execute(stmt)).scalars().all())

            return benchmarks, total
        except Exception as exc:
            logger.error("Failed to list benchmarks: %s", exc)
            raise DatabaseOperationError("list", "benchmarks", str(exc)) from exc

    async def update_status(
        self,
        benchmark_uuid: str,
        status: BenchmarkStatus,
        completed_runs: int | None = None,
        failed_runs: int | None = None,
    ) -> Benchmark:
        """Update benchmark status.

        Args:
            benchmark_uuid: UUID of the benchmark
            status: New status
            completed_runs: Number of completed runs (optional)
            failed_runs: Number of failed runs (optional)

        Returns:
            Updated Benchmark instance

        Raises:
            EntityNotFoundError: If benchmark not found
        """
        benchmark = await self.get_by_uuid(benchmark_uuid)
        if not benchmark:
            raise EntityNotFoundError("benchmark", benchmark_uuid)

        now = datetime.now(UTC)
        benchmark.status = status.value

        if status == BenchmarkStatus.RUNNING and benchmark.started_at is None:
            benchmark.started_at = now
        elif status in (BenchmarkStatus.COMPLETED, BenchmarkStatus.FAILED):
            benchmark.completed_at = now
        elif status == BenchmarkStatus.CANCELLED:
            benchmark.cancelled_at = now

        if completed_runs is not None:
            benchmark.completed_runs = completed_runs
        if failed_runs is not None:
            benchmark.failed_runs = failed_runs

        await self.session.flush()
        logger.info("Updated benchmark %s status to %s", benchmark_uuid, status.value)
        return benchmark

    async def set_operation(
        self,
        benchmark_uuid: str,
        operation_uuid: str | None,
    ) -> Benchmark:
        """Link the benchmark to a backing operation.

        Args:
            benchmark_uuid: UUID of the benchmark
            operation_uuid: UUID of the operation (or None to unlink)

        Returns:
            Updated Benchmark instance

        Raises:
            EntityNotFoundError: If benchmark not found
        """
        benchmark = await self.get_by_uuid(benchmark_uuid)
        if not benchmark:
            raise EntityNotFoundError("benchmark", benchmark_uuid)

        benchmark.operation_uuid = operation_uuid
        await self.session.flush()
        return benchmark

    async def set_total_runs(
        self,
        benchmark_uuid: str,
        total_runs: int,
    ) -> Benchmark:
        """Set the total number of runs for a benchmark.

        Args:
            benchmark_uuid: UUID of the benchmark
            total_runs: Total number of runs

        Returns:
            Updated Benchmark instance

        Raises:
            EntityNotFoundError: If benchmark not found
        """
        benchmark = await self.get_by_uuid(benchmark_uuid)
        if not benchmark:
            raise EntityNotFoundError("benchmark", benchmark_uuid)

        benchmark.total_runs = total_runs
        await self.session.flush()
        return benchmark

    async def delete(
        self,
        benchmark_uuid: str,
        user_id: int,
    ) -> None:
        """Delete a benchmark.

        Args:
            benchmark_uuid: UUID of the benchmark to delete
            user_id: ID of the user requesting deletion

        Raises:
            EntityNotFoundError: If benchmark not found
            AccessDeniedError: If user doesn't own the benchmark
        """
        benchmark = await self.get_by_uuid_for_user(benchmark_uuid, user_id)

        stmt = delete(Benchmark).where(Benchmark.id == benchmark.id)
        await self.session.execute(stmt)
        await self.session.flush()
        logger.info("Deleted benchmark %s", benchmark_uuid)

    # =========================================================================
    # Run CRUD
    # =========================================================================

    async def create_run(
        self,
        benchmark_id: str,
        run_order: int,
        config_hash: str,
        config: dict[str, Any],
    ) -> BenchmarkRun:
        """Create a new benchmark run.

        Args:
            benchmark_id: UUID of the benchmark
            run_order: Order of the run within the benchmark
            config_hash: Hash of the configuration
            config: Configuration for this run

        Returns:
            Created BenchmarkRun instance

        Raises:
            EntityNotFoundError: If benchmark not found
            DatabaseOperationError: For database errors
        """
        # Verify benchmark exists
        benchmark = await self.get_by_uuid(benchmark_id)
        if not benchmark:
            raise EntityNotFoundError("benchmark", benchmark_id)

        run = BenchmarkRun(
            id=str(uuid4()),
            benchmark_id=benchmark_id,
            run_order=run_order,
            config_hash=config_hash,
            config=config,
            status=BenchmarkRunStatus.PENDING.value,
        )

        try:
            self.session.add(run)
            await self.session.flush()
            logger.debug("Created run %s for benchmark %s", run.id, benchmark_id)
            return run
        except Exception as exc:
            logger.error("Failed to create benchmark run: %s", exc)
            raise DatabaseOperationError("create", "benchmark_run", str(exc)) from exc

    async def get_run(self, run_id: str) -> BenchmarkRun | None:
        """Get a benchmark run by ID.

        Args:
            run_id: UUID of the run

        Returns:
            BenchmarkRun instance or None
        """
        stmt: Select[tuple[BenchmarkRun]] = (
            select(BenchmarkRun)
            .where(BenchmarkRun.id == run_id)
            .options(
                selectinload(BenchmarkRun.benchmark),
                selectinload(BenchmarkRun.metrics),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_runs_for_benchmark(
        self,
        benchmark_id: str,
    ) -> list[BenchmarkRun]:
        """Get all runs for a benchmark.

        Args:
            benchmark_id: UUID of the benchmark

        Returns:
            List of BenchmarkRun instances
        """
        stmt: Select[tuple[BenchmarkRun]] = (
            select(BenchmarkRun)
            .where(BenchmarkRun.benchmark_id == benchmark_id)
            .options(selectinload(BenchmarkRun.metrics))
            .order_by(BenchmarkRun.run_order)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_run_status(
        self,
        run_id: str,
        status: BenchmarkRunStatus,
        *,
        status_message: str | None = None,
        indexing_duration_ms: int | None = None,
        evaluation_duration_ms: int | None = None,
        total_duration_ms: int | None = None,
        error_message: str | None = None,
        dense_collection_name: str | None = None,
        sparse_collection_name: str | None = None,
    ) -> BenchmarkRun:
        """Update a run's status and timing information.

        Args:
            run_id: UUID of the run
            status: New status
            status_message: Optional status message
            indexing_duration_ms: Indexing phase duration
            evaluation_duration_ms: Evaluation phase duration
            total_duration_ms: Total duration
            error_message: Error message (for failed runs)
            dense_collection_name: Name of the dense Qdrant collection
            sparse_collection_name: Name of the sparse Qdrant collection

        Returns:
            Updated BenchmarkRun instance

        Raises:
            EntityNotFoundError: If run not found
        """
        run = await self.get_run(run_id)
        if not run:
            raise EntityNotFoundError("benchmark_run", run_id)

        now = datetime.now(UTC)
        run.status = status.value

        if status == BenchmarkRunStatus.INDEXING and run.started_at is None:
            run.started_at = now
        elif status in (BenchmarkRunStatus.COMPLETED, BenchmarkRunStatus.FAILED):
            run.completed_at = now

        if status_message is not None:
            run.status_message = status_message
        if indexing_duration_ms is not None:
            run.indexing_duration_ms = indexing_duration_ms
        if evaluation_duration_ms is not None:
            run.evaluation_duration_ms = evaluation_duration_ms
        if total_duration_ms is not None:
            run.total_duration_ms = total_duration_ms
        if error_message is not None:
            run.error_message = error_message
        if dense_collection_name is not None:
            run.dense_collection_name = dense_collection_name
        if sparse_collection_name is not None:
            run.sparse_collection_name = sparse_collection_name

        await self.session.flush()
        return run

    # =========================================================================
    # Metrics CRUD
    # =========================================================================

    async def add_run_metric(
        self,
        run_id: str,
        metric_name: str,
        metric_value: float,
        k_value: int | None = None,
    ) -> BenchmarkRunMetric:
        """Add a metric to a benchmark run.

        Args:
            run_id: UUID of the run
            metric_name: Name of the metric (e.g., "precision", "recall")
            metric_value: Computed metric value
            k_value: k value for @k metrics (optional)

        Returns:
            Created BenchmarkRunMetric instance

        Raises:
            EntityNotFoundError: If run not found
        """
        run = await self.get_run(run_id)
        if not run:
            raise EntityNotFoundError("benchmark_run", run_id)

        metric = BenchmarkRunMetric(
            run_id=run_id,
            metric_name=metric_name,
            k_value=k_value,
            metric_value=metric_value,
        )

        try:
            self.session.add(metric)
            await self.session.flush()
            return metric
        except Exception as exc:
            logger.error("Failed to add metric to run %s: %s", run_id, exc)
            raise DatabaseOperationError("create", "benchmark_run_metric", str(exc)) from exc

    async def get_metrics_for_run(
        self,
        run_id: str,
    ) -> list[BenchmarkRunMetric]:
        """Get all metrics for a run.

        Args:
            run_id: UUID of the run

        Returns:
            List of BenchmarkRunMetric instances
        """
        stmt: Select[tuple[BenchmarkRunMetric]] = (
            select(BenchmarkRunMetric)
            .where(BenchmarkRunMetric.run_id == run_id)
            .order_by(BenchmarkRunMetric.metric_name, BenchmarkRunMetric.k_value)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # =========================================================================
    # Query Results CRUD
    # =========================================================================

    async def add_query_result(
        self,
        run_id: str,
        query_id: int,
        retrieved_doc_ids: list[str],
        *,
        retrieved_debug: dict[str, Any] | None = None,
        precision_at_k: float | None = None,
        recall_at_k: float | None = None,
        reciprocal_rank: float | None = None,
        ndcg_at_k: float | None = None,
        search_time_ms: int | None = None,
        rerank_time_ms: int | None = None,
    ) -> BenchmarkQueryResult:
        """Add a query result to a benchmark run.

        Args:
            run_id: UUID of the run
            query_id: ID of the benchmark query
            retrieved_doc_ids: List of retrieved document IDs
            retrieved_debug: Optional debug information
            precision_at_k: Precision at k
            recall_at_k: Recall at k
            reciprocal_rank: Reciprocal rank
            ndcg_at_k: NDCG at k
            search_time_ms: Search execution time
            rerank_time_ms: Reranking time

        Returns:
            Created BenchmarkQueryResult instance

        Raises:
            EntityNotFoundError: If run or query not found
        """
        # Verify run exists
        run = await self.get_run(run_id)
        if not run:
            raise EntityNotFoundError("benchmark_run", run_id)

        # Verify query exists
        query_exists = await self.session.scalar(
            select(func.count())
            .select_from(BenchmarkQuery)
            .where(BenchmarkQuery.id == query_id)
        )
        if not query_exists:
            raise EntityNotFoundError("benchmark_query", str(query_id))

        result = BenchmarkQueryResult(
            run_id=run_id,
            benchmark_query_id=query_id,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_debug=retrieved_debug,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            reciprocal_rank=reciprocal_rank,
            ndcg_at_k=ndcg_at_k,
            search_time_ms=search_time_ms,
            rerank_time_ms=rerank_time_ms,
        )

        try:
            self.session.add(result)
            await self.session.flush()
            return result
        except Exception as exc:
            logger.error("Failed to add query result: %s", exc)
            raise DatabaseOperationError("create", "benchmark_query_result", str(exc)) from exc

    async def get_query_results_for_run(
        self,
        run_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[BenchmarkQueryResult], int]:
        """Get query results for a run with pagination.

        Args:
            run_id: UUID of the run
            offset: Pagination offset
            limit: Maximum number of results

        Returns:
            Tuple of (results list, total count)
        """
        stmt: Select[tuple[BenchmarkQueryResult]] = (
            select(BenchmarkQueryResult)
            .where(BenchmarkQueryResult.run_id == run_id)
            .options(selectinload(BenchmarkQueryResult.query))
            .order_by(BenchmarkQueryResult.benchmark_query_id)
            .offset(offset)
            .limit(limit)
        )
        results = list((await self.session.execute(stmt)).scalars().all())

        count_stmt = select(func.count(BenchmarkQueryResult.id)).where(
            BenchmarkQueryResult.run_id == run_id
        )
        total = await self.session.scalar(count_stmt) or 0

        return results, total

    # =========================================================================
    # Aggregation Helpers
    # =========================================================================

    async def get_aggregated_results(
        self,
        benchmark_id: str,
    ) -> dict[str, Any]:
        """Get aggregated results for a benchmark.

        Aggregates metrics across all completed runs for comparison.

        Args:
            benchmark_id: UUID of the benchmark

        Returns:
            Dictionary with aggregated results:
            {
                "benchmark_id": "...",
                "runs": [
                    {
                        "run_id": "...",
                        "config": {...},
                        "status": "completed",
                        "metrics": {
                            "precision@5": 0.75,
                            "recall@10": 0.85,
                            ...
                        },
                        "timing": {
                            "indexing_ms": 1234,
                            "evaluation_ms": 5678,
                            "total_ms": 6912
                        }
                    },
                    ...
                ],
                "summary": {
                    "total_runs": 10,
                    "completed_runs": 8,
                    "failed_runs": 2,
                    "best_run": "...",
                    "best_metrics": {...}
                }
            }
        """
        benchmark = await self.get_by_uuid(benchmark_id)
        if not benchmark:
            raise EntityNotFoundError("benchmark", benchmark_id)

        runs = await self.get_runs_for_benchmark(benchmark_id)

        runs_data = []
        best_run_id = None
        best_metric_value = -1.0

        for run in runs:
            metrics = await self.get_metrics_for_run(run.id)

            # Build metrics dict
            metrics_dict: dict[str, float] = {}
            for m in metrics:
                key = m.metric_name
                if m.k_value is not None:
                    key = f"{m.metric_name}@{m.k_value}"
                metrics_dict[key] = m.metric_value

            run_data = {
                "run_id": run.id,
                "run_order": run.run_order,
                "config": run.config,
                "config_hash": run.config_hash,
                "status": run.status,
                "error_message": run.error_message,
                "metrics": metrics_dict,
                "timing": {
                    "indexing_ms": run.indexing_duration_ms,
                    "evaluation_ms": run.evaluation_duration_ms,
                    "total_ms": run.total_duration_ms,
                },
            }
            runs_data.append(run_data)

            # Track best run by primary metric (ndcg@10 or first available)
            primary_metric = metrics_dict.get("ndcg@10") or metrics_dict.get("ndcg@5")
            if primary_metric and primary_metric > best_metric_value:
                best_metric_value = primary_metric
                best_run_id = run.id

        return {
            "benchmark_id": benchmark_id,
            "runs": runs_data,
            "summary": {
                "total_runs": benchmark.total_runs,
                "completed_runs": benchmark.completed_runs,
                "failed_runs": benchmark.failed_runs,
                "best_run": best_run_id,
                "best_primary_metric": best_metric_value if best_metric_value >= 0 else None,
            },
        }
