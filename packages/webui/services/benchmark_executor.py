"""Benchmark Execution Engine for orchestrating benchmark runs.

This module provides the BenchmarkExecutor class which coordinates the execution
of benchmark evaluations across multiple configuration runs. It handles:
- Sequential run processing for predictable GPU memory management
- Progress reporting via Redis pub/sub
- Idempotent execution (skips completed runs on retry)
- Error isolation (individual run failures don't stop the benchmark)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

from shared.benchmarks.evaluator import ConfigurationEvaluator
from shared.benchmarks.exceptions import BenchmarkEvaluationError
from shared.benchmarks.types import ConfigurationEvaluationResult, RelevanceJudgment, RetrievedChunk, SearchTiming
from shared.database.models import Benchmark, BenchmarkRunStatus, BenchmarkStatus, Collection

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.repositories.benchmark_dataset_repository import BenchmarkDatasetRepository
    from shared.database.repositories.benchmark_repository import BenchmarkRepository
    from shared.database.repositories.collection_repository import CollectionRepository
    from webui.tasks.utils import CeleryTaskWithOperationUpdates

    from .search_service import SearchService

logger = logging.getLogger(__name__)


class BenchmarkExecutor:
    """Orchestrates benchmark execution with idempotent run processing.

    This executor handles the complete lifecycle of a benchmark evaluation:
    1. Loading benchmark configuration and queries
    2. Iterating through runs sequentially
    3. Executing searches via SearchService
    4. Computing metrics via ConfigurationEvaluator
    5. Persisting results to the database
    6. Reporting progress via Redis pub/sub

    The executor is designed for idempotent operation - runs that are already
    COMPLETED or FAILED are skipped on retry, allowing recovery from failures.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        benchmark_repo: BenchmarkRepository,
        benchmark_dataset_repo: BenchmarkDatasetRepository,
        collection_repo: CollectionRepository,
        search_service: SearchService,
        progress_reporter: CeleryTaskWithOperationUpdates,
    ) -> None:
        """Initialize the executor with required dependencies.

        Args:
            db_session: AsyncSession for database operations
            benchmark_repo: Repository for benchmark operations
            benchmark_dataset_repo: Repository for dataset operations
            collection_repo: Repository for collection operations
            search_service: Service for executing searches
            progress_reporter: Helper for sending progress updates
        """
        self.db_session = db_session
        self.benchmark_repo = benchmark_repo
        self.benchmark_dataset_repo = benchmark_dataset_repo
        self.collection_repo = collection_repo
        self.search_service = search_service
        self.progress_reporter = progress_reporter
        self.evaluator = ConfigurationEvaluator()

    async def execute_benchmark(self, benchmark_id: str) -> dict[str, Any]:
        """Execute a complete benchmark evaluation.

        This is the main entry point for benchmark execution. It:
        1. Loads the benchmark and associated data
        2. Iterates through all runs sequentially
        3. Skips completed/failed runs for idempotency
        4. Handles errors per-run without stopping the benchmark
        5. Updates benchmark status based on results

        Args:
            benchmark_id: UUID of the benchmark to execute

        Returns:
            Dictionary with execution summary:
            {
                "benchmark_id": "...",
                "status": "completed|failed|cancelled",
                "total_runs": 10,
                "completed_runs": 8,
                "failed_runs": 2,
                "skipped_runs": 0,
                "error": None  # or error message if all failed
            }
        """
        # Load benchmark
        benchmark = await self.benchmark_repo.get_by_uuid(benchmark_id)
        if not benchmark:
            raise BenchmarkEvaluationError(f"Benchmark not found: {benchmark_id}")

        # Update status to RUNNING
        await self.benchmark_repo.update_status(benchmark_id, BenchmarkStatus.RUNNING)
        await self.db_session.commit()

        # Load mapping and collection
        mapping = await self.benchmark_dataset_repo.get_mapping(cast(int, benchmark.mapping_id))
        if not mapping:
            raise BenchmarkEvaluationError(f"Mapping not found: {benchmark.mapping_id}")

        collection = await self.collection_repo.get_by_uuid(cast(str, mapping.collection_id))
        if not collection:
            raise BenchmarkEvaluationError(f"Collection not found: {mapping.collection_id}")

        # Load queries and relevance data
        dataset_id = cast(str, mapping.dataset_id)
        queries = await self.benchmark_dataset_repo.get_queries_for_dataset(dataset_id)
        relevance_data = await self.benchmark_dataset_repo.get_relevance_for_mapping(cast(int, mapping.id))

        # Build relevance map: query_id -> list[RelevanceJudgment]
        relevance_by_query = self._build_relevance_map(relevance_data)

        # Convert queries to format expected by evaluator
        query_dicts = [{"id": q.id, "query_text": q.query_text} for q in queries]

        # Get k values for metrics (from config_matrix or default)
        k_values = self._get_k_values(benchmark)
        top_k = cast(int, benchmark.top_k) or 10

        # Load runs
        runs = await self.benchmark_repo.get_runs_for_benchmark(benchmark_id)

        completed_count = 0
        failed_count = 0
        skipped_count = 0
        total_runs = len(runs)

        # Send initial progress
        await self._send_progress_update(
            benchmark_id=benchmark_id,
            total_runs=total_runs,
            completed_runs=0,
            current_run=None,
            stage="starting",
        )

        # Process runs sequentially
        for run in runs:
            # Check for cancellation
            if await self._check_cancellation(benchmark_id):
                logger.info("Benchmark %s cancelled, stopping execution", benchmark_id)
                break

            # Skip completed/failed runs for idempotency
            run_status = cast(str, run.status)
            if run_status in (BenchmarkRunStatus.COMPLETED.value, BenchmarkRunStatus.FAILED.value):
                if run_status == BenchmarkRunStatus.COMPLETED.value:
                    completed_count += 1
                else:
                    failed_count += 1
                skipped_count += 1
                logger.debug("Skipping run %s (status: %s)", run.id, run_status)
                continue

            # Execute the run
            try:
                await self._execute_run(
                    run=run,
                    benchmark=benchmark,
                    collection=collection,
                    queries=query_dicts,
                    relevance_by_query=relevance_by_query,
                    k_values=k_values,
                    top_k=top_k,
                    total_runs=total_runs,
                    completed_so_far=completed_count + failed_count,
                )
                completed_count += 1
            except Exception as exc:
                logger.error("Run %s failed: %s", run.id, exc, exc_info=True)
                failed_count += 1

                # Mark run as failed
                await self.benchmark_repo.update_run_status(
                    run_id=str(run.id),
                    status=BenchmarkRunStatus.FAILED,
                    error_message=str(exc)[:1000],  # Truncate long errors
                )
                await self.db_session.commit()

            # Update benchmark progress
            await self.benchmark_repo.update_status(
                benchmark_id,
                BenchmarkStatus.RUNNING,
                completed_runs=completed_count,
                failed_runs=failed_count,
            )
            await self.db_session.commit()

        # Determine final status
        final_status = await self._determine_final_status(benchmark_id, completed_count, failed_count)

        # Update benchmark with final status
        await self.benchmark_repo.update_status(
            benchmark_id,
            final_status,
            completed_runs=completed_count,
            failed_runs=failed_count,
        )
        await self.db_session.commit()

        # Send completion progress
        await self._send_progress_update(
            benchmark_id=benchmark_id,
            total_runs=total_runs,
            completed_runs=completed_count + failed_count,
            current_run=None,
            stage="completed",
        )

        error_msg = None
        if final_status == BenchmarkStatus.FAILED:
            error_msg = "All benchmark runs failed"
        elif final_status == BenchmarkStatus.CANCELLED:
            error_msg = "Benchmark was cancelled"

        return {
            "benchmark_id": benchmark_id,
            "status": final_status.value,
            "total_runs": total_runs,
            "completed_runs": completed_count,
            "failed_runs": failed_count,
            "skipped_runs": skipped_count,
            "error": error_msg,
        }

    async def _execute_run(
        self,
        run: Any,
        benchmark: Benchmark,
        collection: Collection,
        queries: list[dict[str, Any]],
        relevance_by_query: dict[int, list[RelevanceJudgment]],
        k_values: list[int],
        top_k: int,
        total_runs: int,
        completed_so_far: int,
    ) -> None:
        """Execute a single benchmark run.

        Args:
            run: BenchmarkRun instance
            benchmark: Parent Benchmark instance
            collection: Collection to search
            queries: List of query dicts
            relevance_by_query: Relevance judgments by query ID
            k_values: k values for metrics
            top_k: Number of results to retrieve
            total_runs: Total number of runs in benchmark
            completed_so_far: Number of runs already processed
        """
        run_id = str(run.id)
        run_config = cast(dict[str, Any], run.config) or {}

        # Extract search parameters from run config
        search_mode = run_config.get("search_mode", "dense")
        use_reranker = run_config.get("use_reranker", False)
        run_top_k = run_config.get("top_k", top_k)
        rrf_k = run_config.get("rrf_k", 60)
        score_threshold = run_config.get("score_threshold")

        config_summary = self._get_config_summary(run_config)

        # Update run status to INDEXING (preparation phase)
        await self.benchmark_repo.update_run_status(
            run_id=run_id,
            status=BenchmarkRunStatus.INDEXING,
            status_message="Preparing evaluation",
        )
        await self.db_session.commit()

        start_time = time.perf_counter()
        indexing_duration = 0  # No indexing needed for evaluation-only benchmarks

        # Send progress update
        await self._send_progress_update(
            benchmark_id=str(benchmark.id),
            total_runs=total_runs,
            completed_runs=completed_so_far,
            current_run={
                "run_id": run_id,
                "config_summary": config_summary,
                "total_queries": len(queries),
                "completed_queries": 0,
                "stage": "indexing",
            },
        )

        # Update to EVALUATING
        await self.benchmark_repo.update_run_status(
            run_id=run_id,
            status=BenchmarkRunStatus.EVALUATING,
            status_message="Running queries",
            indexing_duration_ms=indexing_duration,
        )
        await self.db_session.commit()

        # Create search function for the evaluator
        search_func = self._create_search_func(
            collection=collection,
            search_mode=search_mode,
            use_reranker=use_reranker,
            rrf_k=rrf_k,
            score_threshold=score_threshold,
        )

        eval_start = time.perf_counter()

        # Send evaluating progress
        await self._send_progress_update(
            benchmark_id=str(benchmark.id),
            total_runs=total_runs,
            completed_runs=completed_so_far,
            current_run={
                "run_id": run_id,
                "config_summary": config_summary,
                "total_queries": len(queries),
                "completed_queries": 0,
                "stage": "evaluating",
            },
        )

        # Run evaluation
        config_hash = cast(str, run.config_hash)
        eval_result = await self.evaluator.evaluate_configuration(
            config_hash=config_hash,
            queries=queries,
            relevance_by_query=relevance_by_query,
            search_func=search_func,
            k_values=k_values,
            top_k=run_top_k,
            include_debug=False,
        )

        eval_duration = int((time.perf_counter() - eval_start) * 1000)
        total_duration = int((time.perf_counter() - start_time) * 1000)

        # Store results
        await self._store_run_results(run_id, eval_result, k_values)

        # Update run status to COMPLETED
        await self.benchmark_repo.update_run_status(
            run_id=run_id,
            status=BenchmarkRunStatus.COMPLETED,
            status_message="Evaluation complete",
            evaluation_duration_ms=eval_duration,
            total_duration_ms=total_duration,
        )
        await self.db_session.commit()

        logger.info(
            "Run %s completed: %d queries, %d ms total",
            run_id,
            eval_result.total_queries,
            total_duration,
        )

    def _create_search_func(
        self,
        collection: Collection,
        search_mode: str,
        use_reranker: bool,
        rrf_k: int,
        score_threshold: float | None,
    ) -> Any:
        """Create a search function closure for the ConfigurationEvaluator.

        Args:
            collection: Collection to search
            search_mode: Search mode (dense, sparse, hybrid)
            use_reranker: Whether to use reranking
            rrf_k: RRF k constant
            score_threshold: Minimum score threshold

        Returns:
            Async function matching SearchFunc signature
        """

        async def search_func(query_text: str, top_k: int) -> tuple[list[RetrievedChunk], SearchTiming]:
            """Execute search and return formatted results."""
            result = await self.search_service.benchmark_search(
                collection=collection,
                query=query_text,
                search_mode=search_mode,  # type: ignore[arg-type]
                use_reranker=use_reranker,
                top_k=top_k,
                rrf_k=rrf_k,
                score_threshold=score_threshold,
            )

            # Convert BenchmarkSearchResult to expected types
            chunks: list[RetrievedChunk] = []
            for chunk in result.chunks:
                chunks.append(
                    RetrievedChunk(
                        doc_id=chunk.get("doc_id", ""),
                        chunk_id=chunk.get("chunk_id", ""),
                        score=chunk.get("score", 0.0),
                    )
                )

            timing = SearchTiming(
                search_time_ms=result.search_time_ms,
                rerank_time_ms=result.rerank_time_ms,
            )

            return chunks, timing

        return search_func

    async def _store_run_results(
        self,
        run_id: str,
        eval_result: ConfigurationEvaluationResult,
        k_values: list[int],
    ) -> None:
        """Persist evaluation results to the database.

        Args:
            run_id: UUID of the run
            eval_result: ConfigurationEvaluationResult from evaluator
            k_values: k values used for metrics
        """
        # Store aggregate metrics
        for metric in eval_result.aggregate_metrics:
            await self.benchmark_repo.add_run_metric(
                run_id=run_id,
                metric_name=metric.name,
                metric_value=metric.value,
                k_value=metric.k_value,
            )

        # Store per-query results (with individual metrics)
        for query_result in eval_result.per_query_results:
            # Extract per-query metrics at the primary k value
            primary_k = k_values[0] if k_values else 10

            precision_at_k = None
            recall_at_k = None
            ndcg_at_k = None
            mrr = None

            for m in query_result.metrics:
                if m.name == "precision" and m.k_value == primary_k:
                    precision_at_k = m.value
                elif m.name == "recall" and m.k_value == primary_k:
                    recall_at_k = m.value
                elif m.name == "ndcg" and m.k_value == primary_k:
                    ndcg_at_k = m.value
                elif m.name == "mrr":
                    mrr = m.value

            await self.benchmark_repo.add_query_result(
                run_id=run_id,
                query_id=query_result.query_id,
                retrieved_doc_ids=query_result.retrieved_doc_ids,
                retrieved_debug=query_result.retrieved_debug,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
                reciprocal_rank=mrr,
                ndcg_at_k=ndcg_at_k,
                search_time_ms=query_result.search_time_ms,
                rerank_time_ms=query_result.rerank_time_ms,
            )

        await self.db_session.flush()

    def _build_relevance_map(
        self,
        relevance_data: list[Any],
    ) -> dict[int, list[RelevanceJudgment]]:
        """Build a mapping from query ID to relevance judgments.

        Args:
            relevance_data: List of BenchmarkRelevance instances

        Returns:
            Dict mapping query_id to list of RelevanceJudgment
        """
        relevance_by_query: dict[int, list[RelevanceJudgment]] = defaultdict(list)

        for rel in relevance_data:
            query_id = cast(int, rel.benchmark_query_id)

            # Use resolved document ID if available, otherwise skip
            doc_id = rel.resolved_document_id
            if not doc_id:
                continue

            judgment = RelevanceJudgment(
                doc_id=str(doc_id),
                relevance_grade=cast(int, rel.relevance_grade),
            )
            relevance_by_query[query_id].append(judgment)

        return dict(relevance_by_query)

    def _get_k_values(self, benchmark: Benchmark) -> list[int]:
        """Extract k values for metrics from benchmark configuration.

        Args:
            benchmark: Benchmark instance

        Returns:
            List of k values (e.g., [5, 10, 20])
        """
        config_matrix = cast(dict[str, Any], benchmark.config_matrix) or {}
        k_values = config_matrix.get("k_values_for_metrics", [5, 10, 20])

        if not k_values:
            k_values = [5, 10, 20]

        return cast(list[int], k_values)

    def _get_config_summary(self, config: dict[str, Any]) -> str:
        """Create a human-readable summary of the run configuration.

        Args:
            config: Run configuration dict

        Returns:
            Summary string (e.g., "dense + reranker")
        """
        parts = []

        search_mode = config.get("search_mode", "dense")
        parts.append(search_mode)

        if config.get("use_reranker"):
            parts.append("reranker")

        top_k = config.get("top_k")
        if top_k:
            parts.append(f"k={top_k}")

        return " + ".join(parts)

    async def _send_progress_update(
        self,
        benchmark_id: str,
        total_runs: int,
        completed_runs: int,
        current_run: dict[str, Any] | None,
        stage: str = "evaluating",
    ) -> None:
        """Send progress update via Redis pub/sub.

        Args:
            benchmark_id: UUID of the benchmark
            total_runs: Total number of runs
            completed_runs: Number of completed runs
            current_run: Current run info or None
            stage: Current stage (starting, evaluating, completed)
        """
        progress_data = {
            "benchmark_id": benchmark_id,
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "current_run": current_run,
            "stage": stage,
        }

        await self.progress_reporter.send_update("benchmark_progress", progress_data)

    async def _check_cancellation(self, benchmark_id: str) -> bool:
        """Check if the benchmark has been cancelled.

        Args:
            benchmark_id: UUID of the benchmark

        Returns:
            True if cancelled, False otherwise
        """
        # Refresh benchmark from database
        benchmark = await self.benchmark_repo.get_by_uuid(benchmark_id)
        if not benchmark:
            return True  # Treat as cancelled if not found

        status_str = cast(str, benchmark.status)
        return bool(status_str == BenchmarkStatus.CANCELLED.value)

    async def _determine_final_status(
        self,
        benchmark_id: str,
        completed_count: int,
        failed_count: int,
    ) -> BenchmarkStatus:
        """Determine the final benchmark status.

        Args:
            benchmark_id: UUID of the benchmark
            completed_count: Number of completed runs
            failed_count: Number of failed runs

        Returns:
            Final BenchmarkStatus
        """
        # Check for cancellation first
        if await self._check_cancellation(benchmark_id):
            return BenchmarkStatus.CANCELLED

        # All failed -> FAILED
        if completed_count == 0 and failed_count > 0:
            return BenchmarkStatus.FAILED

        # At least some completed -> COMPLETED
        return BenchmarkStatus.COMPLETED


__all__ = ["BenchmarkExecutor"]
