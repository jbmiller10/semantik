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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from shared.benchmarks import parse_k_values
from shared.benchmarks.evaluator import ConfigurationEvaluator
from shared.benchmarks.exceptions import BenchmarkCancelledError, BenchmarkEvaluationError
from shared.benchmarks.types import (
    ConfigurationEvaluationResult,
    MetricResult,
    QueryEvaluationResult,
    RelevanceJudgment,
    RetrievedChunk,
    SearchTiming,
)
from shared.database.models import Benchmark, BenchmarkRunStatus, BenchmarkStatus, Collection

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.repositories.benchmark_dataset_repository import BenchmarkDatasetRepository
    from shared.database.repositories.benchmark_repository import BenchmarkRepository
    from shared.database.repositories.collection_repository import CollectionRepository
    from webui.tasks.utils import CeleryTaskWithOperationUpdates

    from .search_service import SearchService

logger = logging.getLogger(__name__)

BENCHMARK_PROGRESS_INTERVAL_QUERIES = 5
BENCHMARK_PROGRESS_INTERVAL_MS = 500
CANCELLED_RUN_ERROR_MESSAGE = "cancelled by user"


@dataclass
class ProgressContext:
    """Holds progress state for a benchmark execution to reduce parameter passing."""

    benchmark_id: str
    total_runs: int
    primary_k: int
    k_values: list[int]
    completed_runs: int = 0
    failed_runs: int = 0
    skipped_runs: int = 0

    def increment_completed(self) -> None:
        self.completed_runs += 1

    def increment_failed(self) -> None:
        self.failed_runs += 1

    def increment_skipped(self) -> None:
        self.skipped_runs += 1


@dataclass
class BenchmarkData:
    """Holds loaded benchmark data to avoid passing many parameters."""

    benchmark: Benchmark
    collection: Collection
    queries: list[dict[str, Any]]
    relevance_by_query: dict[int, list[RelevanceJudgment]]
    runs: list[Any]
    primary_k: int
    k_values: list[int] = field(default_factory=list)
    top_k: int = 10


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

    async def _load_benchmark_data(self, benchmark_id: str) -> BenchmarkData:
        """Load all data needed for benchmark execution.

        Args:
            benchmark_id: UUID of the benchmark to load

        Returns:
            BenchmarkData containing all loaded entities

        Raises:
            BenchmarkEvaluationError: If benchmark, mapping, or collection not found
        """
        benchmark = await self.benchmark_repo.get_by_uuid(benchmark_id)
        if not benchmark:
            raise BenchmarkEvaluationError(f"Benchmark not found: {benchmark_id}")

        mapping = await self.benchmark_dataset_repo.get_mapping(cast(int, benchmark.mapping_id))
        if not mapping:
            raise BenchmarkEvaluationError(f"Mapping not found: {benchmark.mapping_id}")

        collection = await self.collection_repo.get_by_uuid(cast(str, mapping.collection_id))
        if not collection:
            raise BenchmarkEvaluationError(f"Collection not found: {mapping.collection_id}")

        dataset_id = cast(str, mapping.dataset_id)
        queries = await self.benchmark_dataset_repo.get_queries_for_dataset(dataset_id)
        relevance_data = await self.benchmark_dataset_repo.get_relevance_for_mapping(cast(int, mapping.id))
        relevance_by_query = self._build_relevance_map(relevance_data)

        query_dicts = [{"id": q.id, "query_text": q.query_text} for q in queries]
        primary_k, k_values = self._get_metrics_k_values(benchmark)
        top_k = cast(int, benchmark.top_k) or 10
        runs = await self.benchmark_repo.get_runs_for_benchmark(benchmark_id)

        return BenchmarkData(
            benchmark=benchmark,
            collection=collection,
            queries=query_dicts,
            relevance_by_query=relevance_by_query,
            runs=runs,
            primary_k=primary_k,
            k_values=k_values,
            top_k=top_k,
        )

    def _build_execution_result(
        self,
        benchmark_id: str,
        status: BenchmarkStatus,
        ctx: ProgressContext,
    ) -> dict[str, Any]:
        """Build the execution result dictionary.

        Args:
            benchmark_id: UUID of the benchmark
            status: Final benchmark status
            ctx: Progress context with run counts

        Returns:
            Execution result dictionary
        """
        error_msg = None
        if status == BenchmarkStatus.FAILED:
            error_msg = "All benchmark runs failed"
        elif status == BenchmarkStatus.CANCELLED:
            error_msg = "Benchmark was cancelled"

        return {
            "benchmark_id": benchmark_id,
            "status": status.value,
            "total_runs": ctx.total_runs,
            "completed_runs": ctx.completed_runs,
            "failed_runs": ctx.failed_runs,
            "skipped_runs": ctx.skipped_runs,
            "error": error_msg,
        }

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
        # Handle pre-start cancellation
        if await self._check_cancellation(benchmark_id):
            return await self._handle_pre_start_cancellation(benchmark_id)

        # Update status to RUNNING (idempotent; guarded against overwriting CANCELLED)
        await self.benchmark_repo.update_status(benchmark_id, BenchmarkStatus.RUNNING)
        await self.db_session.commit()

        # Load all benchmark data
        data = await self._load_benchmark_data(benchmark_id)
        ctx = ProgressContext(
            benchmark_id=str(data.benchmark.id),
            total_runs=len(data.runs),
            primary_k=data.primary_k,
            k_values=data.k_values,
        )

        # Send initial progress
        await self._send_progress(ctx, BenchmarkStatus.RUNNING, "starting")

        # Process all runs
        await self._process_runs(data, ctx)

        # Finalize benchmark
        final_status = await self._determine_final_status(
            benchmark_id, ctx.completed_runs, ctx.failed_runs
        )
        await self.benchmark_repo.update_status(
            benchmark_id,
            final_status,
            completed_runs=ctx.completed_runs,
            failed_runs=ctx.failed_runs,
        )
        await self.db_session.commit()

        await self._send_progress(ctx, final_status, "completed")

        return self._build_execution_result(benchmark_id, final_status, ctx)

    async def _handle_pre_start_cancellation(self, benchmark_id: str) -> dict[str, Any]:
        """Handle cancellation that occurred before the worker started."""
        runs = await self.benchmark_repo.get_runs_for_benchmark(benchmark_id)
        completed_count = sum(
            1 for r in runs if cast(str, r.status) == BenchmarkRunStatus.COMPLETED.value
        )
        failed_count = sum(
            1 for r in runs if cast(str, r.status) == BenchmarkRunStatus.FAILED.value
        )

        if runs:
            failed_count = await self._mark_remaining_runs_cancelled(
                benchmark_id=benchmark_id,
                completed_runs=completed_count,
                runs=runs,
                starting_from_run_id=str(runs[0].id),
                existing_failed_count=failed_count,
            )

        ctx = ProgressContext(
            benchmark_id=benchmark_id,
            total_runs=len(runs),
            primary_k=10,
            k_values=[10],
            completed_runs=completed_count,
            failed_runs=failed_count,
        )
        return self._build_execution_result(benchmark_id, BenchmarkStatus.CANCELLED, ctx)

    async def _process_runs(self, data: BenchmarkData, ctx: ProgressContext) -> None:
        """Process all benchmark runs sequentially.

        Args:
            data: Loaded benchmark data
            ctx: Progress context for tracking state
        """
        for run in data.runs:
            if await self._check_cancellation(ctx.benchmark_id):
                logger.info("Benchmark %s cancelled, stopping execution", ctx.benchmark_id)
                ctx.failed_runs = await self._mark_remaining_runs_cancelled(
                    benchmark_id=ctx.benchmark_id,
                    completed_runs=ctx.completed_runs,
                    runs=data.runs,
                    starting_from_run_id=str(run.id),
                    existing_failed_count=ctx.failed_runs,
                )
                return

            run_status = cast(str, run.status)
            if run_status in (BenchmarkRunStatus.COMPLETED.value, BenchmarkRunStatus.FAILED.value):
                self._count_skipped_run(ctx, run_status)
                logger.debug("Skipping run %s (status: %s)", run.id, run_status)
                continue

            await self._process_single_run(run, data, ctx)

            await self.benchmark_repo.update_status(
                ctx.benchmark_id,
                BenchmarkStatus.RUNNING,
                completed_runs=ctx.completed_runs,
                failed_runs=ctx.failed_runs,
            )
            await self.db_session.commit()

    def _count_skipped_run(self, ctx: ProgressContext, run_status: str) -> None:
        """Count a skipped run based on its status."""
        if run_status == BenchmarkRunStatus.COMPLETED.value:
            ctx.increment_completed()
        else:
            ctx.increment_failed()
        ctx.increment_skipped()

    async def _process_single_run(
        self,
        run: Any,
        data: BenchmarkData,
        ctx: ProgressContext,
    ) -> None:
        """Process a single benchmark run with error handling.

        Args:
            run: The run to process
            data: Loaded benchmark data
            ctx: Progress context
        """
        try:
            await self._execute_run(run, data, ctx)
            ctx.increment_completed()
        except BenchmarkCancelledError:
            logger.info("Benchmark %s cancelled during run %s", ctx.benchmark_id, run.id)
            ctx.failed_runs = await self._mark_remaining_runs_cancelled(
                benchmark_id=ctx.benchmark_id,
                completed_runs=ctx.completed_runs,
                runs=data.runs,
                starting_from_run_id=str(run.id),
                existing_failed_count=ctx.failed_runs,
            )
        except Exception as exc:
            logger.error("Run %s failed: %s", run.id, exc, exc_info=True)
            ctx.increment_failed()
            await self._handle_run_failure(run, exc, ctx)

    async def _handle_run_failure(
        self,
        run: Any,
        exc: Exception,
        ctx: ProgressContext,
    ) -> None:
        """Handle a failed run by updating status and sending progress.

        Args:
            run: The failed run
            exc: The exception that caused the failure
            ctx: Progress context
        """
        error_message = str(exc)[:1000]

        await self.benchmark_repo.update_run_status(
            run_id=str(run.id),
            status=BenchmarkRunStatus.FAILED,
            error_message=error_message,
        )
        await self.db_session.commit()

        await self._send_progress(
            ctx,
            BenchmarkStatus.RUNNING,
            "evaluating",
            last_completed_run={
                "run_id": str(run.id),
                "run_order": int(run.run_order),
                "config": cast(dict[str, Any], run.config) or {},
                "status": BenchmarkRunStatus.FAILED.value,
                "error_message": error_message,
            },
        )

    async def _execute_run(
        self,
        run: Any,
        data: BenchmarkData,
        ctx: ProgressContext,
    ) -> None:
        """Execute a single benchmark run.

        Args:
            run: BenchmarkRun instance
            data: Loaded benchmark data
            ctx: Progress context for tracking state
        """
        run_id = str(run.id)
        run_config = cast(dict[str, Any], run.config) or {}
        if not run_config:
            logger.warning(
                "Run %s has no configuration, using defaults. This may indicate a bug.",
                run_id,
            )

        run_info = self._build_run_info(run, run_config, len(data.queries))
        search_params = self._extract_search_params(run_config)

        # Update run status to INDEXING (preparation phase)
        await self.benchmark_repo.update_run_status(
            run_id=run_id,
            status=BenchmarkRunStatus.INDEXING,
            status_message="Preparing evaluation",
        )
        await self.db_session.commit()

        start_time = time.perf_counter()
        indexing_duration = 0  # No indexing needed for evaluation-only benchmarks

        await self._send_progress(
            ctx, BenchmarkStatus.RUNNING, "indexing",
            current_run={**run_info, "completed_queries": 0, "stage": "indexing"},
        )

        # Update to EVALUATING
        await self.benchmark_repo.update_run_status(
            run_id=run_id,
            status=BenchmarkRunStatus.EVALUATING,
            status_message="Running queries",
            indexing_duration_ms=indexing_duration,
        )
        await self.db_session.commit()

        search_func = self._create_search_func(
            benchmark_id=ctx.benchmark_id,
            collection=data.collection,
            **search_params,
        )

        eval_start = time.perf_counter()

        await self._send_progress(
            ctx, BenchmarkStatus.RUNNING, "evaluating",
            current_run={**run_info, "completed_queries": 0, "stage": "evaluating"},
        )

        # Create progress callback using closure over ctx and run_info
        async def progress_callback(completed_queries: int, total_queries: int) -> None:
            await self._send_progress(
                ctx, BenchmarkStatus.RUNNING, "evaluating",
                current_run={
                    **run_info,
                    "total_queries": total_queries,
                    "completed_queries": completed_queries,
                    "stage": "evaluating",
                },
            )

        config_hash = cast(str, run.config_hash)
        run_top_k = run_config.get("top_k", data.top_k)
        effective_top_k = max(
            int(run_top_k),
            max(data.k_values) if data.k_values else int(run_top_k),
        )

        eval_result = await self.evaluator.evaluate_configuration(
            config_hash=config_hash,
            queries=data.queries,
            relevance_by_query=data.relevance_by_query,
            search_func=search_func,
            k_values=data.k_values,
            top_k=effective_top_k,
            include_debug=False,
            progress_callback=progress_callback,
            progress_interval_queries=BENCHMARK_PROGRESS_INTERVAL_QUERIES,
            progress_interval_ms=BENCHMARK_PROGRESS_INTERVAL_MS,
        )

        eval_duration = int((time.perf_counter() - eval_start) * 1000)
        total_duration = int((time.perf_counter() - start_time) * 1000)

        await self._store_run_results(run_id, eval_result, data.primary_k)

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

        # Send completion progress (temporarily increment for this message)
        completed_ctx = ProgressContext(
            benchmark_id=ctx.benchmark_id,
            total_runs=ctx.total_runs,
            primary_k=ctx.primary_k,
            k_values=ctx.k_values,
            completed_runs=ctx.completed_runs + 1,
            failed_runs=ctx.failed_runs,
        )

        await self._send_progress(
            completed_ctx, BenchmarkStatus.RUNNING, "evaluating",
            last_completed_run={
                "run_id": run_id,
                "run_order": int(run.run_order),
                "config": run_config,
                "config_summary": run_info["config_summary"],
                "status": BenchmarkRunStatus.COMPLETED.value,
                "metrics": self._metrics_to_structured_dict(eval_result.aggregate_metrics),
                "metrics_flat": self._metrics_to_flat_dict(eval_result.aggregate_metrics),
                "timing": {
                    "search_ms": eval_result.total_search_time_ms,
                    "rerank_ms": eval_result.total_rerank_time_ms,
                    "total_ms": total_duration,
                },
            },
        )

    def _build_run_info(
        self,
        run: Any,
        run_config: dict[str, Any],
        total_queries: int,
    ) -> dict[str, Any]:
        """Build common run info dict used in progress updates."""
        return {
            "run_id": str(run.id),
            "run_order": int(run.run_order),
            "config": run_config,
            "config_summary": self._get_config_summary(run_config),
            "total_queries": total_queries,
        }

    def _extract_search_params(self, run_config: dict[str, Any]) -> dict[str, Any]:
        """Extract search parameters from run configuration."""
        return {
            "search_mode": run_config.get("search_mode", "dense"),
            "use_reranker": run_config.get("use_reranker", False),
            "rrf_k": run_config.get("rrf_k", 60),
            "score_threshold": run_config.get("score_threshold"),
        }

    def _create_search_func(
        self,
        benchmark_id: str,
        collection: Collection,
        search_mode: str,
        use_reranker: bool,
        rrf_k: int,
        score_threshold: float | None,
    ) -> Any:
        """Create a search function closure for the ConfigurationEvaluator.

        Args:
            benchmark_id: UUID for cancellation checks
            collection: Collection to search
            search_mode: Search mode (dense, sparse, hybrid)
            use_reranker: Whether to use reranking
            rrf_k: RRF k constant
            score_threshold: Minimum score threshold

        Returns:
            Async function matching SearchFunc signature
        """

        async def search_func(query_text: str, top_k: int) -> tuple[list[RetrievedChunk], SearchTiming]:
            if await self._check_cancellation(benchmark_id):
                raise BenchmarkCancelledError(CANCELLED_RUN_ERROR_MESSAGE)

            result = await self.search_service.benchmark_search(
                collection=collection,
                query=query_text,
                search_mode=search_mode,  # type: ignore[arg-type]
                use_reranker=use_reranker,
                top_k=top_k,
                rrf_k=rrf_k,
                score_threshold=score_threshold,
            )

            chunks = self._convert_search_chunks(result.chunks)
            timing = SearchTiming(
                search_time_ms=result.search_time_ms,
                rerank_time_ms=result.rerank_time_ms,
            )
            return chunks, timing

        return search_func

    def _convert_search_chunks(self, raw_chunks: list[dict[str, Any]]) -> list[RetrievedChunk]:
        """Convert raw search result chunks to RetrievedChunk objects.

        Args:
            raw_chunks: List of chunk dicts from search service

        Returns:
            List of RetrievedChunk objects, skipping malformed entries
        """
        chunks: list[RetrievedChunk] = []
        for chunk in raw_chunks:
            doc_id = chunk.get("doc_id")
            chunk_id = chunk.get("chunk_id")

            if not doc_id or not chunk_id:
                logger.warning(
                    "Chunk missing required fields: doc_id=%s, chunk_id=%s",
                    doc_id,
                    chunk_id,
                )
                continue

            chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    score=chunk.get("score", 0.0),
                )
            )
        return chunks

    async def _store_run_results(
        self,
        run_id: str,
        eval_result: ConfigurationEvaluationResult,
        primary_k: int,
    ) -> None:
        """Persist evaluation results to the database.

        Args:
            run_id: UUID of the run
            eval_result: ConfigurationEvaluationResult from evaluator
            primary_k: k value used for stored per-query metrics
        """
        for metric in eval_result.aggregate_metrics:
            await self.benchmark_repo.add_run_metric(
                run_id=run_id,
                metric_name=metric.name,
                metric_value=metric.value,
                k_value=metric.k_value,
            )

        for query_result in eval_result.per_query_results:
            metrics = self._extract_query_metrics(query_result, primary_k)
            await self.benchmark_repo.add_query_result(
                run_id=run_id,
                query_id=query_result.query_id,
                retrieved_doc_ids=query_result.retrieved_doc_ids,
                retrieved_debug=query_result.retrieved_debug,
                precision_at_k=metrics.get("precision"),
                recall_at_k=metrics.get("recall"),
                reciprocal_rank=metrics.get("mrr"),
                ndcg_at_k=metrics.get("ndcg"),
                search_time_ms=query_result.search_time_ms,
                rerank_time_ms=query_result.rerank_time_ms,
            )

        await self.db_session.flush()

    def _extract_query_metrics(
        self,
        query_result: QueryEvaluationResult,
        primary_k: int,
    ) -> dict[str, float | None]:
        """Extract per-query metrics at the primary k value.

        Args:
            query_result: Single query evaluation result
            primary_k: k value to extract metrics for

        Returns:
            Dict with precision, recall, ndcg, and mrr values (or None if not found)
        """
        metrics: dict[str, float | None] = {
            "precision": None,
            "recall": None,
            "ndcg": None,
            "mrr": None,
        }

        for m in query_result.metrics:
            if m.name == "mrr":
                metrics["mrr"] = m.value
            elif m.k_value == primary_k and m.name in metrics:
                metrics[m.name] = m.value

        return metrics

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

    def _get_metrics_k_values(self, benchmark: Benchmark) -> tuple[int, list[int]]:
        """Return (primary_k, k_values_for_metrics) for this benchmark.

        primary_k defaults to 10. k_values_for_metrics defaults to [primary_k].
        Ensures primary_k is included and values are positive integers.
        """
        config_matrix = cast(dict[str, Any], benchmark.config_matrix)
        if not config_matrix:
            logger.warning(
                "Benchmark %s has no config_matrix, using defaults. This may indicate a bug.",
                benchmark.id,
            )
            config_matrix = {}

        k_config = parse_k_values(
            raw_primary_k=config_matrix.get("primary_k", 10),
            raw_k_values=config_matrix.get("k_values_for_metrics"),
        )
        return k_config.primary_k, k_config.k_values_for_metrics

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

    async def _send_progress(
        self,
        ctx: ProgressContext,
        status: BenchmarkStatus,
        stage: str,
        current_run: dict[str, Any] | None = None,
        last_completed_run: dict[str, Any] | None = None,
    ) -> None:
        """Send progress update via Redis pub/sub.

        Args:
            ctx: Progress context holding benchmark state
            status: Current benchmark status
            stage: Current stage (starting, indexing, evaluating, completed)
            current_run: Current run info or None
            last_completed_run: Last completed run info or None
        """
        progress_data = {
            "benchmark_id": ctx.benchmark_id,
            "status": status.value,
            "total_runs": ctx.total_runs,
            "completed_runs": ctx.completed_runs,
            "failed_runs": ctx.failed_runs,
            "primary_k": ctx.primary_k,
            "k_values_for_metrics": ctx.k_values,
            "current_run": current_run,
            "last_completed_run": last_completed_run,
            "stage": stage,
        }

        await self.progress_reporter.send_update("benchmark_progress", progress_data)

    def _metrics_to_flat_dict(self, metrics: list[MetricResult]) -> dict[str, float]:
        flat: dict[str, float] = {}
        for metric in metrics:
            key = metric.name
            if metric.k_value is not None:
                key = f"{metric.name}@{metric.k_value}"
            flat[key] = metric.value
        return flat

    def _metrics_to_structured_dict(self, metrics: list[MetricResult]) -> dict[str, Any]:
        structured: dict[str, Any] = {
            "mrr": None,
            "ap": None,
            "precision": {},
            "recall": {},
            "ndcg": {},
        }

        for metric in metrics:
            if metric.name in ("precision", "recall", "ndcg") and metric.k_value is not None:
                structured[metric.name][int(metric.k_value)] = metric.value
            elif metric.name in ("mrr", "ap") and metric.k_value is None:
                structured[metric.name] = metric.value

        # Drop empty metric maps so JSON payloads stay compact
        if not structured["precision"]:
            structured.pop("precision", None)
        if not structured["recall"]:
            structured.pop("recall", None)
        if not structured["ndcg"]:
            structured.pop("ndcg", None)
        if structured.get("ap") is None:
            structured.pop("ap", None)

        return structured

    async def _check_cancellation(self, benchmark_id: str) -> bool:
        """Check if the benchmark has been cancelled.

        Args:
            benchmark_id: UUID of the benchmark

        Returns:
            True if cancelled, False otherwise
        """
        status_str = await self.benchmark_repo.get_status_value(benchmark_id)
        if status_str is None:
            return True  # Treat as cancelled if not found
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

    async def _mark_remaining_runs_cancelled(
        self,
        *,
        benchmark_id: str,
        completed_runs: int,
        runs: list[Any],
        starting_from_run_id: str,
        existing_failed_count: int,
    ) -> int:
        """Mark the current and remaining runs as failed due to cancellation.

        The BenchmarkRunStatus enum does not include a CANCELLED state, so we
        represent cancellation as FAILED with a clear error message.
        """
        started = False
        newly_failed = 0

        for run in runs:
            run_id = str(run.id)
            if run_id == starting_from_run_id:
                started = True
            if not started:
                continue

            run_status = cast(str, run.status)
            if run_status in (BenchmarkRunStatus.COMPLETED.value, BenchmarkRunStatus.FAILED.value):
                continue

            await self.benchmark_repo.update_run_status(
                run_id=run_id,
                status=BenchmarkRunStatus.FAILED,
                status_message="Cancelled by user",
                error_message=CANCELLED_RUN_ERROR_MESSAGE,
            )
            newly_failed += 1

        await self.db_session.commit()

        failed_count = existing_failed_count + newly_failed

        await self.benchmark_repo.update_status(
            benchmark_id,
            BenchmarkStatus.CANCELLED,
            completed_runs=completed_runs,
            failed_runs=failed_count,
        )
        await self.db_session.commit()

        return failed_count


__all__ = ["BenchmarkExecutor"]
