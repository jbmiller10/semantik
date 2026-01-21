from __future__ import annotations

from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from shared.benchmarks.types import ConfigurationEvaluationResult, MetricResult, QueryEvaluationResult
from shared.database.models import BenchmarkRunStatus, BenchmarkStatus
from webui.services.benchmark_executor import BenchmarkExecutor


@pytest.mark.asyncio()
async def test_execute_benchmark_runs_pending_and_skips_completed() -> None:
    db_session = AsyncMock()
    benchmark_repo = AsyncMock()
    benchmark_dataset_repo = AsyncMock()
    collection_repo = AsyncMock()
    search_service = AsyncMock()
    progress_reporter = AsyncMock()

    benchmark = MagicMock()
    benchmark.id = "bench-1"
    benchmark.mapping_id = 1
    benchmark.config_matrix = {"primary_k": 10, "k_values_for_metrics": [10]}
    benchmark.top_k = 10
    benchmark_repo.get_by_uuid.return_value = benchmark
    benchmark_repo.get_status_value.return_value = BenchmarkStatus.RUNNING.value

    mapping = MagicMock()
    mapping.id = 1
    mapping.dataset_id = "ds-1"
    mapping.collection_id = "col-1"
    benchmark_dataset_repo.get_mapping.return_value = mapping

    collection = MagicMock()
    collection.id = "col-1"
    collection_repo.get_by_uuid.return_value = collection

    query = MagicMock()
    query.id = 1
    query.query_text = "hello"
    benchmark_dataset_repo.get_queries_for_dataset.return_value = [query]

    relevance = MagicMock()
    relevance.benchmark_query_id = 1
    relevance.resolved_document_id = "doc-1"
    relevance.relevance_grade = 3
    benchmark_dataset_repo.get_relevance_for_mapping.return_value = [relevance]

    run_completed = MagicMock()
    run_completed.id = "run-0"
    run_completed.status = BenchmarkRunStatus.COMPLETED.value
    run_completed.run_order = 0
    run_completed.config = {}
    run_completed.config_hash = "cfg-0"

    run_pending = MagicMock()
    run_pending.id = "run-1"
    run_pending.status = BenchmarkRunStatus.PENDING.value
    run_pending.run_order = 1
    run_pending.config = {
        "search_mode": "dense",
        "use_reranker": False,
        "top_k": 10,
        "rrf_k": 60,
        "score_threshold": None,
    }
    run_pending.config_hash = "cfg-1"

    benchmark_repo.get_runs_for_benchmark.return_value = [run_completed, run_pending]

    search_result = MagicMock()
    search_result.chunks = [{"doc_id": "doc-1", "chunk_id": "c1", "score": 0.9}]
    search_result.search_time_ms = 12
    search_result.rerank_time_ms = 0
    search_service.benchmark_search.return_value = search_result

    executor = BenchmarkExecutor(
        db_session=db_session,
        benchmark_repo=benchmark_repo,
        benchmark_dataset_repo=benchmark_dataset_repo,
        collection_repo=collection_repo,
        search_service=search_service,
        progress_reporter=progress_reporter,
    )

    async def fake_evaluate_configuration(**kwargs):
        chunks, timing = await kwargs["search_func"]("hello", kwargs["top_k"])
        await kwargs["progress_callback"](1, 1)

        primary_k = int(kwargs["k_values"][0])
        per_query = QueryEvaluationResult(
            query_id=1,
            retrieved_doc_ids=[c["doc_id"] for c in chunks],
            retrieved_debug={"chunks": chunks},
            metrics=[
                MetricResult(name="precision", k_value=primary_k, value=1.0),
                MetricResult(name="recall", k_value=primary_k, value=1.0),
                MetricResult(name="ndcg", k_value=primary_k, value=1.0),
                MetricResult(name="mrr", k_value=None, value=1.0),
            ],
            search_time_ms=timing.search_time_ms,
            rerank_time_ms=timing.rerank_time_ms,
        )

        return ConfigurationEvaluationResult(
            config_hash=kwargs["config_hash"],
            total_queries=1,
            aggregate_metrics=[
                MetricResult(name="mrr", k_value=None, value=0.5),
                MetricResult(name="precision", k_value=primary_k, value=1.0),
            ],
            per_query_results=[per_query],
            total_search_time_ms=timing.search_time_ms,
            total_rerank_time_ms=int(timing.rerank_time_ms or 0),
        )

    executor.evaluator.evaluate_configuration = AsyncMock(side_effect=fake_evaluate_configuration)

    result = await executor.execute_benchmark("bench-1")

    assert result["benchmark_id"] == "bench-1"
    assert result["status"] == BenchmarkStatus.COMPLETED.value
    assert result["total_runs"] == 2
    assert result["completed_runs"] == 2
    assert result["failed_runs"] == 0
    assert result["skipped_runs"] == 1

    benchmark_repo.update_run_status.assert_any_await(
        run_id="run-1",
        status=BenchmarkRunStatus.INDEXING,
        status_message="Preparing evaluation",
    )
    benchmark_repo.update_run_status.assert_any_await(
        run_id="run-1",
        status=BenchmarkRunStatus.EVALUATING,
        status_message="Running queries",
        indexing_duration_ms=0,
    )
    benchmark_repo.update_run_status.assert_any_await(
        run_id="run-1",
        status=BenchmarkRunStatus.COMPLETED,
        status_message="Evaluation complete",
        evaluation_duration_ms=ANY,
        total_duration_ms=ANY,
    )

    benchmark_repo.add_run_metric.assert_awaited()
    benchmark_repo.add_query_result.assert_awaited()
    progress_reporter.send_update.assert_awaited()


@pytest.mark.asyncio()
async def test_execute_benchmark_respects_pre_cancelled_status_and_marks_runs() -> None:
    db_session = AsyncMock()
    benchmark_repo = AsyncMock()
    benchmark_dataset_repo = AsyncMock()
    collection_repo = AsyncMock()
    search_service = AsyncMock()
    progress_reporter = AsyncMock()

    benchmark = MagicMock()
    benchmark.id = "bench-cancelled"
    benchmark.mapping_id = 1
    benchmark_repo.get_by_uuid.return_value = benchmark

    benchmark_repo.get_status_value.return_value = BenchmarkStatus.CANCELLED.value

    run_pending = MagicMock()
    run_pending.id = "run-1"
    run_pending.status = BenchmarkRunStatus.PENDING.value
    run_pending.run_order = 0
    run_pending.config = {}
    run_pending.config_hash = "cfg-1"
    benchmark_repo.get_runs_for_benchmark.return_value = [run_pending]

    executor = BenchmarkExecutor(
        db_session=db_session,
        benchmark_repo=benchmark_repo,
        benchmark_dataset_repo=benchmark_dataset_repo,
        collection_repo=collection_repo,
        search_service=search_service,
        progress_reporter=progress_reporter,
    )

    result = await executor.execute_benchmark("bench-cancelled")

    assert result["status"] == BenchmarkStatus.CANCELLED.value
    assert result["total_runs"] == 1
    assert result["completed_runs"] == 0
    assert result["failed_runs"] == 1

    benchmark_repo.update_run_status.assert_awaited_with(
        run_id="run-1",
        status=BenchmarkRunStatus.FAILED,
        status_message="Cancelled by user",
        error_message="cancelled by user",
    )
    benchmark_repo.update_status.assert_any_await(
        "bench-cancelled",
        BenchmarkStatus.CANCELLED,
        completed_runs=0,
        failed_runs=1,
    )
