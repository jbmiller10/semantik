"""
Tests for BenchmarkExecutor cancellation semantics.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.database.models import BenchmarkRunStatus, BenchmarkStatus
from webui.services.benchmark_executor import BenchmarkExecutor
from webui.services.search_service import BenchmarkSearchResult


class TestBenchmarkExecutorCancellation:
    @pytest.mark.asyncio()
    async def test_cancellation_mid_run_marks_remaining_runs_failed_and_benchmark_cancelled(self) -> None:
        db_session = AsyncMock()
        db_session.commit = AsyncMock()
        db_session.flush = AsyncMock()

        benchmark_repo = AsyncMock()
        benchmark_dataset_repo = AsyncMock()
        collection_repo = AsyncMock()
        search_service = AsyncMock()
        progress_reporter = AsyncMock()
        progress_reporter.send_update = AsyncMock()

        benchmark = MagicMock()
        benchmark.id = "bench-1"
        benchmark.mapping_id = 1
        benchmark.config_matrix = {"primary_k": 5, "k_values_for_metrics": [5]}
        benchmark.top_k = 5

        benchmark_repo.get_by_uuid.return_value = benchmark

        # Cancellation checks: running for initial calls, then cancelled mid-run.
        benchmark_repo.get_status_value.side_effect = ["running", "running", "running", "cancelled", "cancelled"]

        mapping = MagicMock()
        mapping.id = 1
        mapping.dataset_id = "dataset-1"
        mapping.collection_id = "collection-1"
        benchmark_dataset_repo.get_mapping.return_value = mapping

        query1 = MagicMock()
        query1.id = 1
        query1.query_text = "q1"
        query2 = MagicMock()
        query2.id = 2
        query2.query_text = "q2"
        benchmark_dataset_repo.get_queries_for_dataset.return_value = [query1, query2]
        benchmark_dataset_repo.get_relevance_for_mapping.return_value = []

        collection = MagicMock()
        collection.id = "collection-1"
        collection.name = "Test Collection"
        collection_repo.get_by_uuid.return_value = collection

        run = MagicMock()
        run.id = "run-1"
        run.benchmark_id = "bench-1"
        run.run_order = 0
        run.config_hash = "cfg"
        run.config = {"search_mode": "dense", "use_reranker": False, "top_k": 5}
        run.status = BenchmarkRunStatus.PENDING.value
        benchmark_repo.get_runs_for_benchmark.return_value = [run]

        search_service.benchmark_search.return_value = BenchmarkSearchResult(
            chunks=[{"doc_id": "doc-1", "chunk_id": "chunk-1", "score": 1.0}],
            search_time_ms=5,
            rerank_time_ms=None,
            total_results=1,
        )

        executor = BenchmarkExecutor(
            db_session=db_session,
            benchmark_repo=benchmark_repo,
            benchmark_dataset_repo=benchmark_dataset_repo,
            collection_repo=collection_repo,
            search_service=search_service,
            progress_reporter=progress_reporter,
        )

        result = await executor.execute_benchmark("bench-1")

        assert result["status"] == BenchmarkStatus.CANCELLED.value
        assert search_service.benchmark_search.call_count == 1

        # Remaining run marked failed with cancellation message.
        assert any(
            call.kwargs.get("status") == BenchmarkRunStatus.FAILED
            and call.kwargs.get("error_message") == "cancelled by user"
            for call in benchmark_repo.update_run_status.call_args_list
        )

