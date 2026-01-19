from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.database.exceptions import ValidationError
from shared.database.models import BenchmarkStatus
from webui.services.benchmark_service import BenchmarkService


@pytest.mark.asyncio()
async def test_start_benchmark_dispatches_task_and_returns_running(stub_celery_send_task) -> None:
    db_session = AsyncMock()
    benchmark_repo = AsyncMock()
    benchmark_dataset_repo = AsyncMock()
    collection_repo = AsyncMock()
    operation_repo = AsyncMock()
    search_service = AsyncMock()

    benchmark = MagicMock()
    benchmark.id = "bench-1"
    benchmark.mapping_id = 123
    benchmark.status = BenchmarkStatus.PENDING.value
    benchmark_repo.get_by_uuid_for_user.return_value = benchmark

    mapping = MagicMock()
    mapping.id = 123
    mapping.collection_id = "col-1"
    benchmark_dataset_repo.get_mapping.return_value = mapping

    operation = MagicMock()
    operation.uuid = "op-1"
    operation_repo.create.return_value = operation

    updated = MagicMock()
    updated.id = "bench-1"
    updated.status = BenchmarkStatus.RUNNING.value
    benchmark_repo.transition_status_atomically.return_value = updated

    service = BenchmarkService(
        db_session=db_session,
        benchmark_repo=benchmark_repo,
        benchmark_dataset_repo=benchmark_dataset_repo,
        collection_repo=collection_repo,
        operation_repo=operation_repo,
        search_service=search_service,
    )

    result = await service.start_benchmark("bench-1", user_id=1)

    assert result["id"] == "bench-1"
    assert result["status"] == BenchmarkStatus.RUNNING.value
    assert result["operation_uuid"] == "op-1"

    benchmark_repo.transition_status_atomically.assert_awaited_with(
        benchmark_uuid="bench-1",
        from_status=BenchmarkStatus.PENDING,
        to_status=BenchmarkStatus.RUNNING,
        operation_uuid="op-1",
    )
    db_session.commit.assert_awaited()
    assert stub_celery_send_task.called


@pytest.mark.asyncio()
async def test_start_benchmark_raises_validation_error_when_race_lost(stub_celery_send_task) -> None:
    db_session = AsyncMock()
    benchmark_repo = AsyncMock()
    benchmark_dataset_repo = AsyncMock()
    collection_repo = AsyncMock()
    operation_repo = AsyncMock()
    search_service = AsyncMock()

    benchmark = MagicMock()
    benchmark.id = "bench-1"
    benchmark.mapping_id = 123
    benchmark.status = BenchmarkStatus.PENDING.value
    benchmark_repo.get_by_uuid_for_user.return_value = benchmark

    mapping = MagicMock()
    mapping.id = 123
    mapping.collection_id = "col-1"
    benchmark_dataset_repo.get_mapping.return_value = mapping

    operation = MagicMock()
    operation.uuid = "op-1"
    operation_repo.create.return_value = operation

    benchmark_repo.transition_status_atomically.return_value = None

    service = BenchmarkService(
        db_session=db_session,
        benchmark_repo=benchmark_repo,
        benchmark_dataset_repo=benchmark_dataset_repo,
        collection_repo=collection_repo,
        operation_repo=operation_repo,
        search_service=search_service,
    )

    with pytest.raises(ValidationError, match="Benchmark must be in PENDING status to start"):
        await service.start_benchmark("bench-1", user_id=1)

    assert not stub_celery_send_task.called


@pytest.mark.asyncio()
async def test_cancel_benchmark_rejects_invalid_status() -> None:
    db_session = AsyncMock()
    benchmark_repo = AsyncMock()
    benchmark_dataset_repo = AsyncMock()
    collection_repo = AsyncMock()
    operation_repo = AsyncMock()
    search_service = AsyncMock()

    benchmark = MagicMock()
    benchmark.id = "bench-1"
    benchmark.status = BenchmarkStatus.COMPLETED.value
    benchmark_repo.get_by_uuid_for_user.return_value = benchmark

    service = BenchmarkService(
        db_session=db_session,
        benchmark_repo=benchmark_repo,
        benchmark_dataset_repo=benchmark_dataset_repo,
        collection_repo=collection_repo,
        operation_repo=operation_repo,
        search_service=search_service,
    )

    with pytest.raises(ValidationError, match="Cannot cancel benchmark"):
        await service.cancel_benchmark("bench-1", user_id=1)

    benchmark_repo.update_status.assert_not_called()


@pytest.mark.asyncio()
async def test_get_run_query_results_formats_unknown_query() -> None:
    db_session = AsyncMock()
    benchmark_repo = AsyncMock()
    benchmark_dataset_repo = AsyncMock()
    collection_repo = AsyncMock()
    operation_repo = AsyncMock()
    search_service = AsyncMock()

    run = MagicMock()
    run.id = "run-1"
    run.benchmark_id = "bench-1"
    benchmark_repo.get_run.return_value = run

    result_row = MagicMock()
    result_row.benchmark_query_id = 123
    result_row.query = None
    result_row.retrieved_doc_ids = ["doc-1"]
    result_row.precision_at_k = 0.1
    result_row.recall_at_k = 0.2
    result_row.reciprocal_rank = 0.3
    result_row.ndcg_at_k = 0.4
    result_row.search_time_ms = 10
    result_row.rerank_time_ms = 2

    benchmark_repo.get_query_results_for_run.return_value = ([result_row], 1)

    service = BenchmarkService(
        db_session=db_session,
        benchmark_repo=benchmark_repo,
        benchmark_dataset_repo=benchmark_dataset_repo,
        collection_repo=collection_repo,
        operation_repo=operation_repo,
        search_service=search_service,
    )

    response = await service.get_run_query_results("run-1", user_id=1, page=1, per_page=50)

    assert response["run_id"] == "run-1"
    assert response["total"] == 1
    assert response["results"][0]["query_key"] == "unknown"
    assert response["results"][0]["query_text"] == "unknown"
