from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shared.database.exceptions import EntityNotFoundError, ValidationError
from shared.database.models import BenchmarkStatus, MappingStatus
from webui.services.benchmark_service import BenchmarkService


@pytest.mark.asyncio()
async def test_create_benchmark_normalizes_k_values_and_validates_top_k_values() -> None:
    mapping = SimpleNamespace(dataset_id="ds-1", collection_id="col-1", mapping_status=MappingStatus.RESOLVED.value)

    benchmark_dataset_repo = AsyncMock()
    benchmark_dataset_repo.get_mapping.return_value = mapping
    benchmark_dataset_repo.get_by_uuid_for_user.return_value = SimpleNamespace(id="ds-1")

    collection_repo = AsyncMock()
    collection_repo.get_by_uuid.return_value = SimpleNamespace(id="col-1")

    benchmark_repo = AsyncMock()
    benchmark_repo.create.return_value = SimpleNamespace(
        id="bench-1",
        name="Bench",
        description=None,
        owner_id=1,
        mapping_id=10,
        status=BenchmarkStatus.PENDING.value,
        created_at="2025-01-01T00:00:00Z",
    )
    benchmark_repo.create_run = AsyncMock()
    benchmark_repo.set_total_runs = AsyncMock()

    service = BenchmarkService(
        db_session=AsyncMock(),
        benchmark_repo=benchmark_repo,
        benchmark_dataset_repo=benchmark_dataset_repo,
        collection_repo=collection_repo,
        operation_repo=AsyncMock(),
        search_service=AsyncMock(),
    )

    config_matrix = {
        "primary_k": "5",
        "k_values_for_metrics": ["10", 0, -1, "nope", 5],
        "top_k_values": [10, "10", 20],
        "search_modes": ["dense"],
        "use_reranker": [False],
        "rrf_k_values": [60],
        "score_thresholds": [None],
    }

    result = await service.create_benchmark(
        user_id=1,
        mapping_id=10,
        name="Bench",
        description=None,
        config_matrix=config_matrix,
        top_k=10,
        metrics_to_compute=["mrr"],
    )

    assert result["id"] == "bench-1"

    created_matrix = benchmark_repo.create.await_args.kwargs["config_matrix"]
    assert created_matrix["primary_k"] == 5
    assert created_matrix["k_values_for_metrics"] == [5, 10]


@pytest.mark.asyncio()
async def test_create_benchmark_rejects_top_k_values_too_small() -> None:
    mapping = SimpleNamespace(dataset_id="ds-1", collection_id="col-1", mapping_status=MappingStatus.RESOLVED.value)

    benchmark_dataset_repo = AsyncMock()
    benchmark_dataset_repo.get_mapping.return_value = mapping
    benchmark_dataset_repo.get_by_uuid_for_user.return_value = SimpleNamespace(id="ds-1")

    collection_repo = AsyncMock()
    collection_repo.get_by_uuid.return_value = SimpleNamespace(id="col-1")

    service = BenchmarkService(
        db_session=AsyncMock(),
        benchmark_repo=AsyncMock(),
        benchmark_dataset_repo=benchmark_dataset_repo,
        collection_repo=collection_repo,
        operation_repo=AsyncMock(),
        search_service=AsyncMock(),
    )

    with pytest.raises(ValidationError):
        await service.create_benchmark(
            user_id=1,
            mapping_id=10,
            name="Bench",
            description=None,
            config_matrix={
                "primary_k": 10,
                "k_values_for_metrics": [10, 20],
                "top_k_values": [10],  # < required_top_k=20
            },
            top_k=10,
            metrics_to_compute=["mrr"],
        )


@pytest.mark.asyncio()
async def test_start_benchmark_raises_on_race_lost(stub_celery_send_task) -> None:
    benchmark = SimpleNamespace(id="bench-1", mapping_id=10)
    mapping = SimpleNamespace(id=10, collection_id="col-1")

    benchmark_repo = AsyncMock()
    benchmark_repo.get_by_uuid_for_user.return_value = benchmark
    benchmark_repo.transition_status_atomically.return_value = None

    benchmark_dataset_repo = AsyncMock()
    benchmark_dataset_repo.get_mapping.return_value = mapping

    operation_repo = AsyncMock()
    operation_repo.create.return_value = SimpleNamespace(uuid="op-1")

    service = BenchmarkService(
        db_session=AsyncMock(),
        benchmark_repo=benchmark_repo,
        benchmark_dataset_repo=benchmark_dataset_repo,
        collection_repo=AsyncMock(),
        operation_repo=operation_repo,
        search_service=AsyncMock(),
    )

    with pytest.raises(ValidationError):
        await service.start_benchmark("bench-1", user_id=1)

    stub_celery_send_task.assert_not_called()


@pytest.mark.asyncio()
async def test_start_benchmark_dispatches_task(stub_celery_send_task) -> None:
    benchmark = SimpleNamespace(id="bench-1", mapping_id=10)
    mapping = SimpleNamespace(id=10, collection_id="col-1")

    benchmark_repo = AsyncMock()
    benchmark_repo.get_by_uuid_for_user.return_value = benchmark
    benchmark_repo.transition_status_atomically.return_value = SimpleNamespace(
        id="bench-1", status=BenchmarkStatus.RUNNING.value
    )

    benchmark_dataset_repo = AsyncMock()
    benchmark_dataset_repo.get_mapping.return_value = mapping

    operation_repo = AsyncMock()
    operation_repo.create.return_value = SimpleNamespace(uuid="op-1")

    db_session = AsyncMock()

    service = BenchmarkService(
        db_session=db_session,
        benchmark_repo=benchmark_repo,
        benchmark_dataset_repo=benchmark_dataset_repo,
        collection_repo=AsyncMock(),
        operation_repo=operation_repo,
        search_service=AsyncMock(),
    )

    result = await service.start_benchmark("bench-1", user_id=1)
    assert result["status"] == BenchmarkStatus.RUNNING.value
    stub_celery_send_task.assert_called_once()


@pytest.mark.asyncio()
async def test_cancel_benchmark_rejects_non_cancellable_states() -> None:
    benchmark_repo = AsyncMock()
    benchmark_repo.get_by_uuid_for_user.return_value = SimpleNamespace(
        id="bench-1", status=BenchmarkStatus.COMPLETED.value
    )

    service = BenchmarkService(
        db_session=AsyncMock(),
        benchmark_repo=benchmark_repo,
        benchmark_dataset_repo=AsyncMock(),
        collection_repo=AsyncMock(),
        operation_repo=AsyncMock(),
        search_service=AsyncMock(),
    )

    with pytest.raises(ValidationError):
        await service.cancel_benchmark("bench-1", user_id=1)


@pytest.mark.asyncio()
async def test_get_run_query_results_requires_existing_run() -> None:
    benchmark_repo = AsyncMock()
    benchmark_repo.get_run.return_value = None

    service = BenchmarkService(
        db_session=AsyncMock(),
        benchmark_repo=benchmark_repo,
        benchmark_dataset_repo=AsyncMock(),
        collection_repo=AsyncMock(),
        operation_repo=AsyncMock(),
        search_service=AsyncMock(),
    )

    with pytest.raises(EntityNotFoundError):
        await service.get_run_query_results(run_id="run-1", user_id=1)
