"""
Integration tests for BenchmarkRepository.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from shared.database.exceptions import AccessDeniedError, EntityNotFoundError, ValidationError
from shared.database.models import (
    Benchmark,
    BenchmarkDataset,
    BenchmarkDatasetMapping,
    BenchmarkRunStatus,
    BenchmarkStatus,
    MappingStatus,
)
from shared.database.repositories.benchmark_repository import BenchmarkRepository


class TestBenchmarkRepositoryStatusGuards:
    @pytest.mark.asyncio()
    async def test_cancelled_benchmark_status_is_not_overwritten(
        self, db_session, test_user_db, collection_factory
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)

        dataset = BenchmarkDataset(
            id=str(uuid4()),
            name="Dataset",
            description=None,
            owner_id=test_user_db.id,
            query_count=0,
            raw_file_path=None,
            schema_version="1.0",
            meta=None,
        )
        db_session.add(dataset)
        await db_session.commit()

        mapping = BenchmarkDatasetMapping(
            dataset_id=dataset.id,
            collection_id=collection.id,
            mapping_status=MappingStatus.RESOLVED.value,
            mapped_count=0,
            total_count=0,
        )
        db_session.add(mapping)
        await db_session.commit()
        await db_session.refresh(mapping)

        benchmark = Benchmark(
            id=str(uuid4()),
            name="Bench",
            description=None,
            owner_id=test_user_db.id,
            mapping_id=mapping.id,
            operation_uuid=None,
            evaluation_unit="query",
            config_matrix={"search_modes": ["dense"], "top_k_values": [10]},
            config_matrix_hash="hash",
            limits=None,
            collection_snapshot_hash=None,
            reproducibility_metadata=None,
            top_k=10,
            metrics_to_compute=["precision"],
            status=BenchmarkStatus.CANCELLED.value,
            total_runs=0,
            completed_runs=0,
            failed_runs=0,
            created_at=datetime.now(UTC),
            cancelled_at=datetime.now(UTC),
        )
        db_session.add(benchmark)
        await db_session.commit()

        repo = BenchmarkRepository(db_session)

        await repo.update_status(benchmark.id, BenchmarkStatus.COMPLETED)
        await db_session.commit()

        assert await repo.get_status_value(benchmark.id) == BenchmarkStatus.CANCELLED.value


class TestBenchmarkRepositoryAtomicTransitions:
    @pytest.mark.asyncio()
    async def test_transition_status_atomically_succeeds_and_sets_operation(
        self, db_session, test_user_db, collection_factory, operation_factory
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)
        operation_uuid = f"op-{uuid4().hex}"
        await operation_factory(user_id=test_user_db.id, collection_id=collection.id, uuid=operation_uuid)

        dataset = BenchmarkDataset(
            id=str(uuid4()),
            name="Dataset",
            description=None,
            owner_id=test_user_db.id,
            query_count=0,
            raw_file_path=None,
            schema_version="1.0",
            meta=None,
        )
        db_session.add(dataset)
        await db_session.commit()

        mapping = BenchmarkDatasetMapping(
            dataset_id=dataset.id,
            collection_id=collection.id,
            mapping_status=MappingStatus.RESOLVED.value,
            mapped_count=0,
            total_count=0,
        )
        db_session.add(mapping)
        await db_session.commit()
        await db_session.refresh(mapping)

        benchmark = Benchmark(
            id=str(uuid4()),
            name="Bench",
            description=None,
            owner_id=test_user_db.id,
            mapping_id=mapping.id,
            operation_uuid=None,
            evaluation_unit="query",
            config_matrix={"search_modes": ["dense"], "top_k_values": [10]},
            config_matrix_hash="hash",
            limits=None,
            collection_snapshot_hash=None,
            reproducibility_metadata=None,
            top_k=10,
            metrics_to_compute=["precision"],
            status=BenchmarkStatus.PENDING.value,
            total_runs=0,
            completed_runs=0,
            failed_runs=0,
            created_at=datetime.now(UTC),
        )
        db_session.add(benchmark)
        await db_session.commit()

        repo = BenchmarkRepository(db_session)
        transitioned = await repo.transition_status_atomically(
            benchmark_uuid=benchmark.id,
            from_status=BenchmarkStatus.PENDING,
            to_status=BenchmarkStatus.RUNNING,
            operation_uuid=operation_uuid,
        )
        assert transitioned is not None
        await db_session.commit()

        refreshed = await repo.get_by_uuid(benchmark.id)
        assert refreshed is not None
        assert refreshed.status == BenchmarkStatus.RUNNING.value
        assert refreshed.operation_uuid == operation_uuid
        assert refreshed.started_at is not None

    @pytest.mark.asyncio()
    async def test_transition_status_atomically_returns_none_when_status_mismatch(
        self, db_session, test_user_db, collection_factory
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)

        dataset = BenchmarkDataset(
            id=str(uuid4()),
            name="Dataset",
            description=None,
            owner_id=test_user_db.id,
            query_count=0,
            raw_file_path=None,
            schema_version="1.0",
            meta=None,
        )
        db_session.add(dataset)
        await db_session.commit()

        mapping = BenchmarkDatasetMapping(
            dataset_id=dataset.id,
            collection_id=collection.id,
            mapping_status=MappingStatus.RESOLVED.value,
            mapped_count=0,
            total_count=0,
        )
        db_session.add(mapping)
        await db_session.commit()
        await db_session.refresh(mapping)

        benchmark = Benchmark(
            id=str(uuid4()),
            name="Bench",
            description=None,
            owner_id=test_user_db.id,
            mapping_id=mapping.id,
            operation_uuid=None,
            evaluation_unit="query",
            config_matrix={"search_modes": ["dense"], "top_k_values": [10]},
            config_matrix_hash="hash",
            limits=None,
            collection_snapshot_hash=None,
            reproducibility_metadata=None,
            top_k=10,
            metrics_to_compute=["precision"],
            status=BenchmarkStatus.RUNNING.value,
            total_runs=0,
            completed_runs=0,
            failed_runs=0,
            created_at=datetime.now(UTC),
            started_at=datetime.now(UTC),
        )
        db_session.add(benchmark)
        await db_session.commit()

        repo = BenchmarkRepository(db_session)
        transitioned = await repo.transition_status_atomically(
            benchmark_uuid=benchmark.id,
            from_status=BenchmarkStatus.PENDING,
            to_status=BenchmarkStatus.RUNNING,
            operation_uuid="op-1",
        )
        assert transitioned is None


@pytest.mark.asyncio()
async def test_update_status_raises_for_missing_benchmark(db_session) -> None:
    repo = BenchmarkRepository(db_session)
    with pytest.raises(EntityNotFoundError):
        await repo.update_status("missing", BenchmarkStatus.RUNNING)


@pytest.mark.asyncio()
async def test_create_validates_inputs_and_requires_mapping(db_session, test_user_db, collection_factory) -> None:
    collection = await collection_factory(owner_id=test_user_db.id)

    dataset = BenchmarkDataset(
        id=str(uuid4()),
        name="Dataset",
        description=None,
        owner_id=test_user_db.id,
        query_count=0,
        raw_file_path=None,
        schema_version="1.0",
        meta=None,
    )
    db_session.add(dataset)
    await db_session.commit()

    mapping = BenchmarkDatasetMapping(
        dataset_id=dataset.id,
        collection_id=collection.id,
        mapping_status=MappingStatus.RESOLVED.value,
        mapped_count=0,
        total_count=0,
    )
    db_session.add(mapping)
    await db_session.commit()
    await db_session.refresh(mapping)

    repo = BenchmarkRepository(db_session)

    with pytest.raises(ValidationError):
        await repo.create(
            name="   ",
            owner_id=test_user_db.id,
            mapping_id=int(mapping.id),
            config_matrix={"primary_k": 10},
            config_matrix_hash="hash",
            metrics_to_compute=["mrr"],
        )

    with pytest.raises(ValidationError):
        await repo.create(
            name="x",
            owner_id=test_user_db.id,
            mapping_id=int(mapping.id),
            config_matrix={"primary_k": 10},
            config_matrix_hash="hash",
            metrics_to_compute=["mrr"],
            top_k=0,
        )

    with pytest.raises(EntityNotFoundError):
        await repo.create(
            name="x",
            owner_id=test_user_db.id,
            mapping_id=999_999_999,
            config_matrix={"primary_k": 10},
            config_matrix_hash="hash",
            metrics_to_compute=["mrr"],
        )


@pytest.mark.asyncio()
async def test_get_by_uuid_for_user_enforces_ownership(
    db_session, test_user_db, other_user_db, collection_factory
) -> None:
    collection = await collection_factory(owner_id=test_user_db.id)

    dataset = BenchmarkDataset(
        id=str(uuid4()),
        name="Dataset",
        description=None,
        owner_id=test_user_db.id,
        query_count=0,
        raw_file_path=None,
        schema_version="1.0",
        meta=None,
    )
    db_session.add(dataset)
    await db_session.commit()

    mapping = BenchmarkDatasetMapping(
        dataset_id=dataset.id,
        collection_id=collection.id,
        mapping_status=MappingStatus.RESOLVED.value,
        mapped_count=0,
        total_count=0,
    )
    db_session.add(mapping)
    await db_session.commit()
    await db_session.refresh(mapping)

    benchmark = Benchmark(
        id=str(uuid4()),
        name="Bench",
        description=None,
        owner_id=test_user_db.id,
        mapping_id=mapping.id,
        operation_uuid=None,
        evaluation_unit="query",
        config_matrix={"search_modes": ["dense"], "top_k_values": [10]},
        config_matrix_hash="hash",
        limits=None,
        collection_snapshot_hash=None,
        reproducibility_metadata=None,
        top_k=10,
        metrics_to_compute=["precision"],
        status=BenchmarkStatus.PENDING.value,
        total_runs=0,
        completed_runs=0,
        failed_runs=0,
        created_at=datetime.now(UTC),
    )
    db_session.add(benchmark)
    await db_session.commit()

    repo = BenchmarkRepository(db_session)
    with pytest.raises(AccessDeniedError):
        await repo.get_by_uuid_for_user(benchmark.id, other_user_db.id)


@pytest.mark.asyncio()
async def test_get_aggregated_results_formats_metrics_and_selects_best_run(
    db_session, test_user_db, collection_factory
) -> None:
    collection = await collection_factory(owner_id=test_user_db.id)

    dataset = BenchmarkDataset(
        id=str(uuid4()),
        name="Dataset",
        description=None,
        owner_id=test_user_db.id,
        query_count=0,
        raw_file_path=None,
        schema_version="1.0",
        meta=None,
    )
    db_session.add(dataset)
    await db_session.commit()

    mapping = BenchmarkDatasetMapping(
        dataset_id=dataset.id,
        collection_id=collection.id,
        mapping_status=MappingStatus.RESOLVED.value,
        mapped_count=0,
        total_count=0,
    )
    db_session.add(mapping)
    await db_session.commit()
    await db_session.refresh(mapping)

    repo = BenchmarkRepository(db_session)
    benchmark = await repo.create(
        name="Bench",
        owner_id=test_user_db.id,
        mapping_id=int(mapping.id),
        config_matrix={"primary_k": "5", "k_values_for_metrics": ["10", 5, -1, "bad"]},
        config_matrix_hash="hash",
        metrics_to_compute=["precision", "mrr"],
        top_k=10,
    )
    await db_session.commit()

    run1 = await repo.create_run(
        benchmark_id=str(benchmark.id),
        run_order=1,
        config_hash="cfg-1",
        config={"alpha": 1},
    )
    run2 = await repo.create_run(
        benchmark_id=str(benchmark.id),
        run_order=2,
        config_hash="cfg-2",
        config={"alpha": 2},
    )

    await repo.update_run_status(run1.id, BenchmarkRunStatus.INDEXING)
    await repo.update_run_status(
        run1.id,
        BenchmarkRunStatus.COMPLETED,
        indexing_duration_ms=1,
        evaluation_duration_ms=2,
        total_duration_ms=3,
    )

    await repo.update_run_status(
        run2.id, BenchmarkRunStatus.COMPLETED, dense_collection_name="dense", sparse_collection_name="sparse"
    )

    await repo.add_run_metric(run1.id, "ndcg", 0.2, k_value=5)
    await repo.add_run_metric(run2.id, "mrr", 0.7)
    await repo.add_run_metric(run2.id, "precision", 0.9, k_value=5)
    await repo.add_run_metric(run2.id, "precision", 0.1, k_value=10)

    await repo.set_total_runs(str(benchmark.id), 2)
    await repo.update_status(str(benchmark.id), BenchmarkStatus.COMPLETED, completed_runs=2, failed_runs=0)
    await db_session.commit()

    aggregated = await repo.get_aggregated_results(str(benchmark.id))
    assert aggregated["benchmark_id"] == str(benchmark.id)
    assert aggregated["primary_k"] == 5
    assert aggregated["k_values_for_metrics"] == [5, 10]

    assert aggregated["summary"]["best_run"] == run2.id
    assert aggregated["summary"]["best_primary_metric"] == 0.7

    runs = aggregated["runs"]
    assert [r["run_id"] for r in runs] == [run1.id, run2.id]
    assert runs[1]["metrics"]["precision"][5] == 0.9
    assert runs[1]["metrics"]["precision"][10] == 0.1
    assert runs[1]["metrics_flat"]["precision@5"] == 0.9
