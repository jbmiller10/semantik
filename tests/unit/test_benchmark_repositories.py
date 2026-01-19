from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.database.exceptions import (
    DatabaseOperationError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    ValidationError,
)
from shared.database.models import BenchmarkStatus, MappingStatus
from shared.database.repositories.benchmark_dataset_repository import BenchmarkDatasetRepository
from shared.database.repositories.benchmark_repository import BenchmarkRepository


@pytest.mark.asyncio()
async def test_benchmark_dataset_repository_create_validates_name_and_owner() -> None:
    session = AsyncMock()
    repo = BenchmarkDatasetRepository(session)

    with pytest.raises(ValidationError):
        await repo.create(name="  ", owner_id=1)

    session.scalar.return_value = 0
    with pytest.raises(EntityNotFoundError):
        await repo.create(name="ds", owner_id=123)

    session.scalar.return_value = 1
    session.flush = AsyncMock()
    session.add = MagicMock()
    created = await repo.create(name="  ds  ", owner_id=1, query_count=2)

    assert created.name == "ds"
    assert created.owner_id == 1


@pytest.mark.asyncio()
async def test_benchmark_dataset_repository_create_mapping_detects_duplicates() -> None:
    session = AsyncMock()
    repo = BenchmarkDatasetRepository(session)

    repo.get_by_uuid = AsyncMock(return_value=SimpleNamespace(id="ds-1"))

    session.scalar.side_effect = [
        1,  # collection exists
        1,  # mapping already exists
    ]

    with pytest.raises(EntityAlreadyExistsError):
        await repo.create_mapping(dataset_id="ds-1", collection_id="col-1")


@pytest.mark.asyncio()
async def test_benchmark_dataset_repository_update_mapping_status_sets_resolved_at() -> None:
    session = AsyncMock()
    repo = BenchmarkDatasetRepository(session)

    mapping = SimpleNamespace(
        id=10,
        mapping_status=MappingStatus.PENDING.value,
        mapped_count=0,
        total_count=0,
        resolved_at=None,
    )
    repo.get_mapping = AsyncMock(return_value=mapping)

    updated = await repo.update_mapping_status(mapping_id=10, status=MappingStatus.RESOLVED, mapped_count=1, total_count=2)
    assert updated.mapping_status == MappingStatus.RESOLVED.value
    assert updated.mapped_count == 1
    assert updated.total_count == 2
    assert isinstance(updated.resolved_at, datetime)
    assert updated.resolved_at.tzinfo == UTC


@pytest.mark.asyncio()
async def test_benchmark_dataset_repository_add_relevance_hash_and_errors() -> None:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    repo = BenchmarkDatasetRepository(session)

    with pytest.raises(ValidationError):
        await repo.add_relevance(query_id=1, mapping_id=2, doc_ref={"uri": "x"}, relevance_grade=99)

    created = await repo.add_relevance(query_id=1, mapping_id=2, doc_ref={"uri": "x"}, relevance_grade=2, doc_ref_hash=None)
    assert created.doc_ref_hash is not None

    session.flush = AsyncMock(side_effect=RuntimeError("boom"))
    with pytest.raises(DatabaseOperationError):
        await repo.add_relevance(query_id=1, mapping_id=2, doc_ref={"uri": "x"}, relevance_grade=2, doc_ref_hash="h")


@pytest.mark.asyncio()
async def test_benchmark_dataset_repository_count_relevance_wraps_db_errors() -> None:
    session = AsyncMock()
    repo = BenchmarkDatasetRepository(session)
    session.scalar.side_effect = RuntimeError("db down")
    with pytest.raises(DatabaseOperationError):
        await repo.count_relevance_for_mapping(10)


@pytest.mark.asyncio()
async def test_benchmark_repository_update_status_does_not_overwrite_cancelled() -> None:
    session = AsyncMock()
    repo = BenchmarkRepository(session)

    cancelled = SimpleNamespace(id="bench-1", status=BenchmarkStatus.CANCELLED.value)

    async def execute_side_effect(stmt):
        res = MagicMock()
        if getattr(stmt, "__visit_name__", "") == "update":
            res.scalar_one_or_none.return_value = None
            return res
        res.scalar_one_or_none.return_value = cancelled
        return res

    session.execute.side_effect = execute_side_effect

    updated = await repo.update_status("bench-1", BenchmarkStatus.RUNNING)
    assert updated.status == BenchmarkStatus.CANCELLED.value


@pytest.mark.asyncio()
async def test_benchmark_repository_transition_status_atomically_handles_race_and_missing() -> None:
    session = AsyncMock()
    repo = BenchmarkRepository(session)

    res = MagicMock()
    res.scalar_one_or_none.return_value = None
    session.execute.return_value = res

    repo.get_by_uuid = AsyncMock(return_value=SimpleNamespace(id="bench-1", status=BenchmarkStatus.PENDING.value))
    result = await repo.transition_status_atomically(
        benchmark_uuid="bench-1",
        from_status=BenchmarkStatus.PENDING,
        to_status=BenchmarkStatus.RUNNING,
        operation_uuid="op-1",
    )
    assert result is None

    repo.get_by_uuid = AsyncMock(return_value=None)
    with pytest.raises(EntityNotFoundError):
        await repo.transition_status_atomically(
            benchmark_uuid="missing",
            from_status=BenchmarkStatus.PENDING,
            to_status=BenchmarkStatus.RUNNING,
        )


@pytest.mark.asyncio()
async def test_benchmark_repository_add_query_result_validates_run_and_query() -> None:
    session = AsyncMock()
    repo = BenchmarkRepository(session)

    repo.get_run = AsyncMock(return_value=None)
    with pytest.raises(EntityNotFoundError):
        await repo.add_query_result(run_id="run-1", query_id=1, retrieved_doc_ids=[])

    repo.get_run = AsyncMock(return_value=SimpleNamespace(id="run-1"))
    session.scalar.return_value = 0
    with pytest.raises(EntityNotFoundError):
        await repo.add_query_result(run_id="run-1", query_id=999, retrieved_doc_ids=[])
