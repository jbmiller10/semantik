"""
Integration tests for BenchmarkRepository.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from shared.database.models import Benchmark, BenchmarkDataset, BenchmarkDatasetMapping, BenchmarkStatus, MappingStatus
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
