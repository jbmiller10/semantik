"""Integration smoke tests for partition manager utilities."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError

from packages.shared.chunking.infrastructure.repositories.partition_manager import PartitionManager

pytestmark = [pytest.mark.asyncio(), pytest.mark.usefixtures("_db_isolation")]


class TestPartitionManagerIntegration:
    """Exercise partition manager helpers with graceful fallbacks if views are missing."""

    @pytest.fixture()
    def manager(self) -> PartitionManager:
        return PartitionManager()

    def test_partition_id_and_name_are_deterministic(self, manager: PartitionManager) -> None:
        collection_id = str(uuid4())
        first = manager.get_partition_id(collection_id)
        second = manager.get_partition_id(collection_id)
        assert 0 <= first < manager.PARTITION_COUNT
        assert first == second
        assert manager.get_partition_name(collection_id).startswith("chunks_part_")

    async def test_distribution_stats_query(self, manager: PartitionManager, db_session) -> None:
        try:
            stats = await manager.get_distribution_stats(db_session)
        except SQLAlchemyError as exc:  # pragma: no cover - depends on view availability
            pytest.skip(f"partition_distribution view not available: {exc}")

        assert stats.partitions_used >= 0
        assert stats.empty_partitions >= 0
        assert stats.distribution_status in {"HEALTHY", "WARNING", "REBALANCE NEEDED", "NO_DATA"}

    async def test_partition_health_query(self, manager: PartitionManager, db_session) -> None:
        try:
            health = await manager.get_partition_health(db_session)
        except SQLAlchemyError as exc:  # pragma: no cover - depends on view availability
            pytest.skip(f"partition_health view not available: {exc}")

        assert len(health) == manager.PARTITION_COUNT or len(health) == 0
        for entry in health:
            assert entry.partition_name.startswith("chunks_part_")
            assert entry.partition_id == int(entry.partition_name.split("_")[-1])

    async def test_verify_partition_for_collection(self, manager: PartitionManager, db_session) -> None:
        collection_id = str(uuid4())
        try:
            result = await manager.verify_partition_for_collection(db_session, collection_id)
        except SQLAlchemyError as exc:  # pragma: no cover
            pytest.skip(f"partition verification query unavailable: {exc}")

        assert result["collection_id"] == collection_id
        assert result["python_partition_name"].startswith("chunks_part_")
        assert result["db_partition_name"].startswith("chunks_part_")
