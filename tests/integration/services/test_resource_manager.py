"""Integration tests for ResourceManager using real repositories with patched system metrics."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import psutil
import pytest

from packages.shared.database.models import OperationType
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.webui.services.resource_manager import ResourceEstimate, ResourceManager

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class CollectionRepoAdapter:
    repo: CollectionRepository

    async def list_for_user(self, user_id: int):
        return await self.repo.list_for_user(user_id, include_public=False)

    async def get_by_id(self, collection_id: str) -> dict[str, Any] | None:
        collection = await self.repo.get_by_uuid(collection_id)
        if not collection:
            return None
        return {
            "id": collection.id,
            "total_size_bytes": collection.total_size_bytes or 0,
        }


@dataclass
class OperationRepoAdapter:
    repo: OperationRepository

    async def list_by_user(self, user_id: int, since: datetime):
        operations, _ = await self.repo.list_for_user(user_id)
        return [op for op in operations if op.created_at and op.created_at >= since]


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestResourceManagerIntegration:
    """Validate resource manager logic with actual repositories."""

    @pytest.fixture()
    def manager(self, db_session):
        collection_repo = CollectionRepoAdapter(CollectionRepository(db_session))
        operation_repo = OperationRepoAdapter(OperationRepository(db_session))
        
        class _StubQdrantManager:
            async def get_collection_usage(self, _name: str) -> dict[str, int]:  # pragma: no cover - simple stub
                raise RuntimeError("stubbed qdrant unavailable")

        return ResourceManager(
            collection_repo=collection_repo,
            operation_repo=operation_repo,
            qdrant_manager=_StubQdrantManager(),
        )

    async def test_can_create_collection_respects_limit(self, manager, collection_factory, test_user_db, db_session):
        for _ in range(10):
            await collection_factory(owner_id=test_user_db.id)
        await db_session.commit()

        assert await manager.can_create_collection(test_user_db.id) is False

    async def test_can_allocate_checks_system_resources(self, manager, monkeypatch, test_user_db):
        monkeypatch.setattr(
            psutil,
            "virtual_memory",
            lambda: SimpleNamespace(available=200 * 1024 * 1024),
        )
        monkeypatch.setattr(
            psutil,
            "disk_usage",
            lambda _path="/": SimpleNamespace(free=1 * 1024 * 1024 * 1024),
        )

        estimate = ResourceEstimate(memory_mb=500, storage_gb=5)
        assert await manager.can_allocate(test_user_db.id, estimate) is False

    async def test_can_allocate_respects_user_storage_quota(
        self, manager, collection_factory, test_user_db, db_session, monkeypatch
    ):
        monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=4 * 1024 * 1024 * 1024))
        monkeypatch.setattr(psutil, "disk_usage", lambda _path="/": SimpleNamespace(free=200 * 1024 * 1024 * 1024))

        size_per_collection = 1_500_000_000  # ~1.4GB, below int32 threshold
        target_collections = 40  # Accumulate >50GB total usage

        for _ in range(target_collections):
            await collection_factory(owner_id=test_user_db.id, total_size_bytes=size_per_collection)
        await db_session.commit()

        estimate = ResourceEstimate(memory_mb=100, storage_gb=1)
        assert await manager.can_allocate(test_user_db.id, estimate) is False

    async def test_reserve_and_release_reindex(
        self, manager, collection_factory, test_user_db, db_session, monkeypatch
    ):
        monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 1024 * 1024 * 1024))
        monkeypatch.setattr(psutil, "disk_usage", lambda _path="/": SimpleNamespace(free=500 * 1024 * 1024 * 1024))

        collection = await collection_factory(owner_id=test_user_db.id, total_size_bytes=1_500_000_000)
        await db_session.commit()

        reserved = await manager.reserve_for_reindex(collection.id)
        assert reserved is True
        assert f"reindex_{collection.id}" in manager._reserved_resources

        await manager.release_reindex_reservation(collection.id)
        assert f"reindex_{collection.id}" not in manager._reserved_resources

    async def test_get_resource_usage_returns_totals(self, manager, collection_factory, test_user_db, db_session):
        collection = await collection_factory(owner_id=test_user_db.id, total_size_bytes=1024)
        await db_session.commit()

        usage = await manager._get_user_resource_usage(test_user_db.id)
        assert usage["collections"] == 1
        assert usage["storage_bytes"] == collection.total_size_bytes

    async def test_get_resource_usage_fallback_flags_metrics(self, manager, collection_factory, test_user_db, db_session):
        collection = await collection_factory(owner_id=test_user_db.id, total_size_bytes=2048)
        await db_session.commit()

        usage = await manager.get_resource_usage(str(collection.id))
        assert usage["metrics_status"] == "unavailable"
        assert usage["metrics_source"] == "postgres"
        assert usage["storage_bytes"] == collection.total_size_bytes

    async def test_recent_operations_count_reads_from_repository(
        self, manager, collection_factory, test_user_db, db_session
    ):
        collection = await collection_factory(owner_id=test_user_db.id)
        op_repo = manager.operation_repo.repo
        await op_repo.create(collection.id, test_user_db.id, OperationType.INDEX, config={"source": "integration"})
        await db_session.commit()

        count = await manager._get_recent_operations_count(test_user_db.id, hours=1)
        assert count >= 1
