"""Integration tests for ResourceManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.webui.services.resource_manager import ResourceEstimate, ResourceManager

pytestmark = [pytest.mark.asyncio(), pytest.mark.usefixtures("_db_isolation")]


@pytest.fixture()
def resource_manager(db_session: AsyncSession) -> ResourceManager:
    collection_repo = CollectionRepository(db_session)
    operation_repo = OperationRepository(db_session)
    return ResourceManager(collection_repo, operation_repo)


class _FakeVirtualMemory:
    def __init__(self, available: int) -> None:
        self.available = available


class _FakeDiskUsage:
    def __init__(self, free: int) -> None:
        self.free = free


async def test_can_create_collection_respects_limit(
    resource_manager: ResourceManager,
    collection_factory,
    test_user_db,
) -> None:
    for _ in range(9):
        await collection_factory(owner_id=test_user_db.id)

    assert await resource_manager.can_create_collection(test_user_db.id) is True

    await collection_factory(owner_id=test_user_db.id)
    assert await resource_manager.can_create_collection(test_user_db.id) is False


async def test_can_allocate_checks_system_resources(
    resource_manager: ResourceManager,
    monkeypatch,
    test_user_db,
) -> None:
    monkeypatch.setattr("packages.webui.services.resource_manager.psutil.virtual_memory", lambda: _FakeVirtualMemory(available=32 * 1024 * 1024 * 1024))
    monkeypatch.setattr("packages.webui.services.resource_manager.psutil.disk_usage", lambda _: _FakeDiskUsage(free=2 * 1024 * 1024 * 1024 * 1024))

    estimate = ResourceEstimate(memory_mb=1024, storage_gb=1.0)
    assert await resource_manager.can_allocate(test_user_db.id, estimate) is True

    heavy = ResourceEstimate(memory_mb=32 * 1024, storage_gb=5_000)
    assert await resource_manager.can_allocate(test_user_db.id, heavy) is False


async def test_estimate_resources_for_directory(resource_manager: ResourceManager, tmp_path: Path) -> None:
    sample_dir = tmp_path / "docs"
    sample_dir.mkdir()
    (sample_dir / "one.pdf").write_bytes(b"a" * 1024 * 1024)
    (sample_dir / "two.txt").write_bytes(b"b" * 2048)

    estimate = await resource_manager.estimate_resources(str(sample_dir), "BAAI/bge-base-en-v1.5")
    assert estimate.memory_mb > 0
    assert estimate.storage_gb > 0
    assert estimate.cpu_cores >= 1.0


async def test_reserve_and_release_reindex(
    resource_manager: ResourceManager,
    db_session: AsyncSession,
    collection_factory,
    test_user_db,
    monkeypatch,
) -> None:
    collection = await collection_factory(owner_id=test_user_db.id)
    collection.total_size_bytes = 5 * 1024 * 1024 * 1024
    await db_session.commit()

    monkeypatch.setattr("packages.webui.services.resource_manager.psutil.virtual_memory", lambda: _FakeVirtualMemory(available=64 * 1024 * 1024 * 1024))
    monkeypatch.setattr("packages.webui.services.resource_manager.psutil.disk_usage", lambda _: _FakeDiskUsage(free=4 * 1024 * 1024 * 1024 * 1024))

    reserved = await resource_manager.reserve_for_reindex(collection.id)
    assert reserved is True

    await resource_manager.release_reindex_reservation(collection.id)
    assert await resource_manager.reserve_for_reindex("non-existent") is False


async def test_get_resource_usage(resource_manager: ResourceManager, collection_factory, test_user_db) -> None:
    collection = await collection_factory(owner_id=test_user_db.id)
    collection.total_size_bytes = 123456789

    usage = await resource_manager.get_resource_usage(collection.id)
    assert usage["documents"] == collection.document_count
    assert usage["storage_bytes"] == collection.total_size_bytes
