from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.webui.services.resource_manager import ResourceEstimate, ResourceManager


@pytest.mark.asyncio()
async def test_can_create_collection_respects_limit():
    collection_repo = AsyncMock()
    collection_repo.list_for_user.return_value = ([MagicMock(status="ready") for _ in range(3)], 3)
    operation_repo = AsyncMock()

    manager = ResourceManager(collection_repo, operation_repo)
    assert await manager.can_create_collection(user_id=1) is True

    # Simulate hitting limit
    collection_repo.list_for_user.return_value = ([MagicMock(status="ready") for _ in range(12)], 12)
    assert await manager.can_create_collection(user_id=1) is False


@pytest.mark.asyncio()
async def test_can_allocate_checks_system_resources(monkeypatch):
    collection_repo = AsyncMock()
    collection_repo.list_for_user.return_value = ([], 0)
    operation_repo = AsyncMock()

    manager = ResourceManager(collection_repo, operation_repo)

    class FakeVM:
        available = 8 * 1024 * 1024 * 1024  # 8GB

    class FakeDisk:
        free = 100 * 1024 * 1024 * 1024  # 100GB

    monkeypatch.setattr("packages.webui.services.resource_manager.psutil.virtual_memory", lambda: FakeVM)
    monkeypatch.setattr("packages.webui.services.resource_manager.psutil.disk_usage", lambda _: FakeDisk)

    async def fake_usage(_user_id: int) -> dict[str, float]:  # noqa: ARG001
        return {"storage_gb": 0}

    monkeypatch.setattr(manager, "_get_user_resource_usage", fake_usage)

    estimate = ResourceEstimate(memory_mb=512, storage_gb=1.0)
    assert await manager.can_allocate(user_id=1, resources=estimate) is True

    # Request excessive memory
    estimate = ResourceEstimate(memory_mb=20_000, storage_gb=1.0)
    assert await manager.can_allocate(user_id=1, resources=estimate) is False


@pytest.mark.asyncio()
async def test_estimate_resources_directory(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_bytes(b"a" * 2048)

    manager = ResourceManager(AsyncMock(), AsyncMock())
    estimate = await manager.estimate_resources(str(tmp_path), "sentence-transformers/all-MiniLM-L6-v2")

    assert estimate.memory_mb > 0
    assert estimate.storage_gb > 0
    assert estimate.cpu_cores >= 1.0


@pytest.mark.asyncio()
async def test_reserve_and_release_reindex(monkeypatch):
    collection_repo = AsyncMock()
    collection_repo.get_by_id.return_value = {
        "total_size_bytes": 5 * 1024 * 1024 * 1024,
        "id": "col-1",
    }
    operation_repo = AsyncMock()

    manager = ResourceManager(collection_repo, operation_repo)

    async def always_ok(_estimate: ResourceEstimate) -> bool:  # noqa: ARG001
        return True

    monkeypatch.setattr(manager, "_check_system_resources", always_ok)

    assert await manager.reserve_for_reindex("col-1") is True
    assert "reindex_col-1" in manager._reserved_resources

    await manager.release_reindex_reservation("col-1")
    assert not manager._reserved_resources
