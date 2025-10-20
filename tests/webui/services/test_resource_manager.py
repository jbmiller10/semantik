from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from packages.shared.managers import QdrantCollectionNotFoundError
from packages.webui.services.resource_manager import ResourceEstimate, ResourceManager


@pytest.mark.asyncio()
async def test_can_create_collection_respects_limit():
    collection_repo = AsyncMock()
    collection_repo.list_for_user.return_value = ([MagicMock(status="ready") for _ in range(3)], 3)
    operation_repo = AsyncMock()

    manager = ResourceManager(collection_repo, operation_repo, qdrant_manager=None)
    assert await manager.can_create_collection(user_id=1) is True

    # Simulate hitting limit
    collection_repo.list_for_user.return_value = ([MagicMock(status="ready") for _ in range(12)], 12)
    assert await manager.can_create_collection(user_id=1) is False


@pytest.mark.asyncio()
async def test_can_allocate_checks_system_resources(monkeypatch):
    collection_repo = AsyncMock()
    collection_repo.list_for_user.return_value = ([], 0)
    operation_repo = AsyncMock()

    manager = ResourceManager(collection_repo, operation_repo, qdrant_manager=None)

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

    manager = ResourceManager(AsyncMock(), AsyncMock(), qdrant_manager=None)
    estimate = await manager.estimate_resources(str(tmp_path), "sentence-transformers/all-MiniLM-L6-v2")

    assert estimate.memory_mb > 0
    assert estimate.storage_gb > 0
    assert estimate.cpu_cores >= 1.0


@pytest.mark.asyncio()
async def test_reserve_and_release_reindex(monkeypatch):
    collection_repo = AsyncMock()
    collection_repo.get_by_uuid.return_value = {
        "total_size_bytes": 5 * 1024 * 1024 * 1024,
        "id": "col-1",
    }
    operation_repo = AsyncMock()

    manager = ResourceManager(collection_repo, operation_repo, qdrant_manager=None)

    async def always_ok(_estimate: ResourceEstimate) -> bool:  # noqa: ARG001
        return True

    monkeypatch.setattr(manager, "_check_system_resources", always_ok)

    assert await manager.reserve_for_reindex("col-1") is True
    assert "reindex_col-1" in manager._reserved_resources

    await manager.release_reindex_reservation("col-1")
    assert not manager._reserved_resources


@pytest.fixture()
def frozen_resource_manager_clock(monkeypatch):
    """Freeze resource manager wall clock to control cache expiry in tests."""

    class FrozenDateTime(datetime):
        _now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return cls._now.replace(tzinfo=None)
            return cls._now.astimezone(tz)

        @classmethod
        def utcnow(cls):
            return cls._now

        @classmethod
        def advance(cls, seconds: float) -> None:
            cls._now = cls._now + timedelta(seconds=seconds)

    monkeypatch.setattr("packages.webui.services.resource_manager.datetime", FrozenDateTime)
    return FrozenDateTime


@pytest.mark.asyncio()
async def test_get_resource_usage_prefers_qdrant_metrics():
    collection_repo = AsyncMock()
    collection_repo.get_by_uuid.return_value = {
        "id": "col-123",
        "vector_store_name": "collection_col-123",
        "document_count": 5,
        "vector_count": 50,
        "total_size_bytes": 1024,
    }
    operation_repo = AsyncMock()
    qdrant_manager = AsyncMock()
    qdrant_manager.get_collection_usage.return_value = {
        "documents": 42,
        "vectors": 84,
        "storage_bytes": 65_536,
    }

    manager = ResourceManager(collection_repo, operation_repo, qdrant_manager=qdrant_manager)

    usage = await manager.get_resource_usage("col-123")

    qdrant_manager.get_collection_usage.assert_awaited_once_with("collection_col-123")
    assert usage["documents"] == 42
    assert usage["vectors"] == 84
    assert usage["storage_bytes"] == 65_536
    assert usage["storage_gb"] == pytest.approx(65_536 / 1024 / 1024 / 1024)
    assert usage["metrics_status"] == "available"
    assert usage["metrics_source"] == "qdrant"


@pytest.mark.asyncio()
async def test_get_resource_usage_falls_back_when_qdrant_errors():
    collection_repo = AsyncMock()
    collection_repo.get_by_uuid.return_value = {
        "id": "col-456",
        "vector_store_name": "collection_col-456",
        "document_count": 8,
        "vector_count": 16,
        "total_size_bytes": 131_072,
    }
    operation_repo = AsyncMock()
    qdrant_manager = AsyncMock()
    qdrant_manager.get_collection_usage.side_effect = RuntimeError("boom")

    manager = ResourceManager(collection_repo, operation_repo, qdrant_manager=qdrant_manager)

    usage = await manager.get_resource_usage("col-456")

    qdrant_manager.get_collection_usage.assert_awaited_once_with("collection_col-456")
    assert usage["documents"] == 8
    assert usage["vectors"] == 16
    assert usage["storage_bytes"] == 131_072
    assert usage["storage_gb"] == pytest.approx(131_072 / 1024 / 1024 / 1024)
    assert usage["metrics_status"] == "unavailable"
    assert usage["metrics_source"] == "postgres"
    assert usage["metrics_reason"] == "qdrant_unreachable"


@pytest.mark.asyncio()
async def test_get_resource_usage_marks_missing_collection():
    collection_repo = AsyncMock()
    collection_repo.get_by_uuid.return_value = {
        "id": "col-000",
        "vector_store_name": "collection_col-000",
        "document_count": 0,
        "vector_count": 0,
        "total_size_bytes": 0,
    }
    operation_repo = AsyncMock()
    qdrant_manager = AsyncMock()
    qdrant_manager.get_collection_usage.side_effect = QdrantCollectionNotFoundError("collection_col-000")

    manager = ResourceManager(collection_repo, operation_repo, qdrant_manager=qdrant_manager)

    usage = await manager.get_resource_usage("col-000")

    assert usage["metrics_status"] == "unavailable"
    assert usage["metrics_source"] == "postgres"
    assert usage["metrics_reason"] == "collection_not_found"


@pytest.mark.asyncio()
async def test_get_resource_usage_caches_qdrant_metrics(frozen_resource_manager_clock):
    collection_repo = AsyncMock()
    collection_repo.get_by_uuid.return_value = {
        "id": "col-789",
        "vector_store_name": "collection_col-789",
        "document_count": 1,
        "vector_count": 2,
        "total_size_bytes": 512,
    }
    operation_repo = AsyncMock()
    qdrant_manager = AsyncMock()
    qdrant_manager.get_collection_usage.return_value = {
        "documents": 33,
        "vectors": 44,
        "storage_bytes": 55_000,
    }

    manager = ResourceManager(collection_repo, operation_repo, qdrant_manager=qdrant_manager)

    # Ensure any future cache implementation respects a short TTL for the test.
    if hasattr(manager, "RESOURCE_USAGE_CACHE_TTL_SECONDS"):
        manager.RESOURCE_USAGE_CACHE_TTL_SECONDS = 30
    else:
        manager._usage_cache_ttl_seconds = 30

    first_usage = await manager.get_resource_usage("col-789")
    qdrant_manager.get_collection_usage.assert_awaited_once_with("collection_col-789")

    second_usage = await manager.get_resource_usage("col-789")
    assert qdrant_manager.get_collection_usage.await_count == 1
    assert second_usage == first_usage

    frozen_resource_manager_clock.advance(31)

    third_usage = await manager.get_resource_usage("col-789")
    assert qdrant_manager.get_collection_usage.await_count == 2
    assert third_usage == first_usage
