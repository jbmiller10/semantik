"""Unit tests for ResourceManager using mocks."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from webui.services.resource_manager import ResourceEstimate, ResourceManager


class TestResourceEstimate:
    """Tests for ResourceEstimate class."""

    def test_init_defaults(self) -> None:
        """Test ResourceEstimate with default values."""
        estimate = ResourceEstimate()

        assert estimate.memory_mb == 0
        assert estimate.storage_gb == 0.0
        assert estimate.cpu_cores == 0.0
        assert estimate.gpu_memory_mb == 0

    def test_init_with_values(self) -> None:
        """Test ResourceEstimate with custom values."""
        estimate = ResourceEstimate(
            memory_mb=1024,
            storage_gb=10.0,
            cpu_cores=2.0,
            gpu_memory_mb=4096,
        )

        assert estimate.memory_mb == 1024
        assert estimate.storage_gb == 10.0
        assert estimate.cpu_cores == 2.0
        assert estimate.gpu_memory_mb == 4096

    def test_str_representation(self) -> None:
        """Test ResourceEstimate string representation."""
        estimate = ResourceEstimate(memory_mb=1024, storage_gb=5.0, cpu_cores=1.0, gpu_memory_mb=2048)

        result = str(estimate)

        assert "memory=1024MB" in result
        assert "storage=5.0GB" in result
        assert "cpu=1.0" in result
        assert "gpu=2048MB" in result


class TestResourceManager:
    """Unit tests for ResourceManager."""

    @pytest.fixture()
    def mock_collection_repo(self) -> AsyncMock:
        """Create a mock collection repository."""
        return AsyncMock()

    @pytest.fixture()
    def mock_operation_repo(self) -> AsyncMock:
        """Create a mock operation repository."""
        return AsyncMock()

    @pytest.fixture()
    def mock_qdrant_manager(self) -> AsyncMock:
        """Create a mock qdrant manager."""
        return AsyncMock()

    @pytest.fixture()
    def resource_manager(self, mock_collection_repo, mock_operation_repo, mock_qdrant_manager) -> ResourceManager:
        """Create a ResourceManager instance with mocked dependencies."""
        return ResourceManager(
            collection_repo=mock_collection_repo,
            operation_repo=mock_operation_repo,
            qdrant_manager=mock_qdrant_manager,
        )

    @pytest.fixture()
    def sample_collection(self) -> MagicMock:
        """Create a sample collection for testing."""
        collection = MagicMock()
        collection.id = str(uuid4())
        collection.status = "ready"
        collection.total_size_bytes = 1024 * 1024 * 100  # 100MB
        collection.vector_store_name = "test_collection"
        collection.document_count = 10
        collection.vector_count = 100
        return collection

    # --- can_create_collection Tests ---

    @pytest.mark.asyncio()
    async def test_can_create_collection_under_limit(self, resource_manager, mock_collection_repo) -> None:
        """Test can_create_collection returns True when under limit."""
        # Mock only a few collections
        mock_collection_repo.list_for_user.return_value = (
            [MagicMock(status="ready") for _ in range(5)],
            5,
        )

        with patch("webui.services.resource_manager.settings") as mock_settings:
            mock_settings.MAX_COLLECTIONS_PER_USER = 10

            result = await resource_manager.can_create_collection(1)

            assert result is True

    @pytest.mark.asyncio()
    async def test_can_create_collection_at_limit(self, resource_manager, mock_collection_repo) -> None:
        """Test can_create_collection returns False when at limit."""
        mock_collection_repo.list_for_user.return_value = (
            [MagicMock(status="ready") for _ in range(10)],
            10,
        )

        with patch("webui.services.resource_manager.settings") as mock_settings:
            mock_settings.MAX_COLLECTIONS_PER_USER = 10

            result = await resource_manager.can_create_collection(1)

            assert result is False

    @pytest.mark.asyncio()
    async def test_can_create_collection_excludes_deleted(self, resource_manager, mock_collection_repo) -> None:
        """Test can_create_collection excludes deleted collections from count."""
        collections = [MagicMock(status="ready") for _ in range(8)]
        collections.extend([MagicMock(status="deleted") for _ in range(5)])
        mock_collection_repo.list_for_user.return_value = (collections, len(collections))

        with patch("webui.services.resource_manager.settings") as mock_settings:
            mock_settings.MAX_COLLECTIONS_PER_USER = 10

            result = await resource_manager.can_create_collection(1)

            assert result is True  # Only 8 active, under 10

    @pytest.mark.asyncio()
    async def test_can_create_collection_error_returns_false(self, resource_manager, mock_collection_repo) -> None:
        """Test can_create_collection returns False on error."""
        mock_collection_repo.list_for_user.side_effect = Exception("Database error")

        result = await resource_manager.can_create_collection(1)

        assert result is False

    # --- can_allocate Tests ---

    @pytest.mark.asyncio()
    async def test_can_allocate_with_sufficient_resources(self, resource_manager, mock_collection_repo) -> None:
        """Test can_allocate returns True when resources available."""
        mock_collection_repo.list_for_user.return_value = ([], 0)

        estimate = ResourceEstimate(memory_mb=100, storage_gb=0.1)

        with (
            patch("webui.services.resource_manager.psutil") as mock_psutil,
            patch("webui.services.resource_manager.settings") as mock_settings,
        ):
            # Mock plenty of available resources
            mock_psutil.virtual_memory.return_value = MagicMock(available=10 * 1024 * 1024 * 1024)
            mock_psutil.disk_usage.return_value = MagicMock(free=100 * 1024 * 1024 * 1024)
            mock_settings.MAX_STORAGE_GB_PER_USER = 50.0

            result = await resource_manager.can_allocate(1, estimate)

            assert result is True

    @pytest.mark.asyncio()
    async def test_can_allocate_insufficient_memory(self, resource_manager, mock_collection_repo) -> None:
        """Test can_allocate returns False when insufficient memory."""
        mock_collection_repo.list_for_user.return_value = ([], 0)

        estimate = ResourceEstimate(memory_mb=10000)

        with patch("webui.services.resource_manager.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(available=100 * 1024 * 1024)  # Only 100MB
            mock_psutil.disk_usage.return_value = MagicMock(free=100 * 1024 * 1024 * 1024)

            result = await resource_manager.can_allocate(1, estimate)

            assert result is False

    @pytest.mark.asyncio()
    async def test_can_allocate_insufficient_storage(self, resource_manager, mock_collection_repo) -> None:
        """Test can_allocate returns False when insufficient storage."""
        mock_collection_repo.list_for_user.return_value = ([], 0)

        estimate = ResourceEstimate(storage_gb=100.0)

        with patch("webui.services.resource_manager.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(available=10 * 1024 * 1024 * 1024)
            mock_psutil.disk_usage.return_value = MagicMock(free=10 * 1024 * 1024 * 1024)  # Only 10GB

            result = await resource_manager.can_allocate(1, estimate)

            assert result is False

    @pytest.mark.asyncio()
    async def test_can_allocate_error_returns_false(self, resource_manager, mock_collection_repo) -> None:
        """Test can_allocate returns False on error."""
        estimate = ResourceEstimate(memory_mb=100)

        with patch("webui.services.resource_manager.psutil") as mock_psutil:
            mock_psutil.virtual_memory.side_effect = Exception("System error")

            result = await resource_manager.can_allocate(1, estimate)

            assert result is False

    # --- get_resource_usage Tests ---

    @pytest.mark.asyncio()
    async def test_get_resource_usage_from_qdrant(
        self, resource_manager, mock_collection_repo, mock_qdrant_manager, sample_collection
    ) -> None:
        """Test get_resource_usage retrieves usage from Qdrant."""
        mock_collection_repo.get_by_uuid.return_value = sample_collection
        mock_qdrant_manager.get_collection_usage.return_value = {
            "documents": 100,
            "vectors": 1000,
            "storage_bytes": 1024 * 1024,
        }

        result = await resource_manager.get_resource_usage(sample_collection.id)

        assert result["documents"] == 100
        assert result["vectors"] == 1000
        assert result["metrics_source"] == "qdrant"

    @pytest.mark.asyncio()
    async def test_get_resource_usage_cache_hit(
        self, resource_manager, mock_collection_repo, sample_collection
    ) -> None:
        """Test get_resource_usage uses cache."""
        cached_usage = {"documents": 50, "vectors": 500, "storage_bytes": 512}
        resource_manager._usage_cache[sample_collection.id] = (cached_usage, datetime.now(UTC))

        result = await resource_manager.get_resource_usage(sample_collection.id)

        assert result == cached_usage
        mock_collection_repo.get_by_uuid.assert_not_called()

    @pytest.mark.asyncio()
    async def test_get_resource_usage_collection_not_found(self, resource_manager, mock_collection_repo) -> None:
        """Test get_resource_usage returns empty dict for missing collection."""
        mock_collection_repo.get_by_uuid.return_value = None

        result = await resource_manager.get_resource_usage(str(uuid4()))

        assert result == {}

    @pytest.mark.asyncio()
    async def test_get_resource_usage_qdrant_fallback(
        self, resource_manager, mock_collection_repo, mock_qdrant_manager, sample_collection
    ) -> None:
        """Test get_resource_usage falls back to postgres when Qdrant fails."""
        mock_collection_repo.get_by_uuid.return_value = sample_collection
        mock_qdrant_manager.get_collection_usage.side_effect = Exception("Qdrant error")

        result = await resource_manager.get_resource_usage(sample_collection.id)

        assert result["metrics_source"] == "postgres"
        assert result["metrics_status"] == "unavailable"

    @pytest.mark.asyncio()
    async def test_get_resource_usage_no_qdrant_manager(
        self, mock_collection_repo, mock_operation_repo, sample_collection
    ) -> None:
        """Test get_resource_usage works without Qdrant manager."""
        manager = ResourceManager(
            collection_repo=mock_collection_repo,
            operation_repo=mock_operation_repo,
            qdrant_manager=None,
        )
        mock_collection_repo.get_by_uuid.return_value = sample_collection

        result = await manager.get_resource_usage(sample_collection.id)

        assert result["metrics_source"] == "postgres"

    # --- _normalize_usage Tests ---

    def test_normalize_usage_from_qdrant(self, resource_manager, sample_collection) -> None:
        """Test _normalize_usage handles Qdrant format."""
        usage = {"documents": 100, "vectors": 1000, "storage_bytes": 1024}

        result = resource_manager._normalize_usage(usage, sample_collection)

        assert result["documents"] == 100
        assert result["vectors"] == 1000
        assert result["storage_bytes"] == 1024

    def test_normalize_usage_from_points(self, resource_manager, sample_collection) -> None:
        """Test _normalize_usage handles points field."""
        usage = {"points": 500, "disk_usage_bytes": 2048}

        result = resource_manager._normalize_usage(usage, sample_collection)

        assert result["documents"] == 500
        assert result["vectors"] == 500
        assert result["storage_bytes"] == 2048

    def test_normalize_usage_fallback_to_collection(self, resource_manager, sample_collection) -> None:
        """Test _normalize_usage falls back to collection values."""
        sample_collection.document_count = 25
        sample_collection.vector_count = 250
        sample_collection.total_size_bytes = 4096
        usage = {}

        result = resource_manager._normalize_usage(usage, sample_collection)

        assert result["documents"] == 25
        assert result["vectors"] == 250
        assert result["storage_bytes"] == 4096

    def test_normalize_usage_dict_collection(self, resource_manager) -> None:
        """Test _normalize_usage handles dict collection."""
        collection_dict = {"document_count": 10, "vector_count": 100, "total_size_bytes": 512}
        usage = {}

        result = resource_manager._normalize_usage(usage, collection_dict)

        assert result["documents"] == 10
        assert result["vectors"] == 100
        assert result["storage_bytes"] == 512

    # --- _check_system_resources Tests ---

    @pytest.mark.asyncio()
    async def test_check_system_resources_sufficient(self, resource_manager) -> None:
        """Test _check_system_resources returns True when sufficient."""
        estimate = ResourceEstimate(memory_mb=100, storage_gb=0.1)

        with patch("webui.services.resource_manager.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(available=10 * 1024 * 1024 * 1024)
            mock_psutil.disk_usage.return_value = MagicMock(free=100 * 1024 * 1024 * 1024)

            result = await resource_manager._check_system_resources(estimate)

            assert result is True

    @pytest.mark.asyncio()
    async def test_check_system_resources_includes_reservations(self, resource_manager) -> None:
        """Test _check_system_resources accounts for reserved resources."""
        # Reserve 5GB of memory, then try to allocate another 5GB with only 10GB available
        # After accounting for reservations and the 80% threshold, this should fail
        resource_manager._reserved_resources["test"] = ResourceEstimate(memory_mb=8000)
        estimate = ResourceEstimate(memory_mb=5000)

        with patch("webui.services.resource_manager.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(available=10 * 1024 * 1024 * 1024)  # 10GB = ~10240 MB
            mock_psutil.disk_usage.return_value = MagicMock(free=100 * 1024 * 1024 * 1024)

            result = await resource_manager._check_system_resources(estimate)

            # Available after reservations: 10240 - 8000 = 2240 MB
            # 80% of 2240 = 1792 MB, which is less than 5000 MB requested
            # So result should be False
            assert result is False

    @pytest.mark.asyncio()
    async def test_check_system_resources_error_returns_false(self, resource_manager) -> None:
        """Test _check_system_resources returns False on error."""
        estimate = ResourceEstimate()

        with patch("webui.services.resource_manager.psutil") as mock_psutil:
            mock_psutil.virtual_memory.side_effect = Exception("System error")

            result = await resource_manager._check_system_resources(estimate)

            assert result is False

    # --- _get_collection_value Tests ---

    def test_get_collection_value_from_object(self, resource_manager, sample_collection) -> None:
        """Test _get_collection_value retrieves from object attribute."""
        sample_collection.total_size_bytes = 12345

        result = resource_manager._get_collection_value(sample_collection, "total_size_bytes")

        assert result == 12345

    def test_get_collection_value_from_dict(self, resource_manager) -> None:
        """Test _get_collection_value retrieves from dict."""
        collection_dict = {"total_size_bytes": 54321}

        result = resource_manager._get_collection_value(collection_dict, "total_size_bytes")

        assert result == 54321

    def test_get_collection_value_default(self, resource_manager) -> None:
        """Test _get_collection_value returns default for missing key."""
        # Use a simple object without auto-creating attributes (unlike MagicMock)

        class SimpleCollection:
            pass

        collection = SimpleCollection()

        result = resource_manager._get_collection_value(collection, "nonexistent", 999)

        assert result == 999
