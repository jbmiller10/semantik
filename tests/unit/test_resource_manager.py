#!/usr/bin/env python3
"""
Comprehensive test suite for webui/services/resource_manager.py
Tests resource allocation, quotas, and management
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from webui.services.resource_manager import ResourceEstimate, ResourceManager


class TestResourceEstimate:
    """Test ResourceEstimate class"""

    def test_resource_estimate_creation(self):
        """Test creating ResourceEstimate"""
        estimate = ResourceEstimate(memory_mb=1024, storage_gb=2.5, cpu_cores=4.0, gpu_memory_mb=2048)

        assert estimate.memory_mb == 1024
        assert estimate.storage_gb == 2.5
        assert estimate.cpu_cores == 4.0
        assert estimate.gpu_memory_mb == 2048

    def test_resource_estimate_defaults(self):
        """Test ResourceEstimate default values"""
        estimate = ResourceEstimate()

        assert estimate.memory_mb == 0
        assert estimate.storage_gb == 0.0
        assert estimate.cpu_cores == 0.0
        assert estimate.gpu_memory_mb == 0

    def test_resource_estimate_string_representation(self):
        """Test ResourceEstimate string representation"""
        estimate = ResourceEstimate(memory_mb=512, storage_gb=1.0, cpu_cores=2.0, gpu_memory_mb=1024)

        expected = "ResourceEstimate(memory=512MB, storage=1.0GB, cpu=2.0, gpu=1024MB)"
        assert str(estimate) == expected


class TestResourceManager:
    """Test ResourceManager implementation"""

    @pytest.fixture()
    def mock_collection_repo(self):
        """Create a mock CollectionRepository"""
        return AsyncMock()

    @pytest.fixture()
    def mock_operation_repo(self):
        """Create a mock OperationRepository"""
        return AsyncMock()

    @pytest.fixture()
    def resource_manager(self, mock_collection_repo, mock_operation_repo):
        """Create ResourceManager with mocked dependencies"""
        return ResourceManager(
            collection_repo=mock_collection_repo,
            operation_repo=mock_operation_repo,
        )

    @pytest.mark.asyncio()
    async def test_can_create_collection_success(self, resource_manager, mock_collection_repo):
        """Test can_create_collection when under limit"""
        # Mock user has 3 active collections
        mock_collections = [
            {"id": "1", "status": "ready"},
            {"id": "2", "status": "ready"},
            {"id": "3", "status": "processing"},
        ]
        mock_collection_repo.list_by_user.return_value = mock_collections

        # Should be able to create (3 < 10)
        result = await resource_manager.can_create_collection(123)
        assert result is True

    @pytest.mark.asyncio()
    async def test_can_create_collection_at_limit(self, resource_manager, mock_collection_repo):
        """Test can_create_collection when at limit"""
        # Mock user has 10 active collections
        mock_collections = [{"id": str(i), "status": "ready"} for i in range(10)]
        mock_collection_repo.list_by_user.return_value = mock_collections

        # Should not be able to create (10 >= 10)
        result = await resource_manager.can_create_collection(123)
        assert result is False

    @pytest.mark.asyncio()
    async def test_can_create_collection_excludes_deleted(self, resource_manager, mock_collection_repo):
        """Test can_create_collection excludes deleted collections"""
        # Mock user has 9 active + 5 deleted collections
        mock_collections = [{"id": str(i), "status": "ready"} for i in range(9)]
        mock_collections.extend([{"id": str(i + 9), "status": "deleted"} for i in range(5)])
        mock_collection_repo.list_by_user.return_value = mock_collections

        # Should be able to create (9 active < 10)
        result = await resource_manager.can_create_collection(123)
        assert result is True

    @pytest.mark.asyncio()
    @patch("webui.services.resource_manager.psutil.virtual_memory")
    @patch("webui.services.resource_manager.psutil.disk_usage")
    async def test_can_allocate_sufficient_resources(
        self, mock_disk_usage, mock_virtual_memory, resource_manager, mock_collection_repo
    ):
        """Test can_allocate with sufficient system resources"""
        # Mock system resources
        mock_memory = Mock()
        mock_memory.available = 8 * 1024 * 1024 * 1024  # 8GB
        mock_virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.free = 100 * 1024 * 1024 * 1024  # 100GB
        mock_disk_usage.return_value = mock_disk

        # Mock user resource usage
        mock_collections = [{"total_size_bytes": 1024 * 1024 * 1024}]  # 1GB
        mock_collection_repo.list_by_user.return_value = mock_collections

        # Request moderate resources
        resources = ResourceEstimate(memory_mb=1024, storage_gb=5.0)

        result = await resource_manager.can_allocate(123, resources)
        assert result is True

    @pytest.mark.asyncio()
    @patch("webui.services.resource_manager.psutil.virtual_memory")
    async def test_can_allocate_insufficient_memory(self, mock_virtual_memory, resource_manager, mock_collection_repo):
        """Test can_allocate with insufficient memory"""
        # Mock low memory
        mock_memory = Mock()
        mock_memory.available = 1 * 1024 * 1024 * 1024  # 1GB
        mock_virtual_memory.return_value = mock_memory

        # Request too much memory
        resources = ResourceEstimate(memory_mb=2048)  # 2GB

        result = await resource_manager.can_allocate(123, resources)
        assert result is False

    @pytest.mark.asyncio()
    async def test_estimate_resources_single_file(self, resource_manager):
        """Test resource estimation for single file"""
        # Create a temporary file
        import tempfile

        with tempfile.NamedTemporaryFile() as temp_file:
            # Write 10MB of data
            temp_file.write(b"x" * (10 * 1024 * 1024))
            temp_file.flush()

            estimate = await resource_manager.estimate_resources(temp_file.name, "BAAI/bge-base-en-v1.5")

            # Should estimate based on file size and model
            assert estimate.memory_mb > 0
            assert estimate.storage_gb > 0
            assert estimate.cpu_cores >= 1.0
            assert estimate.gpu_memory_mb == 400  # Model size for BGE

    @pytest.mark.asyncio()
    async def test_estimate_resources_directory(self, resource_manager):
        """Test resource estimation for directory"""
        # Create a temporary directory with files
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some files
            for i in range(5):
                file_path = Path(temp_dir) / f"file_{i}.txt"
                file_path.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB each

            estimate = await resource_manager.estimate_resources(temp_dir, "sentence-transformers/all-MiniLM-L6-v2")

            # Should sum all file sizes
            assert estimate.memory_mb > 0
            assert estimate.storage_gb > 0
            assert estimate.cpu_cores >= 1.0
            assert estimate.gpu_memory_mb == 100  # Model size for MiniLM

    @pytest.mark.asyncio()
    async def test_reserve_for_reindex_success(self, resource_manager, mock_collection_repo):
        """Test reserving resources for reindex"""
        # Mock collection
        mock_collection = {
            "id": "test-collection",
            "total_size_bytes": 5 * 1024 * 1024 * 1024,  # 5GB
        }
        mock_collection_repo.get_by_id.return_value = mock_collection

        # Mock sufficient system resources
        with patch.object(resource_manager, "_check_system_resources", return_value=True):
            result = await resource_manager.reserve_for_reindex("test-collection")
            assert result is True

            # Should have reserved resources
            assert "reindex_test-collection" in resource_manager._reserved_resources

    @pytest.mark.asyncio()
    async def test_reserve_for_reindex_insufficient_resources(self, resource_manager, mock_collection_repo):
        """Test reserve_for_reindex with insufficient resources"""
        # Mock collection
        mock_collection = {
            "id": "test-collection",
            "total_size_bytes": 50 * 1024 * 1024 * 1024,  # 50GB
        }
        mock_collection_repo.get_by_id.return_value = mock_collection

        # Mock insufficient system resources
        with patch.object(resource_manager, "_check_system_resources", return_value=False):
            result = await resource_manager.reserve_for_reindex("test-collection")
            assert result is False

            # Should not have reserved resources
            assert "reindex_test-collection" not in resource_manager._reserved_resources

    @pytest.mark.asyncio()
    async def test_release_reindex_reservation(self, resource_manager):
        """Test releasing reindex reservation"""
        # Manually add a reservation
        resource_manager._reserved_resources["reindex_test-collection"] = ResourceEstimate(memory_mb=1024)

        # Release it
        await resource_manager.release_reindex_reservation("test-collection")

        # Should be removed
        assert "reindex_test-collection" not in resource_manager._reserved_resources

    @pytest.mark.asyncio()
    async def test_get_resource_usage(self, resource_manager, mock_collection_repo):
        """Test getting resource usage for collection"""
        # Mock collection with usage info
        mock_collection = {
            "id": "test-collection",
            "document_count": 1000,
            "vector_count": 5000,
            "total_size_bytes": 2 * 1024 * 1024 * 1024,  # 2GB
        }
        mock_collection_repo.get_by_id.return_value = mock_collection

        usage = await resource_manager.get_resource_usage("test-collection")

        assert usage["documents"] == 1000
        assert usage["vectors"] == 5000
        assert usage["storage_bytes"] == 2 * 1024 * 1024 * 1024
        assert usage["storage_gb"] == 2.0

    @pytest.mark.asyncio()
    async def test_get_user_resource_usage(self, resource_manager, mock_collection_repo):
        """Test getting total resource usage for user"""
        # Mock user's collections
        mock_collections = [
            {"total_size_bytes": 1 * 1024 * 1024 * 1024},  # 1GB
            {"total_size_bytes": 2 * 1024 * 1024 * 1024},  # 2GB
            {"total_size_bytes": 5 * 1024 * 1024 * 1024},  # 5GB
        ]
        mock_collection_repo.list_by_user.return_value = mock_collections

        # Call private method directly
        usage = await resource_manager._get_user_resource_usage(123)

        assert usage["collections"] == 3
        assert usage["storage_gb"] == 8.0

    @pytest.mark.asyncio()
    async def test_concurrent_resource_reservation(self, resource_manager, mock_collection_repo):
        """Test concurrent resource reservation with lock"""
        # Mock collections
        mock_collection = {
            "id": "test-collection",
            "total_size_bytes": 1 * 1024 * 1024 * 1024,  # 1GB
        }
        mock_collection_repo.get_by_id.return_value = mock_collection

        # Mock sufficient resources
        with patch.object(resource_manager, "_check_system_resources", return_value=True):
            # Reserve resources concurrently
            results = await asyncio.gather(
                resource_manager.reserve_for_reindex("collection-1"),
                resource_manager.reserve_for_reindex("collection-2"),
                resource_manager.reserve_for_reindex("collection-3"),
            )

            # All should succeed
            assert all(results)

            # All should be reserved
            assert len(resource_manager._reserved_resources) == 3


class TestResourceManagerEdgeCases:
    """Test edge cases for ResourceManager"""

    @pytest.mark.asyncio()
    async def test_estimate_resources_invalid_path(self):
        """Test resource estimation with invalid path"""
        manager = ResourceManager(AsyncMock(), AsyncMock())

        # Should handle non-existent path gracefully
        estimate = await manager.estimate_resources("/non/existent/path", "some-model")

        # When path doesn't exist, no files are found, so size_gb = 0
        # Memory: 0 * 2048 + 1000 (default model) + 500 = 1500MB
        # Storage: 0 * 1.75 = 0GB
        # CPU: max(0/10, 0, 1.0) = 1.0
        assert estimate.memory_mb == 1500  # 1000MB default model + 500MB overhead
        assert estimate.storage_gb == 0.0  # No files found
        assert estimate.cpu_cores == 1.0

    @pytest.mark.asyncio()
    async def test_estimate_resources_unknown_model(self):
        """Test resource estimation with unknown model"""
        manager = ResourceManager(AsyncMock(), AsyncMock())

        import tempfile

        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(b"x" * 1024)  # 1KB
            temp_file.flush()

            estimate = await manager.estimate_resources(temp_file.name, "unknown/model-name")

            # Should use default model size (1000MB)
            assert estimate.gpu_memory_mb == 1000

    @pytest.mark.asyncio()
    @patch("webui.services.resource_manager.psutil.virtual_memory")
    @patch("webui.services.resource_manager.psutil.disk_usage")
    async def test_check_system_resources_with_reserved(self, mock_disk_usage, mock_virtual_memory):
        """Test system resource check considering reserved resources"""
        manager = ResourceManager(AsyncMock(), AsyncMock())

        # Add some reserved resources
        manager._reserved_resources["op1"] = ResourceEstimate(memory_mb=1024, storage_gb=10.0)
        manager._reserved_resources["op2"] = ResourceEstimate(memory_mb=2048, storage_gb=20.0)

        # Mock system resources
        mock_memory = Mock()
        mock_memory.available = 8 * 1024 * 1024 * 1024  # 8GB
        mock_virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.free = 100 * 1024 * 1024 * 1024  # 100GB
        mock_disk_usage.return_value = mock_disk

        # Request resources that would definitely exceed available resources after reserved
        # Available after reserved: 8192MB - 3072MB = 5120MB memory, 100GB - 30GB = 70GB storage
        # 80% of available: 4096MB memory, 56GB storage
        # Request more than that
        estimate = ResourceEstimate(memory_mb=5000, storage_gb=60.0)

        result = await manager._check_system_resources(estimate)

        # Should fail due to reserved resources
        assert result is False

    def test_is_gpu_model(self):
        """Test GPU model detection"""
        manager = ResourceManager(AsyncMock(), AsyncMock())

        # Currently returns True for all models
        assert manager._is_gpu_model("any-model") is True
        assert manager._is_gpu_model("cpu-only-model") is True  # Still returns True


class TestResourceManagerIntegration:
    """Test ResourceManager integration scenarios"""

    @pytest.mark.asyncio()
    async def test_full_allocation_workflow(self):
        """Test complete resource allocation workflow"""
        mock_collection_repo = AsyncMock()
        mock_operation_repo = AsyncMock()
        manager = ResourceManager(mock_collection_repo, mock_operation_repo)

        # Setup mocks
        mock_collection_repo.list_by_user.return_value = []  # No existing collections
        mock_collection_repo.get_by_id.return_value = {
            "id": "test-collection",
            "total_size_bytes": 1024 * 1024 * 1024,  # 1GB
        }

        # Check if user can create collection
        can_create = await manager.can_create_collection(123)
        assert can_create is True

        # Estimate resources for a path
        import tempfile

        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(b"x" * 1024 * 1024)  # 1MB
            temp_file.flush()

            estimate = await manager.estimate_resources(temp_file.name, "BAAI/bge-base-en-v1.5")
            assert estimate.memory_mb > 0

        # Reserve resources for reindex
        with patch.object(manager, "_check_system_resources", return_value=True):
            reserved = await manager.reserve_for_reindex("test-collection")
            assert reserved is True

        # Get resource usage
        usage = await manager.get_resource_usage("test-collection")
        assert "storage_gb" in usage

        # Release reservation
        await manager.release_reindex_reservation("test-collection")
        assert len(manager._reserved_resources) == 0

    @pytest.mark.asyncio()
    async def test_rate_limiting_disabled(self):
        """Test that rate limiting is disabled"""
        mock_collection_repo = AsyncMock()
        mock_operation_repo = AsyncMock()
        manager = ResourceManager(mock_collection_repo, mock_operation_repo)

        # Mock many recent operations
        recent_ops = [{"id": str(i)} for i in range(100)]
        mock_operation_repo.list_by_user.return_value = recent_ops

        # Get recent operations count
        count = await manager._get_recent_operations_count(123, hours=1)
        assert count == 100

        # But can_allocate should not check rate limits
        # (The rate limit code is commented out in the implementation)
        with patch.object(manager, "_get_user_resource_usage") as mock_get_usage:
            mock_get_usage.return_value = {"storage_gb": 1.0}

            with patch("webui.services.resource_manager.psutil.virtual_memory") as mock_mem:
                with patch("webui.services.resource_manager.psutil.disk_usage") as mock_disk:
                    mock_mem.return_value = Mock(available=10 * 1024 * 1024 * 1024)
                    mock_disk.return_value = Mock(free=100 * 1024 * 1024 * 1024)

                    resources = ResourceEstimate(memory_mb=100, storage_gb=1.0)
                    result = await manager.can_allocate(123, resources)

                    # Should succeed despite many recent operations
                    assert result is True
