#!/usr/bin/env python3
"""
Comprehensive test suite for webui/services/resource_manager.py
Tests resource allocation, limits, and quotas for collection operations
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from webui.services.resource_manager import ResourceEstimate, ResourceManager


class TestResourceEstimate:
    """Test ResourceEstimate class"""

    def test_resource_estimate_creation(self):
        """Test creating resource estimate"""
        estimate = ResourceEstimate(memory_mb=1024, storage_gb=10.5, cpu_cores=2.5, gpu_memory_mb=2048)

        assert estimate.memory_mb == 1024
        assert estimate.storage_gb == 10.5
        assert estimate.cpu_cores == 2.5
        assert estimate.gpu_memory_mb == 2048

    def test_resource_estimate_defaults(self):
        """Test resource estimate with default values"""
        estimate = ResourceEstimate()

        assert estimate.memory_mb == 0
        assert estimate.storage_gb == 0.0
        assert estimate.cpu_cores == 0.0
        assert estimate.gpu_memory_mb == 0

    def test_resource_estimate_string_representation(self):
        """Test string representation of resource estimate"""
        estimate = ResourceEstimate(memory_mb=512, storage_gb=5.0, cpu_cores=1.0, gpu_memory_mb=1024)

        str_repr = str(estimate)
        assert "memory=512MB" in str_repr
        assert "storage=5.0GB" in str_repr
        assert "cpu=1.0" in str_repr
        assert "gpu=1024MB" in str_repr


class TestResourceManager:
    """Test ResourceManager implementation"""

    @pytest.fixture()
    def mock_collection_repo(self):
        """Create a mock collection repository"""
        return AsyncMock()

    @pytest.fixture()
    def mock_operation_repo(self):
        """Create a mock operation repository"""
        return AsyncMock()

    @pytest.fixture()
    def resource_manager(self, mock_collection_repo, mock_operation_repo):
        """Create ResourceManager with mocked dependencies"""
        return ResourceManager(
            collection_repo=mock_collection_repo,
            operation_repo=mock_operation_repo,
        )

    @pytest.mark.asyncio()
    async def test_can_create_collection_within_limit(self, resource_manager, mock_collection_repo):
        """Test checking if user can create collection within limit"""
        user_id = 123

        # Mock user has 3 active collections (under limit of 10)
        mock_collections = [{"id": f"coll-{i}", "status": "active"} for i in range(3)]
        mock_collection_repo.list_by_user.return_value = mock_collections

        result = await resource_manager.can_create_collection(user_id)

        assert result is True
        mock_collection_repo.list_by_user.assert_called_once_with(user_id)

    @pytest.mark.asyncio()
    async def test_can_create_collection_at_limit(self, resource_manager, mock_collection_repo):
        """Test checking if user can create collection when at limit"""
        user_id = 123

        # Mock user has 10 active collections (at limit)
        mock_collections = [{"id": f"coll-{i}", "status": "active"} for i in range(10)]
        mock_collection_repo.list_by_user.return_value = mock_collections

        result = await resource_manager.can_create_collection(user_id)

        assert result is False

    @pytest.mark.asyncio()
    async def test_can_create_collection_with_deleted(self, resource_manager, mock_collection_repo):
        """Test that deleted collections don't count toward limit"""
        user_id = 123

        # Mock user has 8 active and 5 deleted collections
        mock_collections = []
        for i in range(8):
            mock_collections.append({"id": f"active-{i}", "status": "active"})
        for i in range(5):
            mock_collections.append({"id": f"deleted-{i}", "status": "deleted"})

        mock_collection_repo.list_by_user.return_value = mock_collections

        result = await resource_manager.can_create_collection(user_id)

        assert result is True  # 8 active < 10 limit

    @pytest.mark.asyncio()
    async def test_can_create_collection_error_handling(self, resource_manager, mock_collection_repo):
        """Test error handling in can_create_collection"""
        user_id = 123

        # Mock repository error
        mock_collection_repo.list_by_user.side_effect = Exception("Database error")

        result = await resource_manager.can_create_collection(user_id)

        assert result is False  # Safe default on error

    @pytest.mark.asyncio()
    @patch("webui.services.resource_manager.psutil.virtual_memory")
    @patch("webui.services.resource_manager.psutil.disk_usage")
    async def test_can_allocate_sufficient_resources(self, mock_disk_usage, mock_virtual_memory, resource_manager):
        """Test resource allocation when sufficient resources available"""
        user_id = 123

        # Mock system resources (plenty available)
        mock_memory = Mock()
        mock_memory.available = 16 * 1024 * 1024 * 1024  # 16GB
        mock_virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.free = 100 * 1024 * 1024 * 1024  # 100GB
        mock_disk_usage.return_value = mock_disk

        # Mock user usage (well under quota)
        resource_manager._get_user_resource_usage = AsyncMock(return_value={"storage_gb": 10.0})

        # Request moderate resources
        resources = ResourceEstimate(
            memory_mb=2048,  # 2GB
            storage_gb=5.0,  # 5GB
        )

        result = await resource_manager.can_allocate(user_id, resources)

        assert result is True

    @pytest.mark.asyncio()
    @patch("webui.services.resource_manager.psutil.virtual_memory")
    @patch("webui.services.resource_manager.psutil.disk_usage")
    async def test_can_allocate_insufficient_memory(self, mock_disk_usage, mock_virtual_memory, resource_manager):
        """Test resource allocation when insufficient memory"""
        user_id = 123

        # Mock low memory
        mock_memory = Mock()
        mock_memory.available = 1 * 1024 * 1024 * 1024  # 1GB
        mock_virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.free = 100 * 1024 * 1024 * 1024  # 100GB
        mock_disk_usage.return_value = mock_disk

        # Mock user usage
        resource_manager._get_user_resource_usage = AsyncMock(return_value={"storage_gb": 10.0})

        # Request more memory than available
        resources = ResourceEstimate(
            memory_mb=2048,  # 2GB (more than 80% of 1GB)
            storage_gb=1.0,
        )

        result = await resource_manager.can_allocate(user_id, resources)

        assert result is False

    @pytest.mark.asyncio()
    @patch("webui.services.resource_manager.psutil.virtual_memory")
    @patch("webui.services.resource_manager.psutil.disk_usage")
    async def test_can_allocate_insufficient_storage(self, mock_disk_usage, mock_virtual_memory, resource_manager):
        """Test resource allocation when insufficient storage"""
        user_id = 123

        # Mock sufficient memory
        mock_memory = Mock()
        mock_memory.available = 16 * 1024 * 1024 * 1024  # 16GB
        mock_virtual_memory.return_value = mock_memory

        # Mock low disk space
        mock_disk = Mock()
        mock_disk.free = 5 * 1024 * 1024 * 1024  # 5GB
        mock_disk_usage.return_value = mock_disk

        # Mock user usage
        resource_manager._get_user_resource_usage = AsyncMock(return_value={"storage_gb": 10.0})

        # Request more storage than available
        resources = ResourceEstimate(
            memory_mb=512,
            storage_gb=10.0,  # More than 80% of 5GB
        )

        result = await resource_manager.can_allocate(user_id, resources)

        assert result is False

    @pytest.mark.asyncio()
    @patch("webui.services.resource_manager.psutil.virtual_memory")
    @patch("webui.services.resource_manager.psutil.disk_usage")
    async def test_can_allocate_user_quota_exceeded(self, mock_disk_usage, mock_virtual_memory, resource_manager):
        """Test resource allocation when user quota would be exceeded"""
        user_id = 123

        # Mock plenty of system resources
        mock_memory = Mock()
        mock_memory.available = 16 * 1024 * 1024 * 1024  # 16GB
        mock_virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.free = 100 * 1024 * 1024 * 1024  # 100GB
        mock_disk_usage.return_value = mock_disk

        # Mock user already using 45GB (near 50GB quota)
        resource_manager._get_user_resource_usage = AsyncMock(return_value={"storage_gb": 45.0})

        # Request would exceed quota
        resources = ResourceEstimate(
            memory_mb=512,
            storage_gb=10.0,  # Would put user at 55GB
        )

        result = await resource_manager.can_allocate(user_id, resources)

        assert result is False

    @pytest.mark.asyncio()
    async def test_reserve_resources(self, resource_manager):
        """Test reserving resources for an operation"""
        operation_id = "op-123"
        resources = ResourceEstimate(memory_mb=1024, storage_gb=5.0)

        result = await resource_manager.reserve_resources(operation_id, resources)

        assert result is True
        assert operation_id in resource_manager._reserved_resources
        assert resource_manager._reserved_resources[operation_id] == resources

    @pytest.mark.asyncio()
    async def test_release_resources(self, resource_manager):
        """Test releasing reserved resources"""
        operation_id = "op-123"
        resources = ResourceEstimate(memory_mb=1024, storage_gb=5.0)

        # Reserve first
        await resource_manager.reserve_resources(operation_id, resources)
        assert operation_id in resource_manager._reserved_resources

        # Release
        await resource_manager.release_resources(operation_id)
        assert operation_id not in resource_manager._reserved_resources

    @pytest.mark.asyncio()
    async def test_release_resources_not_reserved(self, resource_manager):
        """Test releasing resources that weren't reserved"""
        # Should not raise error
        await resource_manager.release_resources("non-existent-op")

    @pytest.mark.asyncio()
    async def test_estimate_operation_resources_scan(self, resource_manager):
        """Test estimating resources for scan operation"""
        source_path = "/test/path"

        # Mock file system stats
        with patch("webui.services.resource_manager.Path") as mock_path_class:
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.is_dir.return_value = True
            mock_path_class.return_value = mock_path

            with patch("webui.services.resource_manager.os.walk") as mock_walk:
                # Mock directory with some files
                mock_walk.return_value = [
                    ("/test/path", [], ["file1.pdf", "file2.txt"]),
                    ("/test/path/sub", [], ["file3.docx"]),
                ]

                with patch("os.path.getsize") as mock_getsize:
                    # Mock file sizes
                    mock_getsize.side_effect = [
                        10 * 1024 * 1024,  # 10MB
                        5 * 1024 * 1024,  # 5MB
                        15 * 1024 * 1024,  # 15MB
                    ]

                    estimate = await resource_manager.estimate_operation_resources(
                        operation_type="scan",
                        source_path=source_path,
                    )

                    # Should estimate based on file count and sizes
                    assert estimate.memory_mb > 0
                    assert estimate.storage_gb > 0

    @pytest.mark.asyncio()
    async def test_estimate_operation_resources_reindex(self, resource_manager, mock_collection_repo):
        """Test estimating resources for reindex operation"""
        collection_id = "coll-123"

        # Mock collection with documents
        mock_collection = {
            "id": collection_id,
            "documents_count": 100,
            "total_size_bytes": 500 * 1024 * 1024,  # 500MB
        }
        mock_collection_repo.get_by_uuid.return_value = mock_collection

        estimate = await resource_manager.estimate_operation_resources(
            operation_type="reindex",
            collection_id=collection_id,
        )

        # Should estimate based on document count
        assert estimate.memory_mb >= 512  # Base memory
        assert estimate.gpu_memory_mb > 0  # For embeddings

    @pytest.mark.asyncio()
    async def test_get_user_resource_usage(self, resource_manager, mock_collection_repo):
        """Test getting user's current resource usage"""
        user_id = 123

        # Mock user's collections
        mock_collections = [
            {"id": "coll-1", "status": "active", "total_size_bytes": 1 * 1024 * 1024 * 1024},
            {"id": "coll-2", "status": "active", "total_size_bytes": 2 * 1024 * 1024 * 1024},
            {"id": "coll-3", "status": "deleted", "total_size_bytes": 5 * 1024 * 1024 * 1024},
        ]
        mock_collection_repo.list_by_user.return_value = mock_collections

        usage = await resource_manager._get_user_resource_usage(user_id)

        # Should sum only active collections
        assert usage["storage_gb"] == 3.0  # 1GB + 2GB, not the deleted 5GB

    @pytest.mark.asyncio()
    async def test_concurrent_resource_operations(self, resource_manager):
        """Test concurrent resource reservation/release"""
        operation_ids = [f"op-{i}" for i in range(5)]
        resources = [ResourceEstimate(memory_mb=512 * i, storage_gb=float(i)) for i in range(1, 6)]

        # Reserve resources concurrently
        tasks = [
            resource_manager.reserve_resources(op_id, res) for op_id, res in zip(operation_ids, resources, strict=False)
        ]
        results = await asyncio.gather(*tasks)

        assert all(results)
        assert len(resource_manager._reserved_resources) == 5

        # Release some resources concurrently
        release_tasks = [resource_manager.release_resources(op_id) for op_id in operation_ids[:3]]
        await asyncio.gather(*release_tasks)

        assert len(resource_manager._reserved_resources) == 2


class TestResourceManagerEdgeCases:
    """Test edge cases and error scenarios"""

    @pytest.fixture()
    def resource_manager(self):
        mock_collection_repo = AsyncMock()
        mock_operation_repo = AsyncMock()
        return ResourceManager(mock_collection_repo, mock_operation_repo)

    @pytest.mark.asyncio()
    async def test_estimate_resources_invalid_path(self, resource_manager):
        """Test estimating resources with invalid path"""
        estimate = await resource_manager.estimate_operation_resources(
            operation_type="scan",
            source_path="/non/existent/path",
        )

        # Should return minimal estimate
        assert estimate.memory_mb == 256
        assert estimate.storage_gb == 0.1

    @pytest.mark.asyncio()
    async def test_estimate_resources_unknown_operation(self, resource_manager):
        """Test estimating resources for unknown operation type"""
        estimate = await resource_manager.estimate_operation_resources(
            operation_type="unknown_op",
        )

        # Should return default estimate
        assert estimate.memory_mb == 512
        assert estimate.storage_gb == 1.0

    @pytest.mark.asyncio()
    @patch("webui.services.resource_manager.psutil.virtual_memory")
    async def test_can_allocate_psutil_error(self, mock_virtual_memory, resource_manager):
        """Test resource allocation when psutil fails"""
        user_id = 123

        # Mock psutil error
        mock_virtual_memory.side_effect = Exception("System error")

        resources = ResourceEstimate(memory_mb=1024)

        # Should handle error gracefully
        result = await resource_manager.can_allocate(user_id, resources)

        assert result is False  # Safe default
