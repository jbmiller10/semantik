"""
Unit tests for the collections API endpoints.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from qdrant_client.models import CollectionInfo
from webui.api.collections import (
    CollectionDetails,
    CollectionRenameRequest,
    CollectionSummary,
    PaginatedFileList,
    delete_collection,
    get_collection_details,
    get_collection_files,
    list_collections,
    rename_collection,
)


def setup_async_mock(mock_obj, method_name, return_value):
    """Helper to set up async mock methods."""
    async def async_mock(*args, **kwargs):
        return return_value
    setattr(mock_obj, method_name, MagicMock(side_effect=async_mock))


@pytest.fixture()
def mock_current_user():
    """Mock authenticated user"""
    return {"id": "user123", "username": "testuser"}




@pytest.fixture()
def mock_qdrant_manager():
    """Create a mock qdrant manager"""
    with patch("webui.api.collections.qdrant_manager") as mock_qm:
        mock_client = MagicMock()
        mock_qm.get_client.return_value = mock_client
        yield mock_qm, mock_client


@pytest.mark.usefixtures("mock_qdrant_manager")
class TestListCollections:
    """Test cases for list_collections endpoint"""

    @pytest.mark.asyncio()
    async def test_list_collections_single_job_per_collection(self, mock_collection_repository, mock_current_user):
        """Test listing collections where each collection has a single job"""
        # Mock repository response
        async def mock_list_collections(*args, **kwargs):
            return [
            {
                "name": "research_papers",
                "total_files": 10,
                "total_vectors": 100,
                "model_name": "text-embedding-ada-002",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-02T00:00:00",
                "job_count": 1,
            },
            {
                "name": "documentation",
                "total_files": 5,
                "total_vectors": 50,
                "model_name": "all-MiniLM-L6-v2",
                "created_at": "2024-01-03T00:00:00",
                "updated_at": "2024-01-04T00:00:00",
                "job_count": 1,
            },
        ]
        
        mock_collection_repository.list_collections.side_effect = mock_list_collections

        # Call the function with repository
        result = await list_collections(
            current_user=mock_current_user,
            collection_repo=mock_collection_repository
        )

        # Verify repository call
        mock_collection_repository.list_collections.assert_called_once_with(user_id="user123")

        # Verify result
        assert len(result) == 2
        assert isinstance(result[0], CollectionSummary)
        assert result[0].name == "research_papers"
        assert result[0].total_files == 10
        assert result[0].total_vectors == 100
        assert result[0].model_name == "text-embedding-ada-002"
        assert result[0].job_count == 1

        assert result[1].name == "documentation"
        assert result[1].total_files == 5
        assert result[1].total_vectors == 50
        assert result[1].model_name == "all-MiniLM-L6-v2"
        assert result[1].job_count == 1

    @pytest.mark.asyncio()
    async def test_list_collections_multiple_jobs_aggregation(self, mock_collection_repository, mock_current_user):
        """Test listing collections with proper aggregation of multiple jobs"""
        # Mock repository response with aggregated data
        setup_async_mock(mock_collection_repository, 'list_collections', [
            {
                "name": "large_dataset",
                "total_files": 150,  # Aggregated from 3 jobs
                "total_vectors": 1500,  # Aggregated
                "model_name": "text-embedding-ada-002",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-05T00:00:00",
                "job_count": 3,  # Multiple jobs
            }
        ])

        # Call the function
        result = await list_collections(
            current_user=mock_current_user,
            collection_repo=mock_collection_repository
        )

        # Verify result shows aggregated data
        assert len(result) == 1
        assert result[0].name == "large_dataset"
        assert result[0].total_files == 150
        assert result[0].total_vectors == 1500
        assert result[0].job_count == 3

    @pytest.mark.asyncio()
    async def test_list_collections_handles_null_values(self, mock_collection_repository, mock_current_user):
        """Test that null values in database response are handled correctly"""
        # Mock database response with null values
        setup_async_mock(mock_collection_repository, 'list_collections', [
            {
                "name": "incomplete_collection",
                "total_files": None,
                "total_vectors": None,
                "model_name": None,
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "job_count": 1,
            }
        ])

        # Call the function
        result = await list_collections(current_user=mock_current_user, collection_repo=mock_collection_repository)

        # Verify null values are converted to defaults
        assert result[0].total_files == 0
        assert result[0].total_vectors == 0
        assert result[0].model_name == "Unknown"

    @pytest.mark.asyncio()
    async def test_list_collections_error_handling(self, mock_collection_repository, mock_current_user):
        """Test error handling in list_collections"""
        # Mock database error
        mock_collection_repository.list_collections.side_effect = Exception("Database error")

        # Verify HTTPException is raised
        with pytest.raises(HTTPException) as exc_info:
            await list_collections(current_user=mock_current_user, collection_repo=mock_collection_repository)

        assert exc_info.value.status_code == 500
        assert "Database error" in str(exc_info.value.detail)


class TestGetCollectionDetails:
    """Test cases for get_collection_details endpoint"""

    @pytest.mark.asyncio()
    async def test_get_collection_details_single_job(self, mock_collection_repository, mock_qdrant_manager, mock_current_user):
        """Test getting details for a collection with a single job"""
        mock_qm, mock_client = mock_qdrant_manager

        # Mock database response
        setup_async_mock(mock_collection_repository, 'get_collection_details', {
            "name": "research_papers",
            "stats": {
                "total_files": 10,
                "total_vectors": 100,
                "total_size": 1000000,
                "job_count": 1,
            },
            "configuration": {
                "model_name": "text-embedding-ada-002",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "quantization": "none",
                "vector_dim": 1536,
                "instruction": None,
            },
            "source_directories": ["/data/papers"],
            "jobs": [
                {
                    "id": "job123",
                    "status": "completed",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-02T00:00:00",
                    "directory_path": "/data/papers",
                    "total_files": 10,
                    "processed_files": 10,
                    "failed_files": 0,
                    "mode": "full",
                }
            ],
        })

        # Mock Qdrant response
        mock_info = MagicMock(spec=CollectionInfo)
        mock_info.points_count = 150  # Different from database count
        mock_info.segments_count = 1
        mock_info.status = "green"
        mock_client.get_collection.return_value = mock_info

        # Call the function
        result = await get_collection_details(collection_name="research_papers", current_user=mock_current_user, collection_repo=mock_collection_repository)

        # Verify database call
        mock_collection_repository.get_collection_details.assert_called_once_with("research_papers", user_id="user123")

        # Verify Qdrant call
        mock_client.get_collection.assert_called_once_with("job_job123")

        # Verify result
        assert isinstance(result, CollectionDetails)
        assert result.name == "research_papers"
        assert result.stats.total_files == 10
        assert result.stats.total_vectors == 150  # Updated from Qdrant
        assert result.configuration.model_name == "text-embedding-ada-002"
        assert len(result.jobs) == 1
        assert result.jobs[0].id == "job123"

    @pytest.mark.asyncio()
    async def test_get_collection_details_multiple_jobs(self, mock_collection_repository, mock_qdrant_manager, mock_current_user):
        """Test getting details for a collection with multiple jobs (parent/child structure)"""
        mock_qm, mock_client = mock_qdrant_manager

        # Mock database response with parent and child jobs
        setup_async_mock(mock_collection_repository, 'get_collection_details', {
            "name": "large_dataset",
            "stats": {
                "total_files": 150,
                "total_vectors": 1000,
                "total_size": 10000000,
                "job_count": 3,
            },
            "configuration": {
                "model_name": "text-embedding-ada-002",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "quantization": "scalar",
                "vector_dim": 1536,
                "instruction": "Extract key concepts",
            },
            "source_directories": ["/data/part1", "/data/part2", "/data/part3"],
            "jobs": [
                {  # Parent job
                    "id": "parent_job",
                    "status": "completed",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T12:00:00",
                    "directory_path": "/data/part1",
                    "total_files": 50,
                    "processed_files": 50,
                    "failed_files": 0,
                    "mode": "full",
                },
                {  # Child job 1
                    "id": "child_job1",
                    "status": "completed",
                    "created_at": "2024-01-02T00:00:00",
                    "updated_at": "2024-01-02T12:00:00",
                    "directory_path": "/data/part2",
                    "total_files": 50,
                    "processed_files": 50,
                    "failed_files": 0,
                    "mode": "update",
                },
                {  # Child job 2
                    "id": "child_job2",
                    "status": "completed",
                    "created_at": "2024-01-03T00:00:00",
                    "updated_at": "2024-01-03T12:00:00",
                    "directory_path": "/data/part3",
                    "total_files": 50,
                    "processed_files": 50,
                    "failed_files": 0,
                    "mode": "update",
                },
            ],
        })

        # Mock Qdrant responses for each job
        mock_info1 = MagicMock(spec=CollectionInfo)
        mock_info1.points_count = 500
        mock_info2 = MagicMock(spec=CollectionInfo)
        mock_info2.points_count = 600
        mock_info3 = MagicMock(spec=CollectionInfo)
        mock_info3.points_count = 700

        mock_client.get_collection.side_effect = [mock_info1, mock_info2, mock_info3]

        # Call the function
        result = await get_collection_details(collection_name="large_dataset", current_user=mock_current_user, collection_repo=mock_collection_repository)

        # Verify Qdrant was called for each job
        assert mock_client.get_collection.call_count == 3
        mock_client.get_collection.assert_any_call("job_parent_job")
        mock_client.get_collection.assert_any_call("job_child_job1")
        mock_client.get_collection.assert_any_call("job_child_job2")

        # Verify result
        assert result.name == "large_dataset"
        assert result.stats.total_files == 150
        assert result.stats.total_vectors == 1800  # 500 + 600 + 700
        assert result.stats.job_count == 3
        assert len(result.jobs) == 3
        assert len(result.source_directories) == 3

    @pytest.mark.asyncio()
    async def test_get_collection_details_not_found(self, mock_collection_repository, mock_current_user):
        """Test getting details for non-existent collection"""
        # Mock database returning None
        setup_async_mock(mock_collection_repository, 'get_collection_details', None)

        # Verify HTTPException is raised
        with pytest.raises(HTTPException) as exc_info:
            await get_collection_details(collection_name="nonexistent", current_user=mock_current_user, collection_repo=mock_collection_repository)

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_get_collection_details_qdrant_error_handling(
        self, mock_collection_repository, mock_qdrant_manager, mock_current_user
    ):
        """Test that Qdrant errors are handled gracefully"""
        mock_qm, mock_client = mock_qdrant_manager

        # Mock database response
        setup_async_mock(mock_collection_repository, 'get_collection_details', {
            "name": "test_collection",
            "stats": {
                "total_files": 10,
                "total_vectors": 100,
                "total_size": 1000000,
                "job_count": 1,
            },
            "configuration": {
                "model_name": "text-embedding-ada-002",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "quantization": "none",
                "vector_dim": 1536,
                "instruction": None,
            },
            "source_directories": ["/data"],
            "jobs": [
                {
                    "id": "job123",
                    "status": "completed",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-02T00:00:00",
                    "directory_path": "/data",
                    "total_files": 10,
                    "processed_files": 10,
                    "failed_files": 0,
                    "mode": "full",
                }
            ],
        })

        # Mock Qdrant error
        mock_client.get_collection.side_effect = Exception("Qdrant connection error")

        # Call the function - should not raise exception
        result = await get_collection_details(collection_name="test_collection", current_user=mock_current_user, collection_repo=mock_collection_repository)

        # Verify result uses database vector count
        assert result.stats.total_vectors == 100  # Falls back to database count


class TestRenameCollection:
    """Test cases for rename_collection endpoint"""

    @pytest.mark.asyncio()
    async def test_rename_collection_success(self, mock_collection_repository, mock_current_user):
        """Test successful collection rename"""
        # Mock successful rename
        setup_async_mock(mock_collection_repository, 'rename_collection', True)

        # Create rename request
        request = CollectionRenameRequest(new_name="new_collection_name")

        # Call the function
        result = await rename_collection(
            collection_name="old_name",
            request=request,
            current_user=mock_current_user,
            collection_repo=mock_collection_repository,
        )

        # Verify database call
        mock_collection_repository.rename_collection.assert_called_once_with(
            old_name="old_name",
            new_name="new_collection_name",
            user_id="user123",
        )

        # Verify result
        assert result["message"] == "Collection renamed successfully"
        assert result["new_name"] == "new_collection_name"

    @pytest.mark.asyncio()
    async def test_rename_collection_validation(self):
        """Test collection name validation"""
        # Test invalid characters
        with pytest.raises(ValueError, match="cannot contain"):
            CollectionRenameRequest(new_name="invalid/name")

        # Test empty name
        with pytest.raises(ValueError, match="cannot be empty"):
            CollectionRenameRequest(new_name="   ")

        # Test valid name with spaces (should be trimmed)
        request = CollectionRenameRequest(new_name="  valid name  ")
        assert request.new_name == "valid name"

    @pytest.mark.asyncio()
    async def test_rename_collection_failure(self, mock_collection_repository, mock_current_user):
        """Test failed collection rename (e.g., name already exists)"""
        # Mock failed rename
        setup_async_mock(mock_collection_repository, 'rename_collection', False)

        # Create rename request
        request = CollectionRenameRequest(new_name="existing_name")

        # Verify HTTPException is raised
        with pytest.raises(HTTPException) as exc_info:
            await rename_collection(
                collection_name="old_name",
                request=request,
                current_user=mock_current_user,
                collection_repo=mock_collection_repository,
            )

        assert exc_info.value.status_code == 400
        assert "Failed to rename" in str(exc_info.value.detail)


class TestDeleteCollection:
    """Test cases for delete_collection endpoint"""

    @pytest.mark.asyncio()
    async def test_delete_collection_single_job(self, mock_collection_repository, mock_qdrant_manager, mock_current_user):
        """Test deleting a collection with a single job"""
        mock_qm, mock_client = mock_qdrant_manager

        # Mock database response
        setup_async_mock(mock_collection_repository, 'delete_collection', {
            "job_ids": ["job123"],
            "qdrant_collections": ["job_job123"],
        })

        # Mock successful Qdrant deletion
        mock_client.delete_collection.return_value = None

        # Mock filesystem (patch shutil)
        with patch("webui.api.collections.shutil.rmtree"), patch("webui.api.collections.Path") as mock_path:
            # Mock path existence
            mock_job_dir = MagicMock()
            mock_job_dir.exists.return_value = True
            mock_output_dir = MagicMock()
            mock_output_dir.exists.return_value = True

            mock_path.return_value.__truediv__.side_effect = [
                mock_job_dir,  # /app/jobs/job123
                mock_output_dir,  # /app/output/job123
            ]

            # Call the function
            result = await delete_collection(
                collection_name="test_collection",
                current_user=mock_current_user,
                collection_repo=mock_collection_repository,
            )

        # Verify database call
        mock_collection_repository.delete_collection.assert_called_once_with(collection_name="test_collection", user_id="user123")

        # Verify Qdrant deletion
        mock_client.delete_collection.assert_called_once_with("job_job123")

        # Verify result
        assert result["deleted"]["jobs"] == 1
        assert result["deleted"]["qdrant_collections"] == 1
        assert result["deleted"]["artifacts"] == 2
        assert len(result["errors"]["qdrant_failures"]) == 0
        assert len(result["errors"]["artifact_failures"]) == 0

    @pytest.mark.asyncio()
    async def test_delete_collection_multiple_jobs(self, mock_collection_repository, mock_qdrant_manager, mock_current_user):
        """Test deleting a collection with multiple jobs"""
        mock_qm, mock_client = mock_qdrant_manager

        # Mock repository response with multiple jobs
        setup_async_mock(mock_collection_repository, 'delete_collection', {
            "job_ids": ["job1", "job2", "job3"],
            "qdrant_collections": ["job_job1", "job_job2", "job_job3"],
        })

        # Mock successful Qdrant deletions
        mock_client.delete_collection.return_value = None

        # Mock filesystem
        with patch("webui.api.collections.shutil.rmtree"), patch("webui.api.collections.Path") as mock_path:
            # All paths exist
            mock_path.return_value.__truediv__.return_value.exists.return_value = True

            # Call the function
            result = await delete_collection(
                collection_name="large_collection",
                current_user=mock_current_user,
                collection_repo=mock_collection_repository
            )

        # Verify Qdrant deletions
        assert mock_client.delete_collection.call_count == 3
        mock_client.delete_collection.assert_any_call("job_job1")
        mock_client.delete_collection.assert_any_call("job_job2")
        mock_client.delete_collection.assert_any_call("job_job3")

        # Verify result
        assert result["deleted"]["jobs"] == 3
        assert result["deleted"]["qdrant_collections"] == 3
        assert result["deleted"]["artifacts"] == 6  # 2 directories per job

    @pytest.mark.asyncio()
    async def test_delete_collection_with_failures(self, mock_collection_repository, mock_qdrant_manager, mock_current_user):
        """Test delete collection with some operations failing"""
        mock_qm, mock_client = mock_qdrant_manager

        # Mock repository response
        setup_async_mock(mock_collection_repository, 'delete_collection', {
            "job_ids": ["job1", "job2"],
            "qdrant_collections": ["job_job1", "job_job2"],
        })

        # Mock one Qdrant deletion failing
        mock_client.delete_collection.side_effect = [
            None,  # First succeeds
            Exception("Qdrant error"),  # Second fails
        ]

        # Mock filesystem with one failure
        with patch("webui.api.collections.shutil.rmtree") as mock_rmtree:
            mock_rmtree.side_effect = [
                None,  # First job dir succeeds
                None,  # First output dir succeeds
                Exception("Permission denied"),  # Second job dir fails
                None,  # Second output dir succeeds
            ]

            with patch("webui.api.collections.Path") as mock_path:
                # All paths exist
                mock_path.return_value.__truediv__.return_value.exists.return_value = True

                # Call the function
                result = await delete_collection(
                    collection_name="problematic_collection",
                    current_user=mock_current_user,
                    collection_repo=mock_collection_repository
                )

        # Verify partial success
        assert result["deleted"]["jobs"] == 2
        assert result["deleted"]["qdrant_collections"] == 1
        assert result["deleted"]["artifacts"] == 3
        assert len(result["errors"]["qdrant_failures"]) == 1
        assert "job_job2" in result["errors"]["qdrant_failures"]
        assert len(result["errors"]["artifact_failures"]) == 1

    @pytest.mark.asyncio()
    async def test_delete_collection_not_found(self, mock_collection_repository, mock_current_user):
        """Test deleting non-existent collection"""
        # Mock empty response from repository
        setup_async_mock(mock_collection_repository, 'delete_collection', {
            "job_ids": [],
            "qdrant_collections": [],
        })

        # Verify HTTPException is raised
        with pytest.raises(HTTPException) as exc_info:
            await delete_collection(
                collection_name="nonexistent",
                current_user=mock_current_user,
                collection_repo=mock_collection_repository
            )

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail)


class TestGetCollectionFiles:
    """Test cases for get_collection_files endpoint"""

    @pytest.mark.asyncio()
    async def test_get_collection_files_paginated(self, mock_collection_repository, mock_current_user):
        """Test getting paginated list of files in a collection"""
        # Mock repository response
        setup_async_mock(mock_collection_repository, 'get_collection_files', {
            "files": [
                {
                    "id": 1,
                    "job_id": "job123",
                    "path": "/data/file1.txt",
                    "size": 1024,
                    "modified": "2024-01-01T00:00:00",
                    "extension": "txt",
                    "status": "processed",
                    "chunks_created": 10,
                    "vectors_created": 10,
                    "collection_name": "test_collection",
                },
                {
                    "id": 2,
                    "job_id": "job123",
                    "path": "/data/file2.pdf",
                    "size": 2048,
                    "modified": "2024-01-02T00:00:00",
                    "extension": "pdf",
                    "status": "processed",
                    "chunks_created": 20,
                    "vectors_created": 20,
                    "collection_name": "test_collection",
                },
            ],
            "total": 50,
            "page": 1,
            "pages": 5,
        })

        # Call the function
        result = await get_collection_files(
            collection_name="test_collection",
            page=1,
            limit=10,
            current_user=mock_current_user,
            collection_repo=mock_collection_repository
        )

        # Verify repository call
        mock_collection_repository.get_collection_files.assert_called_once_with(
            collection_name="test_collection",
            user_id="user123",
            page=1,
            limit=10,
        )

        # Verify result
        assert isinstance(result, PaginatedFileList)
        assert len(result.files) == 2
        assert result.total == 50
        assert result.page == 1
        assert result.pages == 5
        assert result.files[0].path == "/data/file1.txt"
        assert result.files[1].path == "/data/file2.pdf"

    @pytest.mark.asyncio()
    async def test_get_collection_files_error_handling(self, mock_collection_repository, mock_current_user):
        """Test error handling in get_collection_files"""
        # Mock repository error
        mock_collection_repository.get_collection_files.side_effect = Exception("Database error")

        # Verify HTTPException is raised
        with pytest.raises(HTTPException) as exc_info:
            await get_collection_files(
                collection_name="test_collection",
                page=1,
                limit=10,
                current_user=mock_current_user,
                collection_repo=mock_collection_repository
            )

        assert exc_info.value.status_code == 500
        assert "Database error" in str(exc_info.value.detail)
