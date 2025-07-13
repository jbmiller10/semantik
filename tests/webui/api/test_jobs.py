"""Tests for job creation and management endpoints"""

from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient
from shared.config import settings
from webui.api.jobs import ConnectionManager


class TestJobCreation:
    """Test job creation endpoint"""

    def test_create_job_no_files(self, test_client: TestClient, monkeypatch):
        """Test job creation with no supported files"""
        # Mock the scan_directory_async to return empty list to avoid file processing
        mock_scan = AsyncMock(return_value={"files": []})
        monkeypatch.setattr("webui.api.files.scan_directory_async", mock_scan)

        # Create job request
        request_data = {
            "name": "Test Job",
            "description": "Test description",
            "directory_path": "/test/path",
            "model_name": "test-model",
            "chunk_size": 600,
            "chunk_overlap": 200,
            "batch_size": 96,
            "quantization": "float32",
        }

        # Make request - should fail with 500 because no files found
        response = test_client.post("/api/jobs", json=request_data, headers={})

        # Assert response
        assert response.status_code == 500  # HTTPException is wrapped in 500
        assert "No supported files found" in response.json()["detail"]


class TestJobManagement:
    """Test job management endpoints"""

    def test_list_jobs(self, test_client_with_mocks: TestClient, mock_job_repository):
        """Test listing jobs for current user"""
        # Mock repository response
        mock_jobs = [
            {
                "id": "job1",
                "name": "Job 1",
                "status": "completed",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T01:00:00",
                "total_files": 10,
                "processed_files": 10,
                "failed_files": 0,
                "model_name": "model1",
                "directory_path": "/path1",
                "quantization": "float32",
                "batch_size": 96,
                "chunk_size": 600,
                "chunk_overlap": 200,
            },
            {
                "id": "job2",
                "name": "Job 2",
                "status": "processing",
                "created_at": "2024-01-02T00:00:00",
                "updated_at": "2024-01-02T00:30:00",
                "total_files": 5,
                "processed_files": 2,
                "failed_files": 0,
                "current_file": "/path2/file.txt",
                "model_name": "model2",
                "directory_path": "/path2",
            },
        ]

        mock_job_repository.list_jobs = AsyncMock(return_value=mock_jobs)

        response = test_client_with_mocks.get("/api/jobs", headers={})

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == "job1"
        assert data[0]["status"] == "completed"
        assert data[1]["id"] == "job2"
        assert data[1]["status"] == "processing"

    def test_get_job_details(self, test_client_with_mocks: TestClient, mock_job_repository):
        """Test getting specific job details"""
        job_id = "test-job-123"
        mock_job = {
            "id": job_id,
            "name": "Test Job",
            "status": "completed",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T01:00:00",
            "total_files": 10,
            "processed_files": 10,
            "failed_files": 0,
            "error": None,
            "current_file": None,
            "model_name": "test-model",
            "directory_path": "/test/path",
            "quantization": "float32",
            "batch_size": 96,
            "chunk_size": 600,
            "chunk_overlap": 200,
        }

        mock_job_repository.get_job = AsyncMock(return_value=mock_job)

        response = test_client_with_mocks.get(f"/api/jobs/{job_id}", headers={})

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert data["name"] == "Test Job"
        assert data["status"] == "completed"

    def test_get_job_not_found(self, test_client_with_mocks: TestClient, mock_job_repository):
        """Test getting non-existent job"""
        mock_job_repository.get_job = AsyncMock(return_value=None)

        response = test_client_with_mocks.get("/api/jobs/non-existent", headers={})

        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]

    def test_cancel_job(self, test_client_with_mocks: TestClient, mock_job_repository):
        """Test cancelling a running job (task revocation pending implementation)"""
        job_id = "running-job"

        # Mock repository methods
        mock_job_repository.get_job = AsyncMock(return_value={"id": job_id, "status": "processing"})
        mock_job_repository.update_job = AsyncMock()

        response = test_client_with_mocks.post(f"/api/jobs/{job_id}/cancel", headers={})

        assert response.status_code == 200
        assert "Job marked as cancelled (task revocation pending implementation)" in response.json()["message"]

        # Verify job status was updated
        mock_job_repository.update_job.assert_called_once_with(job_id, {"status": "cancelled"})

    def test_cancel_job_invalid_status(self, test_client_with_mocks: TestClient, mock_job_repository):
        """Test cancelling a job that's not running"""
        job_id = "completed-job"

        # Mock repository method
        mock_job_repository.get_job = AsyncMock(return_value={"id": job_id, "status": "completed"})

        response = test_client_with_mocks.post(f"/api/jobs/{job_id}/cancel", headers={})

        assert response.status_code == 400
        assert response.json()["detail"] == "Cannot cancel job in status: completed"

    def test_delete_job(self, test_client_with_mocks: TestClient, mock_job_repository, monkeypatch):
        """Test deleting a job"""
        job_id = "job-to-delete"

        # Mock repository methods
        mock_job_repository.get_job = AsyncMock(return_value={"id": job_id, "status": "completed"})
        mock_job_repository.delete_job = AsyncMock(return_value=True)

        # Mock AsyncQdrantClient - delete endpoint creates its own client
        mock_async_qdrant = AsyncMock()
        mock_async_qdrant.delete_collection = AsyncMock()

        # Mock the AsyncQdrantClient constructor
        mock_async_qdrant_class = Mock(return_value=mock_async_qdrant)
        monkeypatch.setattr("webui.api.jobs.AsyncQdrantClient", mock_async_qdrant_class)

        response = test_client_with_mocks.delete(f"/api/jobs/{job_id}", headers={})

        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]

        # Verify AsyncQdrantClient was created and collection was deleted
        mock_async_qdrant_class.assert_called_once_with(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        mock_async_qdrant.delete_collection.assert_called_once_with(f"job_{job_id}")

        # Verify repository delete
        mock_job_repository.delete_job.assert_called_once_with(job_id)


class TestCollectionOperations:
    """Test collection-related operations"""

    def test_check_collection_exists(self, test_client_with_mocks: TestClient, monkeypatch):
        """Test checking if a job's collection exists"""
        job_id = "test-job"
        collection_name = f"job_{job_id}"

        # Mock qdrant client
        mock_collection = Mock()
        mock_collection.name = collection_name  # Set the .name attribute explicitly
        mock_collections = Mock(collections=[mock_collection])

        mock_qdrant_client = Mock()
        mock_qdrant_client.get_collections.return_value = mock_collections

        # Mock collection info with point count
        mock_collection_info = Mock(points_count=100)
        mock_qdrant_client.get_collection.return_value = mock_collection_info

        monkeypatch.setattr("webui.api.jobs.qdrant_manager.get_client", Mock(return_value=mock_qdrant_client))

        response = test_client_with_mocks.get(f"/api/jobs/{job_id}/collection-exists", headers={})

        assert response.status_code == 200
        data = response.json()
        assert data["exists"] is True
        assert data["collection_name"] == collection_name
        assert data["point_count"] == 100

    def test_get_collection_metadata(self, test_client_with_mocks: TestClient, mock_collection_repository):
        """Test getting collection metadata"""
        collection_name = "Test Collection"

        mock_metadata = {
            "id": "job123",
            "name": collection_name,
            "model_name": "test-model",
            "chunk_size": 600,
            "chunk_overlap": 200,
            "batch_size": 96,
            "quantization": "float32",
            "instruction": "Test instruction",
            "vector_dim": 768,
        }

        mock_collection_repository.get_collection_metadata = AsyncMock(return_value=mock_metadata)

        response = test_client_with_mocks.get(f"/api/jobs/collection-metadata/{collection_name}", headers={})

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == collection_name
        assert data["model_name"] == "test-model"
        assert data["vector_dim"] == 768

    def test_get_collection_metadata_not_found(self, test_client_with_mocks: TestClient, mock_collection_repository):
        """Test getting metadata for non-existent collection"""
        mock_collection_repository.get_collection_metadata = AsyncMock(return_value=None)

        response = test_client_with_mocks.get("/api/jobs/collection-metadata/non-existent", headers={})

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestWebSocketOperations:
    """Test WebSocket-related operations"""

    def test_get_new_job_id(self, test_client: TestClient):
        """Test getting a new job ID for WebSocket connection"""
        response = test_client.get("/api/jobs/new-id", headers={})

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert len(data["job_id"]) == 36  # UUID format

    def test_connection_manager_operations(self):
        """Test ConnectionManager class operations"""
        cm = ConnectionManager()
        job_id = "test-job"

        # Mock WebSocket
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()

        # Test connection tracking
        assert job_id not in cm.active_connections

        # Note: These are async methods, so we'd need to test them properly
        # in an async context, but this tests the basic structure exists


class TestValidation:
    """Test request validation"""

    def test_create_job_invalid_chunk_size(self, test_client: TestClient):
        """Test job creation with invalid chunk size"""
        request_data = {
            "name": "Test Job",
            "directory_path": "/test",
            "model_name": "test-model",
            "chunk_size": 50,  # Too small
        }

        response = test_client.post("/api/jobs", json=request_data, headers={})

        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any(
            (
                "chunk_size must be at least 100" in str(error)
                or "Input should be greater than or equal to 100" in str(error)
            )
            for error in errors
        )

    def test_create_job_invalid_chunk_overlap(self, test_client: TestClient):
        """Test job creation with invalid chunk overlap"""
        request_data = {
            "name": "Test Job",
            "directory_path": "/test",
            "model_name": "test-model",
            "chunk_size": 600,
            "chunk_overlap": 700,  # Larger than chunk_size
        }

        response = test_client.post("/api/jobs", json=request_data, headers={})

        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any("chunk_overlap" in str(error) and "must be less than chunk_size" in str(error) for error in errors)
